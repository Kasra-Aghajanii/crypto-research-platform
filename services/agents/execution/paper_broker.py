"""Paper trading execution agent.

Phase 2, item 6.  Consumes risk-approved ``OrderIntent`` events and simulates
their execution against the **live Hyperliquid order book**, publishing ``Fill``
events.  It never sends anything to the exchange.

Fill model
----------
A market buy walks the ask side of the most recent book snapshot, a market sell
walks the bid side, consuming resting size level by level.  The resulting
volume-weighted price therefore includes real depth-based slippage: a size that
clears three levels prices worse than one that rests inside the top level.  A
configured slippage allowance is then applied on top to stand in for latency and
queue position, and the taker fee is charged on the filled notional.

Safety
------
The broker refuses to start outside paper mode (``TRADING_MODE=paper``), refuses
to fill against a book older than ``max_book_age_s``, and refuses to fill an
order it cannot fully cover with visible depth -- a partial fill is published for
the portion that could be filled rather than inventing liquidity.

Run with::

    python -m services.agents.execution.paper_broker
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from decimal import ROUND_HALF_UP, Decimal
from typing import Final
from uuid import uuid4

from libs.config import Settings
from libs.kafka_client import Topics
from libs.logging_config import configure_logging
from libs.schemas.base import BaseEvent, utc_now
from libs.schemas.market import BookLevel, OrderBookSnapshot, Side
from libs.schemas.trading import Fill, OrderIntent
from services.agents.common.base_agent import BaseAgent, Publication

logger = logging.getLogger(__name__)

SERVICE_NAME: Final[str] = "paper_broker"

_BPS: Final[Decimal] = Decimal(10_000)
_PRICE_QUANTUM: Final[Decimal] = Decimal("0.00000001")
_USD_QUANTUM: Final[Decimal] = Decimal("0.00000001")


class FillRejected(RuntimeError):
    """Raised when a simulated order cannot be filled at all."""


def walk_book(levels: Sequence[BookLevel], size: Decimal) -> tuple[Decimal, Decimal]:
    """Consume ``size`` against ``levels`` and return the fill it produces.

    Args:
        levels: Book levels on the side being taken, best price first.
        size: Requested size in base asset.

    Returns:
        A ``(filled_size, volume_weighted_price)`` tuple. ``filled_size`` is less
        than ``size`` when visible depth runs out.

    Raises:
        FillRejected: If the side is empty or nothing could be filled.
    """
    if not levels:
        raise FillRejected("Book side is empty.")

    remaining = size
    notional = Decimal(0)
    filled = Decimal(0)
    for level in levels:
        if remaining <= 0:
            break
        take = min(remaining, level.size)
        if take <= 0:
            continue
        notional += take * level.price
        filled += take
        remaining -= take

    if filled <= 0:
        raise FillRejected("No resting size available on this side of the book.")
    return filled, notional / filled


class PaperBroker(BaseAgent):
    """Simulates fills for risk-approved orders using live book snapshots.

    Args:
        config: Settings override, mainly for tests.
    """

    name = SERVICE_NAME
    version = "1.0.0"

    def __init__(self, *, config: Settings | None = None) -> None:
        """Initialise the broker and assert paper mode."""
        super().__init__(config=config)
        self.settings.require_paper_mode("PaperBroker")
        self.params = self.settings.paper
        self._books: dict[str, OrderBookSnapshot] = {}
        self.fills_published = 0
        self.orders_rejected = 0

    @property
    def input_topics(self) -> Mapping[str, type[BaseEvent]]:
        """Consume approved orders, and the book snapshots used to price them."""
        return {
            Topics.ORDER_INTENTS: OrderIntent,
            Topics.MARKET_ORDERBOOK: OrderBookSnapshot,
        }

    async def handle(self, topic: str, event: BaseEvent) -> Sequence[Publication]:
        """Cache book snapshots, or simulate one order.

        Args:
            topic: Topic the event arrived on.
            event: The decoded event.

        Returns:
            A single ``(topic, Fill)`` pair, or nothing when the order is rejected.
        """
        if isinstance(event, OrderBookSnapshot):
            self._books[event.symbol] = event
            return ()
        if not isinstance(event, OrderIntent):
            return ()

        try:
            fill = self.simulate_fill(event)
        except FillRejected as exc:
            self.orders_rejected += 1
            logger.warning(
                "Rejected paper order",
                extra={
                    "symbol": event.symbol,
                    "size": str(event.size),
                    "reason": str(exc),
                },
            )
            return ()

        self.fills_published += 1
        logger.info(
            "Simulated fill",
            extra={
                "symbol": fill.symbol,
                "side": fill.side.value,
                "size": str(fill.size),
                "price": str(fill.price),
                "slippage_bps": fill.slippage_bps,
                "fee": str(fill.fee),
            },
        )
        return ((Topics.FILLS, fill.with_correlation(event.correlation_id)),)

    def simulate_fill(self, order: OrderIntent) -> Fill:
        """Simulate the execution of one order against the cached book.

        Args:
            order: The risk-approved order to fill.

        Returns:
            The resulting fill event.

        Raises:
            FillRejected: If there is no fresh book, or no fillable depth.
        """
        book = self._books.get(order.symbol)
        if book is None:
            raise FillRejected(f"No order book cached for {order.symbol}.")

        age_s = (utc_now() - book.occurred_at).total_seconds()
        if age_s > self.params.max_book_age_s:
            raise FillRejected(
                f"Order book for {order.symbol} is stale ({age_s:.1f}s > "
                f"{self.params.max_book_age_s}s)."
            )

        levels = book.asks if order.side is Side.BUY else book.bids
        filled_size, book_price = walk_book(levels, order.size)
        if filled_size < order.size:
            logger.warning(
                "Partial fill: visible depth exhausted",
                extra={
                    "symbol": order.symbol,
                    "requested": str(order.size),
                    "filled": str(filled_size),
                },
            )

        slippage = Decimal(str(self.params.slippage_bps)) / _BPS
        adverse = (1 + slippage) if order.side is Side.BUY else (1 - slippage)
        fill_price = (book_price * adverse).quantize(_PRICE_QUANTUM, rounding=ROUND_HALF_UP)

        reference = book.best_ask if order.side is Side.BUY else book.best_bid
        effective_slippage_bps = 0.0
        if reference is not None and reference > 0:
            drift = (fill_price - reference) / reference * _BPS
            effective_slippage_bps = float(drift if order.side is Side.BUY else -drift)

        notional = filled_size * fill_price
        fee = (notional * Decimal(str(self.params.taker_fee_bps)) / _BPS).quantize(
            _USD_QUANTUM, rounding=ROUND_HALF_UP
        )

        return Fill(
            source=self.name,
            fill_id=uuid4(),
            order_id=order.order_id,
            decision_id=order.decision_id,
            symbol=order.symbol,
            side=order.side,
            size=filled_size,
            price=fill_price,
            fee=fee,
            slippage_bps=round(effective_slippage_bps, 4),
            reference_price=reference,
            is_paper=True,
            reduce_only=order.reduce_only,
        )

    async def on_start(self) -> None:
        """Log the simulation parameters in force."""
        logger.info(
            "Paper broker ready",
            extra={
                "taker_fee_bps": self.params.taker_fee_bps,
                "slippage_bps": self.params.slippage_bps,
                "max_book_age_s": self.params.max_book_age_s,
                "trading_mode": self.settings.trading_mode.value,
            },
        )

    async def on_stop(self) -> None:
        """Log a fill summary on shutdown."""
        logger.info(
            "Paper broker summary",
            extra={"fills": self.fills_published, "rejected": self.orders_rejected},
        )


async def main() -> None:
    """Service entrypoint for ``python -m services.agents.execution.paper_broker``."""
    configure_logging(SERVICE_NAME)
    await PaperBroker.main()


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    asyncio.run(main())


__all__ = ["FillRejected", "PaperBroker", "walk_book"]
