"""Position Monitor -- independent stop-loss, take-profit and trailing exits.

Phase 3, item 1.  This service is what makes a position able to *close*.  Phase 2
attached stops and take-profits to orders but nothing watched them, so a position
could only be closed by the analyst flipping direction.

The monitor runs independently of the analyst and of the decision engine.  It
watches every price tick and, the moment a level is breached, fires a reduce-only
market order straight to the execution layer.  No new signal is required, and no
agent is consulted -- an exit must not wait on analysis.

How it learns about a position
------------------------------
Two events, in order:

1. ``OrderIntent`` (opening) tells it the protective levels risk approved.
2. ``Fill`` tells it the real entry price and size.

It then tracks the position itself rather than trusting periodic snapshots, so a
gap between snapshots cannot leave a position unguarded.  ``PortfolioSnapshot``
is still consumed, but only to *reconcile*: a position the portfolio no longer
holds is dropped, and a position the monitor has never seen is adopted with
whatever protection the snapshot carries.

Trailing stops
--------------
A trailing stop ratchets and never loosens.  The monitor records the best price
seen since entry (``extreme_price``) and derives the stop from it:

* long:  ``stop = extreme * (1 - trailing_pct/100)``, moving up only
* short: ``stop = extreme * (1 + trailing_pct/100)``, moving down only

The effective stop is always the *tighter* of the fixed stop and the trailing
stop, so enabling a trail can only reduce risk. The extreme is persisted, so a
restart does not hand back profit by resetting the ratchet.

Exit ordering
-------------
When both a stop and a take-profit would trigger on the same tick -- possible on
a candle-driven tick or a fast move -- the **stop wins**.  Assuming the
favourable fill in an ambiguous bar is how backtests flatter themselves.

Run with::

    python -m services.execution.position_monitor
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Final
from uuid import UUID, uuid4

from libs.config import Settings
from libs.kafka_client import Topics
from libs.logging_config import configure_logging
from libs.persistence import ProtectionRepository, try_connect
from libs.persistence.database import PostgresDatabase
from libs.schemas.base import BaseEvent, utc_now
from libs.schemas.learning import ExitReason
from libs.schemas.market import OrderBookSnapshot, Side, TradeTick
from libs.schemas.portfolio import PortfolioSnapshot, Position
from libs.schemas.signals import Direction
from libs.schemas.trading import Fill, OrderIntent, OrderType
from services.agents.common.base_agent import BaseAgent, Publication

logger = logging.getLogger(__name__)

SERVICE_NAME: Final[str] = "position_monitor"

_HUNDRED: Final[Decimal] = Decimal(100)


@dataclass(slots=True)
class ProtectedPosition:
    """A position the monitor is guarding.

    Attributes:
        extreme_price: Best price seen since entry -- the high-water mark for a
            long, the low-water mark for a short. Drives the trailing stop.
        exit_pending: Set once an exit order has been fired, so a breach that
            persists across several ticks cannot fire duplicate exits.
    """

    symbol: str
    direction: Direction
    size: Decimal
    entry_price: Decimal
    stop_price: Decimal | None = None
    take_profit_price: Decimal | None = None
    trailing_stop_pct: float | None = None
    extreme_price: Decimal | None = None
    exit_pending: bool = False
    correlation_id: UUID | None = None
    decision_id: UUID | None = None
    opened_at: Any = field(default_factory=utc_now)
    updated_at: Any = field(default_factory=utc_now)

    def observe(self, price: Decimal) -> None:
        """Update the high/low-water mark from a new price.

        Args:
            price: Latest observed price.
        """
        if price <= 0:
            return
        if self.extreme_price is None:
            self.extreme_price = price
        elif self.direction is Direction.LONG:
            self.extreme_price = max(self.extreme_price, price)
        else:
            self.extreme_price = min(self.extreme_price, price)

    @property
    def trailing_stop(self) -> Decimal | None:
        """Return the trailing stop derived from the extreme price, if enabled."""
        if self.trailing_stop_pct is None or self.extreme_price is None:
            return None
        distance = Decimal(str(self.trailing_stop_pct)) / _HUNDRED
        if self.direction is Direction.LONG:
            return self.extreme_price * (Decimal(1) - distance)
        return self.extreme_price * (Decimal(1) + distance)

    @property
    def effective_stop(self) -> Decimal | None:
        """Return the tighter of the fixed and trailing stops.

        Tighter means closer to price on the losing side: the higher stop for a
        long, the lower stop for a short.  A trail can therefore only ever
        reduce risk relative to the fixed stop.
        """
        fixed = self.stop_price
        trailing = self.trailing_stop
        if fixed is None:
            return trailing
        if trailing is None:
            return fixed
        return max(fixed, trailing) if self.direction is Direction.LONG else min(fixed, trailing)

    def breach(self, price: Decimal) -> ExitReason | None:
        """Return the exit that ``price`` triggers, if any.

        The stop is checked before the take-profit: when a single tick could
        satisfy both, the unfavourable outcome is assumed.

        Args:
            price: Latest observed price.

        Returns:
            The triggered :class:`ExitReason`, or ``None``.
        """
        stop = self.effective_stop
        if stop is not None:
            hit = price <= stop if self.direction is Direction.LONG else price >= stop
            if hit:
                trailing = self.trailing_stop
                is_trailing = (
                    trailing is not None
                    and self.trailing_stop_pct is not None
                    and (
                        self.stop_price is None
                        or (
                            trailing > self.stop_price
                            if self.direction is Direction.LONG
                            else trailing < self.stop_price
                        )
                    )
                )
                return ExitReason.TRAILING_STOP if is_trailing else ExitReason.STOP_LOSS

        target = self.take_profit_price
        if target is not None:
            hit = price >= target if self.direction is Direction.LONG else price <= target
            if hit:
                return ExitReason.TAKE_PROFIT
        return None

    def to_row(self) -> dict[str, Any]:
        """Render the record for :class:`ProtectionRepository`."""
        return {
            "symbol": self.symbol,
            "direction": self.direction.value,
            "size": self.size,
            "entry_price": self.entry_price,
            "stop_price": self.stop_price,
            "take_profit_price": self.take_profit_price,
            "trailing_stop_pct": self.trailing_stop_pct,
            "extreme_price": self.extreme_price,
            "exit_pending": self.exit_pending,
            "correlation_id": self.correlation_id,
            "decision_id": self.decision_id,
            "opened_at": self.opened_at,
            "updated_at": utc_now(),
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> ProtectedPosition:
        """Rebuild a record loaded from the database."""
        return cls(
            symbol=row["symbol"],
            direction=Direction(row["direction"]),
            size=row["size"],
            entry_price=row["entry_price"],
            stop_price=row["stop_price"],
            take_profit_price=row["take_profit_price"],
            trailing_stop_pct=row["trailing_stop_pct"],
            extreme_price=row["extreme_price"],
            exit_pending=row["exit_pending"],
            correlation_id=row["correlation_id"],
            decision_id=row["decision_id"],
            opened_at=row["opened_at"],
            updated_at=row["updated_at"],
        )

    @classmethod
    def from_position(cls, position: Position) -> ProtectedPosition:
        """Adopt a position reported by the portfolio manager."""
        return cls(
            symbol=position.symbol,
            direction=position.direction,
            size=position.size,
            entry_price=position.entry_price,
            stop_price=position.stop_price,
            take_profit_price=position.take_profit_price,
            trailing_stop_pct=position.trailing_stop_pct,
            extreme_price=position.mark_price,
            correlation_id=position.opening_correlation_id,
            decision_id=position.opening_decision_id,
            opened_at=position.opened_at,
        )


class PositionMonitor(BaseAgent):
    """Watches open positions and fires exits when protective levels break.

    Args:
        config: Settings override, mainly for tests.
        repository: Protection repository override; ``None`` disables persistence.
    """

    name = SERVICE_NAME
    version = "1.0.0"

    def __init__(
        self,
        *,
        config: Settings | None = None,
        repository: ProtectionRepository | None = None,
    ) -> None:
        """Initialise the monitor with no tracked positions."""
        super().__init__(config=config)
        self.params = self.settings.monitor
        self._positions: dict[str, ProtectedPosition] = {}
        self._pending_orders: dict[UUID, OrderIntent] = {}
        self._repository = repository
        self._database: PostgresDatabase | None = None
        self.exits_fired = 0
        self.ticks_processed = 0

    @property
    def input_topics(self) -> Mapping[str, type[BaseEvent]]:
        """Consume orders, fills, price ticks and portfolio reconciliation."""
        return {
            Topics.ORDER_INTENTS: OrderIntent,
            Topics.FILLS: Fill,
            Topics.MARKET_TRADES: TradeTick,
            Topics.MARKET_ORDERBOOK: OrderBookSnapshot,
            Topics.PORTFOLIO_SNAPSHOTS: PortfolioSnapshot,
        }

    @property
    def tracked(self) -> Mapping[str, ProtectedPosition]:
        """Return the positions currently being guarded."""
        return self._positions

    async def on_start(self) -> None:
        """Connect persistence and rebuild trailing state from the database."""
        if self._repository is None and self.settings.persistence_enabled:
            self._database = await try_connect()
            if self._database is not None:
                self._repository = ProtectionRepository(self._database)
        if self._repository is None:
            logger.warning("Position monitor running without persistence")
            return
        for row in await self._repository.load_all():
            record = ProtectedPosition.from_row(row)
            self._positions[record.symbol] = record
        logger.info(
            "Rebuilt protection state",
            extra={"positions": len(self._positions), "symbols": list(self._positions)},
        )

    async def on_stop(self) -> None:
        """Close the database and log a summary."""
        if self._database is not None:
            await self._database.close()
            self._database = None
        logger.info(
            "Position monitor summary",
            extra={"exits_fired": self.exits_fired, "ticks": self.ticks_processed},
        )

    async def _persist(self, record: ProtectedPosition) -> None:
        """Write one protection record, if persistence is enabled."""
        if self._repository is not None:
            await self._repository.save(record.to_row())

    async def _forget(self, symbol: str) -> None:
        """Drop a position and its persisted protection state."""
        self._positions.pop(symbol, None)
        if self._repository is not None:
            await self._repository.delete(symbol)

    async def handle(self, topic: str, event: BaseEvent) -> Sequence[Publication]:
        """Route one inbound event.

        Args:
            topic: Topic the event arrived on.
            event: The decoded event.

        Returns:
            Reduce-only exit orders to publish, if any level broke.
        """
        if isinstance(event, OrderIntent):
            self._remember_order(event)
            return ()
        if isinstance(event, Fill):
            await self._apply_fill(event)
            return ()
        if isinstance(event, PortfolioSnapshot):
            await self._reconcile(event)
            return ()
        if isinstance(event, TradeTick):
            return await self.on_price(event.symbol, event.price)
        if isinstance(event, OrderBookSnapshot):
            mid = event.mid_price
            if mid is None:
                return ()
            return await self.on_price(event.symbol, mid)
        return ()

    def _remember_order(self, order: OrderIntent) -> None:
        """Cache an opening order so its levels can be attached to the fill."""
        if order.reduce_only:
            return
        self._pending_orders[order.order_id] = order

    async def _apply_fill(self, fill: Fill) -> None:
        """Open, extend or close a tracked position from a fill.

        Args:
            fill: The executed fill.
        """
        existing = self._positions.get(fill.symbol)
        fill_direction = Direction.LONG if fill.side is Side.BUY else Direction.SHORT

        if existing is not None and existing.direction is not fill_direction:
            remaining = existing.size - fill.size
            if remaining <= 0:
                await self._forget(fill.symbol)
                logger.info(
                    "Position closed; protection released", extra={"symbol": fill.symbol}
                )
                return
            existing.size = remaining
            existing.exit_pending = False
            await self._persist(existing)
            return

        order = self._pending_orders.pop(fill.order_id, None)
        if existing is not None:
            existing.size += fill.size
            existing.updated_at = utc_now()
            await self._persist(existing)
            return

        if order is None:
            logger.debug(
                "Fill without a known opening order; awaiting portfolio reconciliation",
                extra={"symbol": fill.symbol, "order_id": str(fill.order_id)},
            )
            return

        record = ProtectedPosition(
            symbol=fill.symbol,
            direction=fill_direction,
            size=fill.size,
            entry_price=fill.price,
            stop_price=order.stop_price,
            take_profit_price=order.take_profit_price,
            trailing_stop_pct=order.trailing_stop_pct or self.params.default_trailing_stop_pct,
            extreme_price=fill.price,
            correlation_id=fill.correlation_id,
            decision_id=fill.decision_id,
        )
        self._positions[fill.symbol] = record
        await self._persist(record)
        logger.info(
            "Guarding new position",
            extra={
                "symbol": record.symbol,
                "direction": record.direction.value,
                "entry": str(record.entry_price),
                "stop": str(record.stop_price),
                "take_profit": str(record.take_profit_price),
                "trailing_pct": record.trailing_stop_pct,
            },
        )

    async def _reconcile(self, snapshot: PortfolioSnapshot) -> None:
        """Align tracked positions with the portfolio's authoritative view.

        Args:
            snapshot: Latest portfolio snapshot.
        """
        live = {position.symbol: position for position in snapshot.positions}

        for symbol in list(self._positions):
            if symbol not in live:
                await self._forget(symbol)
                logger.info("Dropped protection for closed position", extra={"symbol": symbol})

        for symbol, position in live.items():
            if symbol in self._positions:
                continue
            adopted = ProtectedPosition.from_position(position)
            if adopted.trailing_stop_pct is None:
                adopted.trailing_stop_pct = self.params.default_trailing_stop_pct
            self._positions[symbol] = adopted
            await self._persist(adopted)
            logger.warning(
                "Adopted an unguarded position from the portfolio snapshot",
                extra={"symbol": symbol, "stop": str(adopted.stop_price)},
            )

    async def on_price(self, symbol: str, price: Decimal) -> Sequence[Publication]:
        """Apply one price tick to the tracked position for ``symbol``.

        Args:
            symbol: Symbol the tick belongs to.
            price: Observed price.

        Returns:
            A reduce-only exit order if a level broke, otherwise nothing.
        """
        record = self._positions.get(symbol)
        if record is None or price <= 0:
            return ()

        self.ticks_processed += 1
        previous_stop = record.effective_stop
        record.observe(price)

        if record.exit_pending:
            return ()

        reason = record.breach(price)
        if reason is None:
            if record.effective_stop != previous_stop:
                await self._persist(record)
            return ()

        record.exit_pending = True
        await self._persist(record)
        self.exits_fired += 1

        order = OrderIntent(
            source=self.name,
            order_id=uuid4(),
            decision_id=record.decision_id or uuid4(),
            correlation_id=record.correlation_id or uuid4(),
            symbol=symbol,
            side=Side.SELL if record.direction is Direction.LONG else Side.BUY,
            size=record.size,
            order_type=OrderType.MARKET,
            reduce_only=True,
            exit_reason=reason.value,
            is_paper=self.settings.is_paper,
        )
        logger.info(
            "Exit triggered",
            extra={
                "symbol": symbol,
                "reason": reason.value,
                "price": str(price),
                "stop": str(record.effective_stop),
                "take_profit": str(record.take_profit_price),
                "size": str(record.size),
            },
        )
        return ((Topics.ORDER_INTENTS, order),)


async def main() -> None:
    """Service entrypoint for ``python -m services.execution.position_monitor``."""
    configure_logging(SERVICE_NAME)
    await PositionMonitor.main()


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    asyncio.run(main())


__all__ = ["PositionMonitor", "ProtectedPosition"]
