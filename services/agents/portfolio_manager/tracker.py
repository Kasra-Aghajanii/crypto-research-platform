"""Portfolio Manager -- paper equity, PnL and open position tracking.

Phase 2, item 7.  Consumes ``Fill`` events and order book snapshots and
maintains the authoritative portfolio state, publishing a ``PortfolioSnapshot``
whenever it changes.  The risk manager consumes those snapshots, which is what
closes the loop: fills change equity, equity changes what risk will approve next.

Accounting model
----------------
* Positions net per symbol.  Adding in the same direction updates the
  volume-weighted average entry; trading against the position realises PnL on
  the closed portion at the difference between entry and fill price.
* A fill larger than the open position closes it and opens a new one in the
  opposite direction with the remainder.
* Fees are charged to cash immediately and tracked on the position, so realised
  PnL is always net of costs.
* ``equity = cash + unrealised PnL``, and unrealised PnL is marked to the latest
  book mid price.
* ``day_start_equity`` rolls over at the UTC day boundary, which is what the risk
  manager's daily loss limit is measured against.

Run with::

    python -m services.agents.portfolio_manager.tracker
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from datetime import date
from decimal import Decimal
from typing import Any, Final
from uuid import UUID

from libs.config import Settings
from libs.kafka_client import Topics
from libs.logging_config import configure_logging
from libs.persistence import PortfolioRepository, try_connect
from libs.persistence.database import PostgresDatabase
from libs.schemas.base import BaseEvent, utc_now
from libs.schemas.learning import ExitReason, PositionClosed
from libs.schemas.market import OrderBookSnapshot, Side
from libs.schemas.portfolio import PortfolioSnapshot, Position
from libs.schemas.signals import Direction
from libs.schemas.trading import Fill, OrderIntent
from services.agents.common.base_agent import BaseAgent, Publication

logger = logging.getLogger(__name__)

SERVICE_NAME: Final[str] = "portfolio_manager"


class PortfolioTracker:
    """Pure accounting core: applies fills and marks, holds no I/O.

    Keeping the arithmetic free of Kafka makes it directly unit-testable and
    reusable by the future backtester.

    Args:
        starting_equity: Opening cash balance in USD.
        is_paper: Whether this portfolio is simulated.
    """

    def __init__(self, *, starting_equity: Decimal, is_paper: bool = True) -> None:
        """Initialise a flat portfolio."""
        self.starting_equity = starting_equity
        self.cash = starting_equity
        self.day_start_equity = starting_equity
        self.is_paper = is_paper
        self.realized_pnl = Decimal(0)
        self.fees_paid = Decimal(0)
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self._positions: dict[str, Position] = {}
        self._marks: dict[str, Decimal] = {}
        self._day = utc_now().date()

    @property
    def day(self) -> date:
        """Return the UTC day the daily equity baseline belongs to."""
        return self._day

    @property
    def positions(self) -> tuple[Position, ...]:
        """Return open positions, ordered by symbol."""
        return tuple(self._positions[symbol] for symbol in sorted(self._positions))

    @property
    def unrealized_pnl(self) -> Decimal:
        """Return total mark-to-market PnL across open positions."""
        return sum((position.unrealized_pnl for position in self._positions.values()), Decimal(0))

    @property
    def gross_notional(self) -> Decimal:
        """Return the sum of absolute position notionals."""
        return sum((position.notional for position in self._positions.values()), Decimal(0))

    @property
    def equity(self) -> Decimal:
        """Return cash plus unrealised PnL."""
        return self.cash + self.unrealized_pnl

    def mark(self, symbol: str, price: Decimal) -> bool:
        """Update the mark price for a symbol.

        Args:
            symbol: Symbol to mark.
            price: New mark price.

        Returns:
            ``True`` if an open position was revalued.
        """
        if price <= 0:
            return False
        self._marks[symbol] = price
        position = self._positions.get(symbol)
        if position is None:
            return False
        self._positions[symbol] = position.model_copy(
            update={"mark_price": price, "updated_at": utc_now()}
        )
        return True

    def apply_fill(self, fill: Fill, *, order: OrderIntent | None = None) -> PositionClosed | None:
        """Apply a fill to the portfolio.

        Args:
            fill: The executed fill.
            order: The order that produced the fill, when known. An opening
                order carries the protective levels to attach to the position;
                a reduce-only exit order carries the reason it fired.

        Returns:
            A :class:`PositionClosed` event when this fill took a position flat,
            otherwise ``None``. That event is what drives outcome attribution.
        """
        self._roll_day()
        self.trade_count += 1
        self.cash -= fill.fee
        self.fees_paid += fill.fee
        self._marks[fill.symbol] = fill.price

        direction = Direction.LONG if fill.side is Side.BUY else Direction.SHORT
        existing = self._positions.get(fill.symbol)

        if existing is None:
            self._open(fill, direction, order)
            return None
        if existing.direction is direction:
            self._add(existing, fill)
            return None
        return self._reduce_or_flip(existing, fill, direction, order)

    def _open(self, fill: Fill, direction: Direction, order: OrderIntent | None = None) -> None:
        """Open a brand new position from a fill.

        Protective levels come from the opening order, and the correlation id
        comes from the fill: together they let a later closure be attributed
        back to the signal that caused it.
        """
        now = utc_now()
        self._positions[fill.symbol] = Position(
            source=SERVICE_NAME,
            symbol=fill.symbol,
            direction=direction,
            size=fill.size,
            entry_price=fill.price,
            mark_price=fill.price,
            fees_paid=fill.fee,
            opened_at=now,
            updated_at=now,
            stop_price=order.stop_price if order is not None else None,
            take_profit_price=order.take_profit_price if order is not None else None,
            trailing_stop_pct=order.trailing_stop_pct if order is not None else None,
            opening_correlation_id=fill.correlation_id,
            opening_decision_id=fill.decision_id,
        )

    def _add(self, existing: Position, fill: Fill) -> None:
        """Add to a position, updating the volume-weighted average entry."""
        total_size = existing.size + fill.size
        entry = (existing.entry_price * existing.size + fill.price * fill.size) / total_size
        self._positions[fill.symbol] = existing.model_copy(
            update={
                "size": total_size,
                "entry_price": entry,
                "mark_price": fill.price,
                "fees_paid": existing.fees_paid + fill.fee,
                "updated_at": utc_now(),
            }
        )

    def _reduce_or_flip(
        self,
        existing: Position,
        fill: Fill,
        direction: Direction,
        order: OrderIntent | None = None,
    ) -> PositionClosed | None:
        """Close all or part of a position, flipping it if the fill is larger.

        Returns:
            A :class:`PositionClosed` event when the position went flat.
        """
        closed_size = min(existing.size, fill.size)
        pnl = (fill.price - existing.entry_price) * closed_size * Decimal(existing.direction.sign)
        self.realized_pnl += pnl
        self.cash += pnl
        if pnl > 0:
            self.win_count += 1
        elif pnl < 0:
            self.loss_count += 1

        logger.info(
            "Realised PnL",
            extra={
                "symbol": fill.symbol,
                "closed_size": str(closed_size),
                "entry": str(existing.entry_price),
                "exit": str(fill.price),
                "pnl": str(pnl),
            },
        )

        remainder = fill.size - closed_size
        if existing.size > closed_size:
            self._positions[fill.symbol] = existing.model_copy(
                update={
                    "size": existing.size - closed_size,
                    "mark_price": fill.price,
                    "realized_pnl": existing.realized_pnl + pnl,
                    "fees_paid": existing.fees_paid + fill.fee,
                    "updated_at": utc_now(),
                }
            )
            return None

        del self._positions[fill.symbol]
        closure = self._build_closure(existing, fill, closed_size, pnl, order)
        if remainder > 0:
            now = utc_now()
            self._positions[fill.symbol] = Position(
                source=SERVICE_NAME,
                symbol=fill.symbol,
                direction=direction,
                size=remainder,
                entry_price=fill.price,
                mark_price=fill.price,
                fees_paid=fill.fee,
                opened_at=now,
                updated_at=now,
                opening_correlation_id=fill.correlation_id,
                opening_decision_id=fill.decision_id,
            )
        return closure

    @staticmethod
    def _build_closure(
        position: Position,
        fill: Fill,
        closed_size: Decimal,
        pnl: Decimal,
        order: OrderIntent | None,
    ) -> PositionClosed:
        """Build the closure event for a position that has gone flat.

        The ``opening_correlation_id`` is taken from the *position*, not from the
        closing fill: attribution has to reach the signal that opened the trade,
        not the exit that ended it.
        """
        reason = ExitReason.UNKNOWN
        if order is not None and order.exit_reason:
            try:
                reason = ExitReason(order.exit_reason)
            except ValueError:
                reason = ExitReason.UNKNOWN
        elif order is not None and not order.reduce_only:
            reason = ExitReason.SIGNAL_REVERSAL

        return PositionClosed(
            source=SERVICE_NAME,
            symbol=position.symbol,
            direction=position.direction,
            size=closed_size,
            entry_price=position.entry_price,
            exit_price=fill.price,
            realized_pnl=pnl,
            fees_paid=position.fees_paid + fill.fee,
            exit_reason=reason,
            opened_at=position.opened_at,
            closed_at=utc_now(),
            opening_correlation_id=position.opening_correlation_id,
            opening_decision_id=position.opening_decision_id,
            correlation_id=position.opening_correlation_id or fill.correlation_id,
        )

    def _roll_day(self) -> None:
        """Reset the daily equity baseline when the UTC day changes."""
        today: date = utc_now().date()
        if today != self._day:
            self._day = today
            self.day_start_equity = self.equity
            logger.info(
                "Rolled daily equity baseline",
                extra={"day": today.isoformat(), "day_start_equity": str(self.equity)},
            )

    def restore(
        self, *, state: Mapping[str, Any] | None, positions: Sequence[Position]
    ) -> None:
        """Rebuild the tracker from persisted state.

        Called on startup so that a restart does not lose open positions or
        reset cash.  Positions are restored as they were last marked; the next
        book snapshot re-marks them.

        Args:
            state: The persisted account row, or ``None`` for a fresh account.
            positions: Persisted open positions.
        """
        if state is not None:
            self.cash = Decimal(state["cash"])
            self.starting_equity = Decimal(state["starting_equity"])
            self.day_start_equity = Decimal(state["day_start_equity"])
            self.realized_pnl = Decimal(state["realized_pnl"])
            self.fees_paid = Decimal(state["fees_paid"])
            self.trade_count = int(state["trade_count"])
            self.win_count = int(state["win_count"])
            self.loss_count = int(state["loss_count"])
            self._day = state["day_of"]

        self._positions = {position.symbol: position for position in positions}
        self._marks = {position.symbol: position.mark_price for position in positions}
        logger.info(
            "Restored portfolio state",
            extra={
                "cash": str(self.cash),
                "equity": str(self.equity),
                "open_positions": len(self._positions),
                "symbols": list(self._positions),
            },
        )

    def snapshot(self) -> PortfolioSnapshot:
        """Build an immutable snapshot of the current portfolio state."""
        self._roll_day()
        return PortfolioSnapshot(
            source=SERVICE_NAME,
            equity=self.equity,
            cash=self.cash,
            starting_equity=self.starting_equity,
            day_start_equity=self.day_start_equity,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=self.unrealized_pnl,
            fees_paid=self.fees_paid,
            gross_notional=self.gross_notional,
            positions=self.positions,
            trade_count=self.trade_count,
            win_count=self.win_count,
            loss_count=self.loss_count,
            is_paper=self.is_paper,
        )


class PortfolioManagerAgent(BaseAgent):
    """Kafka service wrapping :class:`PortfolioTracker`.

    Publishes a snapshot on every fill, and on a mark that revalues an open
    position (throttled, so a fast book does not flood the topic).

    Args:
        config: Settings override, mainly for tests.
        mark_publish_interval_s: Minimum seconds between mark-driven snapshots.
    """

    name = SERVICE_NAME
    version = "1.0.0"

    def __init__(
        self,
        *,
        config: Settings | None = None,
        mark_publish_interval_s: float = 5.0,
        repository: PortfolioRepository | None = None,
    ) -> None:
        """Initialise the agent and its accounting core."""
        super().__init__(config=config)
        self.tracker = PortfolioTracker(
            starting_equity=Decimal(str(self.settings.paper.starting_equity)),
            is_paper=self.settings.is_paper,
        )
        self._mark_interval = max(0.0, mark_publish_interval_s)
        self._last_mark_publish = 0.0
        self._orders: dict[UUID, OrderIntent] = {}
        self._repository = repository
        self._database: PostgresDatabase | None = None
        self.snapshots_published = 0
        self.closures_published = 0

    @property
    def input_topics(self) -> Mapping[str, type[BaseEvent]]:
        """Consume orders, fills, and book snapshots for mark-to-market."""
        return {
            Topics.ORDER_INTENTS: OrderIntent,
            Topics.FILLS: Fill,
            Topics.MARKET_ORDERBOOK: OrderBookSnapshot,
        }

    async def handle(self, topic: str, event: BaseEvent) -> Sequence[Publication]:
        """Apply a fill or a mark and publish a snapshot when state changed.

        Args:
            topic: Topic the event arrived on.
            event: The decoded event.

        Returns:
            The snapshot to publish, preceded by a closure event when a
            position went flat.
        """
        if isinstance(event, OrderIntent):
            # Cached so the fill can inherit its protective levels and exit reason.
            self._orders[event.order_id] = event
            return ()

        if isinstance(event, Fill):
            order = self._orders.pop(event.order_id, None)
            closure = self.tracker.apply_fill(event, order=order)
            await self._persist_after_fill(event, closure)

            publications: list[Publication] = []
            if closure is not None:
                self.closures_published += 1
                publications.append((Topics.POSITION_CLOSURES, closure))
                logger.info(
                    "Position closed",
                    extra={
                        "symbol": closure.symbol,
                        "exit_reason": closure.exit_reason.value,
                        "net_pnl": str(closure.net_pnl),
                        "return_pct": round(closure.return_pct, 4),
                        "holding_period_s": round(closure.holding_period_s, 1),
                    },
                )
            publications.extend(self._publish_snapshot(event.symbol))
            return publications

        if isinstance(event, OrderBookSnapshot):
            mid = event.mid_price
            if mid is None or not self.tracker.mark(event.symbol, mid):
                return ()
            now = asyncio.get_running_loop().time()
            if self._mark_interval > 0 and (now - self._last_mark_publish) < self._mark_interval:
                return ()
            self._last_mark_publish = now
            return self._publish_snapshot(event.symbol)

        return ()

    async def _persist_after_fill(self, fill: Fill, closure: PositionClosed | None) -> None:
        """Write position, account and closure state after a fill.

        Args:
            fill: The fill just applied.
            closure: The closure it produced, if any.
        """
        if self._repository is None:
            return
        position = self.tracker.snapshot().position_for(fill.symbol)
        if position is not None:
            await self._repository.save_position(position)
        else:
            await self._repository.delete_position(fill.symbol)
        if closure is not None:
            await self._repository.record_closure(closure)
        await self._save_state()

    async def _save_state(self) -> None:
        """Write the singleton account row."""
        if self._repository is None:
            return
        tracker = self.tracker
        await self._repository.save_state(
            cash=tracker.cash,
            starting_equity=tracker.starting_equity,
            day_start_equity=tracker.day_start_equity,
            realized_pnl=tracker.realized_pnl,
            fees_paid=tracker.fees_paid,
            trade_count=tracker.trade_count,
            win_count=tracker.win_count,
            loss_count=tracker.loss_count,
            day_of=tracker.day,
            is_paper=tracker.is_paper,
        )

    def _publish_snapshot(self, symbol: str) -> Sequence[Publication]:
        """Build, log and return the snapshot publication."""
        snapshot = self.tracker.snapshot()
        self.snapshots_published += 1
        logger.info(
            "Portfolio updated",
            extra={
                "symbol": symbol,
                "equity": str(snapshot.equity),
                "realized_pnl": str(snapshot.realized_pnl),
                "unrealized_pnl": str(snapshot.unrealized_pnl),
                "open_positions": snapshot.open_position_count,
                "day_pnl_pct": round(snapshot.day_pnl_pct, 4),
                "leverage": round(snapshot.leverage, 4),
            },
        )
        return ((Topics.PORTFOLIO_SNAPSHOTS, snapshot),)

    async def on_start(self) -> None:
        """Rebuild state from the database, then publish an opening snapshot.

        Risk sizing depends on knowing current equity and open positions, so the
        opening snapshot is published only after the restore -- otherwise a
        restart would briefly tell the risk manager the account is flat.
        """
        if self._repository is None and self.settings.persistence_enabled:
            self._database = await try_connect()
            if self._database is not None:
                self._repository = PortfolioRepository(self._database)

        if self._repository is not None:
            state = await self._repository.load_state()
            positions = await self._repository.load_positions()
            self.tracker.restore(state=state, positions=positions)
            await self._save_state()
        else:
            logger.warning("Portfolio manager running without persistence")

        await self.publish(self._publish_snapshot("*"), key="portfolio")

    async def on_stop(self) -> None:
        """Persist final state, close the database and log statistics."""
        await self._save_state()
        if self._database is not None:
            await self._database.close()
            self._database = None
        snapshot = self.tracker.snapshot()
        logger.info(
            "Portfolio summary",
            extra={
                "equity": str(snapshot.equity),
                "total_return_pct": round(snapshot.total_return_pct, 4),
                "trades": snapshot.trade_count,
                "win_rate": round(snapshot.win_rate, 4),
                "fees_paid": str(snapshot.fees_paid),
            },
        )


async def main() -> None:
    """Service entrypoint for ``python -m services.agents.portfolio_manager.tracker``."""
    configure_logging(SERVICE_NAME)
    await PortfolioManagerAgent.main()


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    asyncio.run(main())


__all__ = ["PortfolioManagerAgent", "PortfolioTracker"]
