"""Repositories: the only place that knows SQL.

Each repository owns one table (or one closely-related pair) and maps between
rows and the immutable schema models.  Services never build SQL themselves.

Writes are idempotent wherever an event could be redelivered -- Kafka gives
at-least-once delivery, so a replayed fill must not be double-counted.  That is
why almost every insert carries ``ON CONFLICT ... DO NOTHING`` or ``DO UPDATE``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from libs.persistence.database import Database
from libs.schemas.learning import AgentOutcome, PositionClosed
from libs.schemas.market import Candle, Side
from libs.schemas.portfolio import PortfolioSnapshot, Position
from libs.schemas.signals import AgentSignal, Direction
from libs.schemas.trading import Fill, OrderIntent, OrderType, TradeDecision

logger = logging.getLogger(__name__)


class CandleRepository:
    """Stores and reads OHLCV history."""

    def __init__(self, database: Database) -> None:
        """Bind the repository to a database."""
        self._db = database

    async def upsert_many(self, candles: Sequence[Candle]) -> int:
        """Insert candles, updating any that already exist.

        Args:
            candles: Candles to persist.

        Returns:
            The number of candles submitted.
        """
        if not candles:
            return 0
        query = """
            INSERT INTO candles (
                symbol, "interval", open_time, close_time,
                open, high, low, close, volume, trade_count, is_closed, source
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            ON CONFLICT (symbol, "interval", open_time) DO UPDATE SET
                close_time  = EXCLUDED.close_time,
                open        = EXCLUDED.open,
                high        = EXCLUDED.high,
                low         = EXCLUDED.low,
                close       = EXCLUDED.close,
                volume      = EXCLUDED.volume,
                trade_count = EXCLUDED.trade_count,
                is_closed   = EXCLUDED.is_closed
        """
        await self._db.executemany(
            query,
            [
                (
                    candle.symbol,
                    candle.interval,
                    candle.open_time,
                    candle.close_time,
                    candle.open,
                    candle.high,
                    candle.low,
                    candle.close,
                    candle.volume,
                    candle.trade_count,
                    candle.is_closed,
                    candle.source,
                )
                for candle in candles
            ],
        )
        return len(candles)

    async def load(
        self,
        symbol: str,
        interval: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[Candle, ...]:
        """Load closed candles for a symbol and interval, oldest first.

        Args:
            symbol: Symbol to load.
            interval: Candle interval.
            start: Inclusive lower bound on ``open_time``.
            end: Exclusive upper bound on ``open_time``.
            limit: Maximum candles to return (most recent when set).

        Returns:
            Candles ordered oldest first.
        """
        clauses = ['symbol = $1', '"interval" = $2', "is_closed = TRUE"]
        args: list[Any] = [symbol, interval]
        if start is not None:
            args.append(start)
            clauses.append(f"open_time >= ${len(args)}")
        if end is not None:
            args.append(end)
            clauses.append(f"open_time < ${len(args)}")

        order = "DESC" if limit is not None else "ASC"
        # Only fixed clause fragments and $N placeholders are interpolated; every
        # caller-supplied value travels as a bound parameter in `args`.
        query = (
            f"SELECT * FROM candles WHERE {' AND '.join(clauses)} "  # noqa: S608
            f"ORDER BY open_time {order}"
        )
        if limit is not None:
            args.append(limit)
            query += f" LIMIT ${len(args)}"

        rows = await self._db.fetch(query, *args)
        candles = [self._to_candle(row) for row in rows]
        if limit is not None:
            candles.reverse()
        return tuple(candles)

    async def coverage(self, symbol: str, interval: str) -> tuple[datetime | None, datetime | None]:
        """Return the first and last stored ``open_time`` for a series."""
        row = await self._db.fetchrow(
            'SELECT MIN(open_time) AS first, MAX(open_time) AS last '
            'FROM candles WHERE symbol = $1 AND "interval" = $2',
            symbol,
            interval,
        )
        if row is None:
            return None, None
        return row["first"], row["last"]

    @staticmethod
    def _to_candle(row: Any) -> Candle:
        """Map a database row to a :class:`Candle`."""
        return Candle(
            source=row["source"],
            symbol=row["symbol"],
            interval=row["interval"],
            open_time=row["open_time"],
            close_time=row["close_time"],
            occurred_at=row["open_time"],
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
            trade_count=row["trade_count"],
            is_closed=row["is_closed"],
        )


class SignalRepository:
    """Stores agent signals and decisions, the inputs to attribution."""

    def __init__(self, database: Database) -> None:
        """Bind the repository to a database."""
        self._db = database

    async def record_signal(self, signal: AgentSignal) -> None:
        """Persist one agent signal, ignoring redeliveries."""
        await self._db.execute(
            """
            INSERT INTO agent_signals (
                signal_id, correlation_id, agent_name, agent_version, symbol,
                direction, confidence, degraded, reference_price, rationale,
                features, occurred_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            ON CONFLICT (signal_id) DO NOTHING
            """,
            signal.event_id,
            signal.correlation_id,
            signal.agent_name,
            signal.agent_version,
            signal.symbol,
            signal.direction.value,
            signal.confidence,
            signal.degraded,
            signal.reference_price,
            signal.rationale,
            json.dumps(signal.features),
            signal.occurred_at,
        )

    async def record_decision(self, decision: TradeDecision) -> None:
        """Persist one trade decision, ignoring redeliveries."""
        await self._db.execute(
            """
            INSERT INTO trade_decisions (
                decision_id, correlation_id, symbol, action, confidence,
                weighted_confidence, reference_price, contributing_signals,
                agent_weights, mode, occurred_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ON CONFLICT (decision_id) DO NOTHING
            """,
            decision.decision_id,
            decision.correlation_id,
            decision.symbol,
            decision.action.value,
            decision.confidence,
            decision.weighted_confidence,
            decision.reference_price,
            list(decision.contributing_signals),
            json.dumps(decision.agent_weights),
            decision.mode,
            decision.occurred_at,
        )

    async def signals_for_correlation(self, correlation_id: UUID) -> tuple[AgentSignal, ...]:
        """Return every signal that shares a correlation id."""
        rows = await self._db.fetch(
            "SELECT * FROM agent_signals WHERE correlation_id = $1 ORDER BY occurred_at",
            correlation_id,
        )
        return tuple(self._to_signal(row) for row in rows)

    @staticmethod
    def _to_signal(row: Any) -> AgentSignal:
        """Map a database row to an :class:`AgentSignal`."""
        features = row["features"]
        return AgentSignal(
            event_id=row["signal_id"],
            correlation_id=row["correlation_id"],
            source=row["agent_name"],
            agent_name=row["agent_name"],
            agent_version=row["agent_version"],
            symbol=row["symbol"],
            direction=Direction(row["direction"]),
            confidence=row["confidence"],
            degraded=row["degraded"],
            reference_price=row["reference_price"],
            rationale=row["rationale"],
            features=json.loads(features) if isinstance(features, str) else dict(features or {}),
            occurred_at=row["occurred_at"],
        )


class ExecutionRepository:
    """Stores orders and fills."""

    def __init__(self, database: Database) -> None:
        """Bind the repository to a database."""
        self._db = database

    async def record_order(self, order: OrderIntent) -> None:
        """Persist an order intent, ignoring redeliveries."""
        await self._db.execute(
            """
            INSERT INTO orders (
                order_id, decision_id, correlation_id, symbol, side, size,
                order_type, limit_price, reduce_only, stop_price,
                take_profit_price, trailing_stop_pct, exit_reason, is_paper,
                status, created_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16)
            ON CONFLICT (order_id) DO NOTHING
            """,
            order.order_id,
            order.decision_id,
            order.correlation_id,
            order.symbol,
            order.side.value,
            order.size,
            order.order_type.value,
            order.limit_price,
            order.reduce_only,
            order.stop_price,
            order.take_profit_price,
            order.trailing_stop_pct,
            order.exit_reason,
            order.is_paper,
            "submitted",
            order.occurred_at,
        )

    async def record_fill(self, fill: Fill) -> None:
        """Persist a fill and mark its order filled."""
        await self._db.execute(
            """
            INSERT INTO fills (
                fill_id, order_id, decision_id, correlation_id, symbol, side,
                size, price, fee, slippage_bps, reference_price, reduce_only,
                is_paper, occurred_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
            ON CONFLICT (fill_id, occurred_at) DO NOTHING
            """,
            fill.fill_id,
            fill.order_id,
            fill.decision_id,
            fill.correlation_id,
            fill.symbol,
            fill.side.value,
            fill.size,
            fill.price,
            fill.fee,
            fill.slippage_bps,
            fill.reference_price,
            fill.reduce_only,
            fill.is_paper,
            fill.occurred_at,
        )
        await self._db.execute(
            "UPDATE orders SET status = 'filled' WHERE order_id = $1", fill.order_id
        )

    async def order_for(self, order_id: UUID) -> OrderIntent | None:
        """Load one order intent by id."""
        row = await self._db.fetchrow("SELECT * FROM orders WHERE order_id = $1", order_id)
        if row is None:
            return None
        return OrderIntent(
            source="database",
            order_id=row["order_id"],
            decision_id=row["decision_id"],
            correlation_id=row["correlation_id"],
            symbol=row["symbol"],
            side=Side(row["side"]),
            size=row["size"],
            order_type=OrderType(row["order_type"]),
            limit_price=row["limit_price"],
            reduce_only=row["reduce_only"],
            stop_price=row["stop_price"],
            take_profit_price=row["take_profit_price"],
            trailing_stop_pct=row["trailing_stop_pct"],
            exit_reason=row["exit_reason"],
            is_paper=row["is_paper"],
            occurred_at=row["created_at"],
        )


class PortfolioRepository:
    """Persists portfolio state so it survives a restart.

    ``positions`` holds open positions only, and ``portfolio_state`` is a
    single row holding cash and counters.  Together they are everything needed
    to rebuild the in-memory tracker exactly.
    """

    def __init__(self, database: Database) -> None:
        """Bind the repository to a database."""
        self._db = database

    async def save_position(self, position: Position) -> None:
        """Insert or update one open position."""
        await self._db.execute(
            """
            INSERT INTO positions (
                symbol, direction, size, entry_price, mark_price, realized_pnl,
                fees_paid, stop_price, take_profit_price, trailing_stop_pct,
                opening_correlation_id, opening_decision_id, opened_at, updated_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
            ON CONFLICT (symbol) DO UPDATE SET
                direction              = EXCLUDED.direction,
                size                   = EXCLUDED.size,
                entry_price            = EXCLUDED.entry_price,
                mark_price             = EXCLUDED.mark_price,
                realized_pnl           = EXCLUDED.realized_pnl,
                fees_paid              = EXCLUDED.fees_paid,
                stop_price             = EXCLUDED.stop_price,
                take_profit_price      = EXCLUDED.take_profit_price,
                trailing_stop_pct      = EXCLUDED.trailing_stop_pct,
                opening_correlation_id = EXCLUDED.opening_correlation_id,
                opening_decision_id    = EXCLUDED.opening_decision_id,
                updated_at             = EXCLUDED.updated_at
            """,
            position.symbol,
            position.direction.value,
            position.size,
            position.entry_price,
            position.mark_price,
            position.realized_pnl,
            position.fees_paid,
            position.stop_price,
            position.take_profit_price,
            position.trailing_stop_pct,
            position.opening_correlation_id,
            position.opening_decision_id,
            position.opened_at,
            position.updated_at,
        )

    async def delete_position(self, symbol: str) -> None:
        """Remove a position that has gone flat."""
        await self._db.execute("DELETE FROM positions WHERE symbol = $1", symbol)

    async def load_positions(self) -> tuple[Position, ...]:
        """Load every open position."""
        rows = await self._db.fetch("SELECT * FROM positions ORDER BY symbol")
        return tuple(
            Position(
                source="database",
                symbol=row["symbol"],
                direction=Direction(row["direction"]),
                size=row["size"],
                entry_price=row["entry_price"],
                mark_price=row["mark_price"],
                realized_pnl=row["realized_pnl"],
                fees_paid=row["fees_paid"],
                stop_price=row["stop_price"],
                take_profit_price=row["take_profit_price"],
                trailing_stop_pct=row["trailing_stop_pct"],
                opening_correlation_id=row["opening_correlation_id"],
                opening_decision_id=row["opening_decision_id"],
                opened_at=row["opened_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        )

    async def save_state(
        self,
        *,
        cash: Decimal,
        starting_equity: Decimal,
        day_start_equity: Decimal,
        realized_pnl: Decimal,
        fees_paid: Decimal,
        trade_count: int,
        win_count: int,
        loss_count: int,
        day_of: Any,
        is_paper: bool,
    ) -> None:
        """Write the singleton account-state row."""
        await self._db.execute(
            """
            INSERT INTO portfolio_state (
                id, cash, starting_equity, day_start_equity, realized_pnl,
                fees_paid, trade_count, win_count, loss_count, day_of, is_paper,
                updated_at
            )
            VALUES (1,$1,$2,$3,$4,$5,$6,$7,$8,$9,$10, NOW())
            ON CONFLICT (id) DO UPDATE SET
                cash             = EXCLUDED.cash,
                starting_equity  = EXCLUDED.starting_equity,
                day_start_equity = EXCLUDED.day_start_equity,
                realized_pnl     = EXCLUDED.realized_pnl,
                fees_paid        = EXCLUDED.fees_paid,
                trade_count      = EXCLUDED.trade_count,
                win_count        = EXCLUDED.win_count,
                loss_count       = EXCLUDED.loss_count,
                day_of           = EXCLUDED.day_of,
                is_paper         = EXCLUDED.is_paper,
                updated_at       = NOW()
            """,
            cash,
            starting_equity,
            day_start_equity,
            realized_pnl,
            fees_paid,
            trade_count,
            win_count,
            loss_count,
            day_of,
            is_paper,
        )

    async def load_state(self) -> dict[str, Any] | None:
        """Load the singleton account-state row, if one exists."""
        row = await self._db.fetchrow("SELECT * FROM portfolio_state WHERE id = 1")
        return dict(row) if row is not None else None

    async def record_snapshot(self, snapshot: PortfolioSnapshot) -> None:
        """Append a portfolio snapshot to the time series."""
        await self._db.execute(
            """
            INSERT INTO portfolio_snapshots (
                recorded_at, equity, cash, realized_pnl, unrealized_pnl,
                fees_paid, gross_notional, open_positions, trade_count,
                win_count, loss_count, is_paper
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
            ON CONFLICT (recorded_at) DO NOTHING
            """,
            snapshot.occurred_at,
            snapshot.equity,
            snapshot.cash,
            snapshot.realized_pnl,
            snapshot.unrealized_pnl,
            snapshot.fees_paid,
            snapshot.gross_notional,
            snapshot.open_position_count,
            snapshot.trade_count,
            snapshot.win_count,
            snapshot.loss_count,
            snapshot.is_paper,
        )

    async def record_closure(self, closure: PositionClosed) -> None:
        """Append a closed position to the terminal-state table."""
        await self._db.execute(
            """
            INSERT INTO position_closures (
                closure_id, symbol, direction, size, entry_price, exit_price,
                realized_pnl, fees_paid, return_pct, exit_reason,
                opening_correlation_id, opening_decision_id, opened_at,
                closed_at, holding_period_s, is_paper
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16)
            ON CONFLICT (closure_id) DO NOTHING
            """,
            closure.event_id,
            closure.symbol,
            closure.direction.value,
            closure.size,
            closure.entry_price,
            closure.exit_price,
            closure.realized_pnl,
            closure.fees_paid,
            closure.return_pct,
            closure.exit_reason.value,
            closure.opening_correlation_id,
            closure.opening_decision_id,
            closure.opened_at,
            closure.closed_at,
            closure.holding_period_s,
            closure.is_paper,
        )


class ProtectionRepository:
    """Persists position-monitor state, including the trailing-stop extreme."""

    def __init__(self, database: Database) -> None:
        """Bind the repository to a database."""
        self._db = database

    async def save(self, record: dict[str, Any]) -> None:
        """Insert or update one protection record."""
        await self._db.execute(
            """
            INSERT INTO position_protection (
                symbol, direction, size, entry_price, stop_price,
                take_profit_price, trailing_stop_pct, extreme_price,
                exit_pending, correlation_id, decision_id, opened_at, updated_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
            ON CONFLICT (symbol) DO UPDATE SET
                direction         = EXCLUDED.direction,
                size              = EXCLUDED.size,
                entry_price       = EXCLUDED.entry_price,
                stop_price        = EXCLUDED.stop_price,
                take_profit_price = EXCLUDED.take_profit_price,
                trailing_stop_pct = EXCLUDED.trailing_stop_pct,
                extreme_price     = EXCLUDED.extreme_price,
                exit_pending      = EXCLUDED.exit_pending,
                correlation_id    = EXCLUDED.correlation_id,
                decision_id       = EXCLUDED.decision_id,
                updated_at        = EXCLUDED.updated_at
            """,
            record["symbol"],
            record["direction"],
            record["size"],
            record["entry_price"],
            record["stop_price"],
            record["take_profit_price"],
            record["trailing_stop_pct"],
            record["extreme_price"],
            record["exit_pending"],
            record["correlation_id"],
            record["decision_id"],
            record["opened_at"],
            record["updated_at"],
        )

    async def delete(self, symbol: str) -> None:
        """Remove protection state for a closed position."""
        await self._db.execute("DELETE FROM position_protection WHERE symbol = $1", symbol)

    async def load_all(self) -> tuple[dict[str, Any], ...]:
        """Load every protection record."""
        rows = await self._db.fetch("SELECT * FROM position_protection ORDER BY symbol")
        return tuple(dict(row) for row in rows)


class PerformanceRepository:
    """The Brier-score ledger behind adaptive trust weights."""

    def __init__(self, database: Database) -> None:
        """Bind the repository to a database."""
        self._db = database

    async def record_outcome(self, outcome: AgentOutcome, correlation_id: UUID | None) -> None:
        """Persist one scored forecast.

        The unique index on ``signal_id`` makes this idempotent: a redelivered
        closure cannot score the same signal twice and skew the mean.

        Args:
            outcome: The scored outcome.
            correlation_id: Correlation id of the resolving chain, for joins.
        """
        await self._db.execute(
            """
            INSERT INTO agent_performance (
                outcome_id, agent_name, agent_version, signal_id, correlation_id,
                symbol, direction, confidence, realized_pnl, was_correct,
                brier_score, exit_reason, holding_period_s, closed_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
            ON CONFLICT (signal_id) DO NOTHING
            """,
            outcome.event_id,
            outcome.agent_name,
            outcome.agent_version,
            outcome.signal_id,
            correlation_id,
            outcome.symbol,
            outcome.direction.value,
            outcome.confidence,
            outcome.realized_pnl,
            outcome.was_correct,
            outcome.brier_score,
            outcome.exit_reason.value,
            outcome.holding_period_s,
            outcome.closed_at,
        )

    async def agent_scores(self, limit_per_agent: int = 200) -> dict[str, list[float]]:
        """Load recent Brier scores per agent, newest first.

        Args:
            limit_per_agent: Maximum scores to load for each agent.

        Returns:
            A mapping of agent name to its recent Brier scores.
        """
        rows = await self._db.fetch(
            """
            SELECT agent_name, brier_score FROM (
                SELECT agent_name, brier_score,
                       ROW_NUMBER() OVER (PARTITION BY agent_name ORDER BY closed_at DESC) AS rn
                FROM agent_performance
            ) ranked
            WHERE rn <= $1
            ORDER BY agent_name
            """,
            limit_per_agent,
        )
        scores: dict[str, list[float]] = {}
        for row in rows:
            scores.setdefault(row["agent_name"], []).append(float(row["brier_score"]))
        return scores

    async def summary(self) -> tuple[dict[str, Any], ...]:
        """Return aggregate accuracy per agent for reporting."""
        rows = await self._db.fetch(
            """
            SELECT agent_name,
                   COUNT(*)                                   AS samples,
                   AVG(brier_score)                           AS mean_brier,
                   AVG(CASE WHEN was_correct THEN 1.0 ELSE 0.0 END) AS hit_rate,
                   SUM(realized_pnl)                          AS total_pnl
            FROM agent_performance
            GROUP BY agent_name
            ORDER BY agent_name
            """
        )
        return tuple(dict(row) for row in rows)


__all__ = [
    "CandleRepository",
    "ExecutionRepository",
    "PerformanceRepository",
    "PortfolioRepository",
    "ProtectionRepository",
    "SignalRepository",
]
