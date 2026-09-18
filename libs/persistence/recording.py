"""Repositories for the market-data recording tables.

These three series are what Phase 5 exists to capture.  Order book depth and
open interest have no historical endpoint on Hyperliquid, so every row written
here is data that could not have been obtained any other way.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from typing import Any

from libs.persistence.database import Database
from libs.schemas.market import FundingRate, OrderBookSnapshot, PerpMetrics

logger = logging.getLogger(__name__)

_IMBALANCE_DEPTHS = (1, 5, 10, 20)


def _levels_json(levels: Sequence[Any], limit: int) -> str:
    """Serialise book levels to compact JSON for storage.

    Args:
        levels: Book levels, best first.
        limit: Maximum levels to retain.

    Returns:
        A JSON array of ``[price, size, order_count]`` triples.
    """
    return json.dumps(
        [[str(level.price), str(level.size), level.order_count] for level in levels[:limit]]
    )


class OrderBookRepository:
    """Persists order book snapshots and the metrics derived from them."""

    def __init__(self, database: Database, *, stored_levels: int = 10) -> None:
        """Bind the repository to a database.

        Args:
            database: Connection to write through.
            stored_levels: Raw levels retained per side, so a later change to the
                imbalance definition can be applied to recorded history.
        """
        self._db = database
        self._stored_levels = stored_levels

    async def record(self, snapshot: OrderBookSnapshot) -> None:
        """Write one snapshot, ignoring an exact-timestamp redelivery."""
        await self._db.execute(*self._statement(snapshot))

    async def record_many(self, snapshots: Sequence[OrderBookSnapshot]) -> int:
        """Write a batch of snapshots in one round trip.

        Args:
            snapshots: Snapshots to persist.

        Returns:
            The number submitted.
        """
        if not snapshots:
            return 0
        query = self._statement(snapshots[0])[0]
        await self._db.executemany(
            query, [self._statement(snapshot)[1:] for snapshot in snapshots]
        )
        return len(snapshots)

    def _statement(self, snapshot: OrderBookSnapshot) -> tuple[Any, ...]:
        """Build the insert statement and its arguments for one snapshot."""
        imbalances = [snapshot.imbalance(depth=depth) for depth in _IMBALANCE_DEPTHS]
        spread = snapshot.spread_bps
        query = """
            INSERT INTO orderbook_snapshots (
                symbol, recorded_at, best_bid, best_ask, mid_price, spread_bps,
                imbalance_1, imbalance_5, imbalance_10, imbalance_20,
                bid_size_total, ask_size_total, bid_levels, ask_levels, source
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13::jsonb,$14::jsonb,$15)
            ON CONFLICT (symbol, recorded_at) DO NOTHING
        """
        return (
            query,
            snapshot.symbol,
            snapshot.occurred_at,
            snapshot.best_bid,
            snapshot.best_ask,
            snapshot.mid_price,
            float(spread) if spread is not None else None,
            *imbalances,
            sum((level.size for level in snapshot.bids), Decimal(0)),
            sum((level.size for level in snapshot.asks), Decimal(0)),
            _levels_json(snapshot.bids, self._stored_levels),
            _levels_json(snapshot.asks, self._stored_levels),
            snapshot.source,
        )

    async def coverage(self, symbol: str) -> tuple[datetime | None, datetime | None, int]:
        """Return the recorded span and row count for a symbol.

        Args:
            symbol: Symbol to report on.

        Returns:
            A ``(first, last, rows)`` tuple.
        """
        row = await self._db.fetchrow(
            "SELECT MIN(recorded_at) AS first, MAX(recorded_at) AS last, COUNT(*) AS rows "
            "FROM orderbook_snapshots WHERE symbol = $1",
            symbol,
        )
        if row is None:
            return None, None, 0
        return row["first"], row["last"], int(row["rows"] or 0)


class PerpMetricsRepository:
    """Persists open interest and the contract state recorded alongside it."""

    def __init__(self, database: Database) -> None:
        """Bind the repository to a database."""
        self._db = database

    async def record_many(self, metrics: Sequence[PerpMetrics]) -> int:
        """Write a batch of metric snapshots.

        Args:
            metrics: Snapshots to persist.

        Returns:
            The number submitted.
        """
        if not metrics:
            return 0
        query = """
            INSERT INTO perp_metrics (
                symbol, recorded_at, open_interest, mark_price, oracle_price,
                mid_price, funding_rate, premium, day_notional_volume,
                day_base_volume, source
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
            ON CONFLICT (symbol, recorded_at) DO NOTHING
        """
        await self._db.executemany(
            query,
            [
                (
                    item.symbol,
                    item.occurred_at,
                    item.open_interest,
                    item.mark_price,
                    item.oracle_price,
                    item.mid_price,
                    item.funding_rate,
                    item.premium,
                    item.day_notional_volume,
                    item.day_base_volume,
                    item.source,
                )
                for item in metrics
            ],
        )
        return len(metrics)

    async def record(self, item: PerpMetrics) -> None:
        """Write one metric snapshot."""
        await self.record_many([item])

    async def coverage(self, symbol: str) -> tuple[datetime | None, datetime | None, int]:
        """Return the recorded span and row count for a symbol."""
        row = await self._db.fetchrow(
            "SELECT MIN(recorded_at) AS first, MAX(recorded_at) AS last, COUNT(*) AS rows "
            "FROM perp_metrics WHERE symbol = $1",
            symbol,
        )
        if row is None:
            return None, None, 0
        return row["first"], row["last"], int(row["rows"] or 0)


class FundingRepository:
    """Persists funding history.

    Unlike the other two series this one *is* backfillable from
    ``fundingHistory``, so it is stored for convenience rather than necessity.
    """

    def __init__(self, database: Database) -> None:
        """Bind the repository to a database."""
        self._db = database

    async def record_many(self, points: Sequence[FundingRate]) -> int:
        """Write a batch of funding observations.

        Args:
            points: Observations to persist.

        Returns:
            The number submitted.
        """
        if not points:
            return 0
        await self._db.executemany(
            """
            INSERT INTO funding_rates (symbol, occurred_at, funding_rate, premium, source)
            VALUES ($1,$2,$3,$4,$5)
            ON CONFLICT (symbol, occurred_at) DO NOTHING
            """,
            [
                (point.symbol, point.occurred_at, point.funding_rate, point.premium, point.source)
                for point in points
            ],
        )
        return len(points)

    async def load(self, symbol: str) -> tuple[FundingRate, ...]:
        """Load stored funding history for a symbol, oldest first."""
        rows = await self._db.fetch(
            "SELECT * FROM funding_rates WHERE symbol = $1 ORDER BY occurred_at", symbol
        )
        return tuple(
            FundingRate(
                source=row["source"],
                symbol=row["symbol"],
                funding_rate=row["funding_rate"],
                premium=row["premium"],
                occurred_at=row["occurred_at"],
            )
            for row in rows
        )


__all__ = ["FundingRepository", "OrderBookRepository", "PerpMetricsRepository"]
