"""Recording tables for the two signals Phase 4 could not backtest.

Hyperliquid publishes no history for order book depth or open interest -- only
live snapshots.  Phase 4 therefore could not measure either signal, and no
amount of later effort can recover the missing past.  These tables exist so that
recording can start now.

Both are hypertables: they are append-only and grow without bound.

Storage notes
-------------
``orderbook_snapshots`` stores derived metrics (which is what the research
actually consumes) *and* the raw top-of-book levels as JSONB.  Keeping the raw
levels means a later change to the imbalance definition can be applied to
already-recorded history instead of requiring a fresh recording run.  At the
default 5-second throttle and 10 stored levels per side, expect roughly 15-20 MB
per symbol per day.

``perp_metrics`` is one row per poll per symbol -- tiny by comparison.
"""

from __future__ import annotations

from typing import Final

MIGRATION_ID: Final[str] = "0002_market_recording"
"""Identifier recorded in ``schema_migrations`` once applied."""

DESCRIPTION: Final[str] = (
    "Order book and perp-metrics recording tables for order-book imbalance and "
    "open-interest research."
)

STATEMENTS: Final[tuple[str, ...]] = (
    # ------------------------------------------------------------------
    # Order book snapshots
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS orderbook_snapshots (
        symbol           TEXT        NOT NULL,
        recorded_at      TIMESTAMPTZ NOT NULL,
        best_bid         NUMERIC(38, 12),
        best_ask         NUMERIC(38, 12),
        mid_price        NUMERIC(38, 12),
        spread_bps       DOUBLE PRECISION,
        imbalance_1      DOUBLE PRECISION,
        imbalance_5      DOUBLE PRECISION,
        imbalance_10     DOUBLE PRECISION,
        imbalance_20     DOUBLE PRECISION,
        bid_size_total   NUMERIC(38, 12),
        ask_size_total   NUMERIC(38, 12),
        bid_levels       JSONB       NOT NULL DEFAULT '[]'::JSONB,
        ask_levels       JSONB       NOT NULL DEFAULT '[]'::JSONB,
        source           TEXT        NOT NULL DEFAULT 'hyperliquid',
        PRIMARY KEY (symbol, recorded_at)
    )
    """,
    "SELECT create_hypertable('orderbook_snapshots', 'recorded_at', if_not_exists => TRUE)",
    "CREATE INDEX IF NOT EXISTS orderbook_snapshots_symbol_time_idx "
    "  ON orderbook_snapshots (symbol, recorded_at DESC)",
    # ------------------------------------------------------------------
    # Perp contract metrics (open interest and friends)
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS perp_metrics (
        symbol              TEXT        NOT NULL,
        recorded_at         TIMESTAMPTZ NOT NULL,
        open_interest       NUMERIC(38, 12) NOT NULL,
        mark_price          NUMERIC(38, 12) NOT NULL,
        oracle_price        NUMERIC(38, 12),
        mid_price           NUMERIC(38, 12),
        funding_rate        NUMERIC(38, 18),
        premium             NUMERIC(38, 18),
        day_notional_volume NUMERIC(38, 12),
        day_base_volume     NUMERIC(38, 12),
        source              TEXT        NOT NULL DEFAULT 'hyperliquid',
        PRIMARY KEY (symbol, recorded_at)
    )
    """,
    "SELECT create_hypertable('perp_metrics', 'recorded_at', if_not_exists => TRUE)",
    "CREATE INDEX IF NOT EXISTS perp_metrics_symbol_time_idx "
    "  ON perp_metrics (symbol, recorded_at DESC)",
    # ------------------------------------------------------------------
    # Funding history (backfillable, unlike the two above)
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS funding_rates (
        symbol       TEXT        NOT NULL,
        occurred_at  TIMESTAMPTZ NOT NULL,
        funding_rate NUMERIC(38, 18) NOT NULL,
        premium      NUMERIC(38, 18),
        source       TEXT        NOT NULL DEFAULT 'hyperliquid',
        PRIMARY KEY (symbol, occurred_at)
    )
    """,
    "SELECT create_hypertable('funding_rates', 'occurred_at', if_not_exists => TRUE)",
)
"""Ordered DDL statements. Idempotent, so re-running the migration is safe."""

__all__ = ["DESCRIPTION", "MIGRATION_ID", "STATEMENTS"]
