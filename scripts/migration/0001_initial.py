"""Initial TimescaleDB schema for the trading platform.

Defines every table the persistence layer writes to.  Three of them are
hypertables -- ``candles``, ``fills`` and ``portfolio_snapshots`` -- because they
are append-only time series that grow without bound; the rest are ordinary
tables holding current or terminal state.

Design notes
------------
* Prices and sizes are ``NUMERIC``, never ``DOUBLE PRECISION``: a float cannot
  represent an exchange price exactly, and rounding errors accumulate into the
  PnL.
* ``positions`` holds only *open* positions, keyed by symbol.  A closed position
  is deleted from it and appended to ``position_closures``, so rebuilding state
  on startup is a single ``SELECT * FROM positions``.
* ``position_protection`` is owned exclusively by the position monitor and holds
  the trailing-stop extreme, so a restart does not reset a ratcheted stop.
* ``agent_performance`` is the Brier-score ledger the trust weights read from.

Every statement is idempotent (``IF NOT EXISTS``), so re-running the migration
is safe.
"""

from __future__ import annotations

from typing import Final

MIGRATION_ID: Final[str] = "0001_initial"
"""Identifier recorded in ``schema_migrations`` once applied."""

DESCRIPTION: Final[str] = "Initial schema: market data, execution, portfolio and learning tables."

CREATE_MIGRATIONS_TABLE: Final[str] = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    migration_id TEXT PRIMARY KEY,
    description  TEXT NOT NULL,
    applied_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

STATEMENTS: Final[tuple[str, ...]] = (
    # ------------------------------------------------------------------
    # Extensions
    # ------------------------------------------------------------------
    "CREATE EXTENSION IF NOT EXISTS timescaledb",
    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS candles (
        symbol      TEXT        NOT NULL,
        "interval"  TEXT        NOT NULL,
        open_time   TIMESTAMPTZ NOT NULL,
        close_time  TIMESTAMPTZ NOT NULL,
        open        NUMERIC(38, 12) NOT NULL,
        high        NUMERIC(38, 12) NOT NULL,
        low         NUMERIC(38, 12) NOT NULL,
        close       NUMERIC(38, 12) NOT NULL,
        volume      NUMERIC(38, 12) NOT NULL DEFAULT 0,
        trade_count INTEGER     NOT NULL DEFAULT 0,
        is_closed   BOOLEAN     NOT NULL DEFAULT TRUE,
        source      TEXT        NOT NULL DEFAULT 'hyperliquid',
        PRIMARY KEY (symbol, "interval", open_time)
    )
    """,
    "SELECT create_hypertable('candles', 'open_time', if_not_exists => TRUE)",
    'CREATE INDEX IF NOT EXISTS candles_symbol_interval_idx '
    '  ON candles (symbol, "interval", open_time DESC)',
    # ------------------------------------------------------------------
    # Signals and decisions (attribution inputs)
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS agent_signals (
        signal_id      UUID        PRIMARY KEY,
        correlation_id UUID        NOT NULL,
        agent_name     TEXT        NOT NULL,
        agent_version  TEXT        NOT NULL,
        symbol         TEXT        NOT NULL,
        direction      TEXT        NOT NULL,
        confidence     DOUBLE PRECISION NOT NULL,
        degraded       BOOLEAN     NOT NULL DEFAULT FALSE,
        reference_price NUMERIC(38, 12),
        rationale      TEXT        NOT NULL DEFAULT '',
        features       JSONB       NOT NULL DEFAULT '{}'::JSONB,
        occurred_at    TIMESTAMPTZ NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS agent_signals_correlation_idx ON agent_signals (correlation_id)",
    "CREATE INDEX IF NOT EXISTS agent_signals_agent_time_idx "
    "  ON agent_signals (agent_name, occurred_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS trade_decisions (
        decision_id          UUID        PRIMARY KEY,
        correlation_id       UUID        NOT NULL,
        symbol               TEXT        NOT NULL,
        action               TEXT        NOT NULL,
        confidence           DOUBLE PRECISION NOT NULL,
        weighted_confidence  DOUBLE PRECISION NOT NULL,
        reference_price      NUMERIC(38, 12),
        contributing_signals UUID[]      NOT NULL DEFAULT '{}',
        agent_weights        JSONB       NOT NULL DEFAULT '{}'::JSONB,
        mode                 TEXT        NOT NULL DEFAULT 'passthrough',
        occurred_at          TIMESTAMPTZ NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS trade_decisions_correlation_idx "
    "  ON trade_decisions (correlation_id)",
    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS orders (
        order_id          UUID        PRIMARY KEY,
        decision_id       UUID,
        correlation_id    UUID        NOT NULL,
        symbol            TEXT        NOT NULL,
        side              TEXT        NOT NULL,
        size              NUMERIC(38, 12) NOT NULL,
        order_type        TEXT        NOT NULL DEFAULT 'market',
        limit_price       NUMERIC(38, 12),
        reduce_only       BOOLEAN     NOT NULL DEFAULT FALSE,
        stop_price        NUMERIC(38, 12),
        take_profit_price NUMERIC(38, 12),
        trailing_stop_pct DOUBLE PRECISION,
        exit_reason       TEXT,
        is_paper          BOOLEAN     NOT NULL DEFAULT TRUE,
        status            TEXT        NOT NULL DEFAULT 'submitted',
        created_at        TIMESTAMPTZ NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS orders_symbol_time_idx ON orders (symbol, created_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS fills (
        fill_id         UUID        NOT NULL,
        order_id        UUID        NOT NULL,
        decision_id     UUID,
        correlation_id  UUID        NOT NULL,
        symbol          TEXT        NOT NULL,
        side            TEXT        NOT NULL,
        size            NUMERIC(38, 12) NOT NULL,
        price           NUMERIC(38, 12) NOT NULL,
        fee             NUMERIC(38, 12) NOT NULL DEFAULT 0,
        slippage_bps    DOUBLE PRECISION NOT NULL DEFAULT 0,
        reference_price NUMERIC(38, 12),
        reduce_only     BOOLEAN     NOT NULL DEFAULT FALSE,
        is_paper        BOOLEAN     NOT NULL DEFAULT TRUE,
        occurred_at     TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (fill_id, occurred_at)
    )
    """,
    "SELECT create_hypertable('fills', 'occurred_at', if_not_exists => TRUE)",
    "CREATE INDEX IF NOT EXISTS fills_symbol_time_idx ON fills (symbol, occurred_at DESC)",
    "CREATE INDEX IF NOT EXISTS fills_correlation_idx ON fills (correlation_id)",
    # ------------------------------------------------------------------
    # Portfolio state
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS positions (
        symbol                 TEXT        PRIMARY KEY,
        direction              TEXT        NOT NULL,
        size                   NUMERIC(38, 12) NOT NULL,
        entry_price            NUMERIC(38, 12) NOT NULL,
        mark_price             NUMERIC(38, 12) NOT NULL,
        realized_pnl           NUMERIC(38, 12) NOT NULL DEFAULT 0,
        fees_paid              NUMERIC(38, 12) NOT NULL DEFAULT 0,
        stop_price             NUMERIC(38, 12),
        take_profit_price      NUMERIC(38, 12),
        trailing_stop_pct      DOUBLE PRECISION,
        opening_correlation_id UUID,
        opening_decision_id    UUID,
        opened_at              TIMESTAMPTZ NOT NULL,
        updated_at             TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS portfolio_state (
        id                INTEGER     PRIMARY KEY DEFAULT 1,
        cash              NUMERIC(38, 12) NOT NULL,
        starting_equity   NUMERIC(38, 12) NOT NULL,
        day_start_equity  NUMERIC(38, 12) NOT NULL,
        realized_pnl      NUMERIC(38, 12) NOT NULL DEFAULT 0,
        fees_paid         NUMERIC(38, 12) NOT NULL DEFAULT 0,
        trade_count       INTEGER     NOT NULL DEFAULT 0,
        win_count         INTEGER     NOT NULL DEFAULT 0,
        loss_count        INTEGER     NOT NULL DEFAULT 0,
        day_of            DATE        NOT NULL,
        is_paper          BOOLEAN     NOT NULL DEFAULT TRUE,
        updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        CONSTRAINT portfolio_state_singleton CHECK (id = 1)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS portfolio_snapshots (
        recorded_at      TIMESTAMPTZ NOT NULL,
        equity           NUMERIC(38, 12) NOT NULL,
        cash             NUMERIC(38, 12) NOT NULL,
        realized_pnl     NUMERIC(38, 12) NOT NULL DEFAULT 0,
        unrealized_pnl   NUMERIC(38, 12) NOT NULL DEFAULT 0,
        fees_paid        NUMERIC(38, 12) NOT NULL DEFAULT 0,
        gross_notional   NUMERIC(38, 12) NOT NULL DEFAULT 0,
        open_positions   INTEGER     NOT NULL DEFAULT 0,
        trade_count      INTEGER     NOT NULL DEFAULT 0,
        win_count        INTEGER     NOT NULL DEFAULT 0,
        loss_count       INTEGER     NOT NULL DEFAULT 0,
        is_paper         BOOLEAN     NOT NULL DEFAULT TRUE,
        PRIMARY KEY (recorded_at)
    )
    """,
    "SELECT create_hypertable('portfolio_snapshots', 'recorded_at', if_not_exists => TRUE)",
    """
    CREATE TABLE IF NOT EXISTS position_closures (
        closure_id             UUID        PRIMARY KEY,
        symbol                 TEXT        NOT NULL,
        direction              TEXT        NOT NULL,
        size                   NUMERIC(38, 12) NOT NULL,
        entry_price            NUMERIC(38, 12) NOT NULL,
        exit_price             NUMERIC(38, 12) NOT NULL,
        realized_pnl           NUMERIC(38, 12) NOT NULL,
        fees_paid              NUMERIC(38, 12) NOT NULL DEFAULT 0,
        return_pct             DOUBLE PRECISION NOT NULL DEFAULT 0,
        exit_reason            TEXT        NOT NULL DEFAULT 'unknown',
        opening_correlation_id UUID,
        opening_decision_id    UUID,
        opened_at              TIMESTAMPTZ NOT NULL,
        closed_at              TIMESTAMPTZ NOT NULL,
        holding_period_s       DOUBLE PRECISION NOT NULL DEFAULT 0,
        is_paper               BOOLEAN     NOT NULL DEFAULT TRUE
    )
    """,
    "CREATE INDEX IF NOT EXISTS position_closures_correlation_idx "
    "  ON position_closures (opening_correlation_id)",
    "CREATE INDEX IF NOT EXISTS position_closures_closed_at_idx "
    "  ON position_closures (closed_at DESC)",
    # ------------------------------------------------------------------
    # Position monitor state (trailing stops survive restarts)
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS position_protection (
        symbol             TEXT        PRIMARY KEY,
        direction          TEXT        NOT NULL,
        size               NUMERIC(38, 12) NOT NULL,
        entry_price        NUMERIC(38, 12) NOT NULL,
        stop_price         NUMERIC(38, 12),
        take_profit_price  NUMERIC(38, 12),
        trailing_stop_pct  DOUBLE PRECISION,
        extreme_price      NUMERIC(38, 12),
        exit_pending       BOOLEAN     NOT NULL DEFAULT FALSE,
        correlation_id     UUID,
        decision_id        UUID,
        opened_at          TIMESTAMPTZ NOT NULL,
        updated_at         TIMESTAMPTZ NOT NULL
    )
    """,
    # ------------------------------------------------------------------
    # Learning: the Brier ledger behind adaptive trust weights
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS agent_performance (
        outcome_id       UUID        PRIMARY KEY,
        agent_name       TEXT        NOT NULL,
        agent_version    TEXT        NOT NULL,
        signal_id        UUID        NOT NULL,
        correlation_id   UUID,
        symbol           TEXT        NOT NULL,
        direction        TEXT        NOT NULL,
        confidence       DOUBLE PRECISION NOT NULL,
        realized_pnl     NUMERIC(38, 12) NOT NULL,
        was_correct      BOOLEAN     NOT NULL,
        brier_score      DOUBLE PRECISION NOT NULL,
        exit_reason      TEXT        NOT NULL DEFAULT 'unknown',
        holding_period_s DOUBLE PRECISION NOT NULL DEFAULT 0,
        closed_at        TIMESTAMPTZ NOT NULL,
        recorded_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS agent_performance_agent_idx "
    "  ON agent_performance (agent_name, closed_at DESC)",
    "CREATE UNIQUE INDEX IF NOT EXISTS agent_performance_signal_uniq "
    "  ON agent_performance (signal_id)",
)
"""Ordered DDL statements. Idempotent, so re-running the migration is safe."""

__all__ = ["CREATE_MIGRATIONS_TABLE", "DESCRIPTION", "MIGRATION_ID", "STATEMENTS"]
