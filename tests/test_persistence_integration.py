"""Integration tests against a real TimescaleDB.

These are the tests that actually execute the SQL.  They are **skipped unless a
database is reachable**, so they are inert on a machine without Docker and run
automatically once ``docker compose up -d`` is going::

    docker compose up -d
    python -m scripts.migrate
    .venv/Scripts/python -m pytest tests/test_persistence_integration.py -v

Everything else in the suite exercises the repositories against a fake
connection, which proves the row mapping but not that PostgreSQL accepts the
statements.  This file closes that gap.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import uuid4

import pytest

from libs.config import settings
from libs.persistence.database import PostgresDatabase
from libs.persistence.recording import (
    FundingRepository,
    OrderBookRepository,
    PerpMetricsRepository,
)
from libs.persistence.repositories import (
    CandleRepository,
    ExecutionRepository,
    PerformanceRepository,
    PortfolioRepository,
    ProtectionRepository,
    SignalRepository,
)
from libs.schemas.learning import AgentOutcome, ExitReason, PositionClosed
from libs.schemas.market import FundingRate, PerpMetrics, Side
from libs.schemas.portfolio import Position
from libs.schemas.signals import AgentSignal, Direction
from libs.schemas.trading import Fill, OrderIntent
from scripts.migrate import MIGRATIONS, apply
from tests.conftest import trending_candles

DSN = os.environ.get("TEST_TIMESCALE_DSN", settings.storage.timescale_dsn)


_REACHABLE: bool | None = None


async def _reachable() -> bool:
    """Return whether the configured database accepts a connection.

    The result is cached for the session: probing once per test would add a
    connection timeout to every skip.
    """
    global _REACHABLE
    if _REACHABLE is not None:
        return _REACHABLE
    database = PostgresDatabase(dsn=DSN)
    try:
        await asyncio.wait_for(database.connect(), timeout=2.0)
    except Exception:  # noqa: BLE001 - any failure means "not available"
        _REACHABLE = False
        return False
    await database.close()
    _REACHABLE = True
    return True


@pytest.fixture
async def database() -> Any:
    """Yield a migrated database, skipping the test when none is reachable."""
    if not await _reachable():
        pytest.skip(f"No TimescaleDB at {DSN}; start docker compose to run these tests.")
    connection = PostgresDatabase(dsn=DSN)
    await connection.connect()
    await apply(connection, MIGRATIONS)
    try:
        yield connection
    finally:
        await connection.close()


class TestMigration:
    """The schema applies and is idempotent."""

    async def test_migration_is_idempotent(self, database: PostgresDatabase) -> None:
        """Re-running the migration changes nothing and raises nothing."""
        assert await apply(database, MIGRATIONS) == []

    async def test_every_expected_table_exists(self, database: PostgresDatabase) -> None:
        """The migration creates every table the repositories write to."""
        rows = await database.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        )
        tables = {row["tablename"] for row in rows}
        expected = {
            "candles",
            "agent_signals",
            "trade_decisions",
            "orders",
            "fills",
            "positions",
            "portfolio_state",
            "portfolio_snapshots",
            "position_closures",
            "position_protection",
            "agent_performance",
        }
        assert expected <= tables


class TestCandleStorage:
    """Candle upsert and read-back."""

    async def test_upsert_then_load(self, database: PostgresDatabase) -> None:
        """Candles written are read back in order."""
        repository = CandleRepository(database)
        candles = trending_candles(20, symbol="ITBTC", interval="1h")
        assert await repository.upsert_many(candles) == 20

        loaded = await repository.load("ITBTC", "1h")
        assert len(loaded) >= 20
        assert loaded[0].open_time <= loaded[-1].open_time

    async def test_upsert_is_idempotent(self, database: PostgresDatabase) -> None:
        """Re-writing the same candles does not duplicate rows."""
        repository = CandleRepository(database)
        candles = trending_candles(10, symbol="ITETH", interval="1h")
        await repository.upsert_many(candles)
        await repository.upsert_many(candles)
        assert len(await repository.load("ITETH", "1h")) == 10

    async def test_coverage_reports_the_stored_range(self, database: PostgresDatabase) -> None:
        """Coverage returns the first and last stored bar."""
        repository = CandleRepository(database)
        candles = trending_candles(15, symbol="ITSOL", interval="1h")
        await repository.upsert_many(candles)
        first, last = await repository.coverage("ITSOL", "1h")
        assert first is not None and last is not None
        assert first <= last


class TestPortfolioPersistence:
    """Open positions and account state survive a restart."""

    async def test_position_round_trip(self, database: PostgresDatabase) -> None:
        """A saved position is loaded back with its protective levels."""
        repository = PortfolioRepository(database)
        now = datetime.now(tz=UTC)
        correlation = uuid4()
        await repository.save_position(
            Position(
                source="test",
                symbol="ITPOS",
                direction=Direction.LONG,
                size=Decimal("1.25"),
                entry_price=Decimal("100"),
                mark_price=Decimal("105"),
                stop_price=Decimal("95"),
                take_profit_price=Decimal("120"),
                trailing_stop_pct=2.5,
                opening_correlation_id=correlation,
                opened_at=now,
                updated_at=now,
            )
        )
        loaded = {p.symbol: p for p in await repository.load_positions()}
        assert "ITPOS" in loaded
        position = loaded["ITPOS"]
        assert position.size == Decimal("1.25")
        assert position.stop_price == Decimal("95")
        assert position.trailing_stop_pct == pytest.approx(2.5)
        assert position.opening_correlation_id == correlation

        await repository.delete_position("ITPOS")
        assert "ITPOS" not in {p.symbol for p in await repository.load_positions()}

    async def test_account_state_round_trip(self, database: PostgresDatabase) -> None:
        """Cash and counters are restored exactly."""
        repository = PortfolioRepository(database)
        await repository.save_state(
            cash=Decimal("9876.54"),
            starting_equity=Decimal("10000"),
            day_start_equity=Decimal("9900"),
            realized_pnl=Decimal("-123.46"),
            fees_paid=Decimal("12.34"),
            trade_count=7,
            win_count=3,
            loss_count=4,
            day_of=datetime.now(tz=UTC).date(),
            is_paper=True,
        )
        state = await repository.load_state()
        assert state is not None
        assert Decimal(state["cash"]) == Decimal("9876.54")
        assert state["trade_count"] == 7

    async def test_closure_is_recorded(self, database: PostgresDatabase) -> None:
        """A closure lands in the terminal-state table."""
        repository = PortfolioRepository(database)
        now = datetime.now(tz=UTC)
        closure = PositionClosed(
            source="test",
            symbol="ITCLOSE",
            direction=Direction.LONG,
            size=Decimal("1"),
            entry_price=Decimal("100"),
            exit_price=Decimal("110"),
            realized_pnl=Decimal("10"),
            fees_paid=Decimal("0.5"),
            exit_reason=ExitReason.TAKE_PROFIT,
            opened_at=now,
            closed_at=now,
            opening_correlation_id=uuid4(),
        )
        await repository.record_closure(closure)
        rows = await database.fetch(
            "SELECT * FROM position_closures WHERE symbol = $1", "ITCLOSE"
        )
        assert rows


class TestExecutionAndAttribution:
    """Orders, fills, signals and the Brier ledger."""

    async def test_order_and_fill_round_trip(self, database: PostgresDatabase) -> None:
        """An order is stored and marked filled by its fill."""
        repository = ExecutionRepository(database)
        order = OrderIntent(
            source="test",
            order_id=uuid4(),
            decision_id=uuid4(),
            symbol="ITEXEC",
            side=Side.BUY,
            size=Decimal("1"),
            stop_price=Decimal("95"),
            trailing_stop_pct=2.0,
        )
        await repository.record_order(order)
        loaded = await repository.order_for(order.order_id)
        assert loaded is not None
        assert loaded.stop_price == Decimal("95")
        assert loaded.trailing_stop_pct == pytest.approx(2.0)

        await repository.record_fill(
            Fill(
                source="test",
                fill_id=uuid4(),
                order_id=order.order_id,
                symbol="ITEXEC",
                side=Side.BUY,
                size=Decimal("1"),
                price=Decimal("100"),
                fee=Decimal("0.05"),
            )
        )
        row = await database.fetchrow(
            "SELECT status FROM orders WHERE order_id = $1", order.order_id
        )
        assert row is not None and row["status"] == "filled"

    async def test_signals_are_found_by_correlation(self, database: PostgresDatabase) -> None:
        """Attribution can recover signals long after they left memory."""
        repository = SignalRepository(database)
        correlation = uuid4()
        signal = AgentSignal(
            source="test",
            correlation_id=correlation,
            agent_name="market_analyst",
            agent_version="1.0.0",
            symbol="ITSIG",
            direction=Direction.LONG,
            confidence=0.77,
            features={"confluence_score": 0.42},
        )
        await repository.record_signal(signal)

        found = await repository.signals_for_correlation(correlation)
        assert len(found) == 1
        assert found[0].confidence == pytest.approx(0.77)
        assert found[0].features["confluence_score"] == pytest.approx(0.42)

    async def test_outcome_is_recorded_once_per_signal(
        self, database: PostgresDatabase
    ) -> None:
        """A redelivered closure cannot score the same signal twice."""
        repository = PerformanceRepository(database)
        outcome = AgentOutcome(
            source="test",
            agent_name="it_agent",
            signal_id=uuid4(),
            symbol="ITPERF",
            direction=Direction.LONG,
            confidence=0.8,
            realized_pnl=Decimal("10"),
            was_correct=True,
            brier_score=0.04,
            closed_at=datetime.now(tz=UTC),
            holding_period_s=60.0,
        )
        await repository.record_outcome(outcome, uuid4())
        await repository.record_outcome(outcome, uuid4())

        rows = await database.fetch(
            "SELECT COUNT(*) AS n FROM agent_performance WHERE signal_id = $1", outcome.signal_id
        )
        assert rows[0]["n"] == 1

    async def test_agent_scores_are_queryable(self, database: PostgresDatabase) -> None:
        """The trust registry can be rehydrated from the ledger."""
        repository = PerformanceRepository(database)
        for _ in range(3):
            await repository.record_outcome(
                AgentOutcome(
                    source="test",
                    agent_name="it_scores",
                    signal_id=uuid4(),
                    symbol="ITPERF",
                    direction=Direction.LONG,
                    confidence=0.9,
                    realized_pnl=Decimal("5"),
                    was_correct=True,
                    brier_score=0.01,
                    closed_at=datetime.now(tz=UTC),
                    holding_period_s=60.0,
                ),
                uuid4(),
            )
        scores = await repository.agent_scores()
        assert len(scores.get("it_scores", [])) >= 3
        assert any(row["agent_name"] == "it_scores" for row in await repository.summary())


class TestProtectionPersistence:
    """Trailing-stop state survives a restart."""

    async def test_protection_round_trip(self, database: PostgresDatabase) -> None:
        """The ratcheted extreme is stored and read back."""
        repository = ProtectionRepository(database)
        now = datetime.now(tz=UTC)
        await repository.save(
            {
                "symbol": "ITPROT",
                "direction": "long",
                "size": Decimal("1"),
                "entry_price": Decimal("100"),
                "stop_price": Decimal("95"),
                "take_profit_price": Decimal("120"),
                "trailing_stop_pct": 3.0,
                "extreme_price": Decimal("130"),
                "exit_pending": False,
                "correlation_id": uuid4(),
                "decision_id": uuid4(),
                "opened_at": now,
                "updated_at": now,
            }
        )
        rows = {row["symbol"]: row for row in await repository.load_all()}
        assert "ITPROT" in rows
        assert rows["ITPROT"]["extreme_price"] == Decimal("130")

        await repository.delete("ITPROT")
        assert "ITPROT" not in {row["symbol"] for row in await repository.load_all()}


class TestRecordingTables:
    """The Phase 5 recording tables, exercised against real PostgreSQL.

    These are the tables capturing data with no historical endpoint, so a schema
    bug here is not recoverable after the fact.
    """

    async def test_orderbook_snapshot_round_trip(self, database: PostgresDatabase) -> None:
        """A snapshot is stored with its derived metrics and raw levels."""
        from tests.conftest import make_book

        repository = OrderBookRepository(database, stored_levels=3)
        snapshot = make_book(symbol="ITBOOK", mid=100.0, levels=8)
        await repository.record(snapshot)

        row = await database.fetchrow(
            "SELECT * FROM orderbook_snapshots WHERE symbol = $1 ORDER BY recorded_at DESC LIMIT 1",
            "ITBOOK",
        )
        assert row is not None
        assert row["mid_price"] == pytest.approx(Decimal("100"))
        assert row["imbalance_5"] == pytest.approx(0.0, abs=1e-9)
        assert row["spread_bps"] is not None
        assert len(json.loads(row["bid_levels"])) == 3

    async def test_orderbook_batch_and_coverage(self, database: PostgresDatabase) -> None:
        """A batch lands and coverage reports the span."""
        from tests.conftest import make_book

        repository = OrderBookRepository(database)
        base = datetime.now(tz=UTC)
        await repository.record_many(
            [
                make_book(symbol="ITCOV", mid=100.0 + i, occurred_at=base + timedelta(seconds=i))
                for i in range(5)
            ]
        )
        first, last, rows = await repository.coverage("ITCOV")
        assert rows >= 5
        assert first is not None and last is not None and first <= last

    async def test_perp_metrics_round_trip(self, database: PostgresDatabase) -> None:
        """Open interest and the contract state around it are stored."""
        repository = PerpMetricsRepository(database)
        await repository.record(
            PerpMetrics(
                source="test",
                symbol="ITOI",
                open_interest=Decimal("34382.5142799999"),
                mark_price=Decimal("78661.0"),
                oracle_price=Decimal("78691.1"),
                funding_rate=Decimal("0.0000125"),
                premium=Decimal("-0.0003401568"),
                day_notional_volume=Decimal("2223906367.2157306671"),
            )
        )
        row = await database.fetchrow(
            "SELECT * FROM perp_metrics WHERE symbol = $1 ORDER BY recorded_at DESC LIMIT 1",
            "ITOI",
        )
        assert row is not None
        # NUMERIC(38, 12) keeps the submitted value exactly, to 12 decimal places.
        assert row["open_interest"] == Decimal("34382.5142799999")
        assert row["funding_rate"] == Decimal("0.000012500000000000")

    async def test_perp_metrics_dedupes_on_timestamp(self, database: PostgresDatabase) -> None:
        """A redelivered poll does not double-count.

        The symbol is unique per run so repeated test runs cannot accumulate
        rows and mask the behaviour under test.
        """
        repository = PerpMetricsRepository(database)
        symbol = f"IT{uuid4().hex[:8].upper()}"
        item = PerpMetrics(
            source="test",
            symbol=symbol,
            open_interest=Decimal("1"),
            mark_price=Decimal("100"),
        )
        await repository.record(item)
        await repository.record(item)
        row = await database.fetchrow(
            "SELECT COUNT(*) AS n FROM perp_metrics WHERE symbol = $1", symbol
        )
        assert row is not None and row["n"] == 1

    async def test_funding_round_trip(self, database: PostgresDatabase) -> None:
        """Funding history stores and loads back."""
        repository = FundingRepository(database)
        symbol = f"IT{uuid4().hex[:8].upper()}"
        now = datetime.now(tz=UTC).replace(microsecond=0)
        await repository.record_many(
            [
                FundingRate(
                    source="test",
                    symbol=symbol,
                    funding_rate=Decimal("0.0000125"),
                    premium=Decimal("-0.00034"),
                    occurred_at=now + timedelta(hours=i),
                )
                for i in range(3)
            ]
        )
        loaded = await repository.load(symbol)
        assert len(loaded) == 3
        assert loaded[0].funding_rate == Decimal("0.000012500000000000")
