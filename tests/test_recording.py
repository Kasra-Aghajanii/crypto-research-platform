"""Tests for the Phase 5 recording path.

These cover the machinery capturing data that cannot be re-obtained: the open
interest poller, the recorder's throttle and retry behaviour, and the repository
row mapping.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from libs.kafka_client import Topics
from libs.persistence.recording import (
    FundingRepository,
    OrderBookRepository,
    PerpMetricsRepository,
)
from libs.schemas.market import FundingRate, PerpMetrics
from services.data_ingestion.market_data.hyperliquid_perp_metrics import (
    PerpMetricsCollector,
    parse_contexts,
)
from services.data_ingestion.market_data.parsing import PayloadError
from services.data_ingestion.recorder import MarketRecorder
from tests.conftest import make_book, make_candle
from tests.test_learning import FakeDatabase

META_PAYLOAD: list[Any] = [
    {
        "universe": [
            {"name": "BTC", "szDecimals": 5},
            {"name": "ETH", "szDecimals": 4},
            {"name": "DOGE", "szDecimals": 0},
        ]
    },
    [
        {
            "funding": "0.0000125",
            "openInterest": "34382.51",
            "premium": "-0.00033",
            "oraclePx": "78691.1",
            "markPx": "78661.0",
            "midPx": "78664.5",
            "dayNtlVlm": "2223906367.2",
            "dayBaseVlm": "28288.9",
        },
        {
            "funding": "0.0000125",
            "openInterest": "990231.7",
            "premium": "-0.00028",
            "oraclePx": "2495.3",
            "markPx": "2494.5",
            "midPx": "2494.55",
            "dayNtlVlm": "961381496.8",
            "dayBaseVlm": "387889.7",
        },
        {"funding": "0.00001", "openInterest": "1.0", "markPx": "0.1"},
    ],
]


class TestPerpMetricsParsing:
    """metaAndAssetCtxs translation."""

    def test_extracts_requested_symbols_only(self) -> None:
        """Only the configured symbols are turned into events."""
        metrics = parse_contexts(META_PAYLOAD, ["BTC", "ETH"])
        assert {m.symbol for m in metrics} == {"BTC", "ETH"}

    def test_maps_fields_by_universe_position(self) -> None:
        """Context rows are matched to coins by their position in the universe."""
        metrics = {m.symbol: m for m in parse_contexts(META_PAYLOAD, ["BTC", "ETH"])}
        assert metrics["BTC"].open_interest == Decimal("34382.51")
        assert metrics["BTC"].mark_price == Decimal("78661.0")
        assert metrics["ETH"].open_interest == Decimal("990231.7")

    def test_basis_is_computed_from_mark_and_oracle(self) -> None:
        """The perp premium over the index is derived, not taken on trust."""
        btc = next(m for m in parse_contexts(META_PAYLOAD, ["BTC"]))
        expected = (Decimal("78661.0") - Decimal("78691.1")) / Decimal("78691.1") * 10_000
        assert btc.basis_bps == pytest.approx(expected, abs=Decimal("0.001"))

    def test_open_interest_notional(self) -> None:
        """Open interest is valued at the mark price."""
        btc = next(m for m in parse_contexts(META_PAYLOAD, ["BTC"]))
        assert btc.open_interest_notional == Decimal("34382.51") * Decimal("78661.0")

    def test_missing_optional_fields_are_none(self) -> None:
        """A context without oracle or premium still parses."""
        metrics = parse_contexts(META_PAYLOAD, ["DOGE"])
        assert len(metrics) == 1
        assert metrics[0].oracle_price is None
        assert metrics[0].basis_bps is None

    def test_unknown_symbol_yields_nothing(self) -> None:
        """A symbol absent from the universe is simply not returned."""
        assert parse_contexts(META_PAYLOAD, ["NOTACOIN"]) == ()

    def test_malformed_payload_raises(self) -> None:
        """A response that is not the two-part structure is an error."""
        with pytest.raises(PayloadError, match="two-element array"):
            parse_contexts({"universe": []}, ["BTC"])

    def test_mismatched_lengths_raise(self) -> None:
        """Universe and contexts must line up or the mapping is meaningless."""
        with pytest.raises(PayloadError, match="lengths differ"):
            parse_contexts([{"universe": [{"name": "BTC"}]}, []], ["BTC"])

    def test_all_symbols_share_one_timestamp(self) -> None:
        """One poll is one instant, so the snapshot is internally consistent."""
        metrics = parse_contexts(META_PAYLOAD, ["BTC", "ETH"])
        assert len({m.occurred_at for m in metrics}) == 1


class TestCollectorLifecycle:
    """The poller's failure behaviour."""

    async def test_poll_before_start_raises(self) -> None:
        """Using the collector without opening it is a programming error."""
        with pytest.raises(RuntimeError, match="before start"):
            await PerpMetricsCollector().poll_once()


class FakeRepo:
    """Repository double that can be told to fail."""

    def __init__(self, *, fail: bool = False) -> None:
        """Start with an empty record and a failure switch."""
        self.rows: list[Any] = []
        self.fail = fail

    async def record_many(self, items: Any) -> int:
        """Record or raise, depending on the switch."""
        if self.fail:
            raise RuntimeError("database unavailable")
        self.rows.extend(items)
        return len(items)

    async def upsert_many(self, items: Any) -> int:
        """Alias used by the candle repository."""
        return await self.record_many(items)


def build_recorder(*, fail_books: bool = False) -> tuple[MarketRecorder, dict[str, FakeRepo]]:
    """Build a recorder wired to repository doubles."""
    recorder = MarketRecorder(database=None)
    repos = {
        "books": FakeRepo(fail=fail_books),
        "metrics": FakeRepo(),
        "funding": FakeRepo(),
        "candles": FakeRepo(),
    }
    recorder._books = repos["books"]  # type: ignore[assignment]
    recorder._metrics = repos["metrics"]  # type: ignore[assignment]
    recorder._funding = repos["funding"]  # type: ignore[assignment]
    recorder._candles = repos["candles"]  # type: ignore[assignment]
    return recorder, repos


def book_at(seconds: float, symbol: str = "BTC") -> Any:
    """Build a book snapshot at a fixed offset from a base instant."""
    base = datetime(2026, 1, 1, tzinfo=UTC)
    return make_book(symbol=symbol, mid=100.0, occurred_at=base + timedelta(seconds=seconds))


class TestRecorderThrottle:
    """Storage throttling for the high-rate order book stream."""

    async def test_snapshots_inside_the_interval_are_dropped(self) -> None:
        """The recorder stores at most one snapshot per symbol per interval."""
        recorder, repos = build_recorder()
        recorder.params = recorder.params.model_copy(update={"orderbook_interval_s": 5.0})

        for offset in (0.0, 1.0, 2.0, 6.0, 7.0, 12.0):
            await recorder.handle(Topics.MARKET_ORDERBOOK, book_at(offset))
        await recorder.flush()

        assert len(repos["books"].rows) == 3
        assert recorder.books_throttled == 3

    async def test_throttle_is_per_symbol(self) -> None:
        """A busy symbol does not suppress a quiet one."""
        recorder, repos = build_recorder()
        recorder.params = recorder.params.model_copy(update={"orderbook_interval_s": 5.0})

        await recorder.handle(Topics.MARKET_ORDERBOOK, book_at(0.0, "BTC"))
        await recorder.handle(Topics.MARKET_ORDERBOOK, book_at(0.5, "ETH"))
        await recorder.flush()

        assert {row.symbol for row in repos["books"].rows} == {"BTC", "ETH"}

    async def test_zero_interval_stores_everything(self) -> None:
        """Turning the throttle off records every snapshot."""
        recorder, repos = build_recorder()
        recorder.params = recorder.params.model_copy(update={"orderbook_interval_s": 0.0})

        for offset in (0.0, 0.1, 0.2):
            await recorder.handle(Topics.MARKET_ORDERBOOK, book_at(offset))
        await recorder.flush()

        assert len(repos["books"].rows) == 3


class TestRecorderDurability:
    """A failed write must not lose irreplaceable rows."""

    async def test_failed_stream_does_not_discard_the_others(self) -> None:
        """A book write failure must not take the open-interest rows with it."""
        recorder, repos = build_recorder(fail_books=True)
        recorder.params = recorder.params.model_copy(update={"orderbook_interval_s": 0.0})

        await recorder.handle(Topics.MARKET_ORDERBOOK, book_at(0.0))
        await recorder.handle(
            Topics.MARKET_PERP_METRICS,
            PerpMetrics(
                source="test",
                symbol="BTC",
                open_interest=Decimal("1"),
                mark_price=Decimal("100"),
            ),
        )
        await recorder.flush()

        assert len(repos["metrics"].rows) == 1, "metrics must be written despite the book failure"

    async def test_failed_rows_are_retained_for_retry(self) -> None:
        """Rows that fail to write go back in the buffer, not to the void."""
        recorder, repos = build_recorder(fail_books=True)
        recorder.params = recorder.params.model_copy(update={"orderbook_interval_s": 0.0})

        await recorder.handle(Topics.MARKET_ORDERBOOK, book_at(0.0))
        await recorder.flush()
        assert recorder._buffered() == 1, "the failed row must be retained"

        repos["books"].fail = False
        await recorder.flush()
        assert len(repos["books"].rows) == 1
        assert recorder._buffered() == 0

    async def test_retry_buffer_is_bounded(self) -> None:
        """A long outage drops the oldest rows rather than exhausting memory."""
        recorder, _ = build_recorder(fail_books=True)
        recorder.params = recorder.params.model_copy(
            update={"orderbook_interval_s": 0.0, "max_retry_buffer": 5}
        )

        for offset in range(20):
            await recorder.handle(Topics.MARKET_ORDERBOOK, book_at(float(offset)))
        await recorder.flush()

        assert recorder._buffered() <= 5

    async def test_only_closed_candles_are_recorded(self) -> None:
        """An in-progress candle is not persisted."""
        recorder, repos = build_recorder()
        await recorder.handle(
            Topics.MARKET_CANDLES,
            make_candle(open_price=1, high=2, low=0.5, close=1.5, is_closed=False),
        )
        await recorder.flush()
        assert repos["candles"].rows == []

    async def test_recorder_publishes_nothing(self) -> None:
        """The recorder is a sink."""
        recorder, _ = build_recorder()
        assert await recorder.handle(Topics.MARKET_ORDERBOOK, book_at(0.0)) == ()


class TestRecordingRepositories:
    """Row mapping, checked against recorded SQL."""

    async def test_orderbook_row_carries_derived_metrics(self) -> None:
        """Imbalance and spread are computed at write time."""
        database = FakeDatabase()
        await OrderBookRepository(database).record(make_book(mid=100.0))
        statements = database.statements_containing("INSERT INTO orderbook_snapshots")
        assert statements
        args = database.calls[0][1]
        assert "imbalance_1" in statements[0]
        assert len(args) == 15

    async def test_orderbook_batch_uses_one_statement(self) -> None:
        """A batch is written in a single round trip."""
        database = FakeDatabase()
        written = await OrderBookRepository(database).record_many(
            [make_book(mid=100.0 + i) for i in range(4)]
        )
        assert written == 4
        assert len(database.calls) == 1

    async def test_stored_levels_are_truncated(self) -> None:
        """Only the configured number of levels is persisted."""
        database = FakeDatabase()
        await OrderBookRepository(database, stored_levels=2).record(
            make_book(mid=100.0, levels=10)
        )
        bid_json = database.calls[0][1][12]
        assert bid_json.count("[") == 3  # outer array plus two levels

    async def test_perp_metrics_batch(self) -> None:
        """Open interest rows are written as a batch."""
        database = FakeDatabase()
        written = await PerpMetricsRepository(database).record_many(
            [
                PerpMetrics(
                    source="test",
                    symbol="BTC",
                    open_interest=Decimal("1"),
                    mark_price=Decimal("100"),
                )
            ]
        )
        assert written == 1
        assert database.statements_containing("INSERT INTO perp_metrics")

    async def test_funding_batch(self) -> None:
        """Funding rows are written as a batch."""
        database = FakeDatabase()
        written = await FundingRepository(database).record_many(
            [
                FundingRate(
                    source="test",
                    symbol="BTC",
                    funding_rate=Decimal("0.0000125"),
                    premium=Decimal("-0.0003"),
                )
            ]
        )
        assert written == 1
        assert database.statements_containing("INSERT INTO funding_rates")

    async def test_empty_batches_do_nothing(self) -> None:
        """Nothing to write means no SQL."""
        database = FakeDatabase()
        assert await OrderBookRepository(database).record_many([]) == 0
        assert await PerpMetricsRepository(database).record_many([]) == 0
        assert await FundingRepository(database).record_many([]) == 0
        assert database.calls == []
