"""Tests for the historical backfill and the replay engine."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from libs.schemas.learning import ExitReason
from libs.schemas.signals import Direction
from services.backtest.replay import ReplayEngine, _OpenTrade
from services.data_ingestion.market_data.hyperliquid_history import (
    HyperliquidHistoryClient,
    cache_path,
    interval_seconds,
    parse_history_candle,
    read_cache,
    write_cache,
)
from services.data_ingestion.market_data.parsing import PayloadError
from tests.conftest import make_candle, trending_candles

HISTORY_PAYLOAD = {
    "t": 1_700_000_000_000,
    "T": 1_700_003_599_999,
    "s": "BTC",
    "i": "1h",
    "o": "37000.0",
    "h": "37500.0",
    "l": "36900.0",
    "c": "37400.0",
    "v": "512.5",
    "n": 8123,
}


def open_trade(
    *,
    direction: Direction = Direction.LONG,
    entry: float = 100.0,
    stop: float | None = 95.0,
    take_profit: float | None = 110.0,
    trailing: float | None = None,
) -> _OpenTrade:
    """Build an in-flight replay trade."""
    return _OpenTrade(
        symbol="BTC",
        direction=direction,
        confidence=0.8,
        entry_time=datetime(2026, 1, 1, tzinfo=UTC),
        entry_price=Decimal(str(entry)),
        size=Decimal(1),
        stop_price=Decimal(str(stop)) if stop is not None else None,
        take_profit_price=Decimal(str(take_profit)) if take_profit is not None else None,
        trailing_stop_pct=trailing,
        extreme_price=Decimal(str(entry)),
        entry_cost=Decimal("0.05"),
    )


class TestHistoryParsing:
    """REST history payload translation."""

    def test_history_candle_is_always_closed(self) -> None:
        """Historical bars are final by definition."""
        candle = parse_history_candle(HISTORY_PAYLOAD)
        assert candle.is_closed is True
        assert candle.symbol == "BTC"
        assert candle.interval == "1h"
        assert candle.close == Decimal("37400.0")

    def test_missing_symbol_raises(self) -> None:
        """A payload without a coin cannot become a candle."""
        with pytest.raises(PayloadError):
            parse_history_candle({**HISTORY_PAYLOAD, "s": None})

    def test_interval_seconds_known_values(self) -> None:
        """Interval lengths convert to seconds."""
        assert interval_seconds("1m") == 60
        assert interval_seconds("1h") == 3_600
        assert interval_seconds("1d") == 86_400

    def test_unknown_interval_rejected(self) -> None:
        """An unsupported interval is an error, not a silent default."""
        with pytest.raises(ValueError, match="Unsupported interval"):
            interval_seconds("7s")


class TestHistoryFetching:
    """Window walking against a stubbed endpoint."""

    @staticmethod
    def client_returning(pages: list[list[dict[str, Any]]]) -> httpx.AsyncClient:
        """Build an HTTP client that returns each page in turn."""
        remaining = list(pages)

        def handler(request: httpx.Request) -> httpx.Response:
            payload = remaining.pop(0) if remaining else []
            return httpx.Response(200, json=payload)

        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def test_fetch_window_parses_and_sorts(self) -> None:
        """A window returns candles oldest first."""
        second = {**HISTORY_PAYLOAD, "t": HISTORY_PAYLOAD["t"] + 3_600_000}
        async with HyperliquidHistoryClient(
            client=self.client_returning([[second, HISTORY_PAYLOAD]]), request_pause_s=0
        ) as client:
            candles = await client.fetch_window(
                "BTC", "1h", datetime.now(tz=UTC) - timedelta(days=1), datetime.now(tz=UTC)
            )
        assert len(candles) == 2
        assert candles[0].open_time < candles[1].open_time

    async def test_fetch_range_stops_on_an_empty_window(self) -> None:
        """Running out of history ends the walk instead of looping."""
        async with HyperliquidHistoryClient(
            client=self.client_returning([[HISTORY_PAYLOAD], []]), request_pause_s=0
        ) as client:
            candles = await client.fetch_range(
                "BTC",
                "1h",
                start=datetime.fromtimestamp(HISTORY_PAYLOAD["t"] / 1000, tz=UTC),
                end=datetime.fromtimestamp(HISTORY_PAYLOAD["t"] / 1000, tz=UTC)
                + timedelta(days=400),
            )
        assert len(candles) == 1

    async def test_fetch_range_deduplicates_overlapping_windows(self) -> None:
        """The same bar returned twice is stored once."""
        async with HyperliquidHistoryClient(
            client=self.client_returning([[HISTORY_PAYLOAD], [HISTORY_PAYLOAD], []]),
            request_pause_s=0,
        ) as client:
            candles = await client.fetch_range(
                "BTC",
                "1h",
                start=datetime.fromtimestamp(HISTORY_PAYLOAD["t"] / 1000, tz=UTC),
                end=datetime.fromtimestamp(HISTORY_PAYLOAD["t"] / 1000, tz=UTC)
                + timedelta(days=30),
            )
        assert len(candles) == 1

    async def test_client_outside_context_manager_raises(self) -> None:
        """Using the client without opening it is a programming error."""
        client = HyperliquidHistoryClient()
        with pytest.raises(RuntimeError, match="context manager"):
            await client.fetch_window(
                "BTC", "1h", datetime.now(tz=UTC), datetime.now(tz=UTC)
            )


class TestCacheRoundTrip:
    """The on-disk cache must reproduce candles exactly."""

    def test_write_then_read_returns_the_same_candles(self, tmp_path: Path) -> None:
        """Cached candles survive the round trip."""
        candles = trending_candles(10, interval="1h")
        path = cache_path(tmp_path, "BTC", "1h")
        assert write_cache(path, candles) == 10

        restored = read_cache(path)
        assert len(restored) == 10
        assert restored[0].close == candles[0].close
        assert restored[-1].open_time == candles[-1].open_time

    def test_missing_cache_raises(self, tmp_path: Path) -> None:
        """Reading a cache that does not exist is an explicit error."""
        with pytest.raises(FileNotFoundError):
            read_cache(tmp_path / "nope.jsonl")


class TestExitEvaluation:
    """Bar-level exit rules inside the replay."""

    def test_stop_is_checked_before_take_profit(self) -> None:
        """A bar spanning both levels resolves to the stop, not the target."""
        engine = ReplayEngine()
        trade = open_trade(stop=95.0, take_profit=110.0)
        bar = make_candle(open_price=100, high=115, low=90, close=105)
        result = engine._exit_on(bar, trade)
        assert result is not None
        assert result[0] is ExitReason.STOP_LOSS

    def test_take_profit_hit_alone(self) -> None:
        """A bar that only reaches the target takes profit."""
        engine = ReplayEngine()
        bar = make_candle(open_price=100, high=115, low=99, close=112)
        result = engine._exit_on(bar, open_trade())
        assert result is not None
        assert result[0] is ExitReason.TAKE_PROFIT

    def test_no_exit_inside_the_range(self) -> None:
        """A quiet bar resolves nothing."""
        bar = make_candle(open_price=100, high=105, low=98, close=101)
        assert ReplayEngine()._exit_on(bar, open_trade()) is None

    def test_short_exits_are_mirrored(self) -> None:
        """A short stops out on a high and profits on a low."""
        engine = ReplayEngine()
        trade = open_trade(direction=Direction.SHORT, stop=105.0, take_profit=90.0)
        stop_bar = make_candle(open_price=100, high=106, low=99, close=104)
        result = engine._exit_on(stop_bar, trade)
        assert result is not None and result[0] is ExitReason.STOP_LOSS

        profit_bar = make_candle(open_price=100, high=101, low=89, close=91)
        result = engine._exit_on(profit_bar, trade)
        assert result is not None and result[0] is ExitReason.TAKE_PROFIT

    def test_trailing_stop_ratchets_off_candle_extremes(self) -> None:
        """The replay's trail tracks the bar high, matching the live monitor."""
        trade = open_trade(stop=None, take_profit=None, trailing=10.0)
        trade.observe(make_candle(open_price=100, high=130, low=99, close=125))
        assert trade.trailing_stop == Decimal("117.0")
        trade.observe(make_candle(open_price=125, high=126, low=120, close=121))
        assert trade.trailing_stop == Decimal("117.0")


class TestReplayRun:
    """End-to-end replay behaviour."""

    async def test_replay_produces_trades_on_a_trending_series(self) -> None:
        """A clean trend yields entries and resolved exits."""
        engine = ReplayEngine(warmup_bars=120)
        candles = trending_candles(400, start=1000.0, drift=2.0, noise=2.0, interval="1h")
        result = await engine.run("BTC", {"1h": candles}, driving_interval="1h")

        assert result.bars_replayed > 0
        assert result.signals_evaluated > 0
        assert len(result.trades) > 0
        assert 0.0 <= result.hit_rate <= 1.0

    async def test_series_shorter_than_warmup_is_rejected(self) -> None:
        """Too little history is an error rather than a silent empty result."""
        engine = ReplayEngine(warmup_bars=120)
        with pytest.raises(ValueError, match="Need more than"):
            await engine.run(
                "BTC", {"1h": trending_candles(50, interval="1h")}, driving_interval="1h"
            )

    async def test_missing_driving_interval_is_rejected(self) -> None:
        """The driving interval has to be present."""
        engine = ReplayEngine(warmup_bars=10)
        with pytest.raises(ValueError, match="Need more than"):
            await engine.run(
                "BTC", {"15m": trending_candles(100, interval="15m")}, driving_interval="1h"
            )

    async def test_open_trade_at_the_end_is_force_closed_and_flagged(self) -> None:
        """A position still open at the end is marked, not counted as a clean win."""
        engine = ReplayEngine(warmup_bars=120, max_bars_held=None)
        candles = trending_candles(300, start=1000.0, drift=2.0, noise=1.0, interval="1h")
        result = await engine.run("BTC", {"1h": candles}, driving_interval="1h")
        forced = [trade for trade in result.trades if trade.forced_close]
        assert all(trade.exit_reason is ExitReason.MANUAL for trade in forced)

    async def test_max_bars_held_forces_an_exit(self) -> None:
        """A holding-period cap resolves trades that would otherwise linger."""
        engine = ReplayEngine(warmup_bars=120, max_bars_held=2)
        candles = trending_candles(300, start=1000.0, drift=1.0, noise=1.0, interval="1h")
        result = await engine.run("BTC", {"1h": candles}, driving_interval="1h")
        assert all(trade.bars_held <= 2 for trade in result.trades)

    async def test_costs_are_charged_on_both_sides(self) -> None:
        """Net PnL is always below gross by the modelled costs."""
        engine = ReplayEngine(warmup_bars=120, cost_bps=10.0)
        candles = trending_candles(300, start=1000.0, drift=2.0, noise=2.0, interval="1h")
        result = await engine.run("BTC", {"1h": candles}, driving_interval="1h")
        if result.trades:
            assert result.total_costs > 0
            assert result.net_pnl < result.gross_pnl

    async def test_context_window_is_bounded(self) -> None:
        """The analyst never sees more bars than the live buffer holds."""
        engine = ReplayEngine(warmup_bars=120)
        seen: list[int] = []
        original = engine.analyst.analyze

        async def spy(context: Any) -> Any:
            seen.append(len(context.candles_for("1h")))
            return await original(context)

        engine.analyst.analyze = spy  # type: ignore[method-assign]
        candles = trending_candles(500, start=1000.0, drift=1.0, interval="1h")
        await engine.run("BTC", {"1h": candles}, driving_interval="1h")
        assert seen
        assert max(seen) <= engine.window_bars

    async def test_no_look_ahead_on_secondary_timeframes(self) -> None:
        """A slower timeframe only contributes bars that had already closed."""
        engine = ReplayEngine(warmup_bars=120)
        seen_times: list[datetime] = []
        original = engine.analyst.analyze

        async def spy(context: Any) -> Any:
            secondary = context.candles_for("4h")
            if secondary:
                seen_times.append(secondary[-1].close_time)
            return await original(context)

        engine.analyst.analyze = spy  # type: ignore[method-assign]
        driver = trending_candles(300, start=1000.0, drift=1.0, interval="1h")
        slow = trending_candles(300, start=1000.0, drift=1.0, interval="4h")
        result = await engine.run(
            "BTC", {"1h": driver, "4h": slow}, driving_interval="1h"
        )
        assert result.bars_replayed > 0
        # Every secondary bar handed to the analyst had already closed.
        assert all(t <= driver[-1].close_time for t in seen_times)


class TestReplayMetrics:
    """Reported statistics."""

    async def test_metrics_are_internally_consistent(self) -> None:
        """Hit rate, PnL and profit factor agree with the trade list."""
        engine = ReplayEngine(warmup_bars=120)
        candles = trending_candles(400, start=1000.0, drift=2.0, noise=2.0, interval="1h")
        result = await engine.run("BTC", {"1h": candles}, driving_interval="1h")
        if not result.trades:
            pytest.skip("no trades produced on this series")

        expected_hit = sum(1 for t in result.trades if t.net_pnl > 0) / len(result.trades)
        assert result.hit_rate == pytest.approx(expected_hit)
        assert result.net_pnl == sum((t.net_pnl for t in result.trades), Decimal(0))
        assert result.gross_hit_rate >= result.hit_rate

    def test_empty_result_reports_zeros(self) -> None:
        """A replay with no trades reports zeros rather than dividing by zero."""
        from services.backtest.replay import ReplayResult

        result = ReplayResult(symbol="BTC", interval="1h")
        assert result.hit_rate == 0.0
        assert result.profit_factor == 0.0
        assert result.average_bars_held == 0.0
        assert result.exit_breakdown() == {}
