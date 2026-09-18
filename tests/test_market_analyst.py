"""Tests for the Market Analyst agent and its support/resistance detection."""

from __future__ import annotations

from decimal import Decimal

import pytest

from libs.schemas.signals import Direction
from services.agents.common.context import AgentContext, MarketContextBuilder
from services.agents.market_analyst.agent import MarketAnalystAgent
from services.agents.market_analyst.levels import cluster_pivots, detect_levels, find_pivots
from tests.conftest import make_book, make_candle, trending_candles


@pytest.fixture
def analyst() -> MarketAnalystAgent:
    """Return a Market Analyst agent built from default settings."""
    return MarketAnalystAgent()


def context_from(candles: dict[str, tuple], *, symbol: str = "BTC", mid: float | None = None):
    """Build an AgentContext directly from candle series."""
    return AgentContext(
        symbol=symbol,
        candles=candles,
        order_book=make_book(symbol=symbol, mid=mid) if mid is not None else None,
    )


class TestPivotDetection:
    """Swing pivot and clustering behaviour."""

    def test_finds_an_obvious_swing_high(self) -> None:
        """A single peak with room on both sides is a confirmed pivot."""
        highs = [1.0, 2.0, 3.0, 10.0, 3.0, 2.0, 1.0]
        lows = [h - 0.5 for h in highs]
        pivots = find_pivots(highs, lows, lookback=3)
        assert any(pivot.kind == "high" and pivot.price == 10.0 for pivot in pivots)

    def test_recent_bars_cannot_be_pivots(self) -> None:
        """A pivot needs bars on both sides, so the last bars are never pivots."""
        highs = [1.0, 2.0, 3.0, 4.0, 99.0]
        lows = [h - 0.5 for h in highs]
        pivots = find_pivots(highs, lows, lookback=3)
        assert all(pivot.index < len(highs) - 3 for pivot in pivots)

    def test_flat_series_has_no_pivots(self) -> None:
        """Without a strict extreme there is nothing to confirm."""
        assert find_pivots([5.0] * 20, [4.0] * 20, lookback=2) == []

    def test_mismatched_lengths_rejected(self) -> None:
        """High and low series must line up."""
        with pytest.raises(ValueError, match="same length"):
            find_pivots([1.0, 2.0], [1.0], lookback=1)

    def test_nearby_pivots_cluster_into_one_level(self) -> None:
        """Pivots within the cluster distance merge and count as touches."""
        pivots = find_pivots(
            [1.0, 2.0, 10.0, 2.0, 1.0, 2.0, 10.02, 2.0, 1.0],
            [0.5] * 9,
            lookback=2,
        )
        levels = cluster_pivots(pivots, cluster_pct=1.0, total_bars=9)
        assert any(level.touches >= 2 for level in levels)

    def test_cluster_pct_must_be_positive(self) -> None:
        """A non-positive cluster distance is rejected."""
        with pytest.raises(ValueError, match="cluster_pct"):
            cluster_pivots([], cluster_pct=0.0, total_bars=10)


class TestLevelDetection:
    """Classification of levels into support and resistance."""

    def test_levels_split_around_current_price(self) -> None:
        """Levels below price are support; levels above are resistance."""
        candles = trending_candles(120, drift=0.0, noise=3.0)
        highs = [float(candle.high) for candle in candles]
        lows = [float(candle.low) for candle in candles]
        price = float(candles[-1].close)
        levels = detect_levels(highs, lows, price, lookback=3, cluster_pct=0.5)
        for level in levels:
            if level.kind == "support":
                assert float(level.price) < price
            else:
                assert float(level.price) > price

    def test_distance_pct_sign_matches_side(self) -> None:
        """Support sits at a negative distance, resistance at a positive one."""
        candles = trending_candles(120, drift=0.0, noise=3.0)
        levels = detect_levels(
            [float(c.high) for c in candles],
            [float(c.low) for c in candles],
            float(candles[-1].close),
        )
        for level in levels:
            assert (level.distance_pct < 0) is (level.kind == "support")

    def test_no_levels_without_data(self) -> None:
        """An empty series yields no levels rather than raising."""
        assert detect_levels([], [], 100.0) == ()


class TestTimeframeAnalysis:
    """Per-timeframe scoring."""

    def test_uptrend_scores_positive(self, analyst: MarketAnalystAgent) -> None:
        """A steady uptrend produces a positive composite score."""
        analysis = analyst.analyze_timeframe("15m", trending_candles(200, drift=0.6))
        assert analysis.composite_score > 0
        assert analysis.trend_score > 0

    def test_downtrend_scores_negative(self, analyst: MarketAnalystAgent) -> None:
        """A steady downtrend produces a negative composite score."""
        analysis = analyst.analyze_timeframe("15m", trending_candles(200, start=300.0, drift=-0.6))
        assert analysis.composite_score < 0
        assert analysis.trend_score < 0

    def test_short_series_returns_zero_score(self, analyst: MarketAnalystAgent) -> None:
        """Below the warm-up bar count nothing is scored."""
        analysis = analyst.analyze_timeframe("15m", trending_candles(10))
        assert analysis.composite_score == 0.0
        assert analysis.candles_used == 10

    def test_all_indicators_are_populated(self, analyst: MarketAnalystAgent) -> None:
        """Every indicator the agent advertises is present on a full series."""
        analysis = analyst.analyze_timeframe("1h", trending_candles(200, drift=0.4))
        for field in ("rsi", "macd", "macd_signal", "ema_fast", "ema_slow", "bb_upper", "atr"):
            assert getattr(analysis, field) is not None, field


class TestSignalGeneration:
    """End-to-end behaviour of ``analyze``."""

    async def test_uptrend_produces_a_long_signal(self, analyst: MarketAnalystAgent) -> None:
        """Agreeing uptrends across timeframes yield a long view."""
        context = context_from(
            {
                "15m": trending_candles(200, drift=0.6, interval="15m"),
                "1h": trending_candles(200, drift=0.6, interval="1h"),
            },
            mid=None,
        )
        signal = await analyst.analyze(context)
        assert signal is not None
        assert signal.direction is Direction.LONG
        assert signal.confidence > 0
        assert signal.signed_confidence > 0

    async def test_downtrend_produces_a_short_signal(self, analyst: MarketAnalystAgent) -> None:
        """Agreeing downtrends yield a short view."""
        context = context_from(
            {
                "15m": trending_candles(200, start=400.0, drift=-0.6, interval="15m"),
                "1h": trending_candles(200, start=400.0, drift=-0.6, interval="1h"),
            }
        )
        signal = await analyst.analyze(context)
        assert signal is not None
        assert signal.direction is Direction.SHORT

    async def test_warmup_returns_no_signal(self, analyst: MarketAnalystAgent) -> None:
        """The agent stays silent until a timeframe has enough history."""
        context = context_from({"15m": trending_candles(20)})
        assert await analyst.analyze(context) is None

    async def test_long_signal_has_stop_below_and_target_above(
        self, analyst: MarketAnalystAgent
    ) -> None:
        """Protective levels are placed on the correct side of the entry."""
        context = context_from({"1h": trending_candles(200, drift=0.6, interval="1h")})
        signal = await analyst.analyze(context)
        assert signal is not None and signal.direction is Direction.LONG
        assert signal.reference_price is not None and signal.suggested_stop is not None
        assert signal.suggested_stop < signal.reference_price
        if signal.suggested_take_profit is not None:
            assert signal.suggested_take_profit > signal.reference_price

    async def test_signal_carries_features_and_expiry(self, analyst: MarketAnalystAgent) -> None:
        """Features feed later scoring; the TTL prevents stale signals trading."""
        context = context_from({"1h": trending_candles(200, drift=0.5, interval="1h")})
        signal = await analyst.analyze(context)
        assert signal is not None
        assert "confluence_score" in signal.features
        assert "timeframe_agreement" in signal.features
        assert signal.valid_until is not None
        assert signal.is_expired() is False

    async def test_book_features_included_when_present(self, analyst: MarketAnalystAgent) -> None:
        """Order book context adds imbalance and spread features."""
        context = context_from({"1h": trending_candles(200, drift=0.5, interval="1h")}, mid=200.0)
        signal = await analyst.analyze(context)
        assert signal is not None
        assert "book_imbalance" in signal.features
        assert "spread_bps" in signal.features

    async def test_conflicting_timeframes_reduce_confidence(
        self, analyst: MarketAnalystAgent
    ) -> None:
        """Disagreement between timeframes lowers confidence versus agreement."""
        agreeing = context_from(
            {
                "15m": trending_candles(200, drift=0.6, interval="15m"),
                "1h": trending_candles(200, drift=0.6, interval="1h"),
            }
        )
        conflicting = context_from(
            {
                "15m": trending_candles(200, start=400.0, drift=-0.6, interval="15m"),
                "1h": trending_candles(200, drift=0.6, interval="1h"),
            }
        )
        agree_signal = await analyst.analyze(agreeing)
        conflict_signal = await analyst.analyze(conflicting)
        assert agree_signal is not None and conflict_signal is not None
        assert conflict_signal.confidence < agree_signal.confidence


class TestContextBuilder:
    """The pre-loading rule: agents receive context, they never fetch."""

    def test_only_closed_candles_are_buffered(self) -> None:
        """An in-progress candle must never reach an agent."""
        builder = MarketContextBuilder(symbols=["BTC"], intervals=["15m"], max_candles=50)
        stored = builder.add_candle(
            make_candle(open_price=1, high=2, low=0.5, close=1.5, is_closed=False)
        )
        assert stored is False
        assert builder.build("BTC").candles_for("15m") == ()

    def test_repeated_open_time_replaces_rather_than_appends(self) -> None:
        """A corrected candle overwrites instead of duplicating the bar."""
        builder = MarketContextBuilder(symbols=["BTC"], intervals=["15m"], max_candles=50)
        builder.add_candle(make_candle(index=0, open_price=1, high=2, low=0.5, close=1.5))
        builder.add_candle(make_candle(index=0, open_price=1, high=2, low=0.5, close=1.9))
        series = builder.build("BTC").candles_for("15m")
        assert len(series) == 1
        assert series[0].close == Decimal("1.9")

    def test_buffer_is_bounded(self) -> None:
        """Buffers cannot grow without limit."""
        builder = MarketContextBuilder(symbols=["BTC"], intervals=["15m"], max_candles=10)
        for i in range(50):
            builder.add_candle(make_candle(index=i, open_price=1, high=2, low=0.5, close=1.5))
        assert len(builder.build("BTC").candles_for("15m")) == 10

    def test_reference_price_prefers_the_book_mid(self) -> None:
        """The freshest price wins: book mid over the last close."""
        builder = MarketContextBuilder(symbols=["BTC"], intervals=["15m"], max_candles=10)
        builder.add_candle(make_candle(open_price=1, high=2, low=0.5, close=1.5))
        builder.add_order_book(make_book(mid=123.0))
        assert builder.build("BTC").reference_price == Decimal("123.0")

    def test_portfolio_is_withheld_unless_requested(self) -> None:
        """An agent only sees portfolio state when it declares that it needs it."""
        builder = MarketContextBuilder(symbols=["BTC"], intervals=["15m"], max_candles=10)
        assert builder.build("BTC", include_portfolio=False).portfolio is None

    def test_context_is_immutable(self) -> None:
        """The context handed to an agent cannot be mutated by it."""
        context = AgentContext(symbol="BTC")
        with pytest.raises(ValueError, match=r"frozen|Instance is frozen"):
            context.symbol = "ETH"  # type: ignore[misc]
