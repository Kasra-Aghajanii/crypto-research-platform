"""Unit tests for the indicator library."""

from __future__ import annotations

import pytest

from services.agents.market_analyst import indicators as ind


class TestMovingAverages:
    """SMA and EMA behaviour."""

    def test_sma_matches_hand_calculation(self) -> None:
        """SMA of a known series matches the arithmetic mean of each window."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = ind.sma(values, 3)
        assert result[:2] == [None, None]
        assert result[2] == pytest.approx(2.0)
        assert result[3] == pytest.approx(3.0)
        assert result[4] == pytest.approx(4.0)

    def test_sma_is_aligned_to_input_length(self) -> None:
        """Output length always equals input length."""
        assert len(ind.sma([1.0] * 10, 4)) == 10

    def test_sma_all_none_when_series_too_short(self) -> None:
        """A series shorter than the period yields no values."""
        assert ind.sma([1.0, 2.0], 5) == [None, None]

    def test_ema_seeds_from_sma_then_smooths(self) -> None:
        """The first EMA value is the seed SMA; later values follow the recursion."""
        values = [2.0, 4.0, 6.0, 8.0, 10.0]
        result = ind.ema(values, 3)
        assert result[2] == pytest.approx(4.0)
        multiplier = 2.0 / 4.0
        assert result[3] == pytest.approx((8.0 - 4.0) * multiplier + 4.0)

    def test_ema_tracks_a_constant_series_exactly(self) -> None:
        """A flat series produces a flat EMA."""
        result = ind.ema([5.0] * 20, 10)
        assert result[-1] == pytest.approx(5.0)

    def test_invalid_period_is_rejected(self) -> None:
        """A period below 1 raises."""
        with pytest.raises(ind.IndicatorError):
            ind.sma([1.0, 2.0], 0)


class TestRsi:
    """RSI edge cases and known values."""

    def test_monotonic_rise_saturates_at_100(self) -> None:
        """An unbroken uptrend has no losses, so RSI pins to 100."""
        values = [float(i) for i in range(1, 40)]
        assert ind.rsi(values, 14)[-1] == pytest.approx(100.0)

    def test_monotonic_fall_saturates_at_zero(self) -> None:
        """An unbroken downtrend has no gains, so RSI pins to 0."""
        values = [float(i) for i in range(40, 1, -1)]
        assert ind.rsi(values, 14)[-1] == pytest.approx(0.0)

    def test_flat_series_is_neutral(self) -> None:
        """With no gains and no losses RSI is defined as neutral 50."""
        assert ind.rsi([10.0] * 30, 14)[-1] == pytest.approx(50.0)

    def test_warmup_region_is_none(self) -> None:
        """RSI is undefined until period+1 bars exist."""
        result = ind.rsi([float(i) for i in range(20)], 14)
        assert result[13] is None
        assert result[14] is not None


class TestMacd:
    """MACD construction."""

    def test_histogram_is_macd_minus_signal(self) -> None:
        """The histogram is exactly the difference of the two lines."""
        values = [100.0 + i for i in range(80)]
        result = ind.macd(values, 12, 26, 9)
        macd_value, signal_value, histogram = (
            result.macd[-1],
            result.signal[-1],
            result.histogram[-1],
        )
        assert macd_value is not None and signal_value is not None and histogram is not None
        assert histogram == pytest.approx(macd_value - signal_value)

    def test_uptrend_gives_positive_macd(self) -> None:
        """A rising series puts the fast EMA above the slow EMA."""
        values = [100.0 + i * 0.8 for i in range(80)]
        macd_value = ind.macd(values).macd[-1]
        assert macd_value is not None and macd_value > 0

    def test_fast_must_be_shorter_than_slow(self) -> None:
        """An inverted period pair raises."""
        with pytest.raises(ind.IndicatorError):
            ind.macd([1.0] * 50, fast=26, slow=12)


class TestBollinger:
    """Bollinger band construction."""

    def test_bands_bracket_the_middle(self) -> None:
        """Upper > middle > lower for a series with variance."""
        values = [100.0 + (i % 5) for i in range(60)]
        bands = ind.bollinger(values, 20, 2.0)
        upper, middle, lower = bands.upper[-1], bands.middle[-1], bands.lower[-1]
        assert upper is not None and middle is not None and lower is not None
        assert upper > middle > lower

    def test_flat_series_collapses_the_bands(self) -> None:
        """Zero variance means zero width and a defined mid %B."""
        bands = ind.bollinger([50.0] * 40, 20, 2.0)
        assert bands.bandwidth[-1] == pytest.approx(0.0)
        assert bands.percent_b[-1] == pytest.approx(0.5)

    def test_percent_b_above_one_when_price_breaks_out(self) -> None:
        """A close above the upper band gives %B > 1."""
        values = [100.0] * 25 + [130.0]
        bands = ind.bollinger(values, 20, 2.0)
        percent_b = bands.percent_b[-1]
        assert percent_b is not None and percent_b > 1.0


class TestAtrAndVolume:
    """ATR, OBV and slope helpers."""

    def test_atr_of_constant_range_equals_that_range(self) -> None:
        """A constant bar range produces an ATR equal to it."""
        closes = [100.0] * 30
        highs = [101.0] * 30
        lows = [99.0] * 30
        assert ind.atr(highs, lows, closes, 14)[-1] == pytest.approx(2.0)

    def test_true_range_uses_the_gap_when_price_jumps(self) -> None:
        """A gap up makes |high - previous close| the true range."""
        ranges = ind.true_range([10.0, 20.0], [9.0, 19.0], [9.5, 19.5])
        assert ranges[1] == pytest.approx(20.0 - 9.5)

    def test_mismatched_series_lengths_are_rejected(self) -> None:
        """ATR inputs must be the same length."""
        with pytest.raises(ind.IndicatorError):
            ind.true_range([1.0, 2.0], [1.0], [1.0, 2.0])

    def test_obv_accumulates_with_direction(self) -> None:
        """OBV adds volume on up closes and subtracts on down closes."""
        obv = ind.on_balance_volume([10.0, 11.0, 10.5], [0.0, 100.0, 50.0])
        assert obv == [0.0, 100.0, 50.0]

    def test_normalized_slope_sign_follows_the_trend(self) -> None:
        """Rising series slope positive, falling series slope negative."""
        assert ind.normalized_slope([1.0, 2.0, 3.0, 4.0], 4) > 0
        assert ind.normalized_slope([4.0, 3.0, 2.0, 1.0], 4) < 0

    def test_normalized_slope_of_flat_series_is_zero(self) -> None:
        """A flat series has no slope."""
        assert ind.normalized_slope([7.0] * 10, 10) == pytest.approx(0.0)


class TestHelpers:
    """Crossover and clamping helpers."""

    def test_crossed_above_detects_only_the_crossing_bar(self) -> None:
        """A cross is reported on the bar where the order flips."""
        fast: ind.Series = [1.0, 3.0]
        slow: ind.Series = [2.0, 2.0]
        assert ind.crossed_above(fast, slow) is True
        assert ind.crossed_below(fast, slow) is False

    def test_no_cross_when_already_above(self) -> None:
        """Staying above is not a crossing."""
        assert ind.crossed_above([3.0, 4.0], [1.0, 2.0]) is False

    def test_cross_helpers_tolerate_warmup_none(self) -> None:
        """Undefined warm-up values never report a crossing."""
        assert ind.crossed_above([None, 1.0], [None, 0.0]) is False

    def test_clamp_bounds_values(self) -> None:
        """Clamp keeps values inside the requested range."""
        assert ind.clamp(5.0) == 1.0
        assert ind.clamp(-5.0) == -1.0
        assert ind.clamp(0.25) == 0.25

    def test_last_defined_skips_trailing_none(self) -> None:
        """The most recent non-None value is returned."""
        assert ind.last_defined([1.0, 2.0, None]) == 2.0
        assert ind.last_defined([None, None]) is None
