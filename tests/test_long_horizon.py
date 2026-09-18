"""Tests for the long-horizon signal library and its corrections.

Two corrections carry the weight of Phase 6's conclusions and are tested
hardest: measuring edges against buy-and-hold rather than zero, and discounting
pooled sample sizes for cross-symbol correlation.  Both push results toward
"nothing found", so both need to be right.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from itertools import pairwise

import pytest

from libs.schemas.market import Candle, FundingRate
from libs.schemas.signals import Direction
from services.research.long_horizon import (
    LONG_HORIZON_SIGNALS,
    MIN_TESTABLE_SAMPLE,
    _rolling_volatility,
    compute_benchmark,
    correlation_adjusted_tests,
    cross_sectional_momentum,
    drawdown_from_high,
    effective_sample_size,
    excess_statistics,
    funding_extreme,
    mean_pairwise_correlation,
    traded_bars,
    volatility_regime,
)
from services.research.signals import SignalWindow

BASE = datetime(2024, 1, 1, tzinfo=UTC)


def daily(prices: list[float], *, symbol: str = "BTC", volume: float = 100.0) -> tuple[Candle, ...]:
    """Build a daily candle series from a close-price path."""
    return tuple(
        Candle(
            source="test",
            symbol=symbol,
            interval="1d",
            open_time=BASE + timedelta(days=i),
            close_time=BASE + timedelta(days=i + 1),
            occurred_at=BASE + timedelta(days=i),
            open=Decimal(str(price)),
            high=Decimal(str(price * 1.01)),
            low=Decimal(str(price * 0.99)),
            close=Decimal(str(price)),
            volume=Decimal(str(volume)),
            trade_count=int(volume),
            is_closed=True,
        )
        for i, price in enumerate(prices)
    )


class Observation:
    """Minimal stand-in for a study observation."""

    def __init__(self, symbol: str, direction: Direction, gross_return: float) -> None:
        """Store the three fields the excess calculation reads."""
        self.symbol = symbol
        self.direction = direction
        self.gross_return = gross_return


class TestTradedBars:
    """Separating real trading from backfilled index prices."""

    def test_drops_the_zero_volume_prefix(self) -> None:
        """Backfilled bars carry no volume and are excluded."""
        series = daily([100.0] * 5, volume=0.0) + daily([101.0] * 5, volume=50.0)
        assert len(traded_bars(series)) == 5

    def test_keeps_everything_once_trading_starts(self) -> None:
        """A later zero-volume day inside traded history is retained."""
        series = daily([100.0] * 2, volume=0.0) + daily([101.0] * 3, volume=10.0)
        assert len(traded_bars(series)) == 3

    def test_all_synthetic_yields_nothing(self) -> None:
        """A series that never traded is unusable."""
        assert traded_bars(daily([100.0] * 5, volume=0.0)) == ()

    def test_all_traded_is_unchanged(self) -> None:
        """A fully traded series passes through."""
        series = daily([100.0] * 5, volume=10.0)
        assert len(traded_bars(series)) == 5


class TestRollingVolatility:
    """The optimised volatility series."""

    def test_matches_a_direct_computation(self) -> None:
        """Rolling sums reproduce the textbook standard deviation."""
        prices = [100 * math.exp(0.001 * i + 0.02 * math.sin(i / 5)) for i in range(200)]
        series = _rolling_volatility(prices, 30)

        window = range(len(prices) - 30, len(prices))
        returns = [math.log(prices[i] / prices[i - 1]) for i in window]
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        expected = math.sqrt(variance) * math.sqrt(365.0)

        assert series[-1] == pytest.approx(expected, rel=1e-9)

    def test_warmup_is_none(self) -> None:
        """Volatility is undefined before the window fills."""
        series = _rolling_volatility([100.0] * 50, 30)
        assert series[10] is None
        assert series[-1] is not None

    def test_constant_prices_have_zero_volatility(self) -> None:
        """A flat series has no dispersion."""
        assert _rolling_volatility([100.0] * 60, 30)[-1] == pytest.approx(0.0)

    def test_short_series_returns_all_none(self) -> None:
        """Too little data yields no estimates rather than an error."""
        assert all(value is None for value in _rolling_volatility([100.0] * 5, 30))


class TestSignalConventions:
    """Each signal fires in the direction its stated convention implies."""

    def test_momentum_is_long_above_the_average(self) -> None:
        """A rising series puts price above its trailing average."""
        window = SignalWindow(candles=daily([100.0 + i for i in range(250)]))
        assert LONG_HORIZON_SIGNALS["momentum_200d"].evaluate(window).direction is Direction.LONG

    def test_momentum_is_short_below_the_average(self) -> None:
        """A falling series puts price below it."""
        window = SignalWindow(candles=daily([500.0 - i for i in range(250)]))
        assert LONG_HORIZON_SIGNALS["momentum_200d"].evaluate(window).direction is Direction.SHORT

    def test_momentum_needs_its_full_lookback(self) -> None:
        """A series shorter than the lookback produces no view."""
        window = SignalWindow(candles=daily([100.0 + i for i in range(50)]))
        assert LONG_HORIZON_SIGNALS["momentum_200d"].evaluate(window).direction is Direction.FLAT

    def test_tsmom_follows_the_trailing_return(self) -> None:
        """Time-series momentum takes the sign of the trailing move."""
        up = SignalWindow(candles=daily([100.0 * (1.002**i) for i in range(150)]))
        down = SignalWindow(candles=daily([300.0 * (0.998**i) for i in range(150)]))
        assert LONG_HORIZON_SIGNALS["tsmom_90d"].evaluate(up).direction is Direction.LONG
        assert LONG_HORIZON_SIGNALS["tsmom_90d"].evaluate(down).direction is Direction.SHORT

    def test_drawdown_is_long_only(self) -> None:
        """The buy-the-dip convention never goes short."""
        prices = [100.0] * 100 + [100.0 - i for i in range(100)]
        window = SignalWindow(candles=daily(prices))
        reading = drawdown_from_high(window)
        assert reading.direction is Direction.LONG

    def test_drawdown_is_flat_near_the_high(self) -> None:
        """A shallow pullback does not trigger the signal."""
        window = SignalWindow(candles=daily([100.0 + i * 0.01 for i in range(250)]))
        assert drawdown_from_high(window).direction is Direction.FLAT

    def test_volatility_regime_is_long_when_calm(self) -> None:
        """Low realised volatility reads bullish under the tested convention."""
        noisy = [100.0 * (1 + 0.05 * math.sin(i)) for i in range(200)]
        calm = [float(noisy[-1])] * 60
        window = SignalWindow(candles=daily(noisy + calm))
        assert volatility_regime(window).direction is Direction.LONG

    def test_funding_extreme_is_contrarian(self) -> None:
        """High funding reads short, low funding reads long."""
        base = [0.0001] * 60
        high = SignalWindow(
            candles=daily([100.0] * 100),
            funding=_funding([*base, 0.01]),
        )
        low = SignalWindow(
            candles=daily([100.0] * 100),
            funding=_funding([*base, -0.01]),
        )
        assert funding_extreme(high).direction is Direction.SHORT
        assert funding_extreme(low).direction is Direction.LONG

    def test_funding_extreme_is_flat_mid_distribution(self) -> None:
        """A typical funding reading produces no view."""
        window = SignalWindow(
            candles=daily([100.0] * 100),
            funding=_funding([0.0001 * (i % 7) for i in range(80)]),
        )
        assert funding_extreme(window).direction is Direction.FLAT

    def test_funding_extreme_without_data_is_flat(self) -> None:
        """No funding history means no opinion."""
        assert funding_extreme(SignalWindow(candles=daily([100.0] * 100))).direction is (
            Direction.FLAT
        )

    def test_every_signal_survives_a_short_window(self) -> None:
        """No signal raises when handed less history than it needs."""
        window = SignalWindow(candles=daily([100.0, 101.0, 102.0]))
        for spec in LONG_HORIZON_SIGNALS.values():
            reading = spec.evaluate(window)
            assert 0.0 <= reading.conviction <= 1.0

    def test_every_signal_declares_a_convention(self) -> None:
        """A result is meaningless without stating which reading was tested."""
        for spec in LONG_HORIZON_SIGNALS.values():
            assert spec.convention and spec.description


def _funding(rates: list[float]) -> tuple[FundingRate, ...]:
    """Build a daily funding series."""
    return tuple(
        FundingRate(
            source="test",
            symbol="BTC",
            funding_rate=Decimal(str(rate)),
            occurred_at=BASE + timedelta(days=i),
        )
        for i, rate in enumerate(rates)
    )


class TestBenchmark:
    """Measuring against buy-and-hold rather than zero."""

    def test_benchmark_is_the_unconditional_forward_return(self) -> None:
        """The benchmark is what every window earned on average."""
        benchmark = compute_benchmark("BTC", [100.0, 110.0, 121.0, 133.1], 1)
        assert benchmark.mean_return == pytest.approx(0.10, rel=1e-9)
        assert benchmark.up_rate == pytest.approx(1.0)

    def test_benchmark_up_rate_counts_positive_windows(self) -> None:
        """Half up, half down reads 50%."""
        benchmark = compute_benchmark("BTC", [100.0, 110.0, 100.0, 110.0, 100.0], 1)
        assert benchmark.up_rate == pytest.approx(0.5)

    def test_always_long_signal_has_zero_excess(self) -> None:
        """A signal that is simply long earns the drift and no more.

        This is the confound the correction exists to remove: without it, a
        long-biased signal in a rising market scores as skill.
        """
        benchmark = compute_benchmark("BTC", [100.0 * 1.01**i for i in range(50)], 1)
        observations = [
            Observation("BTC", Direction.LONG, benchmark.mean_return) for _ in range(40)
        ]
        stats = excess_statistics(observations, {"BTC": benchmark})
        assert stats.mean_excess_bps == pytest.approx(0.0, abs=1e-6)
        assert stats.mean_benchmark_bps == pytest.approx(benchmark.mean_return * 10_000)

    def test_genuine_timing_shows_positive_excess(self) -> None:
        """Beating the drift registers as excess."""
        benchmark = compute_benchmark("BTC", [100.0 * 1.01**i for i in range(50)], 1)
        observations = [
            Observation("BTC", Direction.LONG, benchmark.mean_return + 0.02) for _ in range(40)
        ]
        stats = excess_statistics(observations, {"BTC": benchmark})
        assert stats.mean_excess_bps == pytest.approx(200.0, rel=1e-6)

    def test_short_is_credited_with_the_drift_it_avoided(self) -> None:
        """A short in a rising market is measured against minus the drift."""
        benchmark = compute_benchmark("BTC", [100.0 * 1.01**i for i in range(50)], 1)
        # A short that loses exactly the drift has zero excess: it did what a
        # permanent short would have done, no better and no worse.
        observations = [
            Observation("BTC", Direction.SHORT, -benchmark.mean_return) for _ in range(20)
        ]
        stats = excess_statistics(observations, {"BTC": benchmark})
        assert stats.mean_excess_bps == pytest.approx(0.0, abs=1e-6)

    def test_null_probability_reflects_directional_bias(self) -> None:
        """A long-only signal's null is the market up-rate, not 50%."""
        benchmark = compute_benchmark("BTC", [100.0, 110.0, 121.0, 133.1, 146.4], 1)
        observations = [Observation("BTC", Direction.LONG, 0.05) for _ in range(10)]
        stats = excess_statistics(observations, {"BTC": benchmark})
        assert stats.null_probability == pytest.approx(1.0)
        assert stats.long_fraction == pytest.approx(1.0)

    def test_empty_observations(self) -> None:
        """No observations yields a neutral result rather than a crash."""
        stats = excess_statistics([], {})
        assert stats.n == 0
        assert stats.excess_p_value == 1.0


class TestPoolingCorrection:
    """Correlated symbols do not supply independent information."""

    def test_uncorrelated_symbols_pool_fully(self) -> None:
        """At zero correlation the effective size is the raw pooled count."""
        assert effective_sample_size(100, 10, 0.0) == pytest.approx(1000)

    def test_perfectly_correlated_symbols_add_nothing(self) -> None:
        """At correlation one, twenty symbols are worth one."""
        assert effective_sample_size(100, 20, 1.0) == pytest.approx(100)

    def test_realistic_crypto_correlation_collapses_the_gain(self) -> None:
        """At 0.8 correlation, 20 symbols are worth barely more than one."""
        effective = effective_sample_size(100, 20, 0.8)
        assert effective == pytest.approx(2000 / 16.2, rel=1e-6)
        assert effective < 130

    def test_single_symbol_is_unchanged(self) -> None:
        """One symbol cannot be discounted."""
        assert effective_sample_size(50, 1, 0.9) == pytest.approx(50)

    def test_rejects_zero_symbols(self) -> None:
        """Pooling nothing is invalid."""
        with pytest.raises(ValueError, match="symbols"):
            effective_sample_size(10, 0, 0.5)

    def test_mean_correlation_of_identical_series_is_one(self) -> None:
        """Identical series correlate perfectly."""
        series = [0.01, -0.02, 0.03, 0.01, -0.01, 0.02]
        assert mean_pairwise_correlation({"A": series, "B": series}) == pytest.approx(1.0)

    def test_mean_correlation_of_opposed_series(self) -> None:
        """Mirrored series correlate at minus one."""
        series = [0.01, -0.02, 0.03, 0.01, -0.01]
        mirror = [-value for value in series]
        assert mean_pairwise_correlation({"A": series, "B": mirror}) == pytest.approx(-1.0)

    def test_single_series_has_no_correlation(self) -> None:
        """One series has no pair to correlate with."""
        assert mean_pairwise_correlation({"A": [0.1, 0.2, 0.3]}) == 0.0


class TestAdjustedTests:
    """p-values recomputed at the effective sample size."""

    def test_shrinking_the_sample_weakens_significance(self) -> None:
        """The same effect at fewer effective observations is less significant."""
        excess = [0.01] * 50 + [-0.005] * 50
        _, full = correlation_adjusted_tests(
            excess, wins=60, null_probability=0.5, effective_n=100
        )
        _, reduced = correlation_adjusted_tests(
            excess, wins=60, null_probability=0.5, effective_n=20
        )
        assert reduced > full

    def test_effective_size_is_capped_at_the_raw_count(self) -> None:
        """A correction can shrink a sample, never inflate it."""
        excess = [0.01, -0.005] * 25
        _, capped = correlation_adjusted_tests(
            excess, wins=25, null_probability=0.5, effective_n=10_000
        )
        _, raw = correlation_adjusted_tests(
            excess, wins=25, null_probability=0.5, effective_n=len(excess)
        )
        assert capped == pytest.approx(raw)

    def test_zero_variance_is_not_evidence(self) -> None:
        """A constant excess series cannot be tested."""
        _, p = correlation_adjusted_tests(
            [0.01] * 40, wins=40, null_probability=0.5, effective_n=40
        )
        assert p == 1.0

    def test_tiny_sample_is_not_evidence(self) -> None:
        """Fewer than two observations yields no conclusion."""
        assert correlation_adjusted_tests([0.1], wins=1, null_probability=0.5, effective_n=1) == (
            1.0,
            1.0,
        )


class TestCrossSectional:
    """Long-top / short-bottom ranking."""

    def test_ranking_picks_the_extremes(self) -> None:
        """The best and worst trailing performers form the two legs."""
        closes = {
            "WIN": daily([100.0 * 1.02**i for i in range(60)], symbol="WIN"),
            "MID": daily([100.0] * 60, symbol="MID"),
            "LOSE": daily([100.0 * 0.98**i for i in range(60)], symbol="LOSE"),
            "A": daily([100.0 * 1.01**i for i in range(60)], symbol="A"),
            "B": daily([100.0 * 0.99**i for i in range(60)], symbol="B"),
            "C": daily([100.0] * 60, symbol="C"),
        }
        observations = cross_sectional_momentum(
            closes, lookback=20, horizon=5, quantile=0.2, min_symbols=6
        )
        assert observations
        first = observations[0]
        assert "WIN" in first.long_symbols
        assert "LOSE" in first.short_symbols

    def test_independent_observations_are_spaced_by_the_horizon(self) -> None:
        """Non-overlapping rebalances are one horizon apart."""
        closes = {
            f"S{i}": daily([100.0 * (1 + 0.001 * i) ** j for j in range(200)], symbol=f"S{i}")
            for i in range(8)
        }
        observations = cross_sectional_momentum(closes, lookback=20, horizon=10, min_symbols=6)
        indices = [o.index for o in observations if o.independent]
        assert all(b - a >= 10 for a, b in pairwise(indices))

    def test_costs_are_charged_on_both_legs(self) -> None:
        """The spread pays a round trip on each side."""
        closes = {
            f"S{i}": daily([100.0 * (1 + 0.001 * i) ** j for j in range(120)], symbol=f"S{i}")
            for i in range(8)
        }
        observations = cross_sectional_momentum(
            closes, lookback=20, horizon=5, cost_bps=10.0, min_symbols=6
        )
        assert observations
        for observation in observations:
            assert observation.net_return == pytest.approx(observation.gross_return - 0.002)

    def test_too_few_symbols_yields_nothing(self) -> None:
        """A universe below the minimum is not ranked."""
        closes = {"A": daily([100.0] * 100, symbol="A"), "B": daily([101.0] * 100, symbol="B")}
        assert cross_sectional_momentum(closes, lookback=10, horizon=5, min_symbols=6) == []

    def test_rejects_bad_parameters(self) -> None:
        """Degenerate parameters are errors."""
        closes = {f"S{i}": daily([100.0] * 50, symbol=f"S{i}") for i in range(8)}
        with pytest.raises(ValueError, match="lookback and horizon"):
            cross_sectional_momentum(closes, lookback=0, horizon=5)
        with pytest.raises(ValueError, match="quantile"):
            cross_sectional_momentum(closes, lookback=10, horizon=5, quantile=0.9)


    def test_misaligned_histories_are_aligned_by_date(self) -> None:
        """Symbols with different listing dates are compared on the same day.

        Regression test for a bug that indexed every symbol by the same integer
        offset. With histories of different lengths that compared one symbol's
        first month against another's second year, manufacturing a large and
        highly significant "signal" out of calendar misalignment alone.
        """
        # Two identical price paths; one symbol simply lists 40 days later.
        path = [100.0 * 1.01**i for i in range(120)]
        full = daily(path, symbol="OLD")
        late = daily(path, symbol="NEW")[40:]
        universe = {
            "OLD": full,
            "NEW": late,
            **{f"S{i}": daily(path, symbol=f"S{i}") for i in range(4)},
        }
        observations = cross_sectional_momentum(
            universe, lookback=10, horizon=5, min_symbols=6
        )
        # Every symbol follows the same path, so on any correctly aligned date
        # the spread between best and worst performer is exactly zero.
        assert observations
        assert all(abs(o.gross_return) < 1e-9 for o in observations)


def test_min_testable_sample_is_documented() -> None:
    """The untestable floor is a stated constant, not a magic number."""
    assert MIN_TESTABLE_SAMPLE == 30
