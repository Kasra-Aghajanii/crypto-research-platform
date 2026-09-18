"""Tests for the funding-carry study and the statistical power calculations."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from itertools import pairwise

import pytest

from libs.schemas.market import FundingRate
from services.research.basis import CarryStudy, pool_carry, profile_funding
from services.research.power import (
    bars_per_day,
    days_to_collect,
    normal_cdf,
    normal_quantile,
    samples_for_hit_rate,
    samples_for_mean_return,
    standard_deviation_bps,
)

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def funding_series(
    rates: list[float], premiums: list[float] | None = None, symbol: str = "BTC"
) -> tuple[FundingRate, ...]:
    """Build an hourly funding series."""
    premium_values = premiums if premiums is not None else [0.0] * len(rates)
    return tuple(
        FundingRate(
            source="test",
            symbol=symbol,
            funding_rate=Decimal(str(rate)),
            premium=Decimal(str(premium)),
            occurred_at=BASE + timedelta(hours=i),
        )
        for i, (rate, premium) in enumerate(zip(rates, premium_values, strict=True))
    )


class TestNormalDistribution:
    """The quantile function underpinning every power calculation."""

    def test_cdf_at_zero(self) -> None:
        """The standard normal is centred on zero."""
        assert normal_cdf(0.0) == pytest.approx(0.5)

    def test_quantile_inverts_the_cdf(self) -> None:
        """Quantile and CDF are inverses."""
        for p in (0.025, 0.1, 0.5, 0.9, 0.975):
            assert normal_cdf(normal_quantile(p)) == pytest.approx(p, abs=1e-9)

    def test_known_critical_values(self) -> None:
        """The textbook 95% and 80% points come out right."""
        assert normal_quantile(0.975) == pytest.approx(1.959964, abs=1e-5)
        assert normal_quantile(0.80) == pytest.approx(0.841621, abs=1e-5)

    def test_rejects_out_of_range(self) -> None:
        """A probability outside (0, 1) is invalid."""
        with pytest.raises(ValueError, match=r"\(0, 1\)"):
            normal_quantile(0.0)


class TestSampleSizes:
    """Required sample sizes."""

    def test_smaller_effects_need_more_data(self) -> None:
        """Detecting 52% takes far more observations than detecting 60%."""
        small = samples_for_hit_rate(0.52).observations
        large = samples_for_hit_rate(0.60).observations
        assert small > large * 10

    def test_hit_rate_requirement_is_symmetric(self) -> None:
        """Detecting 45% takes as much data as detecting 55%."""
        assert samples_for_hit_rate(0.55).observations == pytest.approx(
            samples_for_hit_rate(0.45).observations, rel=0.02
        )

    def test_known_hit_rate_requirement(self) -> None:
        """A 55% hit rate at 80% power needs roughly 780 observations."""
        assert samples_for_hit_rate(0.55).observations == pytest.approx(783, rel=0.02)

    def test_zero_effect_is_undetectable(self) -> None:
        """Exactly 50% cannot be distinguished from 50%."""
        with pytest.raises(ValueError, match="50%"):
            samples_for_hit_rate(0.5)

    def test_return_requirement_scales_with_variance(self) -> None:
        """Four times the variance needs four times the sample."""
        low = samples_for_mean_return(10.0, 50.0).observations
        high = samples_for_mean_return(10.0, 100.0).observations
        assert high == pytest.approx(low * 4, rel=0.01)

    def test_return_requirement_scales_inversely_with_effect(self) -> None:
        """Halving the effect quadruples the sample needed."""
        big = samples_for_mean_return(20.0, 100.0).observations
        small = samples_for_mean_return(10.0, 100.0).observations
        assert small == pytest.approx(big * 4, rel=0.01)

    def test_zero_effect_or_deviation_rejected(self) -> None:
        """Degenerate inputs are errors, not infinities."""
        with pytest.raises(ValueError, match="zero effect"):
            samples_for_mean_return(0.0, 10.0)
        with pytest.raises(ValueError, match="Standard deviation"):
            samples_for_mean_return(10.0, 0.0)


class TestCalendarTime:
    """Translating sample sizes into recording time."""

    def test_bars_per_day(self) -> None:
        """Bar counts follow from the interval length."""
        assert bars_per_day(3600) == 24
        assert bars_per_day(60) == 1440

    def test_longer_horizons_cost_proportionally_more_time(self) -> None:
        """Independent observations are spaced one horizon apart."""
        short = days_to_collect(100, horizon_bars=1, interval_seconds=3600)
        long = days_to_collect(100, horizon_bars=20, interval_seconds=3600)
        assert long == pytest.approx(short * 20)

    def test_more_symbols_reduce_calendar_time(self) -> None:
        """Recording three symbols gathers observations three times faster."""
        one = days_to_collect(300, horizon_bars=1, interval_seconds=3600, symbols=1)
        three = days_to_collect(300, horizon_bars=1, interval_seconds=3600, symbols=3)
        assert three == pytest.approx(one / 3)

    def test_rare_signals_take_longer(self) -> None:
        """A signal firing on 10% of bars needs ten times the calendar time."""
        always = days_to_collect(100, horizon_bars=1, interval_seconds=3600, firing_rate=1.0)
        rare = days_to_collect(100, horizon_bars=1, interval_seconds=3600, firing_rate=0.1)
        assert rare == pytest.approx(always * 10)

    def test_known_case(self) -> None:
        """783 observations at h=1 on 1h bars across 3 symbols is about 11 days."""
        days = days_to_collect(783, horizon_bars=1, interval_seconds=3600, symbols=3)
        assert days == pytest.approx(10.9, abs=0.2)

    def test_rejects_bad_arguments(self) -> None:
        """Non-positive inputs are errors."""
        with pytest.raises(ValueError, match="horizon_bars"):
            days_to_collect(10, horizon_bars=0, interval_seconds=3600)
        with pytest.raises(ValueError, match="firing_rate"):
            days_to_collect(10, horizon_bars=1, interval_seconds=3600, firing_rate=0.0)

    def test_standard_deviation_in_bps(self) -> None:
        """A 1% dispersion reads as 100 bps."""
        assert standard_deviation_bps([0.01, -0.01]) == pytest.approx(
            math.sqrt(2 * 0.0001) * 10_000, rel=1e-6
        )
        assert standard_deviation_bps([0.5]) == 0.0


class TestFundingProfile:
    """Descriptive statistics of a funding series."""

    def test_annualises_the_mean_rate(self) -> None:
        """A constant hourly rate annualises by 24 x 365."""
        profile = profile_funding("BTC", funding_series([0.00001] * 100))
        assert profile.annualized_pct == pytest.approx(0.00001 * 24 * 365 * 100)

    def test_counts_positive_fraction(self) -> None:
        """Half positive rates report 50%."""
        profile = profile_funding("BTC", funding_series([0.001, -0.001, 0.001, -0.001]))
        assert profile.positive_fraction == pytest.approx(0.5)

    def test_detects_the_base_rate_floor(self) -> None:
        """Funding pinned at the 0.01%/8h floor is counted."""
        profile = profile_funding("BTC", funding_series([0.0000125] * 8 + [0.00005] * 2))
        assert profile.pinned_fraction == pytest.approx(0.8)

    def test_empty_series_is_all_zero(self) -> None:
        """No data yields a zeroed profile rather than a crash."""
        profile = profile_funding("BTC", ())
        assert profile.observations == 0
        assert profile.annualized_pct == 0.0


class TestCarryArithmetic:
    """The delta-neutral PnL identity."""

    def test_funding_is_summed_over_the_hold(self) -> None:
        """A short perp collects funding each hour it is held."""
        study = CarryStudy(hours=4, cost_bps=0.0)
        result = study.run("BTC", funding_series([0.0001] * 20))
        trade = result.independent[0]
        assert trade.funding_collected == pytest.approx(0.0004)

    def test_basis_widening_is_a_cost(self) -> None:
        """A premium that rises over the hold loses money for the pair."""
        study = CarryStudy(hours=2, cost_bps=0.0)
        premiums = [0.0, 0.0, *([0.001] * 8)]
        result = study.run("BTC", funding_series([0.0] * 10, premiums=premiums))
        trade = result.independent[0]
        assert trade.basis_change == pytest.approx(0.001)
        assert trade.gross_return == pytest.approx(-0.001)

    def test_basis_narrowing_is_a_gain(self) -> None:
        """A premium that falls over the hold gains for the pair."""
        study = CarryStudy(hours=2, cost_bps=0.0)
        result = study.run("BTC", funding_series([0.0] * 10, premiums=[0.001] + [0.0] * 9))
        assert result.independent[0].gross_return == pytest.approx(0.001)

    def test_costs_are_subtracted_once_per_trade(self) -> None:
        """The cost argument is the whole round trip across both legs."""
        study = CarryStudy(hours=2, cost_bps=18.0)
        result = study.run("BTC", funding_series([0.0] * 10))
        assert result.independent[0].net_return == pytest.approx(-0.0018)

    def test_annualisation(self) -> None:
        """A rate held continuously annualises by hours per year."""
        study = CarryStudy(hours=10, cost_bps=0.0)
        result = study.run("BTC", funding_series([0.0001] * 100))
        # 0.001 per 10 hours -> 0.0001/hour -> x 8760
        assert result.annualized_net_pct == pytest.approx(0.0001 * 24 * 365 * 100, rel=1e-6)

    def test_holding_period_must_be_positive(self) -> None:
        """A zero-hour hold is meaningless."""
        with pytest.raises(ValueError, match="Holding period"):
            CarryStudy(hours=0, cost_bps=0.0)

    def test_short_series_is_reported_not_crashed(self) -> None:
        """Too little funding history yields an empty result with a reason."""
        result = CarryStudy(hours=50, cost_bps=0.0).run("BTC", funding_series([0.0001] * 10))
        assert result.sample_size == 0
        assert "insufficient" in result.note


class TestCarryIndependence:
    """Overlapping holding windows must not be counted as independent."""

    def test_independent_trades_are_spaced_by_the_hold(self) -> None:
        """Non-overlapping windows are one holding period apart."""
        study = CarryStudy(hours=10, cost_bps=0.0)
        result = study.run("BTC", funding_series([0.0001] * 100))
        indices = [trade.index for trade in result.independent]
        assert all(b - a >= 10 for a, b in pairwise(indices))

    def test_independent_sample_is_roughly_n_over_hold(self) -> None:
        """A 10-hour hold over 100 hours yields about 9 independent trades."""
        result = CarryStudy(hours=10, cost_bps=0.0).run("BTC", funding_series([0.0001] * 100))
        assert result.sample_size == pytest.approx(9, abs=1)
        assert len(result.trades) > result.sample_size

    def test_tests_use_the_independent_sample(self) -> None:
        """p-values are computed on the non-overlapping subsample only."""
        result = CarryStudy(hours=10, cost_bps=0.0).run("BTC", funding_series([0.0001] * 200))
        assert result.net_return_test.sample_size == result.sample_size


class TestCarryPooling:
    """Combining symbols."""

    def test_pool_concatenates(self) -> None:
        """Pooled trades are the sum of the parts."""
        study = CarryStudy(hours=5, cost_bps=0.0)
        a = study.run("BTC", funding_series([0.0001] * 50))
        b = study.run("ETH", funding_series([0.0001] * 50, symbol="ETH"))
        pooled = pool_carry([a, b])
        assert pooled.sample_size == a.sample_size + b.sample_size
        assert "BTC" in pooled.symbol and "ETH" in pooled.symbol

    def test_pool_rejects_mismatched_holds(self) -> None:
        """Pooling different holding periods would be meaningless."""
        a = CarryStudy(hours=5, cost_bps=0.0).run("BTC", funding_series([0.0001] * 50))
        b = CarryStudy(hours=10, cost_bps=0.0).run("ETH", funding_series([0.0001] * 50))
        with pytest.raises(ValueError, match="Cannot pool"):
            pool_carry([a, b])

    def test_pool_rejects_empty(self) -> None:
        """There is nothing to pool."""
        with pytest.raises(ValueError, match="empty"):
            pool_carry([])
