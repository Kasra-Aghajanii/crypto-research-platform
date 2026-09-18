"""Tests for the signal library and the forward-horizon study.

The study's job is to produce a number that can be trusted, so the tests here
concentrate on the ways it could quietly lie: look-ahead, overlapping samples
counted as independent, costs applied in the wrong direction, and signals
firing on data they should not have.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from itertools import pairwise

import pytest

from libs.schemas.market import Candle, FundingRate
from libs.schemas.signals import Direction
from services.research.signals import (
    BACKTESTABLE_SIGNALS,
    BLOCKED_SIGNALS,
    SIGNALS,
    DataRequirement,
    SignalReading,
    SignalSpec,
    SignalWindow,
    get_signal,
    signal_names,
)
from services.research.study import ForwardHorizonStudy, pool
from tests.conftest import trending_candles


def flat_then_rise(count: int, *, rise_at: int, interval: str = "1h") -> tuple[Candle, ...]:
    """Build a series that is flat and then rises, for exact-return assertions."""
    candles = []
    base = datetime(2026, 1, 1, tzinfo=UTC)
    for i in range(count):
        price = 100.0 if i < rise_at else 110.0
        open_time = base + timedelta(hours=i)
        candles.append(
            Candle(
                source="test",
                symbol="BTC",
                interval=interval,
                open_time=open_time,
                close_time=open_time + timedelta(hours=1),
                occurred_at=open_time,
                open=Decimal(str(price)),
                high=Decimal(str(price + 1)),
                low=Decimal(str(price - 1)),
                close=Decimal(str(price)),
                volume=Decimal("100"),
                is_closed=True,
            )
        )
    return tuple(candles)


def always_long(window: SignalWindow) -> SignalReading:
    """A signal that always calls long with full conviction."""
    return SignalReading(direction=Direction.LONG, conviction=1.0, value=1.0)


def always_flat(window: SignalWindow) -> SignalReading:
    """A signal that never takes a view."""
    return SignalReading(direction=Direction.FLAT, conviction=0.0, value=0.0)


def spec_for(evaluate, *, name: str = "test_signal", min_bars: int = 1) -> SignalSpec:
    """Wrap a callable in a minimal signal spec."""
    return SignalSpec(
        name=name,
        description="test",
        convention="test",
        min_bars=min_bars,
        requires=frozenset({DataRequirement.CANDLES}),
        evaluate=evaluate,
    )


class TestSignalLibrary:
    """The catalogue itself."""

    def test_every_signal_declares_its_convention(self) -> None:
        """A result is meaningless without saying which reading was tested."""
        for spec in SIGNALS.values():
            assert spec.convention
            assert spec.description
            assert spec.min_bars >= 1

    def test_blocked_signals_are_identified(self) -> None:
        """Signals needing unavailable history are flagged, not silently dropped."""
        assert set(BLOCKED_SIGNALS) == {"orderbook_imbalance", "open_interest_change"}
        for name in BLOCKED_SIGNALS:
            spec = get_signal(name)
            assert not spec.is_backtestable
            assert spec.missing_data

    def test_backtestable_signals_need_only_available_data(self) -> None:
        """Everything in the testable set uses candles and/or funding."""
        for name in BACKTESTABLE_SIGNALS:
            assert get_signal(name).is_backtestable

    def test_signal_names_excludes_blocked_by_default(self) -> None:
        """The default run set is what can actually be measured."""
        assert set(signal_names()) == set(BACKTESTABLE_SIGNALS)
        assert set(signal_names(include_blocked=True)) == set(SIGNALS)

    def test_unknown_signal_raises(self) -> None:
        """A typo names the known signals rather than failing obscurely."""
        with pytest.raises(KeyError, match="Unknown signal"):
            get_signal("does_not_exist")

    def test_conviction_maps_to_a_probability_above_a_half(self) -> None:
        """Zero conviction is a coin flip; full conviction stakes everything."""
        assert SignalReading(Direction.LONG, 0.0, 0.0).probability == pytest.approx(0.5)
        assert SignalReading(Direction.LONG, 1.0, 0.0).probability == pytest.approx(1.0)
        assert SignalReading(Direction.LONG, 0.5, 0.0).probability == pytest.approx(0.75)

    def test_conviction_is_clamped(self) -> None:
        """Out-of-range conviction cannot produce an invalid probability."""
        assert SignalReading(Direction.LONG, 5.0, 0.0).probability == pytest.approx(1.0)
        assert SignalReading(Direction.LONG, -5.0, 0.0).probability == pytest.approx(0.5)


class TestSignalBehaviour:
    """Each signal fires in the direction its stated convention implies."""

    def test_rsi_reversion_buys_oversold(self) -> None:
        """A sustained decline drives RSI down and the mean-reversion read long."""
        window = SignalWindow(candles=trending_candles(80, start=500.0, drift=-3.0, noise=0.2))
        assert get_signal("rsi").evaluate(window).direction is Direction.LONG

    def test_rsi_momentum_is_the_opposite_reading(self) -> None:
        """The same decline gives the momentum convention a short."""
        window = SignalWindow(candles=trending_candles(80, start=500.0, drift=-3.0, noise=0.2))
        assert get_signal("rsi_momentum").evaluate(window).direction is Direction.SHORT

    def test_ema_cross_follows_the_trend(self) -> None:
        """A rising series puts the fast EMA above the slow one."""
        window = SignalWindow(candles=trending_candles(120, start=100.0, drift=1.0, noise=0.2))
        assert get_signal("ema_cross").evaluate(window).direction is Direction.LONG

    def test_macd_histogram_follows_momentum(self) -> None:
        """A rising series gives a positive histogram."""
        window = SignalWindow(candles=trending_candles(120, start=100.0, drift=1.0, noise=0.2))
        assert get_signal("macd_histogram").evaluate(window).direction is Direction.LONG

    def test_atr_momentum_follows_the_normalised_move(self) -> None:
        """A strong advance reads long."""
        window = SignalWindow(candles=trending_candles(120, start=100.0, drift=1.0, noise=0.2))
        assert get_signal("atr_momentum").evaluate(window).direction is Direction.LONG

    def test_bollinger_stays_flat_inside_the_bands(self) -> None:
        """A quiet series produces no view."""
        window = SignalWindow(candles=trending_candles(80, start=100.0, drift=0.0, noise=0.3))
        assert get_signal("bollinger_percent_b").evaluate(window).direction is Direction.FLAT

    def test_funding_is_contrarian(self) -> None:
        """Positive funding (crowded long) reads short."""
        window = SignalWindow(
            candles=trending_candles(10),
            funding=(
                FundingRate(
                    source="test",
                    symbol="BTC",
                    funding_rate=Decimal("0.0001"),
                    occurred_at=datetime.now(tz=UTC),
                ),
            ),
        )
        assert get_signal("funding_rate").evaluate(window).direction is Direction.SHORT

    def test_funding_without_data_is_flat(self) -> None:
        """No funding history means no opinion, never a guess."""
        window = SignalWindow(candles=trending_candles(10))
        assert get_signal("funding_rate").evaluate(window).direction is Direction.FLAT

    def test_orderbook_signal_is_flat_without_a_book(self) -> None:
        """The blocked signals degrade to flat rather than inventing a view."""
        window = SignalWindow(candles=trending_candles(10))
        assert get_signal("orderbook_imbalance").evaluate(window).direction is Direction.FLAT

    def test_open_interest_signal_is_flat_without_data(self) -> None:
        """Same for open interest."""
        window = SignalWindow(candles=trending_candles(60))
        assert get_signal("open_interest_change").evaluate(window).direction is Direction.FLAT

    def test_short_history_never_crashes_a_signal(self) -> None:
        """Every signal tolerates a window shorter than it needs."""
        window = SignalWindow(candles=trending_candles(3))
        for spec in SIGNALS.values():
            reading = spec.evaluate(window)
            assert reading is None or 0.0 <= reading.conviction <= 1.0


class TestForwardReturns:
    """The arithmetic of a forward-horizon observation."""

    def test_gross_return_matches_the_price_move(self) -> None:
        """A 10% rise over the horizon is recorded as a 10% gross return."""
        study = ForwardHorizonStudy(horizon=1, cost_bps=0.0)
        candles = flat_then_rise(4, rise_at=2)
        result = study.run(spec_for(always_long), "BTC", candles)
        gains = [o for o in result.observations if o.gross_return > 0]
        assert gains
        assert gains[0].gross_return == pytest.approx(0.10)

    def test_short_direction_flips_the_sign(self) -> None:
        """A short into a rising market records a negative return."""

        def always_short(window: SignalWindow) -> SignalReading:
            return SignalReading(Direction.SHORT, 1.0, -1.0)

        study = ForwardHorizonStudy(horizon=1, cost_bps=0.0)
        result = study.run(spec_for(always_short), "BTC", flat_then_rise(4, rise_at=2))
        losses = [o for o in result.observations if o.gross_return < 0]
        assert losses
        assert losses[0].gross_return == pytest.approx(-0.10)

    def test_costs_are_charged_round_trip(self) -> None:
        """Net return is gross minus twice the per-side cost."""
        study = ForwardHorizonStudy(horizon=1, cost_bps=10.0)
        result = study.run(spec_for(always_long), "BTC", flat_then_rise(6, rise_at=10))
        assert study.round_trip_cost == pytest.approx(0.002)
        for observation in result.observations:
            assert observation.net_return == pytest.approx(observation.gross_return - 0.002)

    def test_flat_signal_produces_no_observations(self) -> None:
        """A signal with no view is not a trade."""
        study = ForwardHorizonStudy(horizon=5)
        result = study.run(spec_for(always_flat), "BTC", trending_candles(100))
        assert result.sample_size == 0
        assert result.independent_size == 0

    def test_horizon_must_be_positive(self) -> None:
        """A zero or negative horizon is meaningless."""
        with pytest.raises(ValueError, match="Horizon"):
            ForwardHorizonStudy(horizon=0)

    def test_insufficient_history_is_reported_not_crashed(self) -> None:
        """Too little data yields an empty result carrying the reason."""
        study = ForwardHorizonStudy(horizon=50)
        result = study.run(spec_for(always_long, min_bars=20), "BTC", trending_candles(30))
        assert result.sample_size == 0
        assert "insufficient history" in result.note


class TestNoLookAhead:
    """The window handed to a signal must not contain the future."""

    def test_window_ends_at_the_decision_bar(self) -> None:
        """A signal never sees a bar at or beyond its own decision point."""
        seen: list[datetime] = []

        def spy(window: SignalWindow) -> SignalReading:
            seen.append(window.candles[-1].close_time)
            return SignalReading(Direction.LONG, 0.5, 1.0)

        candles = trending_candles(120, interval="1h")
        study = ForwardHorizonStudy(horizon=5, window_bars=50)
        result = study.run(spec_for(spy, min_bars=10), "BTC", candles)

        assert len(seen) == result.sample_size
        for observation, last_visible in zip(result.observations, seen, strict=True):
            assert last_visible == candles[observation.index].close_time

    def test_window_is_bounded(self) -> None:
        """A signal sees no more history than the live agent buffer holds."""
        sizes: list[int] = []

        def spy(window: SignalWindow) -> SignalReading:
            sizes.append(len(window.candles))
            return SignalReading(Direction.LONG, 0.5, 1.0)

        study = ForwardHorizonStudy(horizon=5, window_bars=30)
        study.run(spec_for(spy, min_bars=10), "BTC", trending_candles(200))
        assert sizes
        assert max(sizes) <= 30

    def test_funding_never_leaks_from_the_future(self) -> None:
        """Only funding settled by the decision bar is visible."""
        candles = trending_candles(60, interval="1h")
        funding = tuple(
            FundingRate(
                source="test",
                symbol="BTC",
                funding_rate=Decimal("0.00001"),
                occurred_at=candle.close_time,
            )
            for candle in candles
        )
        latest: list[datetime] = []

        def spy(window: SignalWindow) -> SignalReading:
            if window.funding:
                latest.append(window.funding[-1].occurred_at)
            return SignalReading(Direction.LONG, 0.5, 1.0)

        study = ForwardHorizonStudy(horizon=5)
        result = study.run(spec_for(spy, min_bars=5), "BTC", candles, funding=funding)
        assert latest
        for observation, funding_time in zip(result.observations, latest, strict=True):
            assert funding_time <= candles[observation.index].close_time


class TestIndependence:
    """Overlapping observations must not be counted as independent draws."""

    def test_independent_sample_is_spaced_by_the_horizon(self) -> None:
        """Consecutive independent observations are at least H bars apart."""
        study = ForwardHorizonStudy(horizon=10)
        result = study.run(spec_for(always_long), "BTC", trending_candles(300))
        indices = [o.index for o in result.independent]
        assert len(indices) > 1
        assert all(b - a >= 10 for a, b in pairwise(indices))

    def test_independent_sample_is_much_smaller_than_the_raw_one(self) -> None:
        """A signal firing every bar yields roughly n/H independent draws."""
        study = ForwardHorizonStudy(horizon=20)
        result = study.run(spec_for(always_long), "BTC", trending_candles(400))
        assert result.sample_size > result.independent_size
        assert result.independent_size == pytest.approx(result.sample_size / 20, rel=0.15)

    def test_horizon_one_keeps_every_observation(self) -> None:
        """With a one-bar horizon nothing overlaps."""
        study = ForwardHorizonStudy(horizon=1)
        result = study.run(spec_for(always_long), "BTC", trending_candles(100))
        assert result.independent_size == result.sample_size

    def test_p_values_use_only_the_independent_sample(self) -> None:
        """The binomial test's n matches the non-overlapping count."""
        study = ForwardHorizonStudy(horizon=20)
        result = study.run(spec_for(always_long), "BTC", trending_candles(400))
        assert result.hit_rate_test.sample_size == result.independent_size
        assert result.net_return_test.sample_size == result.independent_size


class TestMetrics:
    """Reported statistics are internally consistent."""

    def test_brier_of_a_coin_flip_forecast_is_the_baseline(self) -> None:
        """Zero conviction scores exactly 0.25 whatever happens."""

        def no_conviction(window: SignalWindow) -> SignalReading:
            return SignalReading(Direction.LONG, 0.0, 0.0)

        study = ForwardHorizonStudy(horizon=1)
        result = study.run(spec_for(no_conviction), "BTC", trending_candles(200))
        assert result.brier == pytest.approx(0.25)
        assert result.brier_skill_score == pytest.approx(0.0)

    def test_perfect_confident_forecast_scores_zero(self) -> None:
        """Full conviction on an always-right call is a Brier of 0."""
        study = ForwardHorizonStudy(horizon=1, cost_bps=0.0)
        candles = trending_candles(100, start=100.0, drift=1.0, noise=0.0)
        result = study.run(spec_for(always_long), "BTC", candles)
        assert result.gross_hit_rate == pytest.approx(1.0)
        assert result.brier == pytest.approx(0.0)
        assert result.brier_skill_score == pytest.approx(1.0)

    def test_hit_rate_matches_the_observations(self) -> None:
        """The reported rate is the fraction of winning independent calls."""
        study = ForwardHorizonStudy(horizon=5)
        result = study.run(spec_for(always_long), "BTC", trending_candles(300))
        expected = sum(1 for o in result.independent if o.gross_return > 0) / len(
            result.independent
        )
        assert result.gross_hit_rate == pytest.approx(expected)

    def test_net_hit_rate_never_exceeds_gross(self) -> None:
        """Costs can only turn wins into losses, never the reverse."""
        study = ForwardHorizonStudy(horizon=5, cost_bps=20.0)
        result = study.run(spec_for(always_long), "BTC", trending_candles(300))
        assert result.net_hit_rate <= result.gross_hit_rate

    def test_summary_exposes_the_headline_numbers(self) -> None:
        """The flat summary carries everything the matrix prints."""
        study = ForwardHorizonStudy(horizon=5)
        summary = study.run(spec_for(always_long), "BTC", trending_candles(200)).summary()
        for key in ("signal", "horizon", "n_independent", "brier", "mean_net_bps", "p_hit_rate"):
            assert key in summary

    def test_empty_result_reports_zeros(self) -> None:
        """A study with no observations divides by nothing."""
        study = ForwardHorizonStudy(horizon=5)
        result = study.run(spec_for(always_flat), "BTC", trending_candles(200))
        assert result.gross_hit_rate == 0.0
        assert result.brier == 0.0
        assert result.mean_net_bps == 0.0
        assert result.hit_rate_test.p_value == 1.0


class TestPooling:
    """Combining per-symbol results."""

    def test_pool_concatenates_observations(self) -> None:
        """Pooled sample size is the sum of its parts."""
        study = ForwardHorizonStudy(horizon=5)
        first = study.run(spec_for(always_long), "BTC", trending_candles(200))
        second = study.run(spec_for(always_long), "ETH", trending_candles(200))
        pooled = pool([first, second])
        assert pooled.sample_size == first.sample_size + second.sample_size
        assert pooled.symbols == ("BTC", "ETH")

    def test_pool_rejects_mismatched_studies(self) -> None:
        """Pooling different horizons would be meaningless."""
        a = ForwardHorizonStudy(horizon=5).run(
            spec_for(always_long), "BTC", trending_candles(200)
        )
        b = ForwardHorizonStudy(horizon=20).run(
            spec_for(always_long), "ETH", trending_candles(200)
        )
        with pytest.raises(ValueError, match="Cannot pool"):
            pool([a, b])

    def test_pool_rejects_empty_input(self) -> None:
        """There is nothing to pool."""
        with pytest.raises(ValueError, match="empty"):
            pool([])
