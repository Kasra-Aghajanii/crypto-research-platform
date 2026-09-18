"""Long-horizon signals and the pooling correction they need.

Phase 6.  Phases 4 and 5 established that at 1h bars the round-trip cost (9 bps)
exceeds the entire standard deviation of a one-bar move (5.1 bps), so cost was
the binding constraint rather than the absence of signal.  At multi-day to
multi-week horizons the arithmetic reverses: a 30-day move has a standard
deviation in the hundreds of basis points, and 9 bps stops mattering.

Everything here is **low-turnover by construction** -- a 200-day moving average
crossed a handful of times a year, not a 1h indicator with the periods stretched.

The pooling problem
-------------------
Long horizons produce few independent observations, and the obvious fix is to
pool symbols.  In crypto that fix mostly does not work: majors move together, so
twenty symbols do not supply twenty times the information.  With average
pairwise correlation ``rho`` between contemporaneous forward returns::

    n_effective = (k * n) / (1 + (k - 1) * rho)

At ``rho = 0.8`` and ``k = 20`` that is ``20n / 16.2``, barely more than a single
symbol's worth.  :func:`effective_sample_size` computes it, and the report leads
with it rather than quoting the raw pooled count.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from statistics import fmean
from typing import Final

from libs.schemas.market import Candle
from libs.schemas.signals import Direction
from services.research.signals import (
    DataRequirement,
    SignalReading,
    SignalSpec,
    SignalWindow,
)

logger = logging.getLogger(__name__)

MIN_TESTABLE_SAMPLE: Final[int] = 30
"""Independent observations below which a cell is reported as untestable.

Not a significance threshold -- a floor below which neither a positive nor a
negative conclusion can be drawn, so reporting a p-value would mislead.
"""


def _flat(value: float = 0.0) -> SignalReading:
    """Return a no-opinion reading."""
    return SignalReading(direction=Direction.FLAT, conviction=0.0, value=value)


def _directional(value: float, conviction: float) -> SignalReading:
    """Build a reading whose direction follows the sign of ``value``."""
    clamped = max(0.0, min(1.0, conviction))
    if value > 0:
        return SignalReading(direction=Direction.LONG, conviction=clamped, value=value)
    if value < 0:
        return SignalReading(direction=Direction.SHORT, conviction=clamped, value=value)
    return _flat(value)


def _realized_volatility(closes: Sequence[float], window: int) -> float | None:
    """Return the annualised realised volatility of daily log returns.

    Args:
        closes: Close prices, oldest first.
        window: Number of returns to include.

    Returns:
        Annualised volatility as a fraction, or ``None`` if undefined.
    """
    if len(closes) < window + 1:
        return None
    returns = [
        math.log(closes[i] / closes[i - 1])
        for i in range(len(closes) - window, len(closes))
        if closes[i - 1] > 0 and closes[i] > 0
    ]
    if len(returns) < 2:
        return None
    mean = fmean(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return math.sqrt(variance) * math.sqrt(365.0)


def _rolling_volatility(closes: Sequence[float], window: int) -> list[float | None]:
    """Return annualised realised volatility at every bar, in one pass.

    Computing the percentile of current volatility against its own history needs
    a volatility estimate at every recent bar.  Recomputing each from scratch is
    quadratic and makes the whole study intractable; rolling sums make it linear.

    Args:
        closes: Close prices, oldest first.
        window: Number of returns in each estimate.

    Returns:
        A series aligned to ``closes``, ``None`` where undefined.
    """
    n = len(closes)
    out: list[float | None] = [None] * n
    if n < window + 1 or window < 2:
        return out

    log_returns: list[float] = []
    for i in range(1, n):
        previous, current = closes[i - 1], closes[i]
        log_returns.append(math.log(current / previous) if previous > 0 and current > 0 else 0.0)

    total = sum(log_returns[:window])
    total_squares = sum(value * value for value in log_returns[:window])
    annualiser = math.sqrt(365.0)

    def estimate(sum_x: float, sum_x2: float) -> float:
        """Return the annualised sample standard deviation of the window."""
        mean = sum_x / window
        variance = max(0.0, (sum_x2 - window * mean * mean) / (window - 1))
        return math.sqrt(variance) * annualiser

    out[window] = estimate(total, total_squares)
    for i in range(window, len(log_returns)):
        entering, leaving = log_returns[i], log_returns[i - window]
        total += entering - leaving
        total_squares += entering * entering - leaving * leaving
        out[i + 1] = estimate(total, total_squares)
    return out


# ----------------------------------------------------------------------
# Trend / momentum
# ----------------------------------------------------------------------


def _price_vs_average(window: SignalWindow, lookback: int) -> SignalReading:
    """Price relative to its trailing simple average.

    The classic long-horizon trend filter: above the average is a bull regime.
    Turnover is low because a 200-day average is crossed a handful of times a
    year.
    """
    closes = window.closes
    if len(closes) < lookback + 1:
        return _flat()
    average = fmean(closes[-lookback:])
    if average <= 0:
        return _flat()
    deviation = (closes[-1] - average) / average
    # A 10% deviation from the trailing average is treated as full conviction.
    return _directional(deviation, abs(deviation) / 0.10)


def momentum_60d(window: SignalWindow) -> SignalReading:
    """Price versus its 60-day average."""
    return _price_vs_average(window, 60)


def momentum_120d(window: SignalWindow) -> SignalReading:
    """Price versus its 120-day average."""
    return _price_vs_average(window, 120)


def momentum_200d(window: SignalWindow) -> SignalReading:
    """Price versus its 200-day average."""
    return _price_vs_average(window, 200)


def _trailing_return_sign(window: SignalWindow, lookback: int) -> SignalReading:
    """Sign of the trailing N-day return, scaled by realised volatility.

    Time-series momentum in its plainest form: if the last N days were up, go
    long.  Scaling conviction by volatility makes a 10% move in a quiet regime
    count for more than the same move in a violent one.
    """
    closes = window.closes
    if len(closes) < lookback + 1:
        return _flat()
    past = closes[-1 - lookback]
    if past <= 0:
        return _flat()
    trailing = (closes[-1] - past) / past

    volatility = _realized_volatility(closes, min(lookback, 60))
    if volatility is None or volatility <= 0:
        return _directional(trailing, abs(trailing) / 0.20)
    # Express the move in units of its own volatility over the lookback.
    horizon_sigma = volatility * math.sqrt(lookback / 365.0)
    if horizon_sigma <= 0:
        return _flat()
    return _directional(trailing, abs(trailing / horizon_sigma))


def tsmom_30d(window: SignalWindow) -> SignalReading:
    """Sign of the trailing 30-day return."""
    return _trailing_return_sign(window, 30)


def tsmom_90d(window: SignalWindow) -> SignalReading:
    """Sign of the trailing 90-day return."""
    return _trailing_return_sign(window, 90)


def tsmom_180d(window: SignalWindow) -> SignalReading:
    """Sign of the trailing 180-day return."""
    return _trailing_return_sign(window, 180)


# ----------------------------------------------------------------------
# Regime
# ----------------------------------------------------------------------


def volatility_regime(
    window: SignalWindow, *, vol_window: int = 30, history: int = 250
) -> SignalReading:
    """Realised-volatility percentile against its own trailing distribution.

    Convention tested: **low volatility is bullish**, the risk-on reading. Low
    realised vol regimes have historically coincided with rising prices in many
    asset classes; the opposite (high vol predicts recovery) is the competing
    reading and would flip every sign here.

    Args:
        window: The decision window.
        vol_window: Days of returns in each volatility estimate.
        history: Trailing estimates forming the percentile distribution.

    Returns:
        Long in a low-vol percentile, short in a high one.
    """
    closes = window.closes
    if len(closes) < vol_window + 20:
        return _flat()

    series = [value for value in _rolling_volatility(closes, vol_window) if value is not None]
    if len(series) < 20:
        return _flat()

    current = series[-1]
    distribution = series[-history:]
    percentile = sum(1 for value in distribution if value <= current) / len(distribution)

    # 0.5 is neutral; distance from the median drives both sign and conviction.
    centred = 0.5 - percentile
    return _directional(centred, abs(centred) * 2.0)


def drawdown_from_high(
    window: SignalWindow, *, lookback: int = 180, threshold: float = 0.10
) -> SignalReading:
    """Distance below the trailing high.

    Convention tested: **a deep drawdown is bullish** -- the buy-the-dip reading.
    The opposite convention (drawdowns beget drawdowns) is equally defensible and
    would flip every sign.

    Args:
        window: The decision window.
        lookback: Days over which the high is measured.
        threshold: Drawdown depth below which the signal activates.

    Returns:
        Long once the drawdown exceeds ``threshold``, flat above it.
    """
    closes = window.closes
    if len(closes) < lookback:
        return _flat()
    high = max(closes[-lookback:])
    if high <= 0:
        return _flat()
    drawdown = (high - closes[-1]) / high
    if drawdown < threshold:
        return _flat(drawdown)
    # A 40% drawdown is treated as full conviction.
    return SignalReading(
        direction=Direction.LONG,
        conviction=min(1.0, (drawdown - threshold) / 0.30),
        value=drawdown,
    )


def funding_extreme(
    window: SignalWindow, *, history: int = 90, band: float = 0.15
) -> SignalReading:
    """Funding at an extreme of its own trailing distribution.

    Phase 4 tested the funding *level* and found nothing.  This tests the
    *extremes* instead: only the top and bottom ``band`` of the trailing
    distribution produce a view, on the contrarian reading that extreme funding
    marks crowded positioning.

    Args:
        window: The decision window.
        history: Trailing daily funding observations forming the distribution.
        band: Tail fraction that counts as extreme.

    Returns:
        Short at a high funding percentile, long at a low one, flat between.
    """
    if len(window.funding) < 20:
        return _flat()
    rates = [float(point.funding_rate) for point in window.funding[-history:]]
    if len(rates) < 20:
        return _flat()

    current = rates[-1]
    percentile = sum(1 for value in rates if value <= current) / len(rates)

    if percentile >= 1.0 - band:
        return SignalReading(
            direction=Direction.SHORT,
            conviction=min(1.0, (percentile - (1.0 - band)) / band),
            value=current,
        )
    if percentile <= band:
        return SignalReading(
            direction=Direction.LONG,
            conviction=min(1.0, (band - percentile) / band),
            value=current,
        )
    return _flat(current)


LONG_HORIZON_SIGNALS: Final[dict[str, SignalSpec]] = {
    spec.name: spec
    for spec in (
        SignalSpec(
            name="momentum_60d",
            description="Price versus its 60-day simple average.",
            convention="trend following: above the average is bullish",
            min_bars=61,
            requires=frozenset({DataRequirement.CANDLES}),
            evaluate=momentum_60d,
        ),
        SignalSpec(
            name="momentum_120d",
            description="Price versus its 120-day simple average.",
            convention="trend following: above the average is bullish",
            min_bars=121,
            requires=frozenset({DataRequirement.CANDLES}),
            evaluate=momentum_120d,
        ),
        SignalSpec(
            name="momentum_200d",
            description="Price versus its 200-day simple average.",
            convention="trend following: above the average is bullish",
            min_bars=201,
            requires=frozenset({DataRequirement.CANDLES}),
            evaluate=momentum_200d,
        ),
        SignalSpec(
            name="tsmom_30d",
            description="Sign of the trailing 30-day return, volatility scaled.",
            convention="time-series momentum: a positive trailing return continues",
            min_bars=61,
            requires=frozenset({DataRequirement.CANDLES}),
            evaluate=tsmom_30d,
        ),
        SignalSpec(
            name="tsmom_90d",
            description="Sign of the trailing 90-day return, volatility scaled.",
            convention="time-series momentum: a positive trailing return continues",
            min_bars=121,
            requires=frozenset({DataRequirement.CANDLES}),
            evaluate=tsmom_90d,
        ),
        SignalSpec(
            name="tsmom_180d",
            description="Sign of the trailing 180-day return, volatility scaled.",
            convention="time-series momentum: a positive trailing return continues",
            min_bars=211,
            requires=frozenset({DataRequirement.CANDLES}),
            evaluate=tsmom_180d,
        ),
        SignalSpec(
            name="vol_regime",
            description="30-day realised volatility percentile over a 250-day history.",
            convention="risk-on: low realised volatility is bullish",
            min_bars=120,
            requires=frozenset({DataRequirement.CANDLES}),
            evaluate=volatility_regime,
        ),
        SignalSpec(
            name="drawdown",
            description="Drawdown from the trailing 180-day high, beyond 10%.",
            convention="mean reversion: a deep drawdown is bullish (buy the dip)",
            min_bars=181,
            requires=frozenset({DataRequirement.CANDLES}),
            evaluate=drawdown_from_high,
        ),
        SignalSpec(
            name="funding_extreme",
            description="Daily funding in the top/bottom 15% of its 90-day distribution.",
            convention="contrarian: extreme funding marks crowded positioning",
            min_bars=30,
            requires=frozenset({DataRequirement.CANDLES, DataRequirement.FUNDING}),
            evaluate=funding_extreme,
        ),
    )
}
"""Single-symbol long-horizon signals, keyed by name."""


# ----------------------------------------------------------------------
# Cross-sectional momentum
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CrossSectionalObservation:
    """One rebalance of a long-top / short-bottom portfolio.

    Attributes:
        long_symbols: Symbols in the top quantile at the rebalance.
        short_symbols: Symbols in the bottom quantile.
        gross_return: Equal-weighted long-minus-short return over the horizon.
    """

    index: int
    long_symbols: tuple[str, ...]
    short_symbols: tuple[str, ...]
    gross_return: float
    net_return: float
    independent: bool


def cross_sectional_momentum(
    candles_by_symbol: Mapping[str, Sequence[Candle]],
    *,
    lookback: int,
    horizon: int,
    quantile: float = 0.3,
    cost_bps: float = 9.0,
    min_symbols: int = 6,
) -> list[CrossSectionalObservation]:
    """Rank symbols by trailing return and hold the spread.

    A dollar-neutral long-top / short-bottom portfolio, rebalanced every bar and
    held for ``horizon`` bars.  Unlike the single-symbol signals this is not a
    directional bet on crypto: it is a bet that relative ranking persists, which
    survives even when everything moves together.

    **Series are aligned by date, not by position.**  Symbols listed at different
    times, so their histories have different lengths and different start dates.
    Indexing each by the same integer offset would compare one symbol's June
    2023 against another's April 2024 and manufacture a cross-sectional
    "signal" out of nothing but calendar misalignment. Only dates on which at
    least ``min_symbols`` symbols have both the lookback and the forward window
    available are used.

    Costs are charged on both legs, so the round trip is ``2 * cost_bps`` in
    total across the spread.

    Args:
        candles_by_symbol: Daily candles per symbol, oldest first. Alignment is
            derived from their timestamps.
        lookback: Days of trailing return used for ranking.
        horizon: Holding period in days.
        quantile: Fraction of the universe taken on each side.
        cost_bps: Cost per side in basis points.
        min_symbols: Symbols required before a rebalance is attempted.

    Returns:
        Observations, with the non-overlapping subsample marked.

    Raises:
        ValueError: If the parameters are degenerate.
    """
    if lookback < 1 or horizon < 1:
        raise ValueError("lookback and horizon must be >= 1.")
    if not 0.0 < quantile <= 0.5:
        raise ValueError(f"quantile must be in (0, 0.5], got {quantile}.")
    if not candles_by_symbol:
        return []

    # Build a date -> close map per symbol so ranking compares the same day.
    by_date: dict[str, dict[datetime, float]] = {}
    for symbol, series in candles_by_symbol.items():
        by_date[symbol] = {
            candle.open_time: float(candle.close)
            for candle in series
            if candle.close > 0
        }

    calendar = sorted({day for prices in by_date.values() for day in prices})
    observations: list[CrossSectionalObservation] = []
    next_allowed = -1
    cost = 2.0 * cost_bps / 10_000.0

    for index in range(lookback, len(calendar) - horizon):
        entry_day = calendar[index]
        past_day = calendar[index - lookback]
        exit_day = calendar[index + horizon]

        ranked: list[tuple[float, str]] = []
        forwards: dict[str, float] = {}
        for symbol, prices in by_date.items():
            past = prices.get(past_day)
            now = prices.get(entry_day)
            later = prices.get(exit_day)
            if past is None or now is None or later is None or past <= 0 or now <= 0:
                continue
            ranked.append(((now - past) / past, symbol))
            forwards[symbol] = (later - now) / now

        if len(ranked) < min_symbols:
            continue

        ranked.sort()
        take = max(1, int(len(ranked) * quantile))
        shorts = [symbol for _, symbol in ranked[:take]]
        longs = [symbol for _, symbol in ranked[-take:]]

        long_leg = fmean([forwards[symbol] for symbol in longs])
        short_leg = fmean([forwards[symbol] for symbol in shorts])
        gross = long_leg - short_leg

        independent = index >= next_allowed
        if independent:
            next_allowed = index + horizon

        observations.append(
            CrossSectionalObservation(
                index=index,
                long_symbols=tuple(longs),
                short_symbols=tuple(shorts),
                gross_return=gross,
                net_return=gross - cost,
                independent=independent,
            )
        )
    return observations


# ----------------------------------------------------------------------
# Pooling correction
# ----------------------------------------------------------------------


def mean_pairwise_correlation(series_by_symbol: Mapping[str, Sequence[float]]) -> float:
    """Return the average pairwise correlation across aligned series.

    Args:
        series_by_symbol: Aligned return series per symbol.

    Returns:
        Mean pairwise Pearson correlation, or ``0.0`` with fewer than two usable
        series.
    """
    names = [name for name, series in series_by_symbol.items() if len(series) >= 3]
    if len(names) < 2:
        return 0.0

    length = min(len(series_by_symbol[name]) for name in names)
    trimmed = {name: list(series_by_symbol[name])[-length:] for name in names}
    correlations: list[float] = []

    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            correlation = _pearson(trimmed[first], trimmed[second])
            if correlation is not None:
                correlations.append(correlation)
    return fmean(correlations) if correlations else 0.0


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Return the Pearson correlation of two equal-length series, or ``None``."""
    n = min(len(left), len(right))
    if n < 3:
        return None
    a, b = list(left[-n:]), list(right[-n:])
    mean_a, mean_b = fmean(a), fmean(b)
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b, strict=True))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((y - mean_b) ** 2 for y in b)
    if var_a <= 0 or var_b <= 0:
        return None
    return cov / math.sqrt(var_a * var_b)


def effective_sample_size(
    observations_per_symbol: float, symbols: int, correlation: float
) -> float:
    """Return the independent-equivalent sample size after pooling.

    Pooling ``k`` symbols multiplies the raw count by ``k`` but not the
    information, because the symbols move together.  With average pairwise
    correlation ``rho``::

        n_effective = (k * n) / (1 + (k - 1) * rho)

    At ``rho = 0`` this is the raw pooled count; at ``rho = 1`` it collapses to a
    single symbol's worth.

    Args:
        observations_per_symbol: Independent observations per symbol.
        symbols: Number of symbols pooled.
        correlation: Average pairwise correlation of forward returns.

    Returns:
        The effective sample size.

    Raises:
        ValueError: If ``symbols`` is below one.
    """
    if symbols < 1:
        raise ValueError(f"symbols must be >= 1, got {symbols}.")
    rho = max(0.0, min(1.0, correlation))
    return (symbols * observations_per_symbol) / (1.0 + (symbols - 1) * rho)


def traded_bars(candles: Sequence[Candle]) -> tuple[Candle, ...]:
    """Return only candles that show actual trading.

    Hyperliquid backfills daily candles from an index source for dates before
    the venue existed; those bars carry zero volume and zero trades.  They are
    real prices but not Hyperliquid market data, and funding does not exist for
    them at all.

    Args:
        candles: Candles, oldest first.

    Returns:
        The contiguous tail from the first bar showing volume or trades.
    """
    for index, candle in enumerate(candles):
        if candle.volume > 0 or candle.trade_count > 0:
            return tuple(candles[index:])
    return ()




# ----------------------------------------------------------------------
# Benchmark correction: separating skill from drift
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Benchmark:
    """Unconditional forward-return statistics for one symbol and horizon.

    Attributes:
        mean_return: Mean forward return over every bar, as a fraction.
        up_rate: Fraction of forward windows that were positive.
    """

    symbol: str
    horizon: int
    mean_return: float
    up_rate: float
    windows: int


def compute_benchmark(symbol: str, closes: Sequence[float], horizon: int) -> Benchmark:
    """Measure what doing nothing clever would have earned.

    Args:
        symbol: Symbol being measured.
        closes: Close prices, oldest first.
        horizon: Forward horizon in bars.

    Returns:
        The unconditional benchmark.
    """
    returns = [
        (closes[i + horizon] - closes[i]) / closes[i]
        for i in range(len(closes) - horizon)
        if closes[i] > 0
    ]
    if not returns:
        return Benchmark(symbol=symbol, horizon=horizon, mean_return=0.0, up_rate=0.5, windows=0)
    return Benchmark(
        symbol=symbol,
        horizon=horizon,
        mean_return=fmean(returns),
        up_rate=sum(1 for r in returns if r > 0) / len(returns),
        windows=len(returns),
    )


@dataclass(frozen=True, slots=True)
class ExcessStats:
    """A signal's performance measured against buy-and-hold rather than zero.

    A long-biased signal in a rising market earns the drift whether or not it
    knows anything.  Over the traded Hyperliquid sample the unconditional 7-day
    return is roughly +88 bps for BTC, so a signal "earning" +96 bps is earning
    about +8 bps of skill.  These statistics strip the drift out.

    Attributes:
        mean_excess_bps: Mean return in excess of the direction-adjusted
            benchmark, in basis points. This is the number that means something.
        null_probability: Probability of a win under no skill, given how often
            the signal went long and how often the market rose.
        long_fraction: Share of firings that were long, exposing directional bias.
    """

    n: int
    mean_excess_bps: float
    mean_benchmark_bps: float
    wins: int
    null_probability: float
    long_fraction: float
    excess_p_value: float
    hit_p_value: float
    hit_rate: float


def excess_statistics(
    observations: Sequence[object], benchmarks: Mapping[str, Benchmark]
) -> ExcessStats:
    """Measure observations against their unconditional benchmark.

    Args:
        observations: Independent observations carrying ``symbol``,
            ``direction`` and ``gross_return``.
        benchmarks: Benchmark per symbol for this horizon.

    Returns:
        The benchmark-adjusted statistics.
    """
    from services.research.statistics import binomial_test, one_sample_t_test

    excess: list[float] = []
    benchmark_values: list[float] = []
    null_probabilities: list[float] = []
    wins = 0
    longs = 0

    for observation in observations:
        symbol = observation.symbol  # type: ignore[attr-defined]
        direction = observation.direction  # type: ignore[attr-defined]
        gross = float(observation.gross_return)  # type: ignore[attr-defined]
        benchmark = benchmarks.get(symbol)
        if benchmark is None:
            continue
        sign = float(direction.sign)
        # A long is credited only with what it beat the drift by; a short is
        # credited with the drift it avoided.
        excess.append(gross - sign * benchmark.mean_return)
        benchmark_values.append(sign * benchmark.mean_return)
        null_probabilities.append(
            benchmark.up_rate if sign > 0 else 1.0 - benchmark.up_rate
        )
        if gross > 0:
            wins += 1
        if sign > 0:
            longs += 1

    n = len(excess)
    if n == 0:
        return ExcessStats(0, 0.0, 0.0, 0, 0.5, 0.0, 1.0, 1.0, 0.0)

    null_probability = fmean(null_probabilities)
    return ExcessStats(
        n=n,
        mean_excess_bps=fmean(excess) * 10_000.0,
        mean_benchmark_bps=fmean(benchmark_values) * 10_000.0,
        wins=wins,
        null_probability=null_probability,
        long_fraction=longs / n,
        excess_p_value=one_sample_t_test(excess, 0.0).p_value,
        hit_p_value=binomial_test(wins, n, null_probability).p_value,
        hit_rate=wins / n,
    )


def correlation_adjusted_tests(
    excess: Sequence[float],
    *,
    wins: int,
    null_probability: float,
    effective_n: float,
) -> tuple[float, float]:
    """Recompute both p-values as if the sample were its effective size.

    Pooling correlated symbols inflates the row count without adding
    information.  Reporting a p-value computed on the raw pooled count and
    caveating it in prose invites the reader to ignore the caveat, so the
    correction is applied to the numbers instead: the tests are evaluated with
    the same mean and dispersion but ``effective_n`` observations.

    Args:
        excess: Excess returns over the benchmark.
        wins: Observed wins.
        null_probability: Win probability under no skill.
        effective_n: Correlation-adjusted sample size.

    Returns:
        A ``(hit_p_value, excess_p_value)`` pair.
    """
    import math

    from services.research.statistics import binomial_test, student_t_sf

    raw_n = len(excess)
    if raw_n < 2 or effective_n < 2:
        return 1.0, 1.0

    scaled_n = max(2, round(min(effective_n, raw_n)))

    mean = fmean(excess)
    variance = sum((value - mean) ** 2 for value in excess) / (raw_n - 1)
    if variance <= 0:
        excess_p = 1.0
    else:
        t = mean / math.sqrt(variance / scaled_n)
        excess_p = min(1.0, 2.0 * student_t_sf(abs(t), scaled_n - 1))

    # Scale the win count to the effective sample, preserving the observed rate.
    scaled_wins = round(wins * scaled_n / raw_n)
    hit_p = binomial_test(scaled_wins, scaled_n, null_probability).p_value
    return hit_p, excess_p


__all__ = [
    "LONG_HORIZON_SIGNALS",
    "MIN_TESTABLE_SAMPLE",
    "Benchmark",
    "CrossSectionalObservation",
    "ExcessStats",
    "compute_benchmark",
    "correlation_adjusted_tests",
    "cross_sectional_momentum",
    "drawdown_from_high",
    "effective_sample_size",
    "excess_statistics",
    "funding_extreme",
    "mean_pairwise_correlation",
    "momentum_60d",
    "momentum_120d",
    "momentum_200d",
    "traded_bars",
    "tsmom_30d",
    "tsmom_90d",
    "tsmom_180d",
    "volatility_regime",
]
