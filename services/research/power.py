"""Statistical power: how much data is needed before a result can mean anything.

Phase 4 produced no discoveries.  Before spending months recording order book
and open-interest history, it is worth knowing what sample size would be needed
to detect an edge if one existed -- and therefore how long the recorders have to
run before the question is even answerable.

Two tests are covered, matching the two the research harness reports:

* a binomial test of hit rate against 50%
* a t-test of mean forward return against zero

The return test is the one that matters economically, and it is usually the more
demanding: an edge worth trading has to clear costs, and returns are noisy
relative to their mean at short horizons.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

_SQRT2: Final[float] = math.sqrt(2.0)


def normal_cdf(x: float) -> float:
    """Return the standard normal cumulative distribution at ``x``."""
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def normal_quantile(p: float, *, tolerance: float = 1e-12) -> float:
    """Return the standard normal quantile (probit) for probability ``p``.

    Solved by bisection on :func:`normal_cdf`.  This runs a handful of times per
    report, so clarity beats a rational approximation here.

    Args:
        p: Probability in ``(0, 1)``.
        tolerance: Absolute convergence tolerance.

    Returns:
        The value ``z`` such that ``normal_cdf(z) == p``.

    Raises:
        ValueError: If ``p`` is not strictly between 0 and 1.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}.")
    low, high = -40.0, 40.0
    while high - low > tolerance:
        middle = (low + high) / 2.0
        if normal_cdf(middle) < p:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


@dataclass(frozen=True, slots=True)
class SampleRequirement:
    """Independent observations needed to detect a given effect.

    Attributes:
        effect: The effect size the calculation was run for.
        observations: Independent observations required.
    """

    effect: float
    observations: int


def samples_for_hit_rate(
    hit_rate: float, *, alpha: float = 0.05, power: float = 0.80
) -> SampleRequirement:
    """Return the observations needed to detect a hit rate different from 50%.

    Args:
        hit_rate: The true hit rate to detect, e.g. ``0.53``.
        alpha: Two-sided significance level.
        power: Probability of detecting the effect if it is real.

    Returns:
        The sample requirement.

    Raises:
        ValueError: If ``hit_rate`` is 0.5 (no effect to detect) or out of range.
    """
    if not 0.0 < hit_rate < 1.0:
        raise ValueError(f"hit_rate must be in (0, 1), got {hit_rate}.")
    if hit_rate == 0.5:
        raise ValueError("An effect size of exactly 50% cannot be detected at any sample size.")

    z_alpha = normal_quantile(1.0 - alpha / 2.0)
    z_beta = normal_quantile(power)
    numerator = z_alpha * math.sqrt(0.25) + z_beta * math.sqrt(hit_rate * (1.0 - hit_rate))
    n = (numerator / (hit_rate - 0.5)) ** 2
    return SampleRequirement(effect=hit_rate, observations=math.ceil(n))


def samples_for_mean_return(
    effect_bps: float, standard_deviation_bps: float, *, alpha: float = 0.05, power: float = 0.80
) -> SampleRequirement:
    """Return the observations needed to detect a mean return different from zero.

    Args:
        effect_bps: The true mean return to detect, in basis points.
        standard_deviation_bps: Standard deviation of the returns, in basis points.
        alpha: Two-sided significance level.
        power: Probability of detecting the effect if it is real.

    Returns:
        The sample requirement.

    Raises:
        ValueError: If the effect is zero or the deviation is not positive.
    """
    if effect_bps == 0.0:
        raise ValueError("A zero effect cannot be detected at any sample size.")
    if standard_deviation_bps <= 0.0:
        raise ValueError(
            f"Standard deviation must be positive, got {standard_deviation_bps}."
        )
    z_alpha = normal_quantile(1.0 - alpha / 2.0)
    z_beta = normal_quantile(power)
    n = ((z_alpha + z_beta) * standard_deviation_bps / effect_bps) ** 2
    return SampleRequirement(effect=effect_bps, observations=math.ceil(n))


def bars_per_day(interval_seconds: float) -> float:
    """Return how many bars of a given length occur in a day."""
    if interval_seconds <= 0:
        raise ValueError(f"interval_seconds must be positive, got {interval_seconds}.")
    return 86_400.0 / interval_seconds


def days_to_collect(
    observations: int,
    *,
    horizon_bars: int,
    interval_seconds: float,
    symbols: int = 1,
    firing_rate: float = 1.0,
) -> float:
    """Return the calendar days needed to accumulate an independent sample.

    Independent observations are spaced one horizon apart, so a longer horizon
    costs proportionally more calendar time for the same sample size.

    Args:
        observations: Independent observations required.
        horizon_bars: Forward horizon in bars.
        interval_seconds: Length of one bar in seconds.
        symbols: Symbols recorded in parallel. Note these are correlated, so
            this is an optimistic divisor.
        firing_rate: Fraction of bars on which the signal fires.

    Returns:
        Calendar days of recording required.

    Raises:
        ValueError: If any argument is non-positive.
    """
    if horizon_bars < 1 or symbols < 1:
        raise ValueError("horizon_bars and symbols must be >= 1.")
    if not 0.0 < firing_rate <= 1.0:
        raise ValueError(f"firing_rate must be in (0, 1], got {firing_rate}.")

    per_day = bars_per_day(interval_seconds)
    observations_per_day = (per_day / horizon_bars) * symbols * firing_rate
    return observations / observations_per_day


def standard_deviation_bps(returns: Sequence[float]) -> float:
    """Return the sample standard deviation of returns, in basis points.

    Args:
        returns: Returns as fractions.

    Returns:
        The standard deviation in bps, or ``0.0`` for fewer than two values.
    """
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    variance = sum((value - mean) ** 2 for value in returns) / (n - 1)
    return math.sqrt(variance) * 10_000.0


__all__ = [
    "SampleRequirement",
    "bars_per_day",
    "days_to_collect",
    "normal_cdf",
    "normal_quantile",
    "samples_for_hit_rate",
    "samples_for_mean_return",
    "standard_deviation_bps",
]
