"""Significance testing for signal research.

Dependency-free implementations of the three tests the harness needs.  They are
written out rather than pulled from SciPy so the numbers can be read and checked
against a textbook, and so the research harness has no extra runtime dependency.

Two statistical hazards dominate this kind of study, and both are handled here
rather than left to the caller's judgement:

**Overlapping samples.**  A signal that fires on most bars and is measured over a
100-bar forward horizon produces observations that share 99 of their 100 bars.
Those are not independent draws.  Treating them as independent inflates the
effective sample size by up to the horizon length and makes p-values look
spectacular for pure noise.  The study module therefore feeds these tests a
*non-overlapping* subsample; see :mod:`services.research.study`.

**Multiple comparisons.**  Nine signals across four horizons is thirty-six
tests.  At alpha = 0.05 roughly two will clear the bar by chance alone.
:func:`benjamini_hochberg` controls the false discovery rate across the whole
matrix so that "significant" means something after the search is accounted for.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

_MAX_ITERATIONS = 300
_EPSILON = 3.0e-16
_TINY = 1.0e-300


@dataclass(frozen=True, slots=True)
class TestResult:
    """Outcome of a significance test.

    Attributes:
        statistic: The test statistic (a t value, or an observed count).
        p_value: Two-sided p-value.
        sample_size: Number of independent observations the test used.
    """

    statistic: float
    p_value: float
    sample_size: int

    @property
    def significant_at_05(self) -> bool:
        """Return whether the raw p-value clears 0.05."""
        return self.p_value < 0.05


def _log_beta(a: float, b: float) -> float:
    """Return the natural log of the beta function ``B(a, b)``."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _betacf(a: float, b: float, x: float) -> float:
    """Evaluate the continued fraction for the incomplete beta function.

    Modified Lentz's method, as in *Numerical Recipes*.

    Args:
        a: First shape parameter.
        b: Second shape parameter.
        x: Evaluation point in ``[0, 1]``.

    Returns:
        The continued fraction value.
    """
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _TINY:
        d = _TINY
    d = 1.0 / d
    h = d

    for m in range(1, _MAX_ITERATIONS + 1):
        m2 = 2 * m
        # Even step.
        numerator = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + numerator * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + numerator / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        h *= d * c
        # Odd step.
        numerator = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + numerator * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + numerator / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPSILON:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Return the regularised incomplete beta function ``I_x(a, b)``.

    Args:
        a: First shape parameter, positive.
        b: Second shape parameter, positive.
        x: Evaluation point in ``[0, 1]``.

    Returns:
        ``I_x(a, b)`` in ``[0, 1]``.

    Raises:
        ValueError: If ``x`` lies outside ``[0, 1]`` or a shape is not positive.
    """
    if a <= 0 or b <= 0:
        raise ValueError(f"Beta shape parameters must be positive, got a={a}, b={b}.")
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"x must be in [0, 1], got {x}.")
    if x in (0.0, 1.0):
        return x

    front = math.exp(a * math.log(x) + b * math.log1p(-x) - _log_beta(a, b))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_sf(t: float, degrees_of_freedom: float) -> float:
    """Return the upper-tail probability ``P(T > t)`` for Student's t.

    Args:
        t: The t statistic.
        degrees_of_freedom: Degrees of freedom, positive.

    Returns:
        The survival function value.

    Raises:
        ValueError: If ``degrees_of_freedom`` is not positive.
    """
    if degrees_of_freedom <= 0:
        raise ValueError(f"Degrees of freedom must be positive, got {degrees_of_freedom}.")
    x = degrees_of_freedom / (degrees_of_freedom + t * t)
    tail = 0.5 * regularized_incomplete_beta(0.5 * degrees_of_freedom, 0.5, x)
    return tail if t > 0 else 1.0 - tail


def _log_binomial_pmf(k: int, n: int, p: float) -> float:
    """Return the log probability mass of ``k`` successes in ``n`` trials."""
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    log_choose = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    return log_choose + k * math.log(p) + (n - k) * math.log1p(-p)


def binomial_test(successes: int, trials: int, probability: float = 0.5) -> TestResult:
    """Exact two-sided binomial test.

    Uses the standard "sum of outcomes at most as likely as the observed one"
    definition, evaluated in log space so it stays exact for large ``n`` where a
    direct product of binomial coefficients would overflow.

    Args:
        successes: Number of successes observed.
        trials: Number of trials.
        probability: Null-hypothesis success probability.

    Returns:
        The test result; the statistic is the observed success count.

    Raises:
        ValueError: If the counts are inconsistent or the probability is invalid.
    """
    if trials < 0 or not 0 <= successes <= trials:
        raise ValueError(f"Need 0 <= successes <= trials, got {successes}/{trials}.")
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {probability}.")
    if trials == 0:
        return TestResult(statistic=0.0, p_value=1.0, sample_size=0)

    observed = _log_binomial_pmf(successes, trials, probability)
    # A tolerance keeps floating-point ties (the symmetric outcome) included.
    threshold = observed + 1e-9
    total = 0.0
    for k in range(trials + 1):
        mass = _log_binomial_pmf(k, trials, probability)
        if mass <= threshold:
            total += math.exp(mass)
    return TestResult(
        statistic=float(successes), p_value=min(1.0, total), sample_size=trials
    )


def one_sample_t_test(values: Sequence[float], population_mean: float = 0.0) -> TestResult:
    """Two-sided one-sample t-test.

    Args:
        values: Observations.
        population_mean: Null-hypothesis mean.

    Returns:
        The test result; ``p_value`` is 1.0 when the test is undefined (fewer
        than two observations, or zero variance).
    """
    n = len(values)
    if n < 2:
        return TestResult(statistic=0.0, p_value=1.0, sample_size=n)

    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    if variance <= 0.0:
        # No dispersion: either every observation equals the null, or the test
        # is degenerate. Neither is evidence.
        return TestResult(statistic=0.0, p_value=1.0, sample_size=n)

    standard_error = math.sqrt(variance / n)
    t = (mean - population_mean) / standard_error
    p = 2.0 * student_t_sf(abs(t), n - 1)
    return TestResult(statistic=t, p_value=min(1.0, p), sample_size=n)


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Return which hypotheses survive Benjamini-Hochberg FDR control.

    Controls the expected proportion of false discoveries among the rejected
    hypotheses, which is the right correction when scanning a grid of signals
    and horizons: some are expected to look good by chance.

    Args:
        p_values: Raw p-values, in the caller's order.
        alpha: Target false discovery rate.

    Returns:
        A list of rejection flags aligned to ``p_values``.

    Raises:
        ValueError: If ``alpha`` is outside ``(0, 1]``.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}.")
    n = len(p_values)
    if n == 0:
        return []

    ordered = sorted(range(n), key=lambda index: p_values[index])
    largest_hit = -1
    for rank, index in enumerate(ordered, start=1):
        if p_values[index] <= alpha * rank / n:
            largest_hit = rank

    rejected = [False] * n
    if largest_hit > 0:
        for index in ordered[:largest_hit]:
            rejected[index] = True
    return rejected


def bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Return which hypotheses survive a Bonferroni correction.

    Stricter than Benjamini-Hochberg: it controls the chance of *any* false
    positive rather than their proportion.

    Args:
        p_values: Raw p-values.
        alpha: Family-wise error rate.

    Returns:
        A list of rejection flags aligned to ``p_values``.
    """
    if not p_values:
        return []
    threshold = alpha / len(p_values)
    return [p <= threshold for p in p_values]


__all__ = [
    "TestResult",
    "benjamini_hochberg",
    "binomial_test",
    "bonferroni",
    "one_sample_t_test",
    "regularized_incomplete_beta",
    "student_t_sf",
]
