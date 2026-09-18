"""Tests for the significance machinery.

Every conclusion the research harness reaches rests on these functions, so they
are checked against values computed independently: exact rational arithmetic for
the binomial, closed forms and published reference values for the rest.
"""

from __future__ import annotations

import math
from fractions import Fraction

import pytest

from services.research.statistics import (
    benjamini_hochberg,
    binomial_test,
    bonferroni,
    one_sample_t_test,
    regularized_incomplete_beta,
    student_t_sf,
)


def exact_two_sided_binomial(successes: int, trials: int) -> float:
    """Compute the exact two-sided binomial p-value with rational arithmetic.

    Ground truth for the float implementation: no floating point is involved
    until the final conversion.

    Args:
        successes: Observed successes.
        trials: Number of trials.

    Returns:
        The exact p-value under ``p = 1/2``.
    """
    pmf = [Fraction(math.comb(trials, k), 2**trials) for k in range(trials + 1)]
    return float(sum(p for p in pmf if p <= pmf[successes]))


class TestIncompleteBeta:
    """The building block under both continuous tests."""

    def test_symmetry_identity(self) -> None:
        """``I_x(a,b) == 1 - I_(1-x)(b,a)``."""
        value = regularized_incomplete_beta(2.5, 3.5, 0.4)
        mirror = 1.0 - regularized_incomplete_beta(3.5, 2.5, 0.6)
        assert value == pytest.approx(mirror, abs=1e-12)

    def test_uniform_special_case(self) -> None:
        """``I_x(1,1)`` is the identity, since Beta(1,1) is uniform."""
        for x in (0.1, 0.25, 0.5, 0.9):
            assert regularized_incomplete_beta(1.0, 1.0, x) == pytest.approx(x, abs=1e-12)

    def test_endpoints(self) -> None:
        """The function is 0 at 0 and 1 at 1."""
        assert regularized_incomplete_beta(2.0, 3.0, 0.0) == 0.0
        assert regularized_incomplete_beta(2.0, 3.0, 1.0) == 1.0

    def test_rejects_bad_domain(self) -> None:
        """Out-of-range inputs raise rather than return nonsense."""
        with pytest.raises(ValueError, match=r"x must be in"):
            regularized_incomplete_beta(2.0, 2.0, 1.5)
        with pytest.raises(ValueError, match="positive"):
            regularized_incomplete_beta(0.0, 2.0, 0.5)


class TestStudentT:
    """Student's t survival function."""

    def test_median_is_a_half(self) -> None:
        """The distribution is symmetric about zero."""
        assert student_t_sf(0.0, 10) == pytest.approx(0.5, abs=1e-12)

    def test_large_df_approaches_normal(self) -> None:
        """With huge degrees of freedom it matches the normal tail."""
        assert student_t_sf(1.959964, 1e7) == pytest.approx(0.025, abs=1e-5)

    def test_known_critical_value(self) -> None:
        """T = 2.228 at 10 df is the two-sided 5% critical value."""
        assert 2.0 * student_t_sf(2.228139, 10) == pytest.approx(0.05, abs=1e-5)

    def test_symmetry(self) -> None:
        """The two tails sum to one."""
        assert student_t_sf(1.3, 7) + student_t_sf(-1.3, 7) == pytest.approx(1.0, abs=1e-12)

    def test_rejects_bad_df(self) -> None:
        """Non-positive degrees of freedom are invalid."""
        with pytest.raises(ValueError, match="Degrees of freedom"):
            student_t_sf(1.0, 0)


class TestBinomialTest:
    """Exact two-sided binomial test."""

    @pytest.mark.parametrize(("successes", "trials"), [(60, 100), (70, 100), (5, 10), (13, 20)])
    def test_matches_exact_rational_arithmetic(self, successes: int, trials: int) -> None:
        """The float implementation matches exact rational computation."""
        expected = exact_two_sided_binomial(successes, trials)
        assert binomial_test(successes, trials).p_value == pytest.approx(expected, abs=1e-12)

    def test_symmetric_outcome_is_certain(self) -> None:
        """Exactly half successes cannot be evidence against a fair coin."""
        assert binomial_test(50, 100).p_value == pytest.approx(1.0)

    def test_extreme_outcome_is_tiny(self) -> None:
        """An all-or-nothing result is decisive."""
        assert binomial_test(100, 100).p_value == pytest.approx(2 * 0.5**100)

    def test_large_sample_does_not_overflow(self) -> None:
        """A 15,000-trial test is computed in log space without overflowing."""
        result = binomial_test(7_800, 15_000)
        assert 0.0 < result.p_value < 1.0
        assert result.sample_size == 15_000

    def test_symmetry_of_the_two_tails(self) -> None:
        """The test is symmetric about the null for a fair coin."""
        assert binomial_test(60, 100).p_value == pytest.approx(binomial_test(40, 100).p_value)

    def test_zero_trials_is_not_evidence(self) -> None:
        """An empty sample yields p = 1."""
        assert binomial_test(0, 0).p_value == 1.0

    def test_rejects_inconsistent_counts(self) -> None:
        """More successes than trials is invalid."""
        with pytest.raises(ValueError, match="successes <= trials"):
            binomial_test(11, 10)


class TestOneSampleT:
    """One-sample t-test."""

    def test_known_values(self) -> None:
        """A hand-checkable sample reproduces the textbook statistic."""
        result = one_sample_t_test([1.0, 2.0, 3.0, 4.0, 5.0], 0.0)
        assert result.statistic == pytest.approx(4.2426, abs=1e-4)
        assert result.p_value == pytest.approx(0.01323, abs=1e-4)

    def test_mean_equal_to_null_is_not_significant(self) -> None:
        """Data centred on the null gives t = 0."""
        result = one_sample_t_test([-1.0, 0.0, 1.0], 0.0)
        assert result.statistic == pytest.approx(0.0)
        assert result.p_value == pytest.approx(1.0)

    def test_zero_variance_is_not_evidence(self) -> None:
        """A constant sample has no dispersion, so the test is undefined."""
        assert one_sample_t_test([2.0] * 10, 0.0).p_value == 1.0

    def test_single_observation_is_not_evidence(self) -> None:
        """One point cannot establish anything."""
        assert one_sample_t_test([5.0], 0.0).p_value == 1.0

    def test_sign_of_statistic_follows_the_mean(self) -> None:
        """A mean below the null gives a negative t."""
        assert one_sample_t_test([-3.0, -2.0, -4.0], 0.0).statistic < 0


class TestMultipleComparisons:
    """FDR and family-wise corrections."""

    def test_benjamini_hochberg_textbook_example(self) -> None:
        """The original 1995 worked example rejects the first four."""
        p_values = [
            0.0001,
            0.0004,
            0.0019,
            0.0095,
            0.0201,
            0.0278,
            0.0298,
            0.0344,
            0.0459,
            0.3240,
            0.4262,
            0.5719,
            0.6528,
            0.7590,
            1.000,
        ]
        rejected = benjamini_hochberg(p_values, alpha=0.05)
        assert sum(rejected) == 4
        assert rejected[:4] == [True, True, True, True]

    def test_benjamini_hochberg_is_order_independent(self) -> None:
        """Shuffling the inputs does not change which hypotheses are rejected."""
        p_values = [0.04, 0.001, 0.6, 0.02]
        forward = benjamini_hochberg(p_values)
        reversed_flags = benjamini_hochberg(list(reversed(p_values)))
        assert forward == list(reversed(reversed_flags))

    def test_benjamini_hochberg_rejects_nothing_on_noise(self) -> None:
        """Uniform p-values from pure noise produce no discoveries."""
        noise = [i / 20 for i in range(1, 21)]
        assert not any(benjamini_hochberg(noise, alpha=0.01))

    def test_bonferroni_is_stricter_than_fdr(self) -> None:
        """Bonferroni never rejects more than Benjamini-Hochberg."""
        p_values = [0.001, 0.01, 0.02, 0.03, 0.2]
        assert sum(bonferroni(p_values)) <= sum(benjamini_hochberg(p_values))

    def test_empty_inputs(self) -> None:
        """No hypotheses means no rejections."""
        assert benjamini_hochberg([]) == []
        assert bonferroni([]) == []

    def test_rejects_bad_alpha(self) -> None:
        """An alpha outside (0, 1] is invalid."""
        with pytest.raises(ValueError, match="alpha"):
            benjamini_hochberg([0.01], alpha=0.0)
