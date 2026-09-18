"""Confidence calibration analysis.

A model is *calibrated* when the things it calls with 70% confidence come true
about 70% of the time.  Accuracy and calibration are different properties: an
agent can pick direction well and still be badly calibrated by stating 0.9
confidence on every call.  Since confidence is what the risk manager sizes
against and what the Brier score rewards, calibration is the property that
matters operationally.

This module buckets forecasts by stated confidence and reports the gap between
stated confidence and realised hit rate, plus two summary numbers:

``brier``
    Mean squared error of the forecasts. Lower is better; 0.25 is a coin-flip.
``ece`` (expected calibration error)
    Sample-weighted mean absolute gap between confidence and outcome across
    buckets. 0.0 is perfect calibration.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from services.decision_engine.trust import brier_score


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    """One confidence bucket and how its forecasts actually turned out."""

    lower: float
    upper: float
    count: int
    mean_confidence: float
    hit_rate: float

    @property
    def gap(self) -> float:
        """Return hit rate minus stated confidence.

        Negative means overconfident: the agent claimed more than it delivered.
        """
        return self.hit_rate - self.mean_confidence

    @property
    def label(self) -> str:
        """Return a printable range label."""
        return f"{self.lower:.2f}-{self.upper:.2f}"


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Calibration of a set of confidence/outcome pairs."""

    bins: tuple[CalibrationBin, ...]
    sample_count: int
    brier: float
    ece: float
    mean_confidence: float
    hit_rate: float

    @property
    def overconfidence(self) -> float:
        """Return mean confidence minus hit rate.

        Positive means the agent is overconfident overall.
        """
        return self.mean_confidence - self.hit_rate


def analyze_calibration(
    confidences: Sequence[float], outcomes: Sequence[bool], *, bin_count: int = 5
) -> CalibrationReport:
    """Bucket forecasts by confidence and measure calibration.

    Args:
        confidences: Stated confidences in ``[0, 1]``.
        outcomes: Whether each forecast turned out correct.
        bin_count: Number of equal-width confidence buckets.

    Returns:
        The calibration report; an empty report when there are no samples.

    Raises:
        ValueError: If the inputs differ in length or ``bin_count`` is < 1.
    """
    if len(confidences) != len(outcomes):
        raise ValueError("Confidence and outcome series must be the same length.")
    if bin_count < 1:
        raise ValueError(f"bin_count must be >= 1, got {bin_count}.")

    total = len(confidences)
    if total == 0:
        return CalibrationReport(
            bins=(), sample_count=0, brier=0.0, ece=0.0, mean_confidence=0.0, hit_rate=0.0
        )

    width = 1.0 / bin_count
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bin_count)]
    for confidence, outcome in zip(confidences, outcomes, strict=True):
        index = min(int(confidence / width), bin_count - 1)
        buckets[index].append((confidence, outcome))

    bins: list[CalibrationBin] = []
    ece = 0.0
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        mean_confidence = sum(c for c, _ in bucket) / len(bucket)
        hit_rate = sum(1 for _, o in bucket if o) / len(bucket)
        bins.append(
            CalibrationBin(
                lower=index * width,
                upper=(index + 1) * width,
                count=len(bucket),
                mean_confidence=round(mean_confidence, 4),
                hit_rate=round(hit_rate, 4),
            )
        )
        ece += (len(bucket) / total) * abs(hit_rate - mean_confidence)

    brier = sum(
        brier_score(c, o) for c, o in zip(confidences, outcomes, strict=True)
    ) / total

    return CalibrationReport(
        bins=tuple(bins),
        sample_count=total,
        brier=round(brier, 6),
        ece=round(ece, 6),
        mean_confidence=round(sum(confidences) / total, 4),
        hit_rate=round(sum(1 for o in outcomes if o) / total, 4),
    )


__all__ = ["CalibrationBin", "CalibrationReport", "analyze_calibration"]
