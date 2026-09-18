"""Swing-pivot based support and resistance detection.

The approach is deliberately simple and explainable:

1. Find swing pivots -- bars whose high (or low) is the most extreme within a
   window of ``lookback`` bars on each side.
2. Cluster pivots that sit within ``cluster_pct`` of each other into one level;
   a level touched repeatedly is a stronger level.
3. Score each level by touch count and recency, then classify it as support
   (below current price) or resistance (above it).

Levels are returned as immutable
:class:`~libs.schemas.signals.SupportResistanceLevel` events so they can travel
inside an ``AgentSignal`` unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from statistics import fmean

from libs.schemas.signals import SupportResistanceLevel


@dataclass(frozen=True, slots=True)
class Pivot:
    """A confirmed swing high or swing low.

    Attributes:
        index: Bar index of the pivot within the analysed window.
        price: Pivot price.
        kind: ``"high"`` for a swing high, ``"low"`` for a swing low.
    """

    index: int
    price: float
    kind: str


@dataclass(frozen=True, slots=True)
class PriceLevel:
    """A cluster of pivots treated as one horizontal level.

    Attributes:
        price: Mean price of the clustered pivots.
        touches: Number of pivots in the cluster.
        last_index: Bar index of the most recent pivot in the cluster.
        strength: Normalised strength in ``[0, 1]``.
    """

    price: float
    touches: int
    last_index: int
    strength: float


def find_pivots(highs: Sequence[float], lows: Sequence[float], lookback: int = 3) -> list[Pivot]:
    """Find confirmed swing highs and lows.

    A pivot needs ``lookback`` bars on **both** sides, so the most recent
    ``lookback`` bars can never produce one -- this is what makes a pivot
    confirmed rather than provisional.

    Args:
        highs: High prices, oldest first.
        lows: Low prices, oldest first.
        lookback: Bars required on each side of the pivot.

    Returns:
        Pivots in chronological order.

    Raises:
        ValueError: If the series lengths differ or ``lookback`` is not positive.
    """
    if len(highs) != len(lows):
        raise ValueError("High and low series must be the same length.")
    if lookback < 1:
        raise ValueError(f"Pivot lookback must be >= 1, got {lookback}.")

    pivots: list[Pivot] = []
    for i in range(lookback, len(highs) - lookback):
        window = range(i - lookback, i + lookback + 1)
        if all(highs[i] >= highs[j] for j in window) and any(
            highs[i] > highs[j] for j in window if j != i
        ):
            pivots.append(Pivot(index=i, price=highs[i], kind="high"))
        if all(lows[i] <= lows[j] for j in window) and any(
            lows[i] < lows[j] for j in window if j != i
        ):
            pivots.append(Pivot(index=i, price=lows[i], kind="low"))
    return pivots


def cluster_pivots(
    pivots: Sequence[Pivot], *, cluster_pct: float, total_bars: int
) -> list[PriceLevel]:
    """Merge nearby pivots into levels and score them.

    Strength blends how often a level was touched with how recently it was
    touched, so an old level that was hit once ranks below a level that was
    respected three times in the recent window.

    Args:
        pivots: Pivots to cluster.
        cluster_pct: Percent distance within which pivots merge.
        total_bars: Total bars analysed, used for the recency term.

    Returns:
        Levels sorted by descending strength.

    Raises:
        ValueError: If ``cluster_pct`` is not positive.
    """
    if cluster_pct <= 0:
        raise ValueError(f"cluster_pct must be > 0, got {cluster_pct}.")
    if not pivots:
        return []

    ordered = sorted(pivots, key=lambda pivot: pivot.price)
    clusters: list[list[Pivot]] = [[ordered[0]]]
    for pivot in ordered[1:]:
        current = clusters[-1]
        centre = fmean(member.price for member in current)
        if centre > 0 and abs(pivot.price - centre) / centre * 100.0 <= cluster_pct:
            current.append(pivot)
        else:
            clusters.append([pivot])

    max_touches = max(len(cluster) for cluster in clusters)
    denominator = max(total_bars - 1, 1)

    levels: list[PriceLevel] = []
    for cluster in clusters:
        last_index = max(member.index for member in cluster)
        touch_score = len(cluster) / max_touches
        recency_score = last_index / denominator
        levels.append(
            PriceLevel(
                price=fmean(member.price for member in cluster),
                touches=len(cluster),
                last_index=last_index,
                strength=round(0.65 * touch_score + 0.35 * recency_score, 4),
            )
        )
    return sorted(levels, key=lambda level: level.strength, reverse=True)


def detect_levels(
    highs: Sequence[float],
    lows: Sequence[float],
    current_price: float,
    *,
    lookback: int = 3,
    cluster_pct: float = 0.35,
    max_per_side: int = 3,
    source: str = "market_analyst",
) -> tuple[SupportResistanceLevel, ...]:
    """Detect the strongest support and resistance levels around ``current_price``.

    Args:
        highs: High prices, oldest first.
        lows: Low prices, oldest first.
        current_price: Price used to split supports from resistances.
        lookback: Bars required on each side to confirm a pivot.
        cluster_pct: Percent distance within which pivots merge.
        max_per_side: Levels reported per side.
        source: Event source name stamped on the emitted levels.

    Returns:
        Supports (nearest first) followed by resistances (nearest first).
    """
    if current_price <= 0 or not highs:
        return ()

    pivots = find_pivots(highs, lows, lookback=lookback)
    levels = cluster_pivots(pivots, cluster_pct=cluster_pct, total_bars=len(highs))

    supports = sorted(
        (level for level in levels if level.price < current_price),
        key=lambda level: current_price - level.price,
    )[:max_per_side]
    resistances = sorted(
        (level for level in levels if level.price > current_price),
        key=lambda level: level.price - current_price,
    )[:max_per_side]

    return tuple(
        SupportResistanceLevel(
            source=source,
            price=Decimal(str(round(level.price, 8))),
            kind=kind,
            touches=level.touches,
            strength=level.strength,
            distance_pct=round((level.price - current_price) / current_price * 100.0, 4),
        )
        for kind, group in (("support", supports), ("resistance", resistances))
        for level in group
    )


def nearest_level(
    levels: Sequence[SupportResistanceLevel], kind: str
) -> SupportResistanceLevel | None:
    """Return the closest level of ``kind`` by absolute distance, if any.

    Args:
        levels: Detected levels.
        kind: ``"support"`` or ``"resistance"``.

    Returns:
        The nearest matching level, or ``None``.
    """
    candidates = [level for level in levels if level.kind == kind]
    if not candidates:
        return None
    return min(candidates, key=lambda level: abs(level.distance_pct))


__all__ = [
    "Pivot",
    "PriceLevel",
    "cluster_pivots",
    "detect_levels",
    "find_pivots",
    "nearest_level",
]
