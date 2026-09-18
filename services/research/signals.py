"""The signal library: one indicator per entry, no blending.

Phase 3 measured a weighted blend of four component scores and found it had no
edge.  A blend that fails tells you nothing about which of its parts, if any,
carried information.  This module breaks them apart so each can be measured on
its own.

Every signal declares:

* the **convention** it is being tested under, written out in prose.  Most of
  these indicators admit both a momentum and a mean-reversion reading, and the
  two give opposite answers; a result is meaningless without saying which was
  tested.  Where the Phase 2 analyst used a different convention, that is noted.
* the **data it requires**.  Two of the nine need series Hyperliquid does not
  publish historically, and they are marked here rather than quietly dropped.

A signal returns a :class:`SignalReading` carrying a direction and a *conviction*
in ``[0, 1]``.  The study converts conviction to a forecast probability as
``0.5 + 0.5 * conviction``, so zero conviction means a 50/50 call scoring exactly
the 0.25 Brier baseline, and full conviction stakes everything on the call.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from libs.schemas.market import Candle, FundingRate, OrderBookSnapshot
from libs.schemas.signals import Direction
from services.agents.market_analyst import indicators as ind

logger = logging.getLogger(__name__)


class DataRequirement(StrEnum):
    """A data series a signal needs in order to be evaluated."""

    CANDLES = "candles"
    FUNDING = "funding"
    ORDER_BOOK = "order_book"
    OPEN_INTEREST = "open_interest"


HISTORICALLY_AVAILABLE: Final[frozenset[DataRequirement]] = frozenset(
    {DataRequirement.CANDLES, DataRequirement.FUNDING}
)
"""What Hyperliquid publishes historically.

``candleSnapshot`` and ``fundingHistory`` return series.  ``l2Book`` and
``metaAndAssetCtxs`` return only the current snapshot, so order book depth and
open interest cannot be backtested -- they would have to be recorded forward
from now.
"""


@dataclass(frozen=True, slots=True)
class SignalWindow:
    """Everything a signal is allowed to see at one decision point.

    The window ends at the decision bar inclusive and contains nothing after it,
    which is what makes look-ahead structurally impossible rather than merely
    avoided by convention.

    Attributes:
        candles: Bounded candle history, oldest first, ending at the decision bar.
        funding: Funding observations at or before the decision bar, if loaded.
        order_book: Book snapshot at the decision bar, if available.
        open_interest: Open-interest history ending at the decision bar.
    """

    candles: tuple[Candle, ...]
    funding: tuple[FundingRate, ...] = ()
    order_book: OrderBookSnapshot | None = None
    open_interest: tuple[float, ...] = ()

    @property
    def closes(self) -> list[float]:
        """Return close prices as floats, oldest first."""
        return [float(candle.close) for candle in self.candles]

    @property
    def highs(self) -> list[float]:
        """Return high prices as floats, oldest first."""
        return [float(candle.high) for candle in self.candles]

    @property
    def lows(self) -> list[float]:
        """Return low prices as floats, oldest first."""
        return [float(candle.low) for candle in self.candles]

    @property
    def volumes(self) -> list[float]:
        """Return volumes as floats, oldest first."""
        return [float(candle.volume) for candle in self.candles]


@dataclass(frozen=True, slots=True)
class SignalReading:
    """One signal's view at one bar.

    Attributes:
        direction: The directional call. ``FLAT`` means no position.
        conviction: Strength in ``[0, 1]``; 0 is a coin flip.
        value: The underlying indicator value, for diagnostics.
    """

    direction: Direction
    conviction: float
    value: float

    @property
    def probability(self) -> float:
        """Return the forecast probability that this call is correct."""
        return 0.5 + 0.5 * max(0.0, min(1.0, self.conviction))


@dataclass(frozen=True, slots=True)
class SignalSpec:
    """A named, self-describing signal.

    Attributes:
        convention: Which of the competing readings is being tested.
        requires: Data series the signal needs.
        min_bars: Bars needed before the signal can be evaluated.
    """

    name: str
    description: str
    convention: str
    min_bars: int
    requires: frozenset[DataRequirement]
    evaluate: Callable[[SignalWindow], SignalReading | None]

    @property
    def is_backtestable(self) -> bool:
        """Return whether every required series is available historically."""
        return self.requires <= HISTORICALLY_AVAILABLE

    @property
    def missing_data(self) -> tuple[str, ...]:
        """Return the required series that have no historical source."""
        return tuple(sorted(r.value for r in self.requires - HISTORICALLY_AVAILABLE))


FLAT: Final[SignalReading] = SignalReading(
    direction=Direction.FLAT, conviction=0.0, value=0.0
)
"""A no-opinion reading."""


def _directional(value: float, conviction: float) -> SignalReading:
    """Build a reading whose direction follows the sign of ``value``."""
    clamped = max(0.0, min(1.0, conviction))
    if value > 0:
        return SignalReading(direction=Direction.LONG, conviction=clamped, value=value)
    if value < 0:
        return SignalReading(direction=Direction.SHORT, conviction=clamped, value=value)
    return SignalReading(direction=Direction.FLAT, conviction=0.0, value=value)


# ----------------------------------------------------------------------
# Price-derived signals
# ----------------------------------------------------------------------


def rsi_reversion(window: SignalWindow, *, period: int = 14, band: float = 30.0) -> SignalReading:
    """RSI read as mean reversion: oversold buys, overbought sells.

    This is the textbook reading.  The Phase 2 analyst used the opposite
    (momentum) convention, treating RSI above 50 as bullish; that convention is
    tested separately as ``rsi_momentum``.

    Args:
        window: The decision window.
        period: RSI lookback.
        band: Distance from 50 that counts as stretched.

    Returns:
        A long reading below ``50 - band``, short above ``50 + band``.
    """
    value = ind.last_defined(ind.rsi(window.closes, period))
    if value is None:
        return FLAT
    if value <= 50.0 - band:
        return SignalReading(
            direction=Direction.LONG,
            conviction=min(1.0, (50.0 - band - value) / band),
            value=value,
        )
    if value >= 50.0 + band:
        return SignalReading(
            direction=Direction.SHORT,
            conviction=min(1.0, (value - 50.0 - band) / band),
            value=value,
        )
    return SignalReading(direction=Direction.FLAT, conviction=0.0, value=value)


def rsi_momentum(window: SignalWindow, *, period: int = 14) -> SignalReading:
    """RSI read as momentum: above 50 is bullish.

    This is the convention the Phase 2 analyst used, isolated here so the
    blend's failure can be attributed.

    Args:
        window: The decision window.
        period: RSI lookback.

    Returns:
        A directional reading scaled by distance from 50.
    """
    value = ind.last_defined(ind.rsi(window.closes, period))
    if value is None:
        return FLAT
    centred = (value - 50.0) / 50.0
    return _directional(centred, abs(centred))


def macd_histogram(window: SignalWindow) -> SignalReading:
    """MACD histogram sign, read as momentum.

    Conviction scales with the histogram relative to price, so it is comparable
    across instruments at very different price levels.

    Args:
        window: The decision window.

    Returns:
        A long reading on a positive histogram, short on negative.
    """
    closes = window.closes
    result = ind.macd(closes)
    value = ind.last_defined(result.histogram)
    if value is None or not closes or closes[-1] <= 0:
        return FLAT
    normalised = value / closes[-1] * 200.0
    return _directional(normalised, abs(normalised))


def ema_cross(window: SignalWindow, *, fast: int = 20, slow: int = 50) -> SignalReading:
    """Fast EMA above slow EMA, read as trend following.

    Args:
        window: The decision window.
        fast: Fast EMA period.
        slow: Slow EMA period.

    Returns:
        A long reading when the fast EMA leads, short when it lags.
    """
    closes = window.closes
    fast_value = ind.last_defined(ind.ema(closes, fast))
    slow_value = ind.last_defined(ind.ema(closes, slow))
    if fast_value is None or slow_value is None or slow_value == 0:
        return FLAT
    separation = (fast_value - slow_value) / slow_value * 100.0
    return _directional(separation, abs(separation))


def bollinger_percent_b(
    window: SignalWindow, *, period: int = 20, num_std: float = 2.0
) -> SignalReading:
    """Bollinger %B read as mean reversion.

    Below the lower band is a long, above the upper band a short.  This is the
    textbook reading; note it is the opposite of a breakout interpretation.

    Args:
        window: The decision window.
        period: Band period.
        num_std: Band width in standard deviations.

    Returns:
        A reading that fades band excursions.
    """
    bands = ind.bollinger(window.closes, period, num_std)
    value = ind.last_defined(bands.percent_b)
    if value is None:
        return FLAT
    if value < 0.0:
        return SignalReading(
            direction=Direction.LONG, conviction=min(1.0, -value * 2.0), value=value
        )
    if value > 1.0:
        return SignalReading(
            direction=Direction.SHORT, conviction=min(1.0, (value - 1.0) * 2.0), value=value
        )
    return SignalReading(direction=Direction.FLAT, conviction=0.0, value=value)


def atr_normalized_momentum(
    window: SignalWindow, *, lookback: int = 10, atr_period: int = 14
) -> SignalReading:
    """Price change over ``lookback`` bars, divided by ATR.

    Dividing by ATR makes the move comparable across regimes: a 2% move in a
    quiet market is a much larger signal than the same move in a volatile one.

    Args:
        window: The decision window.
        lookback: Bars over which to measure the move.
        atr_period: ATR lookback.

    Returns:
        A directional reading following the normalised move.
    """
    closes = window.closes
    if len(closes) <= lookback:
        return FLAT
    atr_value = ind.last_defined(ind.atr(window.highs, window.lows, closes, atr_period))
    if atr_value is None or atr_value <= 0:
        return FLAT
    move = (closes[-1] - closes[-1 - lookback]) / atr_value
    # Two ATRs of movement is treated as a full-conviction signal.
    return _directional(move, abs(move) / 2.0)


def obv_divergence(window: SignalWindow, *, lookback: int = 20) -> SignalReading:
    """On-balance-volume divergence against price.

    Price making a new high while OBV does not is read as bearish (the move is
    not backed by volume); price making a new low while OBV does not is bullish.

    Args:
        window: The decision window.
        lookback: Window over which the extremes are compared.

    Returns:
        A reading that fades unconfirmed price extremes.
    """
    closes = window.closes
    if len(closes) < lookback * 2:
        return FLAT
    obv = ind.on_balance_volume(closes, window.volumes)

    recent_prices, prior_prices = closes[-lookback:], closes[-2 * lookback : -lookback]
    recent_obv, prior_obv = obv[-lookback:], obv[-2 * lookback : -lookback]

    price_high_break = max(recent_prices) > max(prior_prices)
    price_low_break = min(recent_prices) < min(prior_prices)
    obv_high_break = max(recent_obv) > max(prior_obv)
    obv_low_break = min(recent_obv) < min(prior_obv)

    scale = max(abs(max(prior_obv)), abs(min(prior_obv)), 1.0)
    magnitude = min(1.0, abs(max(recent_obv) - max(prior_obv)) / scale)

    if price_high_break and not obv_high_break:
        return SignalReading(direction=Direction.SHORT, conviction=magnitude, value=-magnitude)
    if price_low_break and not obv_low_break:
        return SignalReading(direction=Direction.LONG, conviction=magnitude, value=magnitude)
    return SignalReading(direction=Direction.FLAT, conviction=0.0, value=0.0)


# ----------------------------------------------------------------------
# Non-price signals
# ----------------------------------------------------------------------


def funding_rate_contrarian(window: SignalWindow, *, scale: float = 0.0001) -> SignalReading:
    """Funding rate read as a crowding indicator.

    Positive funding means longs are paying shorts, which is read as crowded
    long and therefore bearish.  The reading is contrarian by construction.

    Args:
        window: The decision window.
        scale: Funding rate treated as full conviction (1 bp per hour by default).

    Returns:
        A reading opposing the funding sign.
    """
    if not window.funding:
        return FLAT
    rate = float(window.funding[-1].funding_rate)
    if rate == 0.0:
        return SignalReading(direction=Direction.FLAT, conviction=0.0, value=0.0)
    return _directional(-rate, abs(rate) / scale)


def order_book_imbalance(window: SignalWindow, *, depth: int = 5) -> SignalReading:
    """Resting bid size versus ask size at the top of the book.

    Positive imbalance means resting bids dominate, read as buy pressure.

    **Not backtestable.** Hyperliquid's ``l2Book`` returns only the current
    snapshot; there is no historical depth endpoint. To measure this signal the
    platform would have to record book snapshots forward -- the existing
    ``hyperliquid_orderbook`` collector already produces them, but nothing
    persists them yet.

    Args:
        window: The decision window.
        depth: Levels per side to include.

    Returns:
        A reading following the imbalance, or flat with no book.
    """
    book = window.order_book
    if book is None:
        return FLAT
    imbalance = book.imbalance(depth=depth)
    return _directional(imbalance, abs(imbalance))


def open_interest_change(window: SignalWindow, *, lookback: int = 24) -> SignalReading:
    """Change in open interest, read as confirmation of the price move.

    Rising open interest alongside rising price is read as a strengthening
    trend; rising open interest into falling price as strengthening downside.

    **Not backtestable.** ``metaAndAssetCtxs`` reports current open interest
    only; Hyperliquid publishes no historical series.

    Args:
        window: The decision window.
        lookback: Bars over which the change is measured.

    Returns:
        A reading combining the OI change with the price move, or flat.
    """
    series = window.open_interest
    closes = window.closes
    if len(series) <= lookback or len(closes) <= lookback:
        return FLAT
    previous = series[-1 - lookback]
    if previous <= 0:
        return FLAT
    oi_change = (series[-1] - previous) / previous
    price_change = (closes[-1] - closes[-1 - lookback]) / closes[-1 - lookback]
    if oi_change <= 0:
        return SignalReading(direction=Direction.FLAT, conviction=0.0, value=oi_change)
    return _directional(price_change, min(1.0, abs(oi_change) * 5.0))


SIGNALS: Final[dict[str, SignalSpec]] = {
    spec.name: spec
    for spec in (
        SignalSpec(
            name="rsi",
            description="RSI(14) at oversold/overbought extremes.",
            convention="mean reversion (textbook): oversold buys, overbought sells",
            min_bars=30,
            requires=frozenset({DataRequirement.CANDLES}),
            evaluate=rsi_reversion,
        ),
        SignalSpec(
            name="rsi_momentum",
            description="RSI(14) distance from 50, read as momentum.",
            convention="momentum: above 50 bullish (the Phase 2 analyst's reading)",
            min_bars=30,
            requires=frozenset({DataRequirement.CANDLES}),
            evaluate=rsi_momentum,
        ),
        SignalSpec(
            name="macd_histogram",
            description="MACD(12,26,9) histogram sign.",
            convention="momentum: positive histogram bullish",
            min_bars=60,
            requires=frozenset({DataRequirement.CANDLES}),
            evaluate=macd_histogram,
        ),
        SignalSpec(
            name="ema_cross",
            description="EMA(20) versus EMA(50) separation.",
            convention="trend following: fast above slow bullish",
            min_bars=60,
            requires=frozenset({DataRequirement.CANDLES}),
            evaluate=ema_cross,
        ),
        SignalSpec(
            name="bollinger_percent_b",
            description="Bollinger(20, 2) %B outside the bands.",
            convention="mean reversion: fade excursions beyond the bands",
            min_bars=30,
            requires=frozenset({DataRequirement.CANDLES}),
            evaluate=bollinger_percent_b,
        ),
        SignalSpec(
            name="atr_momentum",
            description="10-bar price change divided by ATR(14).",
            convention="momentum: normalised move continues",
            min_bars=40,
            requires=frozenset({DataRequirement.CANDLES}),
            evaluate=atr_normalized_momentum,
        ),
        SignalSpec(
            name="obv_divergence",
            description="On-balance-volume divergence against 20-bar price extremes.",
            convention="mean reversion: fade price extremes unconfirmed by volume",
            min_bars=60,
            requires=frozenset({DataRequirement.CANDLES}),
            evaluate=obv_divergence,
        ),
        SignalSpec(
            name="funding_rate",
            description="Hourly funding rate as a crowding measure.",
            convention="contrarian: positive funding (crowded long) is bearish",
            min_bars=2,
            requires=frozenset({DataRequirement.CANDLES, DataRequirement.FUNDING}),
            evaluate=funding_rate_contrarian,
        ),
        SignalSpec(
            name="orderbook_imbalance",
            description="Top-of-book resting size imbalance.",
            convention="momentum: bid-heavy book is bullish",
            min_bars=2,
            requires=frozenset({DataRequirement.CANDLES, DataRequirement.ORDER_BOOK}),
            evaluate=order_book_imbalance,
        ),
        SignalSpec(
            name="open_interest_change",
            description="24-bar open interest change confirming the price move.",
            convention="confirmation: rising OI strengthens the prevailing move",
            min_bars=30,
            requires=frozenset({DataRequirement.CANDLES, DataRequirement.OPEN_INTEREST}),
            evaluate=open_interest_change,
        ),
    )
}
"""Every signal under test, keyed by name."""

BACKTESTABLE_SIGNALS: Final[tuple[str, ...]] = tuple(
    name for name, spec in SIGNALS.items() if spec.is_backtestable
)
"""Signals whose inputs Hyperliquid publishes historically."""

BLOCKED_SIGNALS: Final[tuple[str, ...]] = tuple(
    name for name, spec in SIGNALS.items() if not spec.is_backtestable
)
"""Signals that cannot be backtested for lack of historical data."""


def get_signal(name: str) -> SignalSpec:
    """Look up a signal by name.

    Args:
        name: Signal identifier.

    Returns:
        The signal specification.

    Raises:
        KeyError: If no such signal exists.
    """
    try:
        return SIGNALS[name]
    except KeyError:
        raise KeyError(f"Unknown signal {name!r}; known: {sorted(SIGNALS)}") from None


def signal_names(include_blocked: bool = False) -> tuple[str, ...]:
    """Return signal names, optionally including those with no historical data."""
    if include_blocked:
        return tuple(SIGNALS)
    return BACKTESTABLE_SIGNALS


def describe(specs: Sequence[SignalSpec] | None = None) -> str:
    """Render a human-readable table of the signal library."""
    chosen = specs if specs is not None else list(SIGNALS.values())
    lines = []
    for spec in chosen:
        status = "ok" if spec.is_backtestable else f"NO DATA ({', '.join(spec.missing_data)})"
        lines.append(f"  {spec.name:22s} {status:28s} {spec.convention}")
    return "\n".join(lines)


__all__ = [
    "BACKTESTABLE_SIGNALS",
    "BLOCKED_SIGNALS",
    "HISTORICALLY_AVAILABLE",
    "SIGNALS",
    "DataRequirement",
    "SignalReading",
    "SignalSpec",
    "SignalWindow",
    "describe",
    "get_signal",
    "signal_names",
]
