"""Technical indicators, implemented in pure Python.

Every function returns a series **aligned to the input length**, with ``None``
in the warm-up region where the indicator is not yet defined.  Alignment makes
crossover and slope logic straightforward: index ``i`` of any output always
refers to index ``i`` of the input.

The implementations are deliberately dependency-free (no numpy/pandas) so that
indicator maths stays deterministic, trivially unit-testable, and cheap to run
inside an agent's hot path.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean, pstdev

Series = list[float | None]
"""An indicator output aligned to the input, ``None`` during warm-up."""


class IndicatorError(ValueError):
    """Raised when an indicator is called with invalid parameters."""


def _validate(values: Sequence[float], period: int, name: str) -> None:
    """Validate common indicator arguments.

    Args:
        values: Input series.
        period: Lookback period.
        name: Indicator name for error messages.

    Raises:
        IndicatorError: If the period is not a positive integer.
    """
    if period < 1:
        raise IndicatorError(f"{name} period must be >= 1, got {period}.")


def sma(values: Sequence[float], period: int) -> Series:
    """Simple moving average.

    Args:
        values: Input series, oldest first.
        period: Averaging window.

    Returns:
        The SMA series aligned to ``values``.
    """
    _validate(values, period, "SMA")
    out: Series = [None] * len(values)
    if len(values) < period:
        return out
    window_sum = sum(values[:period])
    out[period - 1] = window_sum / period
    for i in range(period, len(values)):
        window_sum += values[i] - values[i - period]
        out[i] = window_sum / period
    return out


def ema(values: Sequence[float], period: int) -> Series:
    """Exponential moving average, seeded with the first SMA.

    Args:
        values: Input series, oldest first.
        period: Smoothing period.

    Returns:
        The EMA series aligned to ``values``.
    """
    _validate(values, period, "EMA")
    out: Series = [None] * len(values)
    if len(values) < period:
        return out
    multiplier = 2.0 / (period + 1.0)
    current = fmean(values[:period])
    out[period - 1] = current
    for i in range(period, len(values)):
        current = (values[i] - current) * multiplier + current
        out[i] = current
    return out


def rsi(values: Sequence[float], period: int = 14) -> Series:
    """Relative Strength Index using Wilder's smoothing.

    Args:
        values: Close prices, oldest first.
        period: Lookback period.

    Returns:
        The RSI series in ``[0, 100]``, aligned to ``values``.
    """
    _validate(values, period, "RSI")
    out: Series = [None] * len(values)
    if len(values) <= period:
        return out

    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        change = values[i] - values[i - 1]
        gains += max(change, 0.0)
        losses += max(-change, 0.0)
    avg_gain = gains / period
    avg_loss = losses / period
    out[period] = _rsi_from_averages(avg_gain, avg_loss)

    for i in range(period + 1, len(values)):
        change = values[i] - values[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(change, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-change, 0.0)) / period
        out[i] = _rsi_from_averages(avg_gain, avg_loss)
    return out


def _rsi_from_averages(avg_gain: float, avg_loss: float) -> float:
    """Convert Wilder averages into an RSI value, handling the zero-loss case."""
    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0.0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


@dataclass(frozen=True, slots=True)
class MacdResult:
    """MACD line, signal line and histogram, each aligned to the input.

    Attributes:
        macd: Fast EMA minus slow EMA.
        signal: EMA of the MACD line.
        histogram: MACD minus signal.
    """

    macd: Series
    signal: Series
    histogram: Series


def macd(
    values: Sequence[float], fast: int = 12, slow: int = 26, signal_period: int = 9
) -> MacdResult:
    """Moving Average Convergence Divergence.

    Args:
        values: Close prices, oldest first.
        fast: Fast EMA period.
        slow: Slow EMA period.
        signal_period: Signal EMA period.

    Returns:
        The MACD, signal and histogram series.

    Raises:
        IndicatorError: If ``fast`` is not shorter than ``slow``.
    """
    if fast >= slow:
        raise IndicatorError(f"MACD fast period ({fast}) must be < slow period ({slow}).")
    fast_ema = ema(values, fast)
    slow_ema = ema(values, slow)

    macd_line: Series = [
        None if f is None or s is None else f - s for f, s in zip(fast_ema, slow_ema, strict=True)
    ]
    defined = [(i, v) for i, v in enumerate(macd_line) if v is not None]
    signal_line: Series = [None] * len(values)
    histogram: Series = [None] * len(values)
    if len(defined) >= signal_period:
        signal_values = ema([v for _, v in defined], signal_period)
        for (index, macd_value), signal_value in zip(defined, signal_values, strict=True):
            signal_line[index] = signal_value
            if signal_value is not None:
                histogram[index] = macd_value - signal_value
    return MacdResult(macd=macd_line, signal=signal_line, histogram=histogram)


@dataclass(frozen=True, slots=True)
class BollingerResult:
    """Bollinger band series aligned to the input.

    Attributes:
        upper: Middle band plus ``num_std`` standard deviations.
        middle: The SMA of the input.
        lower: Middle band minus ``num_std`` standard deviations.
        percent_b: Position of price within the bands (0 = lower, 1 = upper).
        bandwidth: Band width as a fraction of the middle band.
    """

    upper: Series
    middle: Series
    lower: Series
    percent_b: Series
    bandwidth: Series


def bollinger(values: Sequence[float], period: int = 20, num_std: float = 2.0) -> BollingerResult:
    """Bollinger bands around a simple moving average.

    Args:
        values: Close prices, oldest first.
        period: SMA period.
        num_std: Standard deviations for the band offset.

    Returns:
        The band series, plus %B and bandwidth.

    Raises:
        IndicatorError: If ``num_std`` is not positive.
    """
    _validate(values, period, "Bollinger")
    if num_std <= 0:
        raise IndicatorError(f"Bollinger num_std must be > 0, got {num_std}.")

    middle = sma(values, period)
    upper: Series = [None] * len(values)
    lower: Series = [None] * len(values)
    percent_b: Series = [None] * len(values)
    bandwidth: Series = [None] * len(values)

    for i in range(period - 1, len(values)):
        centre = middle[i]
        if centre is None:
            continue
        deviation = pstdev(values[i - period + 1 : i + 1])
        top = centre + num_std * deviation
        bottom = centre - num_std * deviation
        upper[i] = top
        lower[i] = bottom
        span = top - bottom
        percent_b[i] = (values[i] - bottom) / span if span > 0 else 0.5
        bandwidth[i] = span / centre if centre != 0 else 0.0

    return BollingerResult(
        upper=upper, middle=middle, lower=lower, percent_b=percent_b, bandwidth=bandwidth
    )


def true_range(
    highs: Sequence[float], lows: Sequence[float], closes: Sequence[float]
) -> list[float]:
    """True Range for each bar.

    Args:
        highs: High prices, oldest first.
        lows: Low prices, oldest first.
        closes: Close prices, oldest first.

    Returns:
        True Range per bar; the first bar uses the high-low range.

    Raises:
        IndicatorError: If the input series lengths differ.
    """
    if not (len(highs) == len(lows) == len(closes)):
        raise IndicatorError("High, low and close series must be the same length.")
    if not highs:
        return []
    ranges = [highs[0] - lows[0]]
    for i in range(1, len(highs)):
        previous_close = closes[i - 1]
        ranges.append(
            max(
                highs[i] - lows[i],
                abs(highs[i] - previous_close),
                abs(lows[i] - previous_close),
            )
        )
    return ranges


def atr(
    highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int = 14
) -> Series:
    """Average True Range using Wilder's smoothing.

    Args:
        highs: High prices, oldest first.
        lows: Low prices, oldest first.
        closes: Close prices, oldest first.
        period: Smoothing period.

    Returns:
        The ATR series aligned to the inputs.
    """
    _validate(closes, period, "ATR")
    ranges = true_range(highs, lows, closes)
    out: Series = [None] * len(ranges)
    if len(ranges) < period:
        return out
    current = fmean(ranges[:period])
    out[period - 1] = current
    for i in range(period, len(ranges)):
        current = (current * (period - 1) + ranges[i]) / period
        out[i] = current
    return out


def on_balance_volume(closes: Sequence[float], volumes: Sequence[float]) -> list[float]:
    """On-Balance Volume, a running signed volume total.

    Args:
        closes: Close prices, oldest first.
        volumes: Volumes, oldest first.

    Returns:
        The OBV series aligned to the inputs.

    Raises:
        IndicatorError: If the input series lengths differ.
    """
    if len(closes) != len(volumes):
        raise IndicatorError("Close and volume series must be the same length.")
    if not closes:
        return []
    obv = [0.0]
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv.append(obv[-1] + volumes[i])
        elif closes[i] < closes[i - 1]:
            obv.append(obv[-1] - volumes[i])
        else:
            obv.append(obv[-1])
    return obv


def linear_slope(values: Sequence[float]) -> float:
    """Least-squares slope of a series against its index.

    Args:
        values: Input series, oldest first.

    Returns:
        The slope per bar, or ``0.0`` for fewer than two points.
    """
    n = len(values)
    if n < 2:
        return 0.0
    mean_x = (n - 1) / 2.0
    mean_y = fmean(values)
    numerator = sum((i - mean_x) * (values[i] - mean_y) for i in range(n))
    denominator = sum((i - mean_x) ** 2 for i in range(n))
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def normalized_slope(values: Sequence[float], period: int) -> float:
    """Slope of the last ``period`` points, scaled by their mean magnitude.

    Scaling makes the result comparable across instruments of very different
    price levels.

    Args:
        values: Input series, oldest first.
        period: Number of trailing points to fit.

    Returns:
        The scale-free slope, or ``0.0`` when undefined.
    """
    window = list(values[-period:])
    if len(window) < 2:
        return 0.0
    scale = fmean(abs(v) for v in window)
    if scale == 0.0:
        return 0.0
    return linear_slope(window) / scale


def last_defined(series: Series) -> float | None:
    """Return the most recent non-``None`` value of an indicator series."""
    for value in reversed(series):
        if value is not None:
            return value
    return None


def crossed_above(fast: Series, slow: Series) -> bool:
    """Return whether ``fast`` crossed above ``slow`` on the final bar."""
    if len(fast) < 2 or len(slow) < 2:
        return False
    prev_fast, prev_slow = fast[-2], slow[-2]
    curr_fast, curr_slow = fast[-1], slow[-1]
    if None in (prev_fast, prev_slow, curr_fast, curr_slow):
        return False
    assert prev_fast is not None and prev_slow is not None  # noqa: S101 - narrowing for mypy
    assert curr_fast is not None and curr_slow is not None  # noqa: S101 - narrowing for mypy
    return prev_fast <= prev_slow and curr_fast > curr_slow


def crossed_below(fast: Series, slow: Series) -> bool:
    """Return whether ``fast`` crossed below ``slow`` on the final bar."""
    return crossed_above(slow, fast)


def clamp(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    """Clamp ``value`` into the inclusive range ``[lower, upper]``."""
    return max(lower, min(upper, value))


__all__ = [
    "BollingerResult",
    "IndicatorError",
    "MacdResult",
    "Series",
    "atr",
    "bollinger",
    "clamp",
    "crossed_above",
    "crossed_below",
    "ema",
    "last_defined",
    "linear_slope",
    "macd",
    "normalized_slope",
    "on_balance_volume",
    "rsi",
    "sma",
    "true_range",
]
