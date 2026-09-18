"""Market Analyst Agent.

Phase 2, item 3.  The first full analysis agent: it turns pre-loaded market
context into a single ``AgentSignal``.

Method
------
For each configured timeframe the agent computes four component scores, each
normalised to ``[-1, 1]``:

``trend``
    EMA(fast) vs EMA(slow) separation, price position relative to both, and the
    sign of the MACD histogram.
``momentum``
    RSI distance from the neutral 50 line, the MACD histogram *level*, and fresh
    MACD crossovers.
``volatility``
    Bollinger %B mean-reversion pressure, scaled down as the trend strengthens
    (band position only mean-reverts in a range) and replaced by trend-following
    during a bandwidth squeeze.
``volume``
    Relative volume versus its moving average and the slope of on-balance
    volume, used only to *confirm* the other components -- never to originate a
    view on its own.

The four components are blended into a per-timeframe composite, then the
timeframes are combined using the configured weights (higher timeframes dominate)
to produce one confluence score.  That score is mapped onto a confidence in
``[0, 1]`` against ``full_conviction_score`` and scaled by timeframe agreement:
when the 1h and 15m disagree, confidence falls even if the raw score is large.

Support/resistance levels are detected on the slowest timeframe and used to
place the suggested stop and take-profit, with an ATR-based fallback.

Architecture rules honoured here
-------------------------------
* The agent never fetches: it reads only the ``AgentContext`` handed to it.
* It publishes exactly one ``AgentSignal`` per trigger, and the ``SignalAgent``
  base class publishes a neutral ``confidence == 0`` signal if this code raises.
* It talks to no other agent -- its only output is a Kafka event.

Run with::

    python -m services.agents.market_analyst.agent
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from typing import Final

from libs.config import MarketAnalystSettings, Settings, settings
from libs.logging_config import configure_logging
from libs.schemas.base import utc_now
from libs.schemas.market import Candle
from libs.schemas.signals import AgentSignal, Direction, SupportResistanceLevel, TimeframeAnalysis
from services.agents.common.base_agent import SignalAgent
from services.agents.common.context import AgentContext
from services.agents.market_analyst import indicators as ind
from services.agents.market_analyst.levels import detect_levels, nearest_level

logger = logging.getLogger(__name__)

SERVICE_NAME: Final[str] = "market_analyst"

COMPONENT_WEIGHTS: Final[dict[str, float]] = {
    "trend": 0.40,
    "momentum": 0.30,
    "volatility": 0.20,
    "volume": 0.10,
}
"""Blend of the four component scores into a per-timeframe composite."""


@dataclass(frozen=True, slots=True)
class OHLCV:
    """Column-oriented view of a candle series, ready for indicator maths."""

    opens: list[float]
    highs: list[float]
    lows: list[float]
    closes: list[float]
    volumes: list[float]

    @classmethod
    def from_candles(cls, candles: tuple[Candle, ...]) -> OHLCV:
        """Build a column view from an oldest-first candle series."""
        return cls(
            opens=[float(candle.open) for candle in candles],
            highs=[float(candle.high) for candle in candles],
            lows=[float(candle.low) for candle in candles],
            closes=[float(candle.close) for candle in candles],
            volumes=[float(candle.volume) for candle in candles],
        )

    def __len__(self) -> int:
        """Return the number of bars."""
        return len(self.closes)


class MarketAnalystAgent(SignalAgent):
    """Multi-timeframe technical analysis agent.

    Attributes:
        name: Agent identifier used for the consumer group, signals and logs.
        version: Implementation version, recorded on every signal.
    """

    name = SERVICE_NAME
    version = "1.0.0"

    consume_order_book = True
    consume_trades = False
    consume_portfolio = False

    def __init__(self, *, config: Settings | None = None) -> None:
        """Initialise the agent from configuration.

        Args:
            config: Settings override, mainly for tests.
        """
        resolved = config or settings
        self.params: MarketAnalystSettings = resolved.analyst
        self.warmup_candles = resolved.analyst.warmup_candles
        self.signal_ttl_s = resolved.analyst.signal_ttl_s
        self.trigger_intervals = tuple(resolved.hyperliquid.candle_intervals)
        super().__init__(config=resolved)
        self._min_bars = self._required_bars()

    def _required_bars(self) -> int:
        """Return the minimum bars needed before any timeframe can be scored."""
        params = self.params
        return (
            max(
                params.rsi_period + 1,
                params.macd_slow + params.macd_signal,
                params.ema_slow,
                params.bollinger_period,
                params.atr_period + 1,
                params.volume_ma_period,
            )
            + 2
        )

    # ------------------------------------------------------------------
    # Component scores
    # ------------------------------------------------------------------

    def _trend_score(
        self, closes: list[float], ema_fast: ind.Series, ema_slow: ind.Series, hist: ind.Series
    ) -> float:
        """Score trend direction from EMA structure and MACD histogram sign.

        Args:
            closes: Close prices, oldest first.
            ema_fast: Fast EMA series.
            ema_slow: Slow EMA series.
            hist: MACD histogram series.

        Returns:
            A score in ``[-1, 1]``; positive means uptrend.
        """
        fast = ind.last_defined(ema_fast)
        slow = ind.last_defined(ema_slow)
        if fast is None or slow is None or slow == 0:
            return 0.0

        # EMA separation, normalised: 1% separation is treated as a full signal.
        separation = ind.clamp((fast - slow) / slow * 100.0)
        price = closes[-1]
        position = 0.0
        if price > fast > slow:
            position = 1.0
        elif price < fast < slow:
            position = -1.0
        elif price > slow:
            position = 0.35
        elif price < slow:
            position = -0.35

        histogram = ind.last_defined(hist)
        histogram_sign = 0.0
        if histogram is not None and price != 0:
            histogram_sign = ind.clamp(histogram / price * 200.0)

        return ind.clamp(0.45 * separation + 0.35 * position + 0.20 * histogram_sign)

    def _momentum_score(
        self, rsi_series: ind.Series, macd_result: ind.MacdResult, price: float
    ) -> float:
        """Score momentum from RSI positioning and MACD behaviour.

        RSI is treated as a *momentum* reading rather than a contrarian one:
        readings above 50 add to a long score.  Genuine exhaustion (beyond the
        configured overbought/oversold levels) is damped slightly rather than
        halved -- halving it let a decelerating downtrend read as positive
        momentum, because the MACD terms outweighed a heavily damped RSI.

        The MACD contribution is the histogram *level* (is momentum positive or
        negative right now), not its slope.  Slope alone is misleading: in a
        steady decline the histogram rises toward zero, which is decelerating
        downside, not upside.

        Args:
            rsi_series: RSI series.
            macd_result: MACD line, signal and histogram.
            price: Latest close, used to make the histogram scale-free.

        Returns:
            A score in ``[-1, 1]``.
        """
        params = self.params
        score = 0.0

        rsi_value = ind.last_defined(rsi_series)
        if rsi_value is not None:
            centred = (rsi_value - 50.0) / 50.0
            if rsi_value >= params.rsi_overbought or rsi_value <= params.rsi_oversold:
                centred *= 0.8  # stretched, but still directional
            score += 0.45 * ind.clamp(centred)

        histogram = ind.last_defined(macd_result.histogram)
        if histogram is not None and price > 0:
            score += 0.35 * ind.clamp(histogram / price * 200.0)

        if ind.crossed_above(macd_result.macd, macd_result.signal):
            score += 0.2
        elif ind.crossed_below(macd_result.macd, macd_result.signal):
            score -= 0.2

        return ind.clamp(score)

    def _volatility_score(self, bands: ind.BollingerResult, trend_score: float) -> float:
        """Score Bollinger positioning, gated by whether the market is trending.

        Band position is only a mean-reversion signal in a **range**.  In a
        trend, price rides the upper (or lower) band for long stretches, so
        reading %B as reversion pressure would fight the trend -- and does so
        hard enough to cancel it: an unmoderated reversion term dragged a strong
        uptrend's composite down to roughly the same score as a flat market.

        The reversion term is therefore scaled by ``1 - |trend|``: dominant when
        the market is directionless, near zero when a trend is established.  A
        volatility squeeze is handled separately, since expansion out of a
        squeeze continues the trend rather than fading it.

        Args:
            bands: Bollinger band series.
            trend_score: Trend score for the same timeframe.

        Returns:
            A score in ``[-1, 1]``; positive favours the long side.
        """
        percent_b = ind.last_defined(bands.percent_b)
        if percent_b is None:
            return 0.0

        bandwidth_values = [value for value in bands.bandwidth if value is not None]
        if len(bandwidth_values) >= 20:
            recent = bandwidth_values[-1]
            typical = sorted(bandwidth_values[-20:])[10]
            if typical > 0 and recent < typical * 0.75:
                # Squeeze: expansion tends to continue the trend, so follow it instead.
                return ind.clamp(0.5 * trend_score)

        # Below the lower band -> stretched down -> mild long pressure, and vice versa.
        reversion = ind.clamp((0.5 - percent_b) * 2.0)
        range_weight = 1.0 - min(1.0, abs(trend_score))
        return ind.clamp(reversion * range_weight)

    def _volume_score(self, ohlcv: OHLCV, direction_hint: float) -> tuple[float, float, float]:
        """Score volume confirmation of the prevailing direction.

        Args:
            ohlcv: Column view of the candle series.
            direction_hint: Directional score volume should confirm.

        Returns:
            A tuple of ``(score, volume_ratio, obv_slope)``.
        """
        params = self.params
        volume_ma = ind.sma(ohlcv.volumes, params.volume_ma_period)
        average = ind.last_defined(volume_ma)
        ratio = 0.0
        if average is not None and average > 0:
            ratio = ohlcv.volumes[-1] / average

        obv = ind.on_balance_volume(ohlcv.closes, ohlcv.volumes)
        obv_slope = ind.normalized_slope(obv, min(params.volume_ma_period, len(obv)))

        # Volume only confirms: it scales an existing view rather than creating one.
        confirmation = 0.0
        if ratio > 0:
            confirmation = ind.clamp(ratio - 1.0) * (1.0 if direction_hint >= 0 else -1.0)
        score = ind.clamp(0.5 * confirmation + 0.5 * ind.clamp(obv_slope * 5.0))
        return score, ratio, obv_slope

    # ------------------------------------------------------------------
    # Timeframe and confluence analysis
    # ------------------------------------------------------------------

    def analyze_timeframe(self, interval: str, candles: tuple[Candle, ...]) -> TimeframeAnalysis:
        """Compute every indicator and component score for one timeframe.

        Args:
            interval: Timeframe identifier.
            candles: Closed candles, oldest first.

        Returns:
            The per-timeframe analysis, with a zero composite when the series is
            still too short to score.
        """
        params = self.params
        ohlcv = OHLCV.from_candles(candles)
        if len(ohlcv) < self._min_bars:
            return TimeframeAnalysis(
                source=self.name,
                interval=interval,
                candles_used=len(ohlcv),
                close=ohlcv.closes[-1] if ohlcv.closes else 0.0,
            )

        closes = ohlcv.closes
        rsi_series = ind.rsi(closes, params.rsi_period)
        macd_result = ind.macd(closes, params.macd_fast, params.macd_slow, params.macd_signal)
        ema_fast = ind.ema(closes, params.ema_fast)
        ema_slow = ind.ema(closes, params.ema_slow)
        bands = ind.bollinger(closes, params.bollinger_period, params.bollinger_std)
        atr_series = ind.atr(ohlcv.highs, ohlcv.lows, closes, params.atr_period)

        trend = self._trend_score(closes, ema_fast, ema_slow, macd_result.histogram)
        momentum = self._momentum_score(rsi_series, macd_result, closes[-1])
        volatility = self._volatility_score(bands, trend)
        volume, volume_ratio, obv_slope = self._volume_score(ohlcv, trend)

        composite = ind.clamp(
            COMPONENT_WEIGHTS["trend"] * trend
            + COMPONENT_WEIGHTS["momentum"] * momentum
            + COMPONENT_WEIGHTS["volatility"] * volatility
            + COMPONENT_WEIGHTS["volume"] * volume
        )

        return TimeframeAnalysis(
            source=self.name,
            interval=interval,
            candles_used=len(ohlcv),
            close=closes[-1],
            rsi=ind.last_defined(rsi_series),
            macd=ind.last_defined(macd_result.macd),
            macd_signal=ind.last_defined(macd_result.signal),
            macd_histogram=ind.last_defined(macd_result.histogram),
            ema_fast=ind.last_defined(ema_fast),
            ema_slow=ind.last_defined(ema_slow),
            bb_upper=ind.last_defined(bands.upper),
            bb_middle=ind.last_defined(bands.middle),
            bb_lower=ind.last_defined(bands.lower),
            bb_percent_b=ind.last_defined(bands.percent_b),
            bb_bandwidth=ind.last_defined(bands.bandwidth),
            atr=ind.last_defined(atr_series),
            volume_ratio=volume_ratio,
            obv_slope=obv_slope,
            trend_score=round(trend, 4),
            momentum_score=round(momentum, 4),
            volatility_score=round(volatility, 4),
            volume_score=round(volume, 4),
            composite_score=round(composite, 4),
        )

    def _confluence(self, analyses: list[TimeframeAnalysis]) -> tuple[float, float]:
        """Combine timeframe composites into one score plus an agreement factor.

        Args:
            analyses: Per-timeframe analyses that produced a usable score.

        Returns:
            A tuple of ``(weighted_score, agreement)``, both in ``[-1, 1]`` and
            ``[0, 1]`` respectively.
        """
        scored = [item for item in analyses if item.candles_used >= self._min_bars]
        if not scored:
            return 0.0, 0.0

        weights = self.params.timeframe_weights
        total_weight = sum(weights.get(item.interval, 0.1) for item in scored)
        if total_weight <= 0:
            return 0.0, 0.0

        weighted = (
            sum(weights.get(item.interval, 0.1) * item.composite_score for item in scored)
            / total_weight
        )

        directional = [item for item in scored if abs(item.composite_score) >= 0.05]
        if not directional:
            agreement = 0.0
        else:
            positives = sum(1 for item in directional if item.composite_score > 0)
            share = max(positives, len(directional) - positives) / len(directional)
            # Rescale 0.5..1.0 (coin-flip..unanimous) onto 0.0..1.0.
            agreement = (share - 0.5) * 2.0
        return ind.clamp(weighted), max(0.0, min(1.0, agreement))

    def _confidence(self, score: float, agreement: float) -> float:
        """Map a confluence score and timeframe agreement onto ``[0, 1]``.

        A raw composite score is a *blend* of four bounded components, so it
        saturates well below 1.0 even when every component agrees -- a strong,
        unanimous trend lands near 0.5, not 0.9.  Treating that raw value as a
        probability left every signal below the decision engine's confidence
        floor, so the platform would never have traded.

        ``full_conviction_score`` names the score at which the agent is fully
        convinced; confidence rises linearly to 1.0 there and saturates.  It is
        then scaled by timeframe agreement, so a view the timeframes disagree
        about can never reach full confidence.

        Args:
            score: Weighted confluence score in ``[-1, 1]``.
            agreement: Timeframe agreement in ``[0, 1]``.

        Returns:
            Calibrated confidence in ``[0, 1]``.
        """
        conviction = min(1.0, abs(score) / self.params.full_conviction_score)
        return round(conviction * (0.6 + 0.4 * agreement), 4)

    def _protective_levels(
        self,
        direction: Direction,
        price: float,
        levels: tuple[SupportResistanceLevel, ...],
        atr_value: float | None,
    ) -> tuple[Decimal | None, Decimal | None]:
        """Derive stop and take-profit prices for a directional view.

        Prefers the nearest structural level; falls back to a 1.5x ATR stop, and
        finally to the configured default stop distance.

        Args:
            direction: The agent's directional view.
            price: Reference price.
            levels: Detected support/resistance levels.
            atr_value: Latest ATR on the anchor timeframe, if available.

        Returns:
            A ``(stop, take_profit)`` tuple; both ``None`` when flat.
        """
        if direction is Direction.FLAT or price <= 0:
            return None, None

        buffer = (atr_value * 1.5) if atr_value else price * 0.015
        support = nearest_level(levels, "support")
        resistance = nearest_level(levels, "resistance")

        if direction is Direction.LONG:
            stop = float(support.price) * 0.999 if support else price - buffer
            target = float(resistance.price) * 0.999 if resistance else price + 2.0 * buffer
            if stop >= price:
                stop = price - buffer
            if target <= price:
                target = price + 2.0 * buffer
        else:
            stop = float(resistance.price) * 1.001 if resistance else price + buffer
            target = float(support.price) * 1.001 if support else price - 2.0 * buffer
            if stop <= price:
                stop = price + buffer
            if target >= price:
                target = price - 2.0 * buffer

        if target <= 0:
            return Decimal(str(round(stop, 8))), None
        return Decimal(str(round(stop, 8))), Decimal(str(round(target, 8)))

    async def analyze(self, context: AgentContext) -> AgentSignal | None:
        """Produce one ``AgentSignal`` from the pre-loaded context.

        Args:
            context: Read-only market state assembled by the runtime.

        Returns:
            The signal to publish, or ``None`` while still warming up.
        """
        analyses = [
            self.analyze_timeframe(interval, context.candles_for(interval))
            for interval in context.intervals
            if context.candles_for(interval)
        ]
        usable = [item for item in analyses if item.candles_used >= self._min_bars]
        if not usable:
            logger.debug(
                "Warming up; no timeframe has enough history",
                extra={"symbol": context.symbol, "required_bars": self._min_bars},
            )
            return None

        score, agreement = self._confluence(analyses)
        confidence = self._confidence(score, agreement)
        direction = Direction.from_score(score, self.params.min_abs_score)
        if direction is Direction.FLAT:
            confidence = min(confidence, self.params.min_abs_score)

        anchor = max(usable, key=lambda item: self.params.timeframe_weights.get(item.interval, 0.1))
        reference_price = context.reference_price or Decimal(str(anchor.close))
        price = float(reference_price)

        anchor_candles = context.candles_for(anchor.interval)
        levels = detect_levels(
            [float(candle.high) for candle in anchor_candles],
            [float(candle.low) for candle in anchor_candles],
            price,
            lookback=self.params.swing_lookback,
            cluster_pct=self.params.level_cluster_pct,
            max_per_side=self.params.support_resistance_levels,
            source=self.name,
        )
        stop, take_profit = self._protective_levels(direction, price, levels, anchor.atr)

        features: dict[str, float] = {
            "confluence_score": round(score, 4),
            "timeframe_agreement": round(agreement, 4),
            "timeframes_scored": float(len(usable)),
        }
        for item in usable:
            features[f"{item.interval}_composite"] = item.composite_score
            features[f"{item.interval}_trend"] = item.trend_score
            features[f"{item.interval}_momentum"] = item.momentum_score
            if item.rsi is not None:
                features[f"{item.interval}_rsi"] = round(item.rsi, 2)
        if context.order_book is not None:
            features["book_imbalance"] = round(context.order_book.imbalance(), 4)
            spread = context.order_book.spread_bps
            if spread is not None:
                features["spread_bps"] = round(float(spread), 4)

        return AgentSignal(
            source=self.name,
            agent_name=self.name,
            agent_version=self.version,
            symbol=context.symbol,
            direction=direction,
            confidence=confidence,
            rationale=self._rationale(direction, score, agreement, usable, anchor),
            reference_price=reference_price,
            suggested_stop=stop,
            suggested_take_profit=take_profit,
            valid_until=utc_now() + timedelta(seconds=self.signal_ttl_s),
            timeframes=tuple(analyses),
            levels=levels,
            features=features,
        )

    def _rationale(
        self,
        direction: Direction,
        score: float,
        agreement: float,
        usable: list[TimeframeAnalysis],
        anchor: TimeframeAnalysis,
    ) -> str:
        """Build a short human-readable explanation of the view.

        Args:
            direction: The resulting direction.
            score: Weighted confluence score.
            agreement: Timeframe agreement factor.
            usable: Timeframes that contributed a score.
            anchor: The highest-weighted timeframe.

        Returns:
            A one-line rationale.
        """
        per_tf = ", ".join(f"{item.interval}={item.composite_score:+.2f}" for item in usable)
        rsi_text = f"{anchor.rsi:.1f}" if anchor.rsi is not None else "n/a"
        return (
            f"{direction.value.upper()} on confluence {score:+.3f} "
            f"(agreement {agreement:.2f}) across {per_tf}; "
            f"anchor {anchor.interval} RSI {rsi_text}, "
            f"trend {anchor.trend_score:+.2f}, momentum {anchor.momentum_score:+.2f}."
        )


async def main() -> None:
    """Service entrypoint for ``python -m services.agents.market_analyst.agent``."""
    configure_logging(SERVICE_NAME)
    await MarketAnalystAgent.main()


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    asyncio.run(main())


__all__ = ["COMPONENT_WEIGHTS", "OHLCV", "MarketAnalystAgent"]
