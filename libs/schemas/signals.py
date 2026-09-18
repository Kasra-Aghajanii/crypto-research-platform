"""Agent signal contract.

Architecture rule (``CLAUDE.md``): **every agent publishes an ``AgentSignal``**,
and on error it publishes a neutral signal with ``confidence == 0`` rather than
staying silent.  Silence is indistinguishable from "no opinion" downstream, so
the neutral signal keeps the decision engine's view of agent liveness honest and
keeps Brier scoring well defined.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import Field

from libs.schemas.base import BaseEvent, EventType, utc_now


class Direction(StrEnum):
    """Directional view expressed by an agent."""

    LONG = "long"
    SHORT = "short"
    FLAT = "flat"

    @property
    def sign(self) -> int:
        """Return ``+1`` for long, ``-1`` for short and ``0`` for flat."""
        if self is Direction.LONG:
            return 1
        if self is Direction.SHORT:
            return -1
        return 0

    @classmethod
    def from_score(cls, score: float, threshold: float) -> Direction:
        """Map a signed score to a direction using a symmetric threshold.

        Args:
            score: Signed score, conventionally in ``[-1, 1]``.
            threshold: Absolute score required to take a directional view.

        Returns:
            The corresponding :class:`Direction`.
        """
        if score >= threshold:
            return cls.LONG
        if score <= -threshold:
            return cls.SHORT
        return cls.FLAT


class SupportResistanceLevel(BaseEvent):
    """A detected horizontal support or resistance level."""

    event_type: Literal[EventType.AGENT_SIGNAL] = EventType.AGENT_SIGNAL

    price: Decimal = Field(gt=0, description="Level price.")
    kind: Literal["support", "resistance"] = Field(description="Which side of price the level is.")
    touches: int = Field(ge=1, description="Number of pivots clustered into this level.")
    strength: float = Field(ge=0.0, le=1.0, description="Normalised level strength.")
    distance_pct: float = Field(description="Signed distance from current price, in percent.")


class TimeframeAnalysis(BaseEvent):
    """Indicator readings and the derived score for a single timeframe."""

    event_type: Literal[EventType.AGENT_SIGNAL] = EventType.AGENT_SIGNAL

    interval: str = Field(description="Timeframe identifier, e.g. 15m.")
    candles_used: int = Field(ge=0, description="Number of closed candles analysed.")
    close: float = Field(description="Latest close price on this timeframe.")
    rsi: float | None = Field(default=None, description="Latest RSI value.")
    macd: float | None = Field(default=None, description="MACD line.")
    macd_signal: float | None = Field(default=None, description="MACD signal line.")
    macd_histogram: float | None = Field(default=None, description="MACD histogram.")
    ema_fast: float | None = Field(default=None, description="Fast EMA.")
    ema_slow: float | None = Field(default=None, description="Slow EMA.")
    bb_upper: float | None = Field(default=None, description="Upper Bollinger band.")
    bb_middle: float | None = Field(default=None, description="Middle Bollinger band (SMA).")
    bb_lower: float | None = Field(default=None, description="Lower Bollinger band.")
    bb_percent_b: float | None = Field(default=None, description="Position within the bands.")
    bb_bandwidth: float | None = Field(default=None, description="Band width relative to middle.")
    atr: float | None = Field(default=None, description="Average True Range.")
    volume_ratio: float | None = Field(
        default=None, description="Latest volume divided by its moving average."
    )
    obv_slope: float | None = Field(default=None, description="Normalised on-balance-volume slope.")
    trend_score: float = Field(default=0.0, description="EMA/MACD trend component in [-1, 1].")
    momentum_score: float = Field(default=0.0, description="RSI/MACD momentum component.")
    volatility_score: float = Field(default=0.0, description="Bollinger-derived component.")
    volume_score: float = Field(default=0.0, description="Volume-confirmation component.")
    composite_score: float = Field(default=0.0, description="Weighted score for this timeframe.")


class AgentSignal(BaseEvent):
    """The single output contract every analysis agent publishes.

    Attributes:
        agent_name: Stable agent identifier, e.g. ``"market_analyst"``.
        agent_version: Semantic version of the agent implementation.
        direction: Directional view.
        confidence: Calibrated confidence in ``[0, 1]``. ``0`` means "no opinion",
            which is also what an agent emits when it fails.
        degraded: ``True`` when the signal was produced by the error path.
        features: Flat, JSON-serialisable feature map used for later scoring.
    """

    event_type: Literal[EventType.AGENT_SIGNAL] = EventType.AGENT_SIGNAL

    agent_name: str = Field(description="Stable agent identifier.")
    agent_version: str = Field(default="0.1.0", description="Agent implementation version.")
    symbol: str = Field(description="Hyperliquid perp coin symbol.")
    direction: Direction = Field(default=Direction.FLAT, description="Directional view.")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence in [0, 1].")
    rationale: str = Field(default="", description="Human-readable explanation.")
    reference_price: Decimal | None = Field(
        default=None, description="Price the view was formed at."
    )
    suggested_stop: Decimal | None = Field(default=None, description="Suggested stop price.")
    suggested_take_profit: Decimal | None = Field(
        default=None, description="Suggested take-profit price."
    )
    valid_until: datetime | None = Field(
        default=None, description="Instant after which the signal is stale."
    )
    timeframes: tuple[TimeframeAnalysis, ...] = Field(
        default=(), description="Per-timeframe analysis backing the signal."
    )
    levels: tuple[SupportResistanceLevel, ...] = Field(
        default=(), description="Detected support and resistance levels."
    )
    features: dict[str, float] = Field(
        default_factory=dict, description="Flat feature map for scoring and audit."
    )
    degraded: bool = Field(default=False, description="Produced by the agent error path.")
    error: str | None = Field(default=None, description="Error detail when degraded.")

    @property
    def signed_confidence(self) -> float:
        """Return confidence signed by direction: positive long, negative short."""
        return self.confidence * float(self.direction.sign)

    def is_expired(self, now: datetime | None = None) -> bool:
        """Return whether the signal has passed its validity window.

        Args:
            now: Reference instant; defaults to the current UTC time.
        """
        if self.valid_until is None:
            return False
        return (now or utc_now()) > self.valid_until

    @classmethod
    def neutral(
        cls,
        *,
        agent_name: str,
        agent_version: str,
        symbol: str,
        source: str,
        correlation_id: UUID | None = None,
        error: str | None = None,
        ttl_s: float | None = None,
    ) -> AgentSignal:
        """Build the mandatory neutral signal used on the agent error path.

        Args:
            agent_name: Stable agent identifier.
            agent_version: Agent implementation version.
            symbol: Symbol the agent was analysing.
            source: Producing service name.
            correlation_id: Correlation id of the triggering event, if known.
            error: Error detail to attach for observability.
            ttl_s: Optional validity window in seconds.

        Returns:
            A flat signal with ``confidence == 0`` and ``degraded == True``.
        """
        now = utc_now()
        payload: dict[str, Any] = {
            "agent_name": agent_name,
            "agent_version": agent_version,
            "symbol": symbol,
            "source": source,
            "direction": Direction.FLAT,
            "confidence": 0.0,
            "rationale": "Neutral signal emitted on the agent error path.",
            "degraded": True,
            "error": error,
            "occurred_at": now,
            "emitted_at": now,
            "valid_until": now + timedelta(seconds=ttl_s) if ttl_s else None,
        }
        if correlation_id is not None:
            payload["correlation_id"] = correlation_id
        return cls(**payload)


__all__ = ["AgentSignal", "Direction", "SupportResistanceLevel", "TimeframeAnalysis"]
