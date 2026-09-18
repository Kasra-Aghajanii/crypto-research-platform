"""Position lifecycle and agent scoring events.

These two events close the feedback loop that Phase 2 left open:

``PositionClosed``
    Published by the portfolio manager the moment a position goes flat.  It
    carries the realised outcome *and* the ``opening_correlation_id`` -- the
    correlation id of the signal chain that opened the position -- which is what
    makes attribution possible at all.

``AgentOutcome``
    Published by the outcome recorder once a closure has been attributed back to
    the signals responsible.  The decision engine consumes it and feeds the
    Brier score into its :class:`~services.decision_engine.trust.TrustRegistry`,
    so trust weights adapt while the platform runs.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import Field

from libs.schemas.base import BaseEvent, EventType
from libs.schemas.signals import Direction


class ExitReason(StrEnum):
    """Why a position was closed."""

    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    SIGNAL_REVERSAL = "signal_reversal"
    MANUAL = "manual"
    UNKNOWN = "unknown"


class PositionClosed(BaseEvent):
    """A position that has just gone flat, with its realised outcome.

    Attributes:
        opening_correlation_id: Correlation id of the chain that opened the
            position. Attribution joins on this.
        exit_reason: What triggered the close.
        realized_pnl: PnL on the closed size, gross of the fees reported
            separately in ``fees_paid``.
    """

    event_type: Literal[EventType.POSITION_CLOSED] = EventType.POSITION_CLOSED

    symbol: str = Field(description="Hyperliquid perp coin symbol.")
    direction: Direction = Field(description="Direction of the closed position.")
    size: Decimal = Field(gt=0, description="Size closed, in base asset.")
    entry_price: Decimal = Field(gt=0, description="Volume-weighted entry price.")
    exit_price: Decimal = Field(gt=0, description="Price the position closed at.")
    realized_pnl: Decimal = Field(description="Realised PnL on the closed size.")
    fees_paid: Decimal = Field(default=Decimal(0), ge=0, description="Fees attributed in USD.")
    exit_reason: ExitReason = Field(default=ExitReason.UNKNOWN, description="Close trigger.")
    opened_at: datetime = Field(description="When the position was opened.")
    closed_at: datetime = Field(description="When the position went flat.")
    opening_correlation_id: UUID | None = Field(
        default=None, description="Correlation id of the opening signal chain."
    )
    opening_decision_id: UUID | None = Field(
        default=None, description="Decision that opened the position."
    )
    is_paper: bool = Field(default=True, description="Whether this was a simulated position.")

    @property
    def net_pnl(self) -> Decimal:
        """Return realised PnL after the attributed fees."""
        return self.realized_pnl - self.fees_paid

    @property
    def return_pct(self) -> float:
        """Return net PnL as a percentage of the entry notional."""
        cost = self.entry_price * self.size
        if cost == 0:
            return 0.0
        return float(self.net_pnl / cost) * 100.0

    @property
    def holding_period_s(self) -> float:
        """Return how long the position was open, in seconds."""
        return max(0.0, (self.closed_at - self.opened_at).total_seconds())

    @property
    def was_profitable(self) -> bool:
        """Return whether the trade made money after fees."""
        return self.net_pnl > 0


class AgentOutcome(BaseEvent):
    """One agent's scored forecast, resolved by a closed position.

    Attributes:
        brier_score: ``(confidence - outcome)^2``; 0 is perfect, 0.25 is the
            score of a permanent coin-flip, 1.0 is maximally wrong.
        was_correct: Whether the agent's directional call made money.
    """

    event_type: Literal[EventType.AGENT_OUTCOME] = EventType.AGENT_OUTCOME

    agent_name: str = Field(description="Agent being scored.")
    agent_version: str = Field(default="0.0.0", description="Agent version that forecast.")
    signal_id: UUID = Field(description="The signal event being scored.")
    symbol: str = Field(description="Hyperliquid perp coin symbol.")
    direction: Direction = Field(description="Direction the agent called.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence the agent published.")
    realized_pnl: Decimal = Field(description="Net PnL of the resolving trade.")
    was_correct: bool = Field(description="Whether the directional call was profitable.")
    brier_score: float = Field(ge=0.0, le=1.0, description="Brier score of this forecast.")
    exit_reason: ExitReason = Field(default=ExitReason.UNKNOWN, description="How it resolved.")
    closed_at: datetime = Field(description="When the resolving position closed.")
    holding_period_s: float = Field(ge=0.0, description="How long the trade was open.")


__all__ = ["AgentOutcome", "ExitReason", "PositionClosed"]
