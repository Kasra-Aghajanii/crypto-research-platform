"""Decision, risk, order and fill events.

These four events form the execution chain::

    AgentSignal -> TradeDecision -> RiskVerdict -> OrderIntent -> Fill

Each stage carries the previous stage's identifier so the whole chain can be
reconstructed from the log, and every stage shares one ``correlation_id``.
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import Field

from libs.schemas.base import BaseEvent, EventType
from libs.schemas.market import Side
from libs.schemas.signals import Direction


class DecisionAction(StrEnum):
    """Action the decision engine wants the execution layer to take."""

    OPEN_LONG = "open_long"
    OPEN_SHORT = "open_short"
    CLOSE = "close"
    HOLD = "hold"

    @property
    def direction(self) -> Direction:
        """Return the directional exposure implied by this action."""
        if self is DecisionAction.OPEN_LONG:
            return Direction.LONG
        if self is DecisionAction.OPEN_SHORT:
            return Direction.SHORT
        return Direction.FLAT

    @classmethod
    def from_direction(cls, direction: Direction) -> DecisionAction:
        """Map a directional view to the corresponding opening action."""
        if direction is Direction.LONG:
            return cls.OPEN_LONG
        if direction is Direction.SHORT:
            return cls.OPEN_SHORT
        return cls.HOLD


class OrderType(StrEnum):
    """Supported order types. Phase 2 paper trading is market-only."""

    MARKET = "market"
    LIMIT = "limit"


class TradeDecision(BaseEvent):
    """The decision engine's aggregated instruction for one symbol.

    Attributes:
        contributing_signals: Signal ids that produced this decision, so the
            Brier scorer can later attribute the outcome back to each agent.
        weighted_confidence: Confidence after applying agent trust weights.
    """

    event_type: Literal[EventType.TRADE_DECISION] = EventType.TRADE_DECISION

    decision_id: UUID = Field(description="Stable identifier for this decision.")
    symbol: str = Field(description="Hyperliquid perp coin symbol.")
    action: DecisionAction = Field(description="Requested action.")
    confidence: float = Field(ge=0.0, le=1.0, description="Raw aggregated confidence.")
    weighted_confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence after trust weighting."
    )
    reference_price: Decimal | None = Field(default=None, description="Price at decision time.")
    suggested_stop: Decimal | None = Field(default=None, description="Suggested stop price.")
    suggested_take_profit: Decimal | None = Field(
        default=None, description="Suggested take-profit price."
    )
    rationale: str = Field(default="", description="Human-readable explanation.")
    contributing_signals: tuple[UUID, ...] = Field(
        default=(), description="Signal event ids behind this decision."
    )
    agent_weights: dict[str, float] = Field(
        default_factory=dict, description="Trust weight applied per contributing agent."
    )
    mode: Literal["passthrough", "ensemble"] = Field(
        default="passthrough", description="Aggregation mode that produced the decision."
    )


class VetoReason(StrEnum):
    """Enumerated reasons the risk manager rejects a decision."""

    LOW_CONFIDENCE = "low_confidence"
    MAX_POSITION_NOTIONAL = "max_position_notional"
    MAX_OPEN_POSITIONS = "max_open_positions"
    MAX_LEVERAGE = "max_leverage"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    INSUFFICIENT_EQUITY = "insufficient_equity"
    SIZE_BELOW_MINIMUM = "size_below_minimum"
    DUPLICATE_EXPOSURE = "duplicate_exposure"
    STALE_DECISION = "stale_decision"
    NO_MARKET_DATA = "no_market_data"
    KILL_SWITCH = "kill_switch"
    INTERNAL_ERROR = "internal_error"


class RiskVerdict(BaseEvent):
    """The risk manager's approval or veto of a trade decision."""

    event_type: Literal[EventType.RISK_VERDICT] = EventType.RISK_VERDICT

    decision_id: UUID = Field(description="Decision this verdict answers.")
    symbol: str = Field(description="Hyperliquid perp coin symbol.")
    approved: bool = Field(description="Whether the decision may proceed to execution.")
    veto_reasons: tuple[VetoReason, ...] = Field(
        default=(), description="All limits breached, empty when approved."
    )
    approved_size: Decimal = Field(
        default=Decimal(0), ge=0, description="Approved size in base asset."
    )
    approved_notional: Decimal = Field(
        default=Decimal(0), ge=0, description="Approved notional in USD."
    )
    stop_price: Decimal | None = Field(default=None, description="Stop used for sizing.")
    take_profit_price: Decimal | None = Field(default=None, description="Take-profit level.")
    risk_amount: Decimal = Field(
        default=Decimal(0), ge=0, description="USD at risk between entry and stop."
    )
    equity_at_decision: Decimal = Field(
        default=Decimal(0), ge=0, description="Account equity used for sizing."
    )
    rationale: str = Field(default="", description="Human-readable explanation.")

    @property
    def vetoed(self) -> bool:
        """Return ``True`` when the decision was rejected."""
        return not self.approved


class OrderIntent(BaseEvent):
    """A risk-approved instruction for the execution layer to fill."""

    event_type: Literal[EventType.ORDER_INTENT] = EventType.ORDER_INTENT

    order_id: UUID = Field(description="Client order identifier.")
    decision_id: UUID = Field(description="Originating decision.")
    symbol: str = Field(description="Hyperliquid perp coin symbol.")
    side: Side = Field(description="Order side.")
    size: Decimal = Field(gt=0, description="Order size in base asset.")
    order_type: OrderType = Field(default=OrderType.MARKET, description="Order type.")
    limit_price: Decimal | None = Field(default=None, description="Limit price, if applicable.")
    reduce_only: bool = Field(default=False, description="Whether the order may only reduce.")
    stop_price: Decimal | None = Field(default=None, description="Protective stop to attach.")
    take_profit_price: Decimal | None = Field(default=None, description="Take-profit to attach.")
    trailing_stop_pct: float | None = Field(
        default=None,
        gt=0,
        description="Trailing stop distance as a percent of the best price seen since entry.",
    )
    exit_reason: str | None = Field(
        default=None, description="Set on reduce-only exits to record what triggered them."
    )
    is_paper: bool = Field(default=True, description="Whether this order is simulated.")


class Fill(BaseEvent):
    """A completed (simulated or real) execution."""

    event_type: Literal[EventType.FILL] = EventType.FILL

    fill_id: UUID = Field(description="Unique fill identifier.")
    order_id: UUID = Field(description="Order that produced this fill.")
    decision_id: UUID | None = Field(default=None, description="Originating decision, if any.")
    symbol: str = Field(description="Hyperliquid perp coin symbol.")
    side: Side = Field(description="Executed side.")
    size: Decimal = Field(gt=0, description="Filled size in base asset.")
    price: Decimal = Field(gt=0, description="Average fill price including slippage.")
    fee: Decimal = Field(default=Decimal(0), ge=0, description="Fee paid in USD.")
    slippage_bps: float = Field(default=0.0, description="Modelled slippage in basis points.")
    reference_price: Decimal | None = Field(
        default=None, description="Book price before slippage was applied."
    )
    is_paper: bool = Field(default=True, description="Whether this fill was simulated.")
    reduce_only: bool = Field(default=False, description="Whether the fill reduced a position.")

    @property
    def notional(self) -> Decimal:
        """Return the USD notional transacted."""
        return self.size * self.price


__all__ = [
    "DecisionAction",
    "Fill",
    "OrderIntent",
    "OrderType",
    "RiskVerdict",
    "TradeDecision",
    "VetoReason",
]
