"""Shared Pydantic event models.

Architecture rule (``CLAUDE.md``): every service imports its models from here
and never redefines them locally.  All models inherit the immutable
:class:`~libs.schemas.base.BaseEvent` envelope.
"""

from __future__ import annotations

from libs.schemas.base import SCHEMA_VERSION, BaseEvent, EventType, utc_now
from libs.schemas.learning import AgentOutcome, ExitReason, PositionClosed
from libs.schemas.market import (
    BookLevel,
    Candle,
    FundingRate,
    OrderBookSnapshot,
    PerpMetrics,
    Side,
    TradeTick,
)
from libs.schemas.portfolio import PortfolioSnapshot, Position
from libs.schemas.signals import (
    AgentSignal,
    Direction,
    SupportResistanceLevel,
    TimeframeAnalysis,
)
from libs.schemas.trading import (
    DecisionAction,
    Fill,
    OrderIntent,
    OrderType,
    RiskVerdict,
    TradeDecision,
    VetoReason,
)

__all__ = [
    "SCHEMA_VERSION",
    "AgentOutcome",
    "AgentSignal",
    "BaseEvent",
    "BookLevel",
    "Candle",
    "DecisionAction",
    "Direction",
    "EventType",
    "ExitReason",
    "Fill",
    "FundingRate",
    "OrderBookSnapshot",
    "OrderIntent",
    "OrderType",
    "PerpMetrics",
    "PortfolioSnapshot",
    "Position",
    "PositionClosed",
    "RiskVerdict",
    "Side",
    "SupportResistanceLevel",
    "TimeframeAnalysis",
    "TradeDecision",
    "TradeTick",
    "VetoReason",
    "utc_now",
]
