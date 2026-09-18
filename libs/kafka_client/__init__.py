"""Kafka (Redpanda) client and the platform's topic registry.

Architecture rule (``CLAUDE.md``): **no agent calls another agent directly** --
all inter-service communication flows through the topics declared below, and
every topic in the platform is listed in this module.

Topic naming: ``<domain>.<entity>``.  Every producer keys messages by symbol so
that per-symbol ordering is preserved within a partition.
"""

from __future__ import annotations

from typing import Final

from libs.kafka_client.consumer import KafkaConsumer
from libs.kafka_client.producer import KafkaProducer
from libs.kafka_client.serialization import EventDecodeError, deserialize, serialize
from libs.schemas.base import BaseEvent
from libs.schemas.learning import AgentOutcome, PositionClosed
from libs.schemas.market import Candle, OrderBookSnapshot, PerpMetrics, TradeTick
from libs.schemas.portfolio import PortfolioSnapshot
from libs.schemas.signals import AgentSignal
from libs.schemas.trading import Fill, OrderIntent, RiskVerdict, TradeDecision


class Topics:
    """Every Kafka topic used by the platform.

    Market data is produced by the ingestion services; the remaining topics form
    the execution chain ``signals -> decisions -> risk -> orders -> fills ->
    portfolio``.
    """

    MARKET_CANDLES: Final[str] = "market.candles"
    """Closed and in-progress OHLCV candles from Hyperliquid."""

    MARKET_TRADES: Final[str] = "market.trades"
    """Individual trade prints from the Hyperliquid tape."""

    MARKET_ORDERBOOK: Final[str] = "market.orderbook"
    """Depth-limited order book snapshots."""

    AGENT_SIGNALS: Final[str] = "agents.signals"
    """AgentSignal events from every analysis agent."""

    TRADE_DECISIONS: Final[str] = "decisions.trades"
    """TradeDecision events from the decision engine."""

    RISK_VERDICTS: Final[str] = "risk.verdicts"
    """RiskVerdict approvals and vetoes from the risk manager."""

    ORDER_INTENTS: Final[str] = "execution.orders"
    """Risk-approved OrderIntent events for the execution layer."""

    FILLS: Final[str] = "execution.fills"
    """Fill events from the paper broker (or, later, the live broker)."""

    PORTFOLIO_SNAPSHOTS: Final[str] = "portfolio.snapshots"
    """PortfolioSnapshot events from the portfolio manager."""

    POSITION_CLOSURES: Final[str] = "portfolio.closures"
    """PositionClosed events, the input to outcome attribution."""

    MARKET_PERP_METRICS: Final[str] = "market.perp_metrics"
    """Open interest, mark/oracle price, premium and funding snapshots."""

    AGENT_OUTCOMES: Final[str] = "agents.outcomes"
    """AgentOutcome events carrying Brier scores back to the decision engine."""


TOPIC_MODELS: Final[dict[str, type[BaseEvent]]] = {
    Topics.MARKET_CANDLES: Candle,
    Topics.MARKET_TRADES: TradeTick,
    Topics.MARKET_ORDERBOOK: OrderBookSnapshot,
    Topics.AGENT_SIGNALS: AgentSignal,
    Topics.TRADE_DECISIONS: TradeDecision,
    Topics.RISK_VERDICTS: RiskVerdict,
    Topics.ORDER_INTENTS: OrderIntent,
    Topics.FILLS: Fill,
    Topics.PORTFOLIO_SNAPSHOTS: PortfolioSnapshot,
    Topics.POSITION_CLOSURES: PositionClosed,
    Topics.MARKET_PERP_METRICS: PerpMetrics,
    Topics.AGENT_OUTCOMES: AgentOutcome,
}
"""Canonical mapping of topic name to the event class it carries."""

ALL_TOPICS: Final[tuple[str, ...]] = tuple(TOPIC_MODELS)
"""Every topic name, for provisioning and administration."""

__all__ = [
    "ALL_TOPICS",
    "TOPIC_MODELS",
    "EventDecodeError",
    "KafkaConsumer",
    "KafkaProducer",
    "Topics",
    "deserialize",
    "serialize",
]
