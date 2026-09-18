"""Base event contract shared by every message on the bus.

Architecture rule (``CLAUDE.md``): **all events are immutable**.  ``BaseEvent``
is declared ``frozen=True`` and every event in the platform inherits from it, so
an event that has been published can never be mutated in place by a downstream
consumer.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1
"""Current wire-format version. Bump when a breaking field change ships."""


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC timestamp."""
    return datetime.now(tz=UTC)


class EventType(StrEnum):
    """Discriminator for every event type carried on the bus."""

    CANDLE = "candle"
    TRADE_TICK = "trade_tick"
    ORDER_BOOK = "order_book"
    FUNDING_RATE = "funding_rate"
    PERP_METRICS = "perp_metrics"
    AGENT_SIGNAL = "agent_signal"
    TRADE_DECISION = "trade_decision"
    RISK_VERDICT = "risk_verdict"
    ORDER_INTENT = "order_intent"
    FILL = "fill"
    PORTFOLIO_SNAPSHOT = "portfolio_snapshot"
    POSITION_CLOSED = "position_closed"
    AGENT_OUTCOME = "agent_outcome"


class BaseEvent(BaseModel):
    """Immutable envelope for every event published to Kafka.

    Attributes:
        event_id: Unique identifier for this event instance.
        event_type: Discriminator identifying the concrete event class.
        schema_version: Wire-format version, for forward compatibility.
        source: Name of the service or agent that produced the event.
        occurred_at: When the underlying fact happened (exchange time when known).
        emitted_at: When this process serialised the event.
        correlation_id: Ties a causal chain together (signal -> decision -> fill).
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        ser_json_timedelta="float",
        populate_by_name=True,
    )

    event_id: UUID = Field(default_factory=uuid4, description="Unique event identifier.")
    event_type: EventType = Field(description="Concrete event discriminator.")
    schema_version: int = Field(default=SCHEMA_VERSION, description="Wire-format version.")
    source: str = Field(description="Producing service or agent name.")
    occurred_at: datetime = Field(
        default_factory=utc_now, description="When the underlying fact happened (UTC)."
    )
    emitted_at: datetime = Field(
        default_factory=utc_now, description="When this process emitted the event (UTC)."
    )
    correlation_id: UUID = Field(
        default_factory=uuid4, description="Correlates a causal chain of events."
    )

    def with_correlation(self, correlation_id: UUID) -> Self:
        """Return a copy of this event carrying a different correlation id.

        Args:
            correlation_id: Correlation id to attach to the copy.

        Returns:
            A new, still-immutable event instance.
        """
        return self.model_copy(update={"correlation_id": correlation_id})

    def to_wire(self) -> dict[str, Any]:
        """Serialise the event to a JSON-compatible dictionary.

        Computed fields are excluded so the result can be validated back
        into the model, which forbids extra keys.
        """
        return self.model_dump(mode="json", exclude=set(type(self).model_computed_fields))


__all__ = ["SCHEMA_VERSION", "BaseEvent", "EventType", "utc_now"]
