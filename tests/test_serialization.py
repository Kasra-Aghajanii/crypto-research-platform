"""Wire-format round-trip tests for every event on the bus.

Regression coverage for a bug that would have broken the platform in production:
``model_dump_json`` includes computed fields, and every event model sets
``extra="forbid"``, so any event with a computed property serialised fine and
then failed to decode on the way back in.  Consumers would have logged and
skipped every candle, fill, signal, verdict and snapshot as a poison message.

Phase 2's tests never round-tripped through the transport, so nothing caught it.
These tests do, for every registered topic model.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from libs.kafka_client import TOPIC_MODELS, deserialize, serialize
from libs.schemas.base import BaseEvent
from libs.schemas.learning import AgentOutcome, ExitReason, PositionClosed
from libs.schemas.market import FundingRate, PerpMetrics, Side, TradeTick
from libs.schemas.portfolio import PortfolioSnapshot, Position
from libs.schemas.signals import AgentSignal, Direction
from libs.schemas.trading import DecisionAction, Fill, OrderIntent, RiskVerdict, TradeDecision
from tests.conftest import make_book, make_candle

NOW = datetime.now(tz=UTC)


def sample_position() -> Position:
    """Build a position with every optional field populated."""
    return Position(
        source="test",
        symbol="BTC",
        direction=Direction.LONG,
        size=Decimal("1.5"),
        entry_price=Decimal("100"),
        mark_price=Decimal("110"),
        stop_price=Decimal("95"),
        take_profit_price=Decimal("120"),
        trailing_stop_pct=2.0,
        opening_correlation_id=uuid4(),
        opening_decision_id=uuid4(),
        opened_at=NOW,
        updated_at=NOW,
    )


def sample_events() -> dict[str, BaseEvent]:
    """Build one populated instance of every event type on the bus."""
    return {
        "Candle": make_candle(open_price=1, high=2, low=0.5, close=1.5),
        "OrderBookSnapshot": make_book(),
        "TradeTick": TradeTick(
            source="test",
            symbol="BTC",
            price=Decimal("100.5"),
            size=Decimal("0.25"),
            side=Side.BUY,
            trade_id="12345",
        ),
        "PerpMetrics": PerpMetrics(
            source="test",
            symbol="BTC",
            open_interest=Decimal("34382.51"),
            mark_price=Decimal("78661.0"),
            oracle_price=Decimal("78691.1"),
            mid_price=Decimal("78664.5"),
            funding_rate=Decimal("0.0000125"),
            premium=Decimal("-0.00033"),
            day_notional_volume=Decimal("2223906367.2"),
            day_base_volume=Decimal("28288.9"),
        ),
        "FundingRate": FundingRate(
            source="test",
            symbol="BTC",
            funding_rate=Decimal("0.0000125"),
            premium=Decimal("-0.00033"),
        ),
        "AgentSignal": AgentSignal(
            source="test",
            agent_name="market_analyst",
            symbol="BTC",
            direction=Direction.LONG,
            confidence=0.8,
            reference_price=Decimal("100"),
        ),
        "TradeDecision": TradeDecision(
            source="test",
            decision_id=uuid4(),
            symbol="BTC",
            action=DecisionAction.OPEN_LONG,
            confidence=0.8,
            weighted_confidence=0.8,
            contributing_signals=(uuid4(),),
            agent_weights={"market_analyst": 1.0},
        ),
        "RiskVerdict": RiskVerdict(
            source="test",
            decision_id=uuid4(),
            symbol="BTC",
            approved=True,
            approved_size=Decimal("1"),
            approved_notional=Decimal("100"),
        ),
        "OrderIntent": OrderIntent(
            source="test",
            order_id=uuid4(),
            decision_id=uuid4(),
            symbol="BTC",
            side=Side.BUY,
            size=Decimal("1"),
            stop_price=Decimal("95"),
            trailing_stop_pct=2.0,
            exit_reason=None,
        ),
        "Fill": Fill(
            source="test",
            fill_id=uuid4(),
            order_id=uuid4(),
            symbol="BTC",
            side=Side.BUY,
            size=Decimal("1"),
            price=Decimal("100"),
            fee=Decimal("0.5"),
        ),
        "Position": sample_position(),
        "PortfolioSnapshot": PortfolioSnapshot(
            source="test",
            equity=Decimal("10000"),
            cash=Decimal("9900"),
            starting_equity=Decimal("10000"),
            day_start_equity=Decimal("10000"),
            positions=(sample_position(),),
        ),
        "PositionClosed": PositionClosed(
            source="test",
            symbol="BTC",
            direction=Direction.LONG,
            size=Decimal("1"),
            entry_price=Decimal("100"),
            exit_price=Decimal("110"),
            realized_pnl=Decimal("10"),
            fees_paid=Decimal("0.5"),
            exit_reason=ExitReason.TAKE_PROFIT,
            opened_at=NOW,
            closed_at=NOW,
            opening_correlation_id=uuid4(),
        ),
        "AgentOutcome": AgentOutcome(
            source="test",
            agent_name="market_analyst",
            signal_id=uuid4(),
            symbol="BTC",
            direction=Direction.LONG,
            confidence=0.8,
            realized_pnl=Decimal("10"),
            was_correct=True,
            brier_score=0.04,
            closed_at=NOW,
            holding_period_s=120.0,
        ),
    }


@pytest.mark.parametrize("name", sorted(sample_events()))
def test_event_round_trips_through_the_wire(name: str) -> None:
    """Every event survives serialise -> deserialise unchanged."""
    event = sample_events()[name]
    restored = deserialize(serialize(event), type(event))
    assert restored == event


@pytest.mark.parametrize("name", sorted(sample_events()))
def test_computed_fields_are_not_on_the_wire(name: str) -> None:
    """Derived values are excluded, which is what makes decoding possible."""
    event = sample_events()[name]
    payload = serialize(event).decode("utf-8")
    for field in type(event).model_computed_fields:
        assert f'"{field}"' not in payload, f"{name} leaked computed field {field}"


def test_every_registered_topic_model_round_trips() -> None:
    """No topic in the registry carries an event that cannot be decoded."""
    events = {type(event).__name__: event for event in sample_events().values()}
    missing = [
        model.__name__ for model in TOPIC_MODELS.values() if model.__name__ not in events
    ]
    assert not missing, f"topic models without round-trip coverage: {missing}"


def test_to_wire_output_can_be_revalidated() -> None:
    """``to_wire`` produces a dict the model itself accepts back."""
    event = sample_events()["Fill"]
    assert type(event).model_validate(event.to_wire()) == event


def test_nested_positions_survive_a_snapshot_round_trip() -> None:
    """A snapshot's nested positions decode with their protective levels."""
    snapshot = sample_events()["PortfolioSnapshot"]
    assert isinstance(snapshot, PortfolioSnapshot)
    restored = deserialize(serialize(snapshot), PortfolioSnapshot)
    position = restored.position_for("BTC")
    assert position is not None
    assert position.stop_price == Decimal("95")
    assert position.trailing_stop_pct == 2.0


def test_computed_values_recompute_after_a_round_trip() -> None:
    """Excluding computed fields loses nothing: they are re-derived."""
    fill = sample_events()["Fill"]
    assert isinstance(fill, Fill)
    restored = deserialize(serialize(fill), Fill)
    assert restored.notional == fill.notional
