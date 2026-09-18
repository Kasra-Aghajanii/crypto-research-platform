"""Tests for the position monitor: stops, take-profits and trailing exits."""

from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

import pytest

from libs.kafka_client import Topics
from libs.schemas.learning import ExitReason
from libs.schemas.market import Side
from libs.schemas.signals import Direction
from libs.schemas.trading import Fill, OrderIntent
from services.execution.position_monitor import PositionMonitor, ProtectedPosition
from tests.conftest import make_book
from tests.test_execution_chain import make_portfolio, make_position


def protected(
    *,
    direction: Direction = Direction.LONG,
    entry: float = 100.0,
    stop: float | None = 95.0,
    take_profit: float | None = 110.0,
    trailing: float | None = None,
    size: float = 1.0,
) -> ProtectedPosition:
    """Build a guarded position."""
    return ProtectedPosition(
        symbol="BTC",
        direction=direction,
        size=Decimal(str(size)),
        entry_price=Decimal(str(entry)),
        stop_price=Decimal(str(stop)) if stop is not None else None,
        take_profit_price=Decimal(str(take_profit)) if take_profit is not None else None,
        trailing_stop_pct=trailing,
        extreme_price=Decimal(str(entry)),
    )


def opening_order(
    *,
    side: Side = Side.BUY,
    stop: float | None = 95.0,
    take_profit: float | None = 110.0,
    trailing: float | None = None,
    size: float = 1.0,
) -> OrderIntent:
    """Build an opening order carrying protective levels."""
    return OrderIntent(
        source="test",
        order_id=uuid4(),
        decision_id=uuid4(),
        symbol="BTC",
        side=side,
        size=Decimal(str(size)),
        stop_price=Decimal(str(stop)) if stop is not None else None,
        take_profit_price=Decimal(str(take_profit)) if take_profit is not None else None,
        trailing_stop_pct=trailing,
    )


def fill_for(order: OrderIntent, price: float = 100.0) -> Fill:
    """Build the fill produced by an order."""
    return Fill(
        source="test",
        fill_id=uuid4(),
        order_id=order.order_id,
        decision_id=order.decision_id,
        symbol=order.symbol,
        side=order.side,
        size=order.size,
        price=Decimal(str(price)),
    )


class TestBreachDetection:
    """The rules that decide when a level has broken."""

    def test_long_stop_triggers_at_or_below(self) -> None:
        """A long stops out when price reaches the stop."""
        record = protected()
        assert record.breach(Decimal("95")) is ExitReason.STOP_LOSS
        assert record.breach(Decimal("94")) is ExitReason.STOP_LOSS
        assert record.breach(Decimal("96")) is None

    def test_long_take_profit_triggers_at_or_above(self) -> None:
        """A long takes profit when price reaches the target."""
        assert protected().breach(Decimal("110")) is ExitReason.TAKE_PROFIT
        assert protected().breach(Decimal("109")) is None

    def test_short_levels_are_mirrored(self) -> None:
        """A short stops out above entry and profits below."""
        record = protected(direction=Direction.SHORT, stop=105.0, take_profit=90.0)
        assert record.breach(Decimal("105")) is ExitReason.STOP_LOSS
        assert record.breach(Decimal("90")) is ExitReason.TAKE_PROFIT
        assert record.breach(Decimal("100")) is None

    def test_stop_wins_when_both_levels_break(self) -> None:
        """An ambiguous tick resolves to the unfavourable outcome."""
        record = protected(stop=100.0, take_profit=100.0)
        assert record.breach(Decimal("100")) is ExitReason.STOP_LOSS

    def test_no_levels_means_no_exit(self) -> None:
        """A position with no protection never self-closes."""
        record = protected(stop=None, take_profit=None)
        assert record.breach(Decimal("1")) is None
        assert record.breach(Decimal("10000")) is None


class TestTrailingStop:
    """Ratcheting behaviour of the trailing stop."""

    def test_trailing_stop_follows_a_rising_long(self) -> None:
        """The stop moves up as the high-water mark rises."""
        record = protected(stop=None, take_profit=None, trailing=10.0)
        record.observe(Decimal("120"))
        assert record.trailing_stop == Decimal("108.0")

    def test_trailing_stop_never_loosens(self) -> None:
        """A pullback does not lower an already-ratcheted stop."""
        record = protected(stop=None, take_profit=None, trailing=10.0)
        record.observe(Decimal("120"))
        high_water = record.trailing_stop
        record.observe(Decimal("110"))
        assert record.trailing_stop == high_water

    def test_short_trailing_stop_ratchets_downward(self) -> None:
        """For a short, the stop follows the low-water mark down."""
        record = protected(
            direction=Direction.SHORT, stop=None, take_profit=None, trailing=10.0, entry=100.0
        )
        record.observe(Decimal("80"))
        assert record.trailing_stop == Decimal("88.0")
        record.observe(Decimal("90"))
        assert record.trailing_stop == Decimal("88.0")

    def test_effective_stop_is_the_tighter_of_the_two(self) -> None:
        """Adding a trail can only reduce risk, never increase it."""
        record = protected(stop=95.0, trailing=10.0)
        assert record.effective_stop == Decimal("95")  # trail is at 90, looser
        record.observe(Decimal("120"))
        assert record.effective_stop == Decimal("108.0")  # trail now tighter

    def test_trailing_breach_is_reported_as_a_trailing_stop(self) -> None:
        """The exit reason distinguishes a trail from the original stop."""
        record = protected(stop=95.0, take_profit=None, trailing=10.0)
        record.observe(Decimal("120"))
        assert record.breach(Decimal("107")) is ExitReason.TRAILING_STOP

    def test_fixed_stop_breach_is_reported_as_a_stop_loss(self) -> None:
        """Before the trail overtakes it, a breach is the fixed stop."""
        record = protected(stop=95.0, take_profit=None, trailing=10.0)
        assert record.breach(Decimal("94")) is ExitReason.STOP_LOSS


class TestMonitorLifecycle:
    """How the monitor learns about, guards and releases positions."""

    async def test_order_then_fill_starts_guarding(self) -> None:
        """Protective levels come from the order, the entry from the fill."""
        monitor = PositionMonitor()
        order = opening_order()
        await monitor.handle(Topics.ORDER_INTENTS, order)
        await monitor.handle(Topics.FILLS, fill_for(order, 100.0))

        record = monitor.tracked["BTC"]
        assert record.entry_price == Decimal("100")
        assert record.stop_price == Decimal("95")
        assert record.take_profit_price == Decimal("110")

    async def test_stop_breach_emits_a_reduce_only_exit(self) -> None:
        """A breach fires a reduce-only market order on the opposite side."""
        monitor = PositionMonitor()
        order = opening_order()
        await monitor.handle(Topics.ORDER_INTENTS, order)
        await monitor.handle(Topics.FILLS, fill_for(order))

        publications = await monitor.on_price("BTC", Decimal("94"))
        assert len(publications) == 1
        topic, exit_order = publications[0]
        assert topic == Topics.ORDER_INTENTS
        assert isinstance(exit_order, OrderIntent)
        assert exit_order.reduce_only is True
        assert exit_order.side is Side.SELL
        assert exit_order.size == Decimal("1")
        assert exit_order.exit_reason == ExitReason.STOP_LOSS.value

    async def test_short_exit_buys_back(self) -> None:
        """Closing a short is a buy."""
        monitor = PositionMonitor()
        order = opening_order(side=Side.SELL, stop=105.0, take_profit=90.0)
        await monitor.handle(Topics.ORDER_INTENTS, order)
        await monitor.handle(Topics.FILLS, fill_for(order))

        publications = await monitor.on_price("BTC", Decimal("106"))
        _, exit_order = publications[0]
        assert isinstance(exit_order, OrderIntent)
        assert exit_order.side is Side.BUY

    async def test_exit_fires_only_once(self) -> None:
        """A breach that persists across ticks does not duplicate the exit."""
        monitor = PositionMonitor()
        order = opening_order()
        await monitor.handle(Topics.ORDER_INTENTS, order)
        await monitor.handle(Topics.FILLS, fill_for(order))

        first = await monitor.on_price("BTC", Decimal("94"))
        second = await monitor.on_price("BTC", Decimal("93"))
        third = await monitor.on_price("BTC", Decimal("92"))
        assert len(first) == 1
        assert second == ()
        assert third == ()
        assert monitor.exits_fired == 1

    async def test_no_exit_while_price_is_inside_the_levels(self) -> None:
        """Ticks between the levels produce nothing."""
        monitor = PositionMonitor()
        order = opening_order()
        await monitor.handle(Topics.ORDER_INTENTS, order)
        await monitor.handle(Topics.FILLS, fill_for(order))
        assert await monitor.on_price("BTC", Decimal("102")) == ()

    async def test_ticks_for_unknown_symbols_are_ignored(self) -> None:
        """A price for a symbol with no position does nothing."""
        assert await PositionMonitor().on_price("DOGE", Decimal("1")) == ()

    async def test_closing_fill_releases_protection(self) -> None:
        """Once the position is flat the monitor stops guarding it."""
        monitor = PositionMonitor()
        order = opening_order()
        await monitor.handle(Topics.ORDER_INTENTS, order)
        await monitor.handle(Topics.FILLS, fill_for(order))
        assert "BTC" in monitor.tracked

        closing = Fill(
            source="test",
            fill_id=uuid4(),
            order_id=uuid4(),
            symbol="BTC",
            side=Side.SELL,
            size=Decimal("1"),
            price=Decimal("105"),
            reduce_only=True,
        )
        await monitor.handle(Topics.FILLS, closing)
        assert "BTC" not in monitor.tracked

    async def test_partial_close_keeps_guarding_the_remainder(self) -> None:
        """Reducing a position leaves the rest protected at the smaller size."""
        monitor = PositionMonitor()
        order = opening_order(size=2.0)
        await monitor.handle(Topics.ORDER_INTENTS, order)
        await monitor.handle(Topics.FILLS, fill_for(order))

        partial = Fill(
            source="test",
            fill_id=uuid4(),
            order_id=uuid4(),
            symbol="BTC",
            side=Side.SELL,
            size=Decimal("1"),
            price=Decimal("105"),
            reduce_only=True,
        )
        await monitor.handle(Topics.FILLS, partial)
        assert monitor.tracked["BTC"].size == Decimal("1")

    async def test_snapshot_drops_positions_the_portfolio_no_longer_holds(self) -> None:
        """Reconciliation removes stale protection."""
        monitor = PositionMonitor()
        order = opening_order()
        await monitor.handle(Topics.ORDER_INTENTS, order)
        await monitor.handle(Topics.FILLS, fill_for(order))

        await monitor.handle(Topics.PORTFOLIO_SNAPSHOTS, make_portfolio(positions=()))
        assert monitor.tracked == {}

    async def test_snapshot_adopts_an_unguarded_position(self) -> None:
        """A position the monitor never saw is adopted rather than left naked."""
        monitor = PositionMonitor()
        position = make_position().model_copy(update={"stop_price": Decimal("95")})
        await monitor.handle(Topics.PORTFOLIO_SNAPSHOTS, make_portfolio(positions=(position,)))

        assert "BTC" in monitor.tracked
        assert monitor.tracked["BTC"].stop_price == Decimal("95")

    async def test_book_mid_drives_the_price_check(self) -> None:
        """An order book snapshot is a valid price source for exits."""
        monitor = PositionMonitor()
        order = opening_order()
        await monitor.handle(Topics.ORDER_INTENTS, order)
        await monitor.handle(Topics.FILLS, fill_for(order))

        publications = await monitor.handle(
            Topics.MARKET_ORDERBOOK, make_book(symbol="BTC", mid=94.0)
        )
        assert len(publications) == 1

    async def test_reduce_only_orders_are_not_treated_as_openings(self) -> None:
        """An exit order must not be cached as a new position's protection."""
        monitor = PositionMonitor()
        exit_order = opening_order().model_copy(update={"reduce_only": True})
        await monitor.handle(Topics.ORDER_INTENTS, exit_order)
        assert monitor._pending_orders == {}


class TestPersistenceRoundTrip:
    """Protection state must survive a restart."""

    def test_record_survives_a_row_round_trip(self) -> None:
        """Serialising and rebuilding preserves the trailing extreme."""
        record = protected(trailing=5.0)
        record.observe(Decimal("130"))
        rebuilt = ProtectedPosition.from_row(record.to_row())

        assert rebuilt.extreme_price == Decimal("130")
        assert rebuilt.trailing_stop == record.trailing_stop
        assert rebuilt.effective_stop == record.effective_stop

    def test_rebuilt_record_keeps_the_ratchet(self) -> None:
        """A restart must not hand back profit by resetting the trail."""
        record = protected(stop=95.0, take_profit=None, trailing=10.0)
        record.observe(Decimal("120"))
        rebuilt = ProtectedPosition.from_row(record.to_row())
        assert rebuilt.breach(Decimal("107")) is ExitReason.TRAILING_STOP

    def test_adopted_position_carries_attribution_ids(self) -> None:
        """Adoption preserves the correlation id needed for attribution."""
        correlation = uuid4()
        position = make_position().model_copy(update={"opening_correlation_id": correlation})
        record = ProtectedPosition.from_position(position)
        assert record.correlation_id == correlation


@pytest.mark.parametrize(
    ("direction", "trailing_pct", "extreme", "expected"),
    [
        (Direction.LONG, 1.0, "200", "198.0"),
        (Direction.LONG, 50.0, "200", "100.0"),
        (Direction.SHORT, 1.0, "50", "50.5"),
        (Direction.SHORT, 20.0, "50", "60.0"),
    ],
)
def test_trailing_stop_arithmetic(
    direction: Direction, trailing_pct: float, extreme: str, expected: str
) -> None:
    """The trailing stop sits exactly ``trailing_pct`` from the extreme."""
    record = protected(direction=direction, stop=None, take_profit=None, trailing=trailing_pct)
    record.extreme_price = Decimal(extreme)
    assert record.trailing_stop == Decimal(expected)
