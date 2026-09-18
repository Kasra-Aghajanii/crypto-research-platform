"""Tests for the decision -> risk -> execution -> portfolio chain."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import pytest

from libs.config import Settings, TradingMode
from libs.schemas.market import Side
from libs.schemas.portfolio import PortfolioSnapshot, Position
from libs.schemas.signals import AgentSignal, Direction
from libs.schemas.trading import (
    DecisionAction,
    Fill,
    OrderIntent,
    OrderType,
    TradeDecision,
    VetoReason,
)
from services.agents.execution.paper_broker import FillRejected, PaperBroker, walk_book
from services.agents.portfolio_manager.tracker import PortfolioTracker
from services.agents.risk_manager.agent import RiskManagerAgent
from services.decision_engine.engine import DecisionEngine
from services.decision_engine.trust import TrustRegistry, brier_score
from tests.conftest import make_book


def make_signal(
    *,
    agent: str = "market_analyst",
    direction: Direction = Direction.LONG,
    confidence: float = 0.8,
    price: float = 100.0,
    stop: float | None = 98.0,
    degraded: bool = False,
    valid_for_s: float | None = 90.0,
) -> AgentSignal:
    """Build a signal for the decision engine to consume."""
    return AgentSignal(
        source="test",
        agent_name=agent,
        agent_version="1.0.0",
        symbol="BTC",
        direction=direction,
        confidence=confidence,
        reference_price=Decimal(str(price)),
        suggested_stop=Decimal(str(stop)) if stop is not None else None,
        suggested_take_profit=Decimal("110"),
        degraded=degraded,
        valid_until=(
            datetime.now(tz=UTC) + timedelta(seconds=valid_for_s)
            if valid_for_s is not None
            else None
        ),
    )


def make_decision(
    *,
    action: DecisionAction = DecisionAction.OPEN_LONG,
    confidence: float = 0.8,
    price: float = 100.0,
    stop: float | None = 98.0,
) -> TradeDecision:
    """Build a decision for the risk manager to gate."""
    return TradeDecision(
        source="test",
        decision_id=uuid4(),
        symbol="BTC",
        action=action,
        confidence=confidence,
        weighted_confidence=confidence,
        reference_price=Decimal(str(price)),
        suggested_stop=Decimal(str(stop)) if stop is not None else None,
        suggested_take_profit=Decimal("110"),
    )


def make_portfolio(
    *,
    equity: float = 10_000.0,
    day_start: float = 10_000.0,
    positions: tuple[Position, ...] = (),
    gross: float = 0.0,
) -> PortfolioSnapshot:
    """Build a portfolio snapshot for the risk manager to size against."""
    return PortfolioSnapshot(
        source="test",
        equity=Decimal(str(equity)),
        cash=Decimal(str(equity)),
        starting_equity=Decimal("10000"),
        day_start_equity=Decimal(str(day_start)),
        gross_notional=Decimal(str(gross)),
        positions=positions,
    )


def make_position(
    *, direction: Direction = Direction.LONG, size: float = 1.0, price: float = 100.0
) -> Position:
    """Build an open position."""
    now = datetime.now(tz=UTC)
    return Position(
        source="test",
        symbol="BTC",
        direction=direction,
        size=Decimal(str(size)),
        entry_price=Decimal(str(price)),
        mark_price=Decimal(str(price)),
        opened_at=now,
        updated_at=now,
    )


class TestDecisionEngine:
    """Passthrough-mode guards."""

    def test_actionable_signal_becomes_a_decision(self) -> None:
        """A confident, fresh signal from the passthrough agent produces a decision."""
        decision = DecisionEngine()._decide(make_signal())
        assert decision is not None
        assert decision.action is DecisionAction.OPEN_LONG
        assert decision.mode == "passthrough"
        assert decision.contributing_signals

    def test_short_signal_maps_to_open_short(self) -> None:
        """Direction maps straight through to the opening action."""
        decision = DecisionEngine()._decide(make_signal(direction=Direction.SHORT))
        assert decision is not None and decision.action is DecisionAction.OPEN_SHORT

    def test_other_agents_are_ignored_in_passthrough(self) -> None:
        """Only the configured passthrough agent can move the engine."""
        assert DecisionEngine()._decide(make_signal(agent="sentiment_agent")) is None

    def test_neutral_signal_never_trades(self) -> None:
        """A confidence-zero error signal must not open a position."""
        assert DecisionEngine()._decide(make_signal(confidence=0.0, degraded=True)) is None

    def test_flat_direction_produces_no_decision(self) -> None:
        """No view means no decision."""
        assert DecisionEngine()._decide(make_signal(direction=Direction.FLAT)) is None

    def test_low_confidence_is_filtered(self) -> None:
        """Confidence below the floor is not actionable."""
        assert DecisionEngine()._decide(make_signal(confidence=0.1)) is None

    def test_expired_signal_is_dropped(self) -> None:
        """A stale view must never reach the risk manager."""
        stale = make_signal().model_copy(
            update={"valid_until": datetime.now(tz=UTC) - timedelta(seconds=1)}
        )
        assert DecisionEngine()._decide(stale) is None

    async def test_repeat_direction_is_suppressed_within_cooldown(self) -> None:
        """The same view on the next candle close does not re-fire."""
        engine = DecisionEngine()
        first = await engine.handle("agents.signals", make_signal())
        second = await engine.handle("agents.signals", make_signal())
        assert len(first) == 1
        assert second == ()

    async def test_direction_flip_bypasses_the_cooldown(self) -> None:
        """A reversal is always allowed through, cooldown or not."""
        engine = DecisionEngine()
        await engine.handle("agents.signals", make_signal())
        flipped = await engine.handle("agents.signals", make_signal(direction=Direction.SHORT))
        assert len(flipped) == 1

    def test_trust_weight_is_recorded_on_the_decision(self) -> None:
        """The decision carries the weight used, for later attribution."""
        decision = DecisionEngine()._decide(make_signal())
        assert decision is not None
        assert decision.agent_weights["market_analyst"] == pytest.approx(1.0)


class TestTrustRegistry:
    """Brier scoring and weight adaptation."""

    def test_brier_score_of_a_perfect_forecast_is_zero(self) -> None:
        """Full confidence in a correct call scores 0."""
        assert brier_score(1.0, True) == pytest.approx(0.0)

    def test_brier_score_of_a_confident_miss_is_one(self) -> None:
        """Full confidence in a wrong call scores 1."""
        assert brier_score(1.0, False) == pytest.approx(1.0)

    def test_coin_flip_scores_the_baseline(self) -> None:
        """A 50/50 forecast always scores the 0.25 baseline."""
        assert brier_score(0.5, True) == pytest.approx(0.25)

    def test_out_of_range_forecast_rejected(self) -> None:
        """A forecast outside [0, 1] is invalid."""
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            brier_score(1.5, True)

    def test_weight_stays_default_before_enough_samples(self) -> None:
        """A weight does not adapt on thin evidence."""
        registry = TrustRegistry(default_weight=1.0, min_samples=30)
        registry.record_outcome("a", confidence=0.9, was_correct=True)
        assert registry.weight_for("a") == pytest.approx(1.0)

    def test_accurate_agent_earns_a_higher_weight(self) -> None:
        """Consistently correct high-confidence calls raise the weight."""
        registry = TrustRegistry(default_weight=1.0, min_samples=10)
        for _ in range(20):
            registry.record_outcome("sharp", confidence=0.9, was_correct=True)
        assert registry.weight_for("sharp") > 1.0

    def test_inaccurate_agent_loses_weight(self) -> None:
        """Consistently wrong high-confidence calls cut the weight."""
        registry = TrustRegistry(default_weight=1.0, min_samples=10)
        for _ in range(20):
            registry.record_outcome("wrong", confidence=0.9, was_correct=False)
        assert registry.weight_for("wrong") < 1.0

    def test_weight_is_clamped(self) -> None:
        """Weights never escape the configured bounds."""
        registry = TrustRegistry(default_weight=1.0, min_samples=1, min_weight=0.25, max_weight=2.0)
        for _ in range(50):
            registry.record_outcome("perfect", confidence=1.0, was_correct=True)
        assert registry.weight_for("perfect") <= 2.0


class TestRiskManager:
    """Sizing arithmetic and veto rules."""

    def test_size_risks_the_configured_percent_of_equity(self) -> None:
        """Size is chosen so a stop-out costs risk_per_trade_pct of equity."""
        agent = RiskManagerAgent()
        agent._portfolio = make_portfolio(equity=10_000.0)
        verdict, order = agent.evaluate(make_decision(price=100.0, stop=98.0))
        assert verdict.approved and order is not None
        # 1% of 10,000 = $100 risk over a $2 stop distance = 50 units, capped by notional.
        assert verdict.risk_amount <= Decimal("100.01")
        assert order.side is Side.BUY

    def test_position_notional_cap_is_enforced(self) -> None:
        """A tight stop cannot buy an unbounded position."""
        agent = RiskManagerAgent()
        agent._portfolio = make_portfolio(equity=100_000.0)
        verdict, _ = agent.evaluate(make_decision(price=100.0, stop=99.99))
        assert verdict.approved
        assert verdict.approved_notional <= Decimal(str(agent.limits.max_position_notional))

    def test_low_confidence_is_vetoed(self) -> None:
        """Confidence below the risk floor is rejected."""
        agent = RiskManagerAgent()
        agent._portfolio = make_portfolio()
        verdict, order = agent.evaluate(make_decision(confidence=0.1))
        assert not verdict.approved and order is None
        assert VetoReason.LOW_CONFIDENCE in verdict.veto_reasons

    def test_daily_loss_limit_halts_new_entries(self) -> None:
        """Past the daily loss limit the kill switch stops new risk."""
        agent = RiskManagerAgent()
        agent._portfolio = make_portfolio(equity=9_000.0, day_start=10_000.0)
        verdict, _ = agent.evaluate(make_decision())
        assert VetoReason.DAILY_LOSS_LIMIT in verdict.veto_reasons

    def test_max_open_positions_is_enforced(self) -> None:
        """A new symbol is refused once the position count is full."""
        agent = RiskManagerAgent()
        agent.limits = agent.limits.model_copy(update={"max_open_positions": 1})
        agent._portfolio = make_portfolio(
            positions=(make_position().model_copy(update={"symbol": "ETH"}),)
        )
        verdict, _ = agent.evaluate(make_decision())
        assert VetoReason.MAX_OPEN_POSITIONS in verdict.veto_reasons

    def test_opposing_position_is_vetoed(self) -> None:
        """Phase 2 does not flip a position in a single step."""
        agent = RiskManagerAgent()
        agent._portfolio = make_portfolio(positions=(make_position(direction=Direction.SHORT),))
        verdict, _ = agent.evaluate(make_decision(action=DecisionAction.OPEN_LONG))
        assert VetoReason.DUPLICATE_EXPOSURE in verdict.veto_reasons

    def test_leverage_ceiling_is_enforced(self) -> None:
        """No headroom under the leverage cap means no new exposure."""
        agent = RiskManagerAgent()
        agent._portfolio = make_portfolio(equity=1_000.0, gross=3_000.0)
        verdict, _ = agent.evaluate(make_decision())
        assert VetoReason.MAX_LEVERAGE in verdict.veto_reasons

    def test_missing_price_is_vetoed(self) -> None:
        """Without a reference price nothing can be sized."""
        agent = RiskManagerAgent()
        agent._portfolio = make_portfolio()
        decision = make_decision().model_copy(update={"reference_price": None})
        verdict, _ = agent.evaluate(decision)
        assert VetoReason.NO_MARKET_DATA in verdict.veto_reasons

    def test_stop_on_the_wrong_side_falls_back_to_default_distance(self) -> None:
        """A long stop above entry is ignored in favour of the configured distance."""
        agent = RiskManagerAgent()
        agent._portfolio = make_portfolio()
        verdict, _ = agent.evaluate(make_decision(price=100.0, stop=105.0))
        assert verdict.approved
        assert verdict.stop_price is not None and verdict.stop_price < Decimal("100")

    def test_short_decision_produces_a_sell_order(self) -> None:
        """A short decision becomes a sell-side order intent."""
        agent = RiskManagerAgent()
        agent._portfolio = make_portfolio()
        decision = make_decision(action=DecisionAction.OPEN_SHORT, price=100.0, stop=102.0)
        verdict, order = agent.evaluate(decision)
        assert verdict.approved and order is not None
        assert order.side is Side.SELL
        assert order.order_type is OrderType.MARKET

    def test_hold_action_is_not_executed(self) -> None:
        """A hold never reaches the execution layer."""
        agent = RiskManagerAgent()
        verdict, order = agent.evaluate(make_decision(action=DecisionAction.HOLD))
        assert not verdict.approved and order is None


class TestPaperBroker:
    """Fill simulation against the live book."""

    def test_walk_book_averages_across_levels(self) -> None:
        """Size larger than the top level prices at the volume-weighted average."""
        book = make_book(mid=100.0, spread=0.02, level_size=1.0, levels=3)
        filled, price = walk_book(book.asks, Decimal("2"))
        assert filled == Decimal("2")
        expected = (book.asks[0].price + book.asks[1].price) / 2
        assert price == pytest.approx(expected)

    def test_walk_book_reports_partial_when_depth_runs_out(self) -> None:
        """Visible depth is never exceeded; the shortfall is reported."""
        book = make_book(level_size=1.0, levels=2)
        filled, _ = walk_book(book.asks, Decimal("10"))
        assert filled == Decimal("2")

    def test_walk_book_rejects_an_empty_side(self) -> None:
        """An empty side cannot fill anything."""
        with pytest.raises(FillRejected):
            walk_book((), Decimal("1"))

    def test_buy_fills_at_or_above_the_best_ask(self) -> None:
        """Slippage is always adverse for a market buy."""
        broker = PaperBroker()
        book = make_book(mid=100.0)
        broker._books["BTC"] = book
        order = OrderIntent(
            source="test",
            order_id=uuid4(),
            decision_id=uuid4(),
            symbol="BTC",
            side=Side.BUY,
            size=Decimal("0.5"),
        )
        fill = broker.simulate_fill(order)
        assert book.best_ask is not None and fill.price >= book.best_ask
        assert fill.slippage_bps >= 0
        assert fill.is_paper is True

    def test_sell_fills_at_or_below_the_best_bid(self) -> None:
        """Slippage is always adverse for a market sell too."""
        broker = PaperBroker()
        book = make_book(mid=100.0)
        broker._books["BTC"] = book
        order = OrderIntent(
            source="test",
            order_id=uuid4(),
            decision_id=uuid4(),
            symbol="BTC",
            side=Side.SELL,
            size=Decimal("0.5"),
        )
        fill = broker.simulate_fill(order)
        assert book.best_bid is not None and fill.price <= book.best_bid

    def test_fee_is_charged_on_notional(self) -> None:
        """The taker fee is a fraction of the filled notional."""
        broker = PaperBroker()
        broker._books["BTC"] = make_book(mid=100.0)
        order = OrderIntent(
            source="test",
            order_id=uuid4(),
            decision_id=uuid4(),
            symbol="BTC",
            side=Side.BUY,
            size=Decimal("1"),
        )
        fill = broker.simulate_fill(order)
        expected = fill.notional * Decimal(str(broker.params.taker_fee_bps)) / Decimal(10_000)
        assert fill.fee == pytest.approx(expected, abs=Decimal("0.0001"))

    def test_stale_book_is_refused(self) -> None:
        """A fill is never invented against an outdated book."""
        broker = PaperBroker()
        broker._books["BTC"] = make_book(
            mid=100.0, occurred_at=datetime.now(tz=UTC) - timedelta(seconds=600)
        )
        order = OrderIntent(
            source="test",
            order_id=uuid4(),
            decision_id=uuid4(),
            symbol="BTC",
            side=Side.BUY,
            size=Decimal("1"),
        )
        with pytest.raises(FillRejected, match="stale"):
            broker.simulate_fill(order)

    def test_unknown_symbol_is_refused(self) -> None:
        """Without a book there is no price to fill at."""
        broker = PaperBroker()
        order = OrderIntent(
            source="test",
            order_id=uuid4(),
            decision_id=uuid4(),
            symbol="DOGE",
            side=Side.BUY,
            size=Decimal("1"),
        )
        with pytest.raises(FillRejected, match="No order book"):
            broker.simulate_fill(order)

    def test_broker_refuses_to_run_in_live_mode(self) -> None:
        """The paper broker is paper-only and says so loudly."""
        live = Settings(trading_mode=TradingMode.LIVE)
        with pytest.raises(RuntimeError, match="paper-only"):
            PaperBroker(config=live)


class TestPortfolioTracker:
    """Position accounting and PnL."""

    @staticmethod
    def fill(
        side: Side, size: float, price: float, fee: float = 0.0, symbol: str = "BTC"
    ) -> Fill:
        """Build a fill for the tracker to apply."""
        return Fill(
            source="test",
            fill_id=uuid4(),
            order_id=uuid4(),
            symbol=symbol,
            side=side,
            size=Decimal(str(size)),
            price=Decimal(str(price)),
            fee=Decimal(str(fee)),
        )

    def tracker(self) -> PortfolioTracker:
        """Return a tracker with 10,000 starting equity."""
        return PortfolioTracker(starting_equity=Decimal("10000"))

    def test_first_fill_opens_a_position(self) -> None:
        """A buy with no existing position opens a long."""
        tracker = self.tracker()
        tracker.apply_fill(self.fill(Side.BUY, 1.0, 100.0))
        position = tracker.snapshot().position_for("BTC")
        assert position is not None
        assert position.direction is Direction.LONG
        assert position.entry_price == Decimal("100")

    def test_adding_averages_the_entry_price(self) -> None:
        """A second buy produces a volume-weighted average entry."""
        tracker = self.tracker()
        tracker.apply_fill(self.fill(Side.BUY, 1.0, 100.0))
        tracker.apply_fill(self.fill(Side.BUY, 1.0, 120.0))
        position = tracker.snapshot().position_for("BTC")
        assert position is not None
        assert position.size == Decimal("2")
        assert position.entry_price == Decimal("110")

    def test_closing_realises_pnl_into_cash(self) -> None:
        """Selling a long books the gain to realised PnL and cash."""
        tracker = self.tracker()
        tracker.apply_fill(self.fill(Side.BUY, 1.0, 100.0))
        tracker.apply_fill(self.fill(Side.SELL, 1.0, 110.0))
        snapshot = tracker.snapshot()
        assert snapshot.realized_pnl == Decimal("10")
        assert snapshot.cash == Decimal("10010")
        assert snapshot.open_position_count == 0
        assert snapshot.win_count == 1

    def test_losing_trade_is_counted_as_a_loss(self) -> None:
        """A negative close increments the loss counter."""
        tracker = self.tracker()
        tracker.apply_fill(self.fill(Side.BUY, 1.0, 100.0))
        tracker.apply_fill(self.fill(Side.SELL, 1.0, 90.0))
        snapshot = tracker.snapshot()
        assert snapshot.realized_pnl == Decimal("-10")
        assert snapshot.loss_count == 1

    def test_partial_close_leaves_the_remainder_open(self) -> None:
        """Selling half a long leaves half open at the original entry."""
        tracker = self.tracker()
        tracker.apply_fill(self.fill(Side.BUY, 2.0, 100.0))
        tracker.apply_fill(self.fill(Side.SELL, 1.0, 110.0))
        position = tracker.snapshot().position_for("BTC")
        assert position is not None
        assert position.size == Decimal("1")
        assert tracker.realized_pnl == Decimal("10")

    def test_oversized_opposing_fill_flips_the_position(self) -> None:
        """Selling more than the long closes it and opens a short."""
        tracker = self.tracker()
        tracker.apply_fill(self.fill(Side.BUY, 1.0, 100.0))
        tracker.apply_fill(self.fill(Side.SELL, 3.0, 110.0))
        position = tracker.snapshot().position_for("BTC")
        assert position is not None
        assert position.direction is Direction.SHORT
        assert position.size == Decimal("2")
        assert tracker.realized_pnl == Decimal("10")

    def test_short_pnl_is_signed_correctly(self) -> None:
        """A short profits when price falls."""
        tracker = self.tracker()
        tracker.apply_fill(self.fill(Side.SELL, 1.0, 100.0))
        tracker.apply_fill(self.fill(Side.BUY, 1.0, 90.0))
        assert tracker.realized_pnl == Decimal("10")

    def test_fees_reduce_cash_and_are_tracked(self) -> None:
        """Fees are charged immediately and reported separately."""
        tracker = self.tracker()
        tracker.apply_fill(self.fill(Side.BUY, 1.0, 100.0, fee=0.5))
        snapshot = tracker.snapshot()
        assert snapshot.fees_paid == Decimal("0.5")
        assert snapshot.cash == Decimal("9999.5")

    def test_marking_moves_unrealised_pnl_and_equity(self) -> None:
        """Equity follows the mark price while a position is open."""
        tracker = self.tracker()
        tracker.apply_fill(self.fill(Side.BUY, 1.0, 100.0))
        tracker.mark("BTC", Decimal("120"))
        snapshot = tracker.snapshot()
        assert snapshot.unrealized_pnl == Decimal("20")
        assert snapshot.equity == Decimal("10020")

    def test_gross_notional_and_leverage(self) -> None:
        """Gross notional sums position values; leverage divides by equity."""
        tracker = self.tracker()
        tracker.apply_fill(self.fill(Side.BUY, 10.0, 100.0))
        snapshot = tracker.snapshot()
        assert snapshot.gross_notional == Decimal("1000")
        assert snapshot.leverage == pytest.approx(0.1)

    def test_win_rate_reflects_closed_trades(self) -> None:
        """Win rate counts only resolved trades."""
        tracker = self.tracker()
        tracker.apply_fill(self.fill(Side.BUY, 1.0, 100.0))
        tracker.apply_fill(self.fill(Side.SELL, 1.0, 110.0))
        tracker.apply_fill(self.fill(Side.BUY, 1.0, 100.0))
        tracker.apply_fill(self.fill(Side.SELL, 1.0, 95.0))
        assert tracker.snapshot().win_rate == pytest.approx(0.5)

    def test_positions_are_tracked_per_symbol(self) -> None:
        """Two symbols keep independent positions."""
        tracker = self.tracker()
        tracker.apply_fill(self.fill(Side.BUY, 1.0, 100.0, symbol="BTC"))
        tracker.apply_fill(self.fill(Side.BUY, 2.0, 50.0, symbol="ETH"))
        snapshot = tracker.snapshot()
        assert snapshot.open_position_count == 2
        assert snapshot.position_for("ETH") is not None
