"""Tests for outcome attribution, persistence and calibration analysis."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

import pytest

from libs.kafka_client import Topics
from libs.persistence.repositories import PortfolioRepository
from libs.schemas.learning import AgentOutcome, ExitReason, PositionClosed
from libs.schemas.market import Side
from libs.schemas.signals import AgentSignal, Direction
from libs.schemas.trading import DecisionAction, Fill, OrderIntent, TradeDecision
from services.agents.learning.outcome_recorder import OutcomeRecorder
from services.agents.portfolio_manager.tracker import PortfolioManagerAgent, PortfolioTracker
from services.backtest.calibration import analyze_calibration
from services.decision_engine.engine import DecisionEngine


class FakeDatabase:
    """Records SQL instead of executing it, and replays canned rows."""

    def __init__(self, rows: list[Any] | None = None) -> None:
        """Start with an optional canned result set."""
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.rows = rows or []

    async def execute(self, query: str, *args: Any) -> str:
        """Record a statement."""
        self.calls.append((query, args))
        return "OK"

    async def executemany(self, query: str, args: Any) -> None:
        """Record a batch statement."""
        self.calls.append((query, tuple(args)))

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        """Return the canned rows."""
        self.calls.append((query, args))
        return self.rows

    async def fetchrow(self, query: str, *args: Any) -> Any | None:
        """Return the first canned row."""
        self.calls.append((query, args))
        return self.rows[0] if self.rows else None

    def statements_containing(self, fragment: str) -> list[str]:
        """Return recorded statements containing ``fragment``."""
        return [query for query, _ in self.calls if fragment in query]


def make_closure(
    *,
    correlation_id: UUID | None = None,
    pnl: float = 50.0,
    fees: float = 1.0,
    reason: ExitReason = ExitReason.TAKE_PROFIT,
) -> PositionClosed:
    """Build a closed position."""
    opened = datetime.now(tz=UTC) - timedelta(hours=3)
    return PositionClosed(
        source="test",
        symbol="BTC",
        direction=Direction.LONG,
        size=Decimal("1"),
        entry_price=Decimal("100"),
        exit_price=Decimal("150"),
        realized_pnl=Decimal(str(pnl)),
        fees_paid=Decimal(str(fees)),
        exit_reason=reason,
        opened_at=opened,
        closed_at=datetime.now(tz=UTC),
        opening_correlation_id=correlation_id,
    )


def make_signal(
    *,
    correlation_id: UUID,
    confidence: float = 0.8,
    agent: str = "market_analyst",
    degraded: bool = False,
) -> AgentSignal:
    """Build a signal carrying a known correlation id."""
    return AgentSignal(
        source="test",
        correlation_id=correlation_id,
        agent_name=agent,
        agent_version="1.0.0",
        symbol="BTC",
        direction=Direction.LONG,
        confidence=confidence,
        degraded=degraded,
    )


class TestOutcomeAttribution:
    """Joining a closed position back to the signals that caused it."""

    async def test_profitable_close_scores_the_signal_correct(self) -> None:
        """A winning trade marks its signal correct and scores it well."""
        recorder = OutcomeRecorder()
        correlation = uuid4()
        await recorder.handle(Topics.AGENT_SIGNALS, make_signal(correlation_id=correlation))

        publications = await recorder.handle(
            Topics.POSITION_CLOSURES, make_closure(correlation_id=correlation, pnl=50.0)
        )
        assert len(publications) == 1
        topic, outcome = publications[0]
        assert topic == Topics.AGENT_OUTCOMES
        assert isinstance(outcome, AgentOutcome)
        assert outcome.was_correct is True
        assert outcome.brier_score == pytest.approx((0.8 - 1.0) ** 2)

    async def test_losing_close_scores_the_signal_wrong(self) -> None:
        """A losing trade marks the call incorrect."""
        recorder = OutcomeRecorder()
        correlation = uuid4()
        await recorder.handle(Topics.AGENT_SIGNALS, make_signal(correlation_id=correlation))
        publications = await recorder.handle(
            Topics.POSITION_CLOSURES, make_closure(correlation_id=correlation, pnl=-50.0)
        )
        _, outcome = publications[0]
        assert isinstance(outcome, AgentOutcome)
        assert outcome.was_correct is False
        assert outcome.brier_score == pytest.approx(0.8**2)

    async def test_fees_can_turn_a_gross_win_into_a_loss(self) -> None:
        """Correctness is judged after costs, not before."""
        recorder = OutcomeRecorder()
        correlation = uuid4()
        await recorder.handle(Topics.AGENT_SIGNALS, make_signal(correlation_id=correlation))
        publications = await recorder.handle(
            Topics.POSITION_CLOSURES,
            make_closure(correlation_id=correlation, pnl=1.0, fees=5.0),
        )
        _, outcome = publications[0]
        assert isinstance(outcome, AgentOutcome)
        assert outcome.was_correct is False

    async def test_decision_contributing_signals_drive_attribution(self) -> None:
        """When a decision names its signals, those are the ones scored."""
        recorder = OutcomeRecorder()
        correlation = uuid4()
        signal = make_signal(correlation_id=correlation)
        await recorder.handle(Topics.AGENT_SIGNALS, signal)
        decision = TradeDecision(
            source="test",
            correlation_id=correlation,
            decision_id=uuid4(),
            symbol="BTC",
            action=DecisionAction.OPEN_LONG,
            confidence=0.8,
            weighted_confidence=0.8,
            contributing_signals=(signal.event_id,),
        )
        await recorder.handle(Topics.TRADE_DECISIONS, decision)

        publications = await recorder.handle(
            Topics.POSITION_CLOSURES, make_closure(correlation_id=correlation)
        )
        _, outcome = publications[0]
        assert isinstance(outcome, AgentOutcome)
        assert outcome.signal_id == signal.event_id

    async def test_degraded_signals_are_not_scored(self) -> None:
        """A neutral error signal expresses no view, so it earns no score."""
        recorder = OutcomeRecorder()
        correlation = uuid4()
        await recorder.handle(
            Topics.AGENT_SIGNALS,
            make_signal(correlation_id=correlation, confidence=0.0, degraded=True),
        )
        publications = await recorder.handle(
            Topics.POSITION_CLOSURES, make_closure(correlation_id=correlation)
        )
        assert not publications

    async def test_unattributable_closure_is_counted_not_crashed(self) -> None:
        """A closure with no matching signal is reported, not fatal."""
        recorder = OutcomeRecorder()
        publications = await recorder.handle(
            Topics.POSITION_CLOSURES, make_closure(correlation_id=uuid4())
        )
        assert publications == ()
        assert recorder.unattributed == 1

    async def test_closure_without_a_correlation_id_is_unattributed(self) -> None:
        """Without the join key there is nothing to attribute to."""
        recorder = OutcomeRecorder()
        publications = await recorder.handle(
            Topics.POSITION_CLOSURES, make_closure(correlation_id=None)
        )
        assert publications == ()

    async def test_multiple_signals_on_one_chain_are_all_scored(self) -> None:
        """Every agent that contributed gets its own outcome."""
        recorder = OutcomeRecorder()
        correlation = uuid4()
        await recorder.handle(
            Topics.AGENT_SIGNALS, make_signal(correlation_id=correlation, agent="market_analyst")
        )
        await recorder.handle(
            Topics.AGENT_SIGNALS, make_signal(correlation_id=correlation, agent="sentiment")
        )
        publications = await recorder.handle(
            Topics.POSITION_CLOSURES, make_closure(correlation_id=correlation)
        )
        agents = {
            outcome.agent_name for _, outcome in publications if isinstance(outcome, AgentOutcome)
        }
        assert agents == {"market_analyst", "sentiment"}


class TestTrustLoopClosure:
    """The decision engine must actually learn from recorded outcomes."""

    def make_outcome(self, *, correct: bool, confidence: float = 0.9) -> AgentOutcome:
        """Build a scored outcome."""
        return AgentOutcome(
            source="test",
            agent_name="market_analyst",
            signal_id=uuid4(),
            symbol="BTC",
            direction=Direction.LONG,
            confidence=confidence,
            realized_pnl=Decimal("10") if correct else Decimal("-10"),
            was_correct=correct,
            brier_score=(confidence - (1.0 if correct else 0.0)) ** 2,
            closed_at=datetime.now(tz=UTC),
            holding_period_s=100.0,
        )

    async def test_outcomes_feed_the_trust_registry(self) -> None:
        """An outcome event updates the agent's recorded track record."""
        engine = DecisionEngine()
        await engine.handle(Topics.AGENT_OUTCOMES, self.make_outcome(correct=True))
        assert engine.outcomes_seen == 1
        samples, _, _ = engine.trust.stats()["market_analyst"]
        assert samples == 1

    async def test_a_good_record_raises_the_weight(self) -> None:
        """Sustained accuracy earns an agent more influence."""
        engine = DecisionEngine()
        for _ in range(engine.trust._min_samples + 5):
            await engine.handle(Topics.AGENT_OUTCOMES, self.make_outcome(correct=True))
        assert engine.trust.weight_for("market_analyst") > 1.0

    async def test_a_bad_record_lowers_the_weight(self) -> None:
        """Sustained inaccuracy costs an agent influence."""
        engine = DecisionEngine()
        for _ in range(engine.trust._min_samples + 5):
            await engine.handle(Topics.AGENT_OUTCOMES, self.make_outcome(correct=False))
        assert engine.trust.weight_for("market_analyst") < 1.0

    async def test_weight_flows_into_the_next_decision(self) -> None:
        """The learned weight is what scales the next decision's confidence."""
        engine = DecisionEngine()
        for _ in range(engine.trust._min_samples + 5):
            await engine.handle(Topics.AGENT_OUTCOMES, self.make_outcome(correct=False))

        signal = AgentSignal(
            source="test",
            agent_name="market_analyst",
            symbol="BTC",
            direction=Direction.LONG,
            confidence=0.9,
            reference_price=Decimal("100"),
        )
        decision = engine._decide(signal)
        assert decision is not None
        assert decision.weighted_confidence < decision.confidence


class TestClosureGeneration:
    """The portfolio manager must emit closures that can be attributed."""

    @staticmethod
    def fill(side: Side, size: float, price: float, correlation: UUID | None = None) -> Fill:
        """Build a fill, optionally on a known correlation chain."""
        payload: dict[str, Any] = {
            "source": "test",
            "fill_id": uuid4(),
            "order_id": uuid4(),
            "symbol": "BTC",
            "side": side,
            "size": Decimal(str(size)),
            "price": Decimal(str(price)),
        }
        if correlation is not None:
            payload["correlation_id"] = correlation
        return Fill(**payload)

    def test_closing_a_position_produces_a_closure(self) -> None:
        """Going flat emits a closure carrying the realised PnL."""
        tracker = PortfolioTracker(starting_equity=Decimal("10000"))
        assert tracker.apply_fill(self.fill(Side.BUY, 1, 100)) is None
        closure = tracker.apply_fill(self.fill(Side.SELL, 1, 110))
        assert closure is not None
        assert closure.realized_pnl == Decimal("10")
        assert closure.was_profitable is True

    def test_closure_carries_the_opening_correlation_id(self) -> None:
        """Attribution needs the id of the chain that *opened* the trade."""
        tracker = PortfolioTracker(starting_equity=Decimal("10000"))
        opening = uuid4()
        tracker.apply_fill(self.fill(Side.BUY, 1, 100, correlation=opening))
        closure = tracker.apply_fill(self.fill(Side.SELL, 1, 110, correlation=uuid4()))
        assert closure is not None
        assert closure.opening_correlation_id == opening

    def test_partial_close_does_not_emit_a_closure(self) -> None:
        """A closure means flat, not merely smaller."""
        tracker = PortfolioTracker(starting_equity=Decimal("10000"))
        tracker.apply_fill(self.fill(Side.BUY, 2, 100))
        assert tracker.apply_fill(self.fill(Side.SELL, 1, 110)) is None

    def test_exit_reason_comes_from_the_closing_order(self) -> None:
        """The monitor's reason is carried through to the closure."""
        tracker = PortfolioTracker(starting_equity=Decimal("10000"))
        tracker.apply_fill(self.fill(Side.BUY, 1, 100))
        exit_order = OrderIntent(
            source="test",
            order_id=uuid4(),
            decision_id=uuid4(),
            symbol="BTC",
            side=Side.SELL,
            size=Decimal("1"),
            reduce_only=True,
            exit_reason=ExitReason.TRAILING_STOP.value,
        )
        closure = tracker.apply_fill(self.fill(Side.SELL, 1, 110), order=exit_order)
        assert closure is not None
        assert closure.exit_reason is ExitReason.TRAILING_STOP

    def test_opening_order_levels_attach_to_the_position(self) -> None:
        """Protective levels are recorded on the position, not lost."""
        tracker = PortfolioTracker(starting_equity=Decimal("10000"))
        order = OrderIntent(
            source="test",
            order_id=uuid4(),
            decision_id=uuid4(),
            symbol="BTC",
            side=Side.BUY,
            size=Decimal("1"),
            stop_price=Decimal("95"),
            take_profit_price=Decimal("115"),
            trailing_stop_pct=2.0,
        )
        tracker.apply_fill(self.fill(Side.BUY, 1, 100), order=order)
        position = tracker.snapshot().position_for("BTC")
        assert position is not None
        assert position.stop_price == Decimal("95")
        assert position.take_profit_price == Decimal("115")
        assert position.trailing_stop_pct == 2.0

    async def test_agent_publishes_closure_before_snapshot(self) -> None:
        """Downstream sees the closure first, then the resulting state."""
        agent = PortfolioManagerAgent()
        await agent.handle(Topics.FILLS, self.fill(Side.BUY, 1, 100))
        publications = await agent.handle(Topics.FILLS, self.fill(Side.SELL, 1, 110))
        assert [topic for topic, _ in publications] == [
            Topics.POSITION_CLOSURES,
            Topics.PORTFOLIO_SNAPSHOTS,
        ]


class TestPortfolioPersistence:
    """State must survive a process restart."""

    def test_restore_rebuilds_cash_and_positions(self) -> None:
        """A restarted tracker resumes exactly where it left off."""
        original = PortfolioTracker(starting_equity=Decimal("10000"))
        original.apply_fill(
            Fill(
                source="test",
                fill_id=uuid4(),
                order_id=uuid4(),
                symbol="BTC",
                side=Side.BUY,
                size=Decimal("2"),
                price=Decimal("100"),
                fee=Decimal("1"),
            )
        )
        snapshot = original.snapshot()

        restored = PortfolioTracker(starting_equity=Decimal("10000"))
        restored.restore(
            state={
                "cash": original.cash,
                "starting_equity": original.starting_equity,
                "day_start_equity": original.day_start_equity,
                "realized_pnl": original.realized_pnl,
                "fees_paid": original.fees_paid,
                "trade_count": original.trade_count,
                "win_count": original.win_count,
                "loss_count": original.loss_count,
                "day_of": original.day,
            },
            positions=snapshot.positions,
        )

        assert restored.cash == original.cash
        assert restored.equity == original.equity
        assert restored.snapshot().open_position_count == 1
        assert restored.snapshot().position_for("BTC") is not None

    def test_restored_position_can_still_be_closed(self) -> None:
        """A position recovered from the database still books PnL correctly."""
        original = PortfolioTracker(starting_equity=Decimal("10000"))
        original.apply_fill(
            Fill(
                source="test",
                fill_id=uuid4(),
                order_id=uuid4(),
                symbol="BTC",
                side=Side.BUY,
                size=Decimal("1"),
                price=Decimal("100"),
            )
        )
        restored = PortfolioTracker(starting_equity=Decimal("10000"))
        restored.restore(state=None, positions=original.snapshot().positions)

        closure = restored.apply_fill(
            Fill(
                source="test",
                fill_id=uuid4(),
                order_id=uuid4(),
                symbol="BTC",
                side=Side.SELL,
                size=Decimal("1"),
                price=Decimal("120"),
            )
        )
        assert closure is not None
        assert closure.realized_pnl == Decimal("20")

    async def test_repository_writes_positions_and_deletes_on_close(self) -> None:
        """The repository issues an upsert on open and a delete on close."""
        database = FakeDatabase()
        repository = PortfolioRepository(database)
        tracker = PortfolioTracker(starting_equity=Decimal("10000"))
        tracker.apply_fill(
            Fill(
                source="test",
                fill_id=uuid4(),
                order_id=uuid4(),
                symbol="BTC",
                side=Side.BUY,
                size=Decimal("1"),
                price=Decimal("100"),
            )
        )
        position = tracker.snapshot().position_for("BTC")
        assert position is not None
        await repository.save_position(position)
        await repository.delete_position("BTC")

        assert database.statements_containing("INSERT INTO positions")
        assert database.statements_containing("DELETE FROM positions")

    async def test_closure_is_persisted(self) -> None:
        """Closures are appended to the terminal-state table."""
        database = FakeDatabase()
        await PortfolioRepository(database).record_closure(make_closure(correlation_id=uuid4()))
        assert database.statements_containing("INSERT INTO position_closures")


class TestCalibration:
    """Confidence calibration measurement."""

    def test_perfect_calibration_has_zero_error(self) -> None:
        """Forecasts that match reality score an ECE of zero."""
        report = analyze_calibration([1.0] * 10, [True] * 10, bin_count=2)
        assert report.ece == pytest.approx(0.0)
        assert report.brier == pytest.approx(0.0)

    def test_overconfidence_is_detected(self) -> None:
        """Claiming 0.9 and hitting 0.5 reports positive overconfidence."""
        report = analyze_calibration([0.9] * 10, [True] * 5 + [False] * 5)
        assert report.overconfidence == pytest.approx(0.4)
        assert report.ece == pytest.approx(0.4)

    def test_underconfidence_is_negative(self) -> None:
        """Claiming less than delivered reports negative overconfidence."""
        report = analyze_calibration([0.4] * 10, [True] * 10)
        assert report.overconfidence == pytest.approx(-0.6)

    def test_coin_flip_scores_the_baseline_brier(self) -> None:
        """A 0.5 forecast always scores 0.25."""
        report = analyze_calibration([0.5] * 8, [True] * 4 + [False] * 4)
        assert report.brier == pytest.approx(0.25)

    def test_bins_partition_the_samples(self) -> None:
        """Every sample lands in exactly one bucket."""
        report = analyze_calibration([0.1, 0.3, 0.5, 0.7, 0.9], [True] * 5, bin_count=5)
        assert sum(bucket.count for bucket in report.bins) == 5

    def test_empty_input_is_an_empty_report(self) -> None:
        """No samples means no report rather than a crash."""
        report = analyze_calibration([], [])
        assert report.sample_count == 0
        assert report.bins == ()

    def test_mismatched_lengths_rejected(self) -> None:
        """Confidences and outcomes must line up."""
        with pytest.raises(ValueError, match="same length"):
            analyze_calibration([0.5], [True, False])

    def test_bin_count_must_be_positive(self) -> None:
        """A non-positive bin count is invalid."""
        with pytest.raises(ValueError, match="bin_count"):
            analyze_calibration([0.5], [True], bin_count=0)
