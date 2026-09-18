"""End-to-end test of the Phase 2 vertical slice.

Wires every stage together in-process -- no Kafka, no exchange -- by passing the
event each stage publishes directly into the next:

    exchange payloads -> collectors -> context -> Market Analyst -> AgentSignal
    -> Decision Engine -> TradeDecision -> Risk Manager -> RiskVerdict +
    OrderIntent -> Paper Broker -> Fill -> Portfolio Tracker -> PortfolioSnapshot

This is the test that proves the contracts line up: every stage consumes exactly
what the previous stage produces.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from libs.kafka_client import Topics
from libs.schemas.learning import PositionClosed
from libs.schemas.market import Side
from libs.schemas.portfolio import PortfolioSnapshot
from libs.schemas.signals import AgentSignal, Direction
from libs.schemas.trading import Fill, OrderIntent, RiskVerdict, TradeDecision
from services.agents.common.context import MarketContextBuilder
from services.agents.execution.paper_broker import PaperBroker
from services.agents.market_analyst.agent import MarketAnalystAgent
from services.agents.portfolio_manager.tracker import PortfolioManagerAgent
from services.agents.risk_manager.agent import RiskManagerAgent
from services.data_ingestion.market_data.hyperliquid_collector import HyperliquidCollector
from services.decision_engine.engine import DecisionEngine
from tests.conftest import FakeProducer, make_book, trending_candles
from tests.test_execution_chain import make_portfolio


class TestVerticalSlice:
    """The full signal-to-fill path."""

    async def test_uptrend_flows_all_the_way_to_a_filled_position(self) -> None:
        """A rising market produces a long signal that becomes a paper position."""
        symbol = "BTC"

        # --- Stage 1/2: ingestion output feeds the agent's context ---------
        builder = MarketContextBuilder(
            symbols=[symbol], intervals=["15m", "1h"], max_candles=250
        )
        for interval in ("15m", "1h"):
            for candle in trending_candles(200, drift=0.6, interval=interval, symbol=symbol):
                builder.add_candle(candle)
        book = make_book(
            symbol=symbol,
            mid=float(trending_candles(200, drift=0.6)[-1].close),
            level_size=100.0,  # deep enough that the risk-sized order fills in full
        )
        builder.add_order_book(book)
        context = builder.build(symbol)

        # --- Stage 3: Market Analyst --------------------------------------
        signal = await MarketAnalystAgent().analyze(context)
        assert isinstance(signal, AgentSignal)
        assert signal.direction is Direction.LONG
        assert signal.confidence > 0

        # --- Stage 4: Decision Engine -------------------------------------
        decision = DecisionEngine()._decide(signal)
        assert isinstance(decision, TradeDecision), "confident long signal should decide"
        assert decision.symbol == symbol

        # --- Stage 5: Risk Manager ----------------------------------------
        risk = RiskManagerAgent()
        verdict, order = risk.evaluate(decision)
        assert isinstance(verdict, RiskVerdict)
        assert verdict.approved, verdict.rationale
        assert isinstance(order, OrderIntent)
        assert order.side is Side.BUY
        assert order.size > 0
        assert order.is_paper is True

        # --- Stage 6: Paper Broker ----------------------------------------
        broker = PaperBroker()
        broker._books[symbol] = book
        fill = broker.simulate_fill(order)
        assert isinstance(fill, Fill)
        assert fill.size == order.size
        assert book.best_ask is not None and fill.price >= book.best_ask
        assert fill.fee > 0

        # --- Stage 7: Portfolio Manager -----------------------------------
        portfolio_agent = PortfolioManagerAgent()
        publications = await portfolio_agent.handle(Topics.FILLS, fill)
        assert len(publications) == 1
        topic, snapshot = publications[0]
        assert topic == Topics.PORTFOLIO_SNAPSHOTS
        assert isinstance(snapshot, PortfolioSnapshot)

        position = snapshot.position_for(symbol)
        assert position is not None
        assert position.direction is Direction.LONG
        assert position.size == fill.size
        assert snapshot.cash == snapshot.starting_equity - fill.fee
        assert snapshot.open_position_count == 1

    async def test_risk_veto_stops_the_chain_before_execution(self) -> None:
        """A vetoed decision produces a verdict but never an order or a fill."""
        risk = RiskManagerAgent()
        risk._portfolio = None
        decision = DecisionEngine()._decide(
            AgentSignal(
                source="test",
                agent_name="market_analyst",
                symbol="BTC",
                direction=Direction.LONG,
                confidence=0.9,
                reference_price=Decimal("100"),
                suggested_stop=Decimal("98"),
            )
        )
        assert decision is not None
        # Force a breach: no leverage headroom left.
        risk._portfolio = make_portfolio(equity=1_000.0, gross=3_000.0)
        verdict, order = risk.evaluate(decision)
        assert not verdict.approved
        assert order is None
        assert verdict.veto_reasons

    async def test_degenerate_stop_falls_back_instead_of_sizing_enormously(self) -> None:
        """A near-zero stop distance must not size a position off ~zero risk."""
        risk = RiskManagerAgent()
        risk._portfolio = make_portfolio(equity=10_000.0)
        decision = DecisionEngine()._decide(
            AgentSignal(
                source="test",
                agent_name="market_analyst",
                symbol="BTC",
                direction=Direction.LONG,
                confidence=0.9,
                reference_price=Decimal("100"),
                suggested_stop=Decimal("99.999999"),
            )
        )
        assert decision is not None
        verdict, order = risk.evaluate(decision)
        assert verdict.approved and order is not None
        assert verdict.stop_price is not None
        # The 0.000001% stop is discarded for the configured default distance.
        assert verdict.stop_price < Decimal("99.9")
        assert verdict.risk_amount > Decimal("1")

    async def test_degraded_signal_never_reaches_execution(self) -> None:
        """The neutral error signal is inert by design."""
        neutral = AgentSignal.neutral(
            agent_name="market_analyst",
            agent_version="1.0.0",
            symbol="BTC",
            source="market_analyst",
            error="boom",
        )
        assert neutral.confidence == 0.0
        assert neutral.degraded is True
        assert DecisionEngine()._decide(neutral) is None

    async def test_agent_error_path_publishes_a_neutral_signal(self) -> None:
        """A failure inside analysis still publishes an AgentSignal, per the rules."""

        class BrokenAnalyst(MarketAnalystAgent):
            """Analyst whose analysis always raises."""

            async def analyze(self, context) -> None:  # type: ignore[override]
                """Fail deliberately."""
                raise RuntimeError("indicator exploded")

        agent = BrokenAnalyst()
        candles = trending_candles(200, drift=0.5, interval="1h")
        for candle in candles[:-1]:
            agent.context_builder.add_candle(candle)

        # The runtime, not handle(), owns error isolation: handle() raises and
        # BaseAgent.run routes the exception to on_error, which must publish the
        # mandatory neutral signal.
        with pytest.raises(RuntimeError, match="indicator exploded"):
            await agent.handle(Topics.MARKET_CANDLES, candles[-1])

        publications = list(
            await agent.on_error(
                Topics.MARKET_CANDLES, candles[-1], RuntimeError("indicator exploded")
            )
        )

        assert len(publications) == 1
        topic, event = publications[0]
        assert topic == Topics.AGENT_SIGNALS
        assert isinstance(event, AgentSignal)
        assert event.confidence == 0.0
        assert event.direction is Direction.FLAT
        assert event.degraded is True

    async def test_round_trip_realises_pnl_through_the_portfolio(self) -> None:
        """Opening and closing a paper position books PnL net of fees."""
        symbol = "BTC"
        broker = PaperBroker()
        portfolio = PortfolioManagerAgent()

        entry_book = make_book(symbol=symbol, mid=100.0, level_size=100.0)
        broker._books[symbol] = entry_book
        buy = OrderIntent(
            source="test",
            order_id=__import__("uuid").uuid4(),
            decision_id=__import__("uuid").uuid4(),
            symbol=symbol,
            side=Side.BUY,
            size=Decimal("1"),
        )
        await portfolio.handle(Topics.FILLS, broker.simulate_fill(buy))

        exit_book = make_book(symbol=symbol, mid=110.0, level_size=100.0)
        broker._books[symbol] = exit_book
        sell = buy.model_copy(
            update={"order_id": __import__("uuid").uuid4(), "side": Side.SELL, "reduce_only": True}
        )
        publications = await portfolio.handle(Topics.FILLS, broker.simulate_fill(sell))

        # A closing fill now publishes the closure first, then the snapshot.
        topics = [topic for topic, _ in publications]
        assert topics == [Topics.POSITION_CLOSURES, Topics.PORTFOLIO_SNAPSHOTS]

        closure = publications[0][1]
        assert isinstance(closure, PositionClosed)
        assert closure.symbol == symbol
        assert closure.was_profitable is True

        snapshot = publications[1][1]
        assert isinstance(snapshot, PortfolioSnapshot)
        assert snapshot.open_position_count == 0
        assert snapshot.realized_pnl > Decimal("9")  # ~10 gross, minus slippage
        assert snapshot.fees_paid > 0
        assert snapshot.equity > snapshot.starting_equity

    async def test_collector_output_is_consumable_by_the_context_builder(self) -> None:
        """The collector's published candles are exactly what the agent buffers."""
        producer = FakeProducer()
        collector = HyperliquidCollector(producer=producer)
        base = {
            "t": 1_700_000_000_000,
            "T": 1_700_000_899_999,
            "s": "BTC",
            "i": "15m",
            "o": "100",
            "h": "101",
            "l": "99",
            "c": "100.5",
            "v": "10",
            "n": 5,
        }
        await collector.handle_candle(base)
        await collector.handle_candle({**base, "t": base["t"] + 900_000, "T": base["T"] + 900_000})

        published = producer.events_on(Topics.MARKET_CANDLES)
        assert len(published) == 1

        builder = MarketContextBuilder(symbols=["BTC"], intervals=["15m"], max_candles=10)
        assert builder.add_candle(published[0]) is True
        assert len(builder.build("BTC").candles_for("15m")) == 1
