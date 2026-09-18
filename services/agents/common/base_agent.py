"""BaseAgent: the shared lifecycle for every service on the bus.

Two layers live here:

``BaseAgent``
    Consume/produce lifecycle, graceful shutdown, and per-message error
    isolation.  A failure while handling one message never takes the process
    down and never blocks the next message.

``SignalAgent``
    The contract for *analysis* agents.  It enforces two architecture rules from
    ``CLAUDE.md``: the agent receives a pre-loaded
    :class:`~services.agents.common.context.AgentContext` (it never fetches), and
    every invocation publishes exactly one ``AgentSignal`` -- a neutral one with
    ``confidence == 0`` when the agent errors.

Execution-side services (risk manager, paper broker, portfolio tracker) extend
``BaseAgent`` directly: their output contract is a verdict, an order or a
snapshot rather than an ``AgentSignal``.
"""

from __future__ import annotations

import abc
import asyncio
import contextlib
import logging
import signal
from collections.abc import Mapping, Sequence

from libs.config import Settings, settings
from libs.kafka_client import KafkaConsumer, KafkaProducer, Topics
from libs.schemas.base import BaseEvent
from libs.schemas.market import Candle, OrderBookSnapshot, TradeTick
from libs.schemas.portfolio import PortfolioSnapshot
from libs.schemas.signals import AgentSignal
from services.agents.common.context import AgentContext, MarketContextBuilder

logger = logging.getLogger(__name__)

Publication = tuple[str, BaseEvent]
"""A ``(topic, event)`` pair the runtime should publish."""


class BaseAgent(abc.ABC):
    """Kafka consume/produce lifecycle shared by every agent and service.

    Subclasses declare which topics they read via :attr:`input_topics` and
    implement :meth:`handle`, returning the events to publish.  The runtime owns
    connection management, ordering (messages are processed one at a time per
    consumer) and error isolation.

    Attributes:
        name: Stable agent identifier used for the consumer group and logs.
        version: Semantic version of the agent implementation.
    """

    name: str = "base_agent"
    version: str = "0.1.0"

    def __init__(self, *, config: Settings | None = None) -> None:
        """Initialise the agent.

        Args:
            config: Settings override, mainly for tests.
        """
        self.settings = config or settings
        self.log = logging.getLogger(self.name)
        self._producer = KafkaProducer(client_id=self.name)
        self._consumer = KafkaConsumer(group_id=self.name, topic_models=self.input_topics)
        self._stopping = asyncio.Event()
        self._started = False

    @property
    @abc.abstractmethod
    def input_topics(self) -> Mapping[str, type[BaseEvent]]:
        """Return the topics this agent consumes, mapped to their event classes."""

    @abc.abstractmethod
    async def handle(self, topic: str, event: BaseEvent) -> Sequence[Publication]:
        """Process one inbound event.

        Args:
            topic: Topic the event arrived on.
            event: The decoded event.

        Returns:
            Zero or more ``(topic, event)`` pairs to publish.
        """

    async def on_start(self) -> None:  # noqa: B027 - optional hook, not abstract
        """Hook for subclass startup work. Runs after Kafka is connected."""

    async def on_stop(self) -> None:  # noqa: B027 - optional hook, not abstract
        """Hook for subclass cleanup. Runs before Kafka is disconnected."""

    async def on_error(self, topic: str, event: BaseEvent, exc: Exception) -> Sequence[Publication]:
        """Handle a failure raised by :meth:`handle`.

        The default implementation logs and swallows the error.  ``SignalAgent``
        overrides it to publish the mandatory neutral signal.

        Args:
            topic: Topic the failing event arrived on.
            event: The event being processed when the error occurred.
            exc: The raised exception.

        Returns:
            Zero or more compensating publications.
        """
        self.log.exception(
            "Unhandled error while processing event",
            extra={"topic": topic, "event_type": event.event_type, "error": str(exc)},
        )
        return ()

    async def publish(self, publications: Sequence[Publication], *, key: str | None = None) -> None:
        """Publish a batch of ``(topic, event)`` pairs.

        Args:
            publications: Pairs to publish, in order.
            key: Partition key; defaults to the event's symbol when it has one.
        """
        for topic, event in publications:
            partition_key = key or getattr(event, "symbol", None)
            await self._producer.publish(topic, event, key=partition_key)

    async def start(self) -> None:
        """Connect to Kafka and run subclass startup."""
        if self._started:
            return
        await self._producer.start()
        await self._consumer.start()
        self._started = True
        await self.on_start()
        self.log.info(
            "Agent started",
            extra={
                "agent": self.name,
                "version": self.version,
                "topics": list(self.input_topics),
                "trading_mode": self.settings.trading_mode.value,
            },
        )

    async def stop(self) -> None:
        """Run subclass cleanup and disconnect from Kafka."""
        self._stopping.set()
        if not self._started:
            return
        await self.on_stop()
        await self._consumer.stop()
        await self._producer.stop()
        self._started = False
        self.log.info("Agent stopped", extra={"agent": self.name})

    async def run(self) -> None:
        """Run the consume loop until cancelled or stopped.

        Each message is handled in isolation: an exception is routed to
        :meth:`on_error` and the loop continues with the next message.
        """
        await self.start()
        try:
            async for topic, event in self._consumer.events():
                if self._stopping.is_set():
                    break
                try:
                    publications = await self.handle(topic, event)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - isolation is the point
                    publications = await self.on_error(topic, event, exc)
                if publications:
                    await self.publish(publications)
        finally:
            await self.stop()

    @classmethod
    async def main(cls) -> None:
        """Entrypoint helper: run the agent until SIGINT/SIGTERM.

        Signal handlers are installed when the platform supports them (they are
        unavailable on Windows event loops, where Ctrl-C surfaces as
        ``KeyboardInterrupt`` instead).
        """
        agent = cls()
        loop = asyncio.get_running_loop()
        for sig_name in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, sig_name, None)
            if sig is None:
                continue
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, lambda: asyncio.create_task(agent.stop()))
        try:
            await agent.run()
        except KeyboardInterrupt:  # pragma: no cover - interactive shutdown
            await agent.stop()


class SignalAgent(BaseAgent):
    """Base class for analysis agents that publish ``AgentSignal`` events.

    The runtime feeds market data into a
    :class:`~services.agents.common.context.MarketContextBuilder`, and calls
    :meth:`analyze` with a pre-loaded context each time a candle closes on a
    trigger interval.  Subclasses implement only :meth:`analyze`.
    """

    trigger_intervals: tuple[str, ...] = ()
    """Intervals whose close triggers analysis. Empty means every interval."""

    consume_order_book: bool = True
    """Whether the agent's context should include order book snapshots."""

    consume_trades: bool = False
    """Whether the agent's context should include the trade tape."""

    consume_portfolio: bool = False
    """Whether the agent is permitted to see portfolio state."""

    warmup_candles: int = 200
    """Candles buffered per interval before analysis is attempted."""

    signal_ttl_s: float = 90.0
    """Validity window stamped on emitted signals."""

    def __init__(self, *, config: Settings | None = None) -> None:
        """Initialise the agent and its context builder."""
        super().__init__(config=config)
        self.context_builder = MarketContextBuilder(
            symbols=self.settings.hyperliquid.symbols,
            intervals=self.settings.hyperliquid.candle_intervals,
            max_candles=self.warmup_candles,
        )

    @property
    def input_topics(self) -> Mapping[str, type[BaseEvent]]:
        """Return market data topics, plus portfolio when the agent needs it."""
        topics: dict[str, type[BaseEvent]] = {Topics.MARKET_CANDLES: Candle}
        if self.consume_order_book:
            topics[Topics.MARKET_ORDERBOOK] = OrderBookSnapshot
        if self.consume_trades:
            topics[Topics.MARKET_TRADES] = TradeTick
        if self.consume_portfolio:
            topics[Topics.PORTFOLIO_SNAPSHOTS] = PortfolioSnapshot
        return topics

    @abc.abstractmethod
    async def analyze(self, context: AgentContext) -> AgentSignal | None:
        """Form a view from a pre-loaded context.

        Args:
            context: Read-only market state assembled by the runtime.

        Returns:
            The signal to publish, or ``None`` to stay silent (for example while
            still warming up).
        """

    async def handle(self, topic: str, event: BaseEvent) -> Sequence[Publication]:
        """Route market data into the context builder and trigger analysis."""
        if isinstance(event, OrderBookSnapshot):
            self.context_builder.add_order_book(event)
            return ()
        if isinstance(event, TradeTick):
            self.context_builder.add_trade(event)
            return ()
        if isinstance(event, PortfolioSnapshot):
            self.context_builder.set_portfolio(event)
            return ()
        if not isinstance(event, Candle):
            return ()

        stored = self.context_builder.add_candle(event)
        if not stored or not self._is_trigger(event.interval):
            return ()

        context = self.context_builder.build(
            event.symbol, include_portfolio=self.consume_portfolio
        )
        signal_event = await self.analyze(context)
        if signal_event is None:
            return ()
        return ((Topics.AGENT_SIGNALS, signal_event.with_correlation(event.correlation_id)),)

    def _is_trigger(self, interval: str) -> bool:
        """Return whether a close on ``interval`` should trigger analysis."""
        return not self.trigger_intervals or interval in self.trigger_intervals

    async def on_error(self, topic: str, event: BaseEvent, exc: Exception) -> Sequence[Publication]:
        """Publish the mandatory neutral signal when analysis fails.

        Architecture rule: an agent that errors must still publish an
        ``AgentSignal`` with ``confidence == 0`` so that downstream consumers can
        distinguish "no opinion" from "agent is dead".
        """
        await super().on_error(topic, event, exc)
        symbol = getattr(event, "symbol", None)
        if symbol is None:
            return ()
        neutral = AgentSignal.neutral(
            agent_name=self.name,
            agent_version=self.version,
            symbol=str(symbol),
            source=self.name,
            correlation_id=event.correlation_id,
            error=f"{type(exc).__name__}: {exc}",
            ttl_s=self.signal_ttl_s,
        )
        return ((Topics.AGENT_SIGNALS, neutral),)


__all__ = ["AgentContext", "BaseAgent", "Publication", "SignalAgent"]
