"""Async Kafka consumer wrapper that yields validated event models."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from types import TracebackType
from typing import Self

from aiokafka import AIOKafkaConsumer

from libs.config import KafkaSettings, settings
from libs.kafka_client.serialization import EventDecodeError, deserialize
from libs.schemas.base import BaseEvent

logger = logging.getLogger(__name__)


class KafkaConsumer:
    """Typed wrapper around :class:`aiokafka.AIOKafkaConsumer`.

    Each subscribed topic is mapped to the event class its payloads decode to.
    Messages that fail validation are logged and skipped rather than crashing
    the consuming service -- a poison message must never take an agent down.

    Example:
        >>> consumer = KafkaConsumer(
        ...     group_id="market_analyst",
        ...     topic_models={Topics.MARKET_CANDLES: Candle},
        ... )
        >>> async with consumer:
        ...     async for topic, event in consumer.events():
        ...         ...
    """

    def __init__(
        self,
        *,
        group_id: str,
        topic_models: Mapping[str, type[BaseEvent]],
        config: KafkaSettings | None = None,
        auto_offset_reset: str | None = None,
    ) -> None:
        """Initialise the consumer.

        Args:
            group_id: Consumer group suffix; the configured prefix is prepended.
            topic_models: Mapping of topic name to the event class it carries.
            config: Kafka settings override, mainly for tests.
            auto_offset_reset: Override for the configured offset reset policy.

        Raises:
            ValueError: If ``topic_models`` is empty.
        """
        if not topic_models:
            raise ValueError("KafkaConsumer requires at least one topic.")
        self._config = config or settings.kafka
        self._group_id = f"{self._config.consumer_group_prefix}.{group_id}"
        self._topic_models = dict(topic_models)
        self._auto_offset_reset = auto_offset_reset or self._config.auto_offset_reset
        self._consumer: AIOKafkaConsumer | None = None

    @property
    def topics(self) -> Sequence[str]:
        """Return the subscribed topic names."""
        return tuple(self._topic_models)

    async def start(self) -> None:
        """Subscribe and join the consumer group. Safe to call more than once."""
        if self._consumer is not None:
            return
        consumer = AIOKafkaConsumer(
            *self._topic_models,
            bootstrap_servers=self._config.bootstrap_servers,
            group_id=self._group_id,
            client_id=self._config.client_id,
            auto_offset_reset=self._auto_offset_reset,
            enable_auto_commit=True,
            max_poll_records=self._config.max_poll_records,
        )
        await consumer.start()
        self._consumer = consumer
        logger.info(
            "Kafka consumer started",
            extra={"group_id": self._group_id, "topics": list(self._topic_models)},
        )

    async def stop(self) -> None:
        """Leave the group and close the underlying consumer."""
        if self._consumer is None:
            return
        await self._consumer.stop()
        self._consumer = None
        logger.info("Kafka consumer stopped", extra={"group_id": self._group_id})

    async def events(self) -> AsyncIterator[tuple[str, BaseEvent]]:
        """Yield ``(topic, event)`` pairs until the consumer is stopped.

        Yields:
            The topic name and the decoded event.

        Raises:
            RuntimeError: If the consumer has not been started.
        """
        if self._consumer is None:
            raise RuntimeError("KafkaConsumer.events called before start().")
        async for message in self._consumer:
            model = self._topic_models.get(message.topic)
            if model is None:
                logger.warning("Message from unsubscribed topic", extra={"topic": message.topic})
                continue
            try:
                event = deserialize(message.value, model)
            except EventDecodeError:
                logger.exception(
                    "Skipping undecodable message",
                    extra={"topic": message.topic, "offset": message.offset},
                )
                continue
            yield message.topic, event

    async def __aenter__(self) -> Self:
        """Start the consumer for use as an async context manager."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Stop the consumer on context exit."""
        await self.stop()


__all__ = ["KafkaConsumer"]
