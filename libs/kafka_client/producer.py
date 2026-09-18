"""Async Kafka producer wrapper used by every publishing service."""

from __future__ import annotations

import logging
from types import TracebackType
from typing import Self

from aiokafka import AIOKafkaProducer

from libs.config import KafkaSettings, settings
from libs.kafka_client.serialization import encode_key, serialize
from libs.schemas.base import BaseEvent

logger = logging.getLogger(__name__)


class KafkaProducer:
    """Thin, typed wrapper around :class:`aiokafka.AIOKafkaProducer`.

    The wrapper owns serialisation so that callers only ever hand it immutable
    :class:`~libs.schemas.base.BaseEvent` instances.

    Example:
        >>> async with KafkaProducer(client_id="market_analyst") as producer:
        ...     await producer.publish(Topics.AGENT_SIGNALS, signal, key=signal.symbol)
    """

    def __init__(
        self,
        *,
        client_id: str | None = None,
        config: KafkaSettings | None = None,
    ) -> None:
        """Initialise the producer.

        Args:
            client_id: Kafka client id; defaults to the configured platform id.
            config: Kafka settings override, mainly for tests.
        """
        self._config = config or settings.kafka
        self._client_id = client_id or self._config.client_id
        self._producer: AIOKafkaProducer | None = None

    async def start(self) -> None:
        """Connect to the broker. Safe to call more than once."""
        if self._producer is not None:
            return
        producer = AIOKafkaProducer(
            bootstrap_servers=self._config.bootstrap_servers,
            client_id=self._client_id,
            linger_ms=self._config.producer_linger_ms,
            enable_idempotence=True,
            acks="all",
        )
        await producer.start()
        self._producer = producer
        logger.info(
            "Kafka producer started", extra={"client_id": self._client_id, "role": "producer"}
        )

    async def stop(self) -> None:
        """Flush and close the underlying producer."""
        if self._producer is None:
            return
        await self._producer.stop()
        self._producer = None
        logger.info("Kafka producer stopped", extra={"client_id": self._client_id})

    async def publish(self, topic: str, event: BaseEvent, *, key: str | None = None) -> None:
        """Publish one event.

        Args:
            topic: Destination topic.
            event: Event to publish.
            key: Partition key. Pass the symbol to keep per-symbol ordering.

        Raises:
            RuntimeError: If the producer has not been started.
        """
        if self._producer is None:
            raise RuntimeError("KafkaProducer.publish called before start().")
        await self._producer.send_and_wait(topic, value=serialize(event), key=encode_key(key))
        logger.debug(
            "Published event",
            extra={"topic": topic, "event_type": event.event_type, "key": key},
        )

    async def publish_many(
        self, topic: str, events: list[BaseEvent], *, key: str | None = None
    ) -> None:
        """Publish a batch of events to the same topic.

        Args:
            topic: Destination topic.
            events: Events to publish, in order.
            key: Partition key shared by the batch.
        """
        for event in events:
            await self.publish(topic, event, key=key)

    async def __aenter__(self) -> Self:
        """Start the producer for use as an async context manager."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Stop the producer on context exit."""
        await self.stop()


__all__ = ["KafkaProducer"]
