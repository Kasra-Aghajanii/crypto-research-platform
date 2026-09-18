"""Hyperliquid L2 order book collector.

Phase 2, item 2.  Replaces the Phase 1 Binance ``orderbook.py``.

Hyperliquid's ``l2Book`` channel pushes a full depth snapshot on every change,
so -- unlike Binance's diff-based depth stream -- there is no local book to
maintain and no sequence-gap resynchronisation to get wrong.  This service
truncates each snapshot to the configured depth, converts it into an immutable
:class:`~libs.schemas.market.OrderBookSnapshot`, and publishes it to
``market.orderbook`` keyed by symbol.

An optional throttle caps how often snapshots are republished per symbol, so a
fast-moving book cannot flood the bus.

Run with::

    python -m services.data_ingestion.market_data.hyperliquid_orderbook
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Final

from libs.config import Settings, settings
from libs.kafka_client import KafkaProducer, Topics
from libs.logging_config import configure_logging
from libs.schemas.market import BookLevel, OrderBookSnapshot
from services.data_ingestion.market_data.hyperliquid_ws import (
    HyperliquidWebSocketClient,
    Subscription,
)
from services.data_ingestion.market_data.parsing import PayloadError, to_datetime, to_decimal

logger = logging.getLogger(__name__)

SERVICE_NAME: Final[str] = "hyperliquid_orderbook"


def parse_levels(raw_levels: Any, *, depth: int) -> tuple[BookLevel, ...]:
    """Convert one side of the book into truncated, immutable levels.

    Args:
        raw_levels: List of ``{"px": ..., "sz": ..., "n": ...}`` objects.
        depth: Maximum number of levels to retain.

    Returns:
        Levels in exchange order (best first), truncated to ``depth``.

    Raises:
        PayloadError: If the payload is not a list of level objects.
    """
    if raw_levels is None:
        return ()
    if not isinstance(raw_levels, list):
        raise PayloadError(f"Book side is not a list: {raw_levels!r}")

    levels: list[BookLevel] = []
    for entry in raw_levels[:depth]:
        if not isinstance(entry, dict):
            raise PayloadError(f"Book level is not an object: {entry!r}")
        levels.append(
            BookLevel(
                source=SERVICE_NAME,
                price=to_decimal(entry.get("px"), "px"),
                size=to_decimal(entry.get("sz"), "sz"),
                order_count=int(entry.get("n", 0) or 0),
            )
        )
    return tuple(levels)


def parse_order_book(payload: dict[str, Any], *, depth: int) -> OrderBookSnapshot:
    """Convert a Hyperliquid ``l2Book`` payload into a snapshot event.

    Payload shape::

        {"coin": "BTC", "time": 1700000000123,
         "levels": [[{"px": "36999", "sz": "1.2", "n": 3}, ...],   # bids
                    [{"px": "37001", "sz": "0.8", "n": 2}, ...]]}  # asks

    Args:
        payload: The ``data`` object from the l2Book channel.
        depth: Maximum levels retained per side.

    Returns:
        The parsed order book snapshot.

    Raises:
        PayloadError: If the payload is missing its coin or levels.
    """
    symbol = payload.get("coin")
    if not isinstance(symbol, str):
        raise PayloadError(f"Order book payload missing coin: {payload!r}")

    levels = payload.get("levels")
    if not isinstance(levels, list) or len(levels) < 2:
        raise PayloadError(f"Order book payload missing both sides: {payload!r}")

    raw_time = payload.get("time")
    occurred_at = to_datetime(raw_time, "time") if raw_time is not None else None
    snapshot_kwargs: dict[str, Any] = {
        "source": SERVICE_NAME,
        "symbol": symbol,
        "bids": parse_levels(levels[0], depth=depth),
        "asks": parse_levels(levels[1], depth=depth),
    }
    if occurred_at is not None:
        snapshot_kwargs["occurred_at"] = occurred_at
    return OrderBookSnapshot(**snapshot_kwargs)


class HyperliquidOrderBookCollector:
    """Streams Hyperliquid L2 book snapshots onto the Kafka bus.

    Args:
        config: Settings override, mainly for tests.
        producer: Kafka producer override, mainly for tests.
        client: WebSocket client override, mainly for tests.
        min_publish_interval_s: Per-symbol throttle. ``0`` publishes every update.
    """

    def __init__(
        self,
        *,
        config: Settings | None = None,
        producer: KafkaProducer | None = None,
        client: HyperliquidWebSocketClient | None = None,
        min_publish_interval_s: float = 0.5,
    ) -> None:
        """Build the collector and subscribe to every configured symbol."""
        self.settings = config or settings
        self._hl = self.settings.hyperliquid
        self._depth = self._hl.orderbook_depth
        self._min_interval = max(0.0, min_publish_interval_s)
        self._producer = producer or KafkaProducer(client_id=SERVICE_NAME)
        self._client = client or HyperliquidWebSocketClient(
            subscriptions=[Subscription(type="l2Book", coin=symbol) for symbol in self._hl.symbols],
            config=self._hl,
        )
        self._last_published: dict[str, float] = {}
        self.snapshots_published = 0
        self.snapshots_throttled = 0

    def _should_publish(self, symbol: str, *, now: float | None = None) -> bool:
        """Return whether the throttle allows publishing for ``symbol`` now."""
        if self._min_interval <= 0.0:
            return True
        current = now if now is not None else time.monotonic()
        last = self._last_published.get(symbol)
        if last is not None and (current - last) < self._min_interval:
            return False
        self._last_published[symbol] = current
        return True

    async def handle_book(self, payload: dict[str, Any]) -> None:
        """Parse and publish one book snapshot, subject to the throttle.

        A snapshot with an empty side is dropped: it cannot produce a mid price
        and would only mislead the paper broker's fill model.

        Args:
            payload: The ``data`` object from the l2Book channel.
        """
        snapshot = parse_order_book(payload, depth=self._depth)
        if not snapshot.bids or not snapshot.asks:
            logger.warning("Dropping one-sided book snapshot", extra={"symbol": snapshot.symbol})
            return
        if not self._should_publish(snapshot.symbol):
            self.snapshots_throttled += 1
            return
        await self._producer.publish(Topics.MARKET_ORDERBOOK, snapshot, key=snapshot.symbol)
        self.snapshots_published += 1
        logger.debug(
            "Published book snapshot",
            extra={
                "symbol": snapshot.symbol,
                "best_bid": str(snapshot.best_bid),
                "best_ask": str(snapshot.best_ask),
                "spread_bps": float(snapshot.spread_bps or 0),
            },
        )

    async def run(self) -> None:
        """Consume the l2Book stream until cancelled."""
        await self._producer.start()
        logger.info(
            "Order book collector starting",
            extra={
                "symbols": self._hl.symbols,
                "depth": self._depth,
                "throttle_s": self._min_interval,
                "ws_url": self._hl.ws_url,
            },
        )
        try:
            async for channel, data in self._client.messages():
                if channel != "l2Book" or not isinstance(data, dict):
                    continue
                try:
                    await self.handle_book(data)
                except PayloadError as exc:
                    logger.warning("Skipping malformed book payload", extra={"error": str(exc)})
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Close the WebSocket and flush the producer."""
        await self._client.close()
        await self._producer.stop()
        logger.info(
            "Order book collector stopped",
            extra={
                "snapshots_published": self.snapshots_published,
                "snapshots_throttled": self.snapshots_throttled,
            },
        )


async def main() -> None:
    """Service entrypoint for ``python -m ...hyperliquid_orderbook``."""
    configure_logging(SERVICE_NAME)
    collector = HyperliquidOrderBookCollector()
    try:
        await collector.run()
    except KeyboardInterrupt:  # pragma: no cover - interactive shutdown
        await collector.stop()


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    asyncio.run(main())


__all__ = ["HyperliquidOrderBookCollector", "parse_levels", "parse_order_book"]
