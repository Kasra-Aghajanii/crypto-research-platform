"""Shared test fixtures and synthetic market data builders."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from libs.schemas.base import BaseEvent
from libs.schemas.market import BookLevel, Candle, OrderBookSnapshot

BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)


class FakeProducer:
    """In-memory stand-in for :class:`libs.kafka_client.KafkaProducer`."""

    def __init__(self) -> None:
        """Start with an empty publication log."""
        self.published: list[tuple[str, BaseEvent, str | None]] = []
        self.started = False

    async def start(self) -> None:
        """Mark the producer started."""
        self.started = True

    async def stop(self) -> None:
        """Mark the producer stopped."""
        self.started = False

    async def publish(self, topic: str, event: BaseEvent, *, key: str | None = None) -> None:
        """Record a publication instead of sending it."""
        self.published.append((topic, event, key))

    def events_on(self, topic: str) -> list[BaseEvent]:
        """Return every event published to ``topic``."""
        return [event for published_topic, event, _ in self.published if published_topic == topic]


@pytest.fixture
def fake_producer() -> FakeProducer:
    """Return a fresh in-memory producer."""
    return FakeProducer()


def make_candle(
    *,
    symbol: str = "BTC",
    interval: str = "15m",
    index: int = 0,
    open_price: float,
    high: float,
    low: float,
    close: float,
    volume: float = 100.0,
    is_closed: bool = True,
) -> Candle:
    """Build one synthetic candle at a deterministic timestamp."""
    open_time = BASE_TIME + timedelta(minutes=15 * index)
    return Candle(
        source="test",
        symbol=symbol,
        interval=interval,
        open_time=open_time,
        close_time=open_time + timedelta(minutes=15),
        occurred_at=open_time,
        open=Decimal(str(open_price)),
        high=Decimal(str(high)),
        low=Decimal(str(low)),
        close=Decimal(str(close)),
        volume=Decimal(str(volume)),
        trade_count=10,
        is_closed=is_closed,
    )


def trending_candles(
    count: int,
    *,
    start: float = 100.0,
    drift: float = 0.5,
    noise: float = 0.4,
    symbol: str = "BTC",
    interval: str = "15m",
    rising_volume: bool = True,
) -> tuple[Candle, ...]:
    """Build a deterministic trending candle series.

    Args:
        count: Number of candles.
        start: Starting price.
        drift: Price change per bar; negative produces a downtrend.
        noise: Amplitude of a deterministic sine wobble.
        symbol: Symbol to stamp on the candles.
        interval: Interval to stamp on the candles.
        rising_volume: Whether volume should grow with the trend.

    Returns:
        Candles oldest first.
    """
    candles = []
    price = start
    for i in range(count):
        wobble = math.sin(i / 3.0) * noise
        open_price = price
        close = price + drift + wobble
        high = max(open_price, close) + abs(noise)
        low = min(open_price, close) - abs(noise)
        volume = 100.0 + (i * 2.0 if rising_volume else 0.0)
        candles.append(
            make_candle(
                symbol=symbol,
                interval=interval,
                index=i,
                open_price=round(open_price, 6),
                high=round(high, 6),
                low=round(max(low, 0.01), 6),
                close=round(close, 6),
                volume=volume,
            )
        )
        price = close
    return tuple(candles)


def make_book(
    *,
    symbol: str = "BTC",
    mid: float = 100.0,
    spread: float = 0.02,
    level_size: float = 1.0,
    levels: int = 5,
    occurred_at: datetime | None = None,
) -> OrderBookSnapshot:
    """Build a symmetric synthetic order book around ``mid``."""
    half = spread / 2.0
    bids = tuple(
        BookLevel(
            source="test",
            price=Decimal(str(round(mid - half - i * spread, 8))),
            size=Decimal(str(level_size)),
            order_count=1,
        )
        for i in range(levels)
    )
    asks = tuple(
        BookLevel(
            source="test",
            price=Decimal(str(round(mid + half + i * spread, 8))),
            size=Decimal(str(level_size)),
            order_count=1,
        )
        for i in range(levels)
    )
    return OrderBookSnapshot(
        source="test",
        symbol=symbol,
        bids=bids,
        asks=asks,
        occurred_at=occurred_at or datetime.now(tz=UTC),
    )
