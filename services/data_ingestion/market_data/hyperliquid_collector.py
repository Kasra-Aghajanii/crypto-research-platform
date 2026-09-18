"""Hyperliquid market data collector -- candles and trade prints.

Phase 2, item 1.  Replaces the Phase 1 Binance WebSocket collector
(``collector.py``): Binance blocks Iranian nationals under KYC/OFAC, whereas
Hyperliquid is a smart contract with public, key-free market data endpoints.

Responsibilities:

* Subscribe to the ``candle`` channel for every configured symbol and interval,
  and to the ``trades`` channel for every configured symbol.
* Translate the exchange payloads into immutable
  :class:`~libs.schemas.market.Candle` and :class:`~libs.schemas.market.TradeTick`
  events.
* Decide when a candle is **closed**.  Hyperliquid streams repeated updates for
  the in-progress candle; a candle is only final once a candle with a later open
  time arrives, so the collector holds the newest candle back and emits it as
  closed on rollover.
* Publish to ``market.candles`` and ``market.trades``, keyed by symbol.

Run with::

    python -m services.data_ingestion.market_data.hyperliquid_collector
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Final

from libs.config import Settings, settings
from libs.kafka_client import KafkaProducer, Topics
from libs.logging_config import configure_logging
from libs.schemas.market import Candle, Side, TradeTick
from services.data_ingestion.market_data.hyperliquid_ws import (
    HyperliquidWebSocketClient,
    Subscription,
)
from services.data_ingestion.market_data.parsing import PayloadError, to_datetime, to_decimal

logger = logging.getLogger(__name__)

SERVICE_NAME: Final[str] = "hyperliquid_collector"

_SIDE_CODES: Final[dict[str, Side]] = {"B": Side.BUY, "A": Side.SELL}
"""Hyperliquid encodes the aggressor as B (bid lifted) or A (ask hit)."""


def parse_candle(payload: dict[str, Any], *, is_closed: bool) -> Candle:
    """Convert a Hyperliquid ``candle`` payload into a :class:`Candle`.

    Payload shape::

        {"t": 1700000000000, "T": 1700000059999, "s": "BTC", "i": "1m",
         "o": "37000.0", "h": "37050.0", "l": "36990.0", "c": "37020.0",
         "v": "12.34", "n": 421}

    Args:
        payload: The ``data`` object from the candle channel.
        is_closed: Whether this candle should be marked final.

    Returns:
        The parsed candle event.

    Raises:
        PayloadError: If a required field is missing or malformed.
    """
    symbol = payload.get("s")
    interval = payload.get("i")
    if not isinstance(symbol, str) or not isinstance(interval, str):
        raise PayloadError(f"Candle payload missing symbol/interval: {payload!r}")
    open_time = to_datetime(payload.get("t"), "t")
    return Candle(
        source=SERVICE_NAME,
        symbol=symbol,
        interval=interval,
        open_time=open_time,
        close_time=to_datetime(payload.get("T"), "T"),
        occurred_at=open_time,
        open=to_decimal(payload.get("o"), "o"),
        high=to_decimal(payload.get("h"), "h"),
        low=to_decimal(payload.get("l"), "l"),
        close=to_decimal(payload.get("c"), "c"),
        volume=to_decimal(payload.get("v", 0), "v"),
        trade_count=int(payload.get("n", 0) or 0),
        is_closed=is_closed,
    )


def parse_trade(payload: dict[str, Any]) -> TradeTick:
    """Convert one Hyperliquid ``trades`` entry into a :class:`TradeTick`.

    Payload shape::

        {"coin": "BTC", "side": "B", "px": "37020.0", "sz": "0.05",
         "time": 1700000000123, "tid": 987654321}

    Args:
        payload: One trade object from the trades channel array.

    Returns:
        The parsed trade event.

    Raises:
        PayloadError: If a required field is missing or malformed.
    """
    symbol = payload.get("coin")
    if not isinstance(symbol, str):
        raise PayloadError(f"Trade payload missing coin: {payload!r}")
    side_code = str(payload.get("side", "")).upper()
    side = _SIDE_CODES.get(side_code)
    if side is None:
        raise PayloadError(f"Unknown trade side {payload.get('side')!r}")
    trade_id = payload.get("tid") or payload.get("hash")
    return TradeTick(
        source=SERVICE_NAME,
        symbol=symbol,
        price=to_decimal(payload.get("px"), "px"),
        size=to_decimal(payload.get("sz"), "sz"),
        side=side,
        trade_id=str(trade_id) if trade_id is not None else None,
        occurred_at=to_datetime(payload.get("time"), "time"),
    )


class HyperliquidCollector:
    """Streams Hyperliquid candles and trades onto the Kafka bus.

    Args:
        config: Settings override, mainly for tests.
        producer: Kafka producer override, mainly for tests.
        client: WebSocket client override, mainly for tests.
    """

    def __init__(
        self,
        *,
        config: Settings | None = None,
        producer: KafkaProducer | None = None,
        client: HyperliquidWebSocketClient | None = None,
    ) -> None:
        """Build the collector and its subscription set."""
        self.settings = config or settings
        self._hl = self.settings.hyperliquid
        self._producer = producer or KafkaProducer(client_id=SERVICE_NAME)
        self._client = client or HyperliquidWebSocketClient(
            subscriptions=self._build_subscriptions(), config=self._hl
        )
        self._open_candles: dict[tuple[str, str], Candle] = {}
        self.candles_published = 0
        self.trades_published = 0

    def _build_subscriptions(self) -> list[Subscription]:
        """Return one candle subscription per symbol/interval, plus trades."""
        subscriptions = [
            Subscription(type="candle", coin=symbol, interval=interval)
            for symbol in self._hl.symbols
            for interval in self._hl.candle_intervals
        ]
        subscriptions.extend(
            Subscription(type="trades", coin=symbol) for symbol in self._hl.symbols
        )
        return subscriptions

    async def _publish_candle(self, candle: Candle) -> None:
        """Publish a candle keyed by symbol and count it."""
        await self._producer.publish(Topics.MARKET_CANDLES, candle, key=candle.symbol)
        self.candles_published += 1

    async def handle_candle(self, payload: dict[str, Any]) -> None:
        """Process one candle update, emitting the previous candle on rollover.

        Hyperliquid re-sends the in-progress candle as it updates.  The
        collector keeps the latest version per ``(symbol, interval)`` and only
        marks it closed once a candle with a later open time arrives, so
        downstream agents never analyse a candle that can still change.

        Args:
            payload: The ``data`` object from the candle channel.
        """
        candle = parse_candle(payload, is_closed=False)
        key = (candle.symbol, candle.interval)
        previous = self._open_candles.get(key)

        if previous is not None and candle.open_time > previous.open_time:
            closed = previous.model_copy(update={"is_closed": True})
            await self._publish_candle(closed)
            logger.debug(
                "Closed candle",
                extra={
                    "symbol": closed.symbol,
                    "interval": closed.interval,
                    "close": str(closed.close),
                },
            )
        elif previous is not None and candle.open_time < previous.open_time:
            logger.debug(
                "Ignoring stale candle update",
                extra={"symbol": candle.symbol, "interval": candle.interval},
            )
            return

        self._open_candles[key] = candle
        if self._hl.publish_unclosed_candles:
            await self._publish_candle(candle)

    async def handle_trades(self, payload: Any) -> None:
        """Process a batch of trade prints.

        Args:
            payload: The ``data`` array from the trades channel.
        """
        entries = payload if isinstance(payload, list) else [payload]
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            try:
                trade = parse_trade(entry)
            except PayloadError:
                logger.warning("Skipping malformed trade", extra={"payload": entry})
                continue
            await self._producer.publish(Topics.MARKET_TRADES, trade, key=trade.symbol)
            self.trades_published += 1

    async def run(self) -> None:
        """Consume the WebSocket stream until cancelled.

        A malformed message is logged and skipped; connection failures are
        handled by the transport, which reconnects and re-subscribes.
        """
        await self._producer.start()
        logger.info(
            "Collector starting",
            extra={
                "symbols": self._hl.symbols,
                "intervals": self._hl.candle_intervals,
                "ws_url": self._hl.ws_url,
                "trading_mode": self.settings.trading_mode.value,
            },
        )
        try:
            async for channel, data in self._client.messages():
                try:
                    if channel == "candle" and isinstance(data, dict):
                        await self.handle_candle(data)
                    elif channel == "trades":
                        await self.handle_trades(data)
                    else:
                        logger.debug("Ignoring channel", extra={"channel": channel})
                except PayloadError as exc:
                    logger.warning(
                        "Skipping malformed payload",
                        extra={"channel": channel, "error": str(exc)},
                    )
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Close the WebSocket and flush the producer."""
        await self._client.close()
        await self._producer.stop()
        logger.info(
            "Collector stopped",
            extra={
                "candles_published": self.candles_published,
                "trades_published": self.trades_published,
            },
        )


async def main() -> None:
    """Service entrypoint for ``python -m ...hyperliquid_collector``."""
    configure_logging(SERVICE_NAME)
    collector = HyperliquidCollector()
    try:
        await collector.run()
    except KeyboardInterrupt:  # pragma: no cover - interactive shutdown
        await collector.stop()


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    asyncio.run(main())


__all__ = ["HyperliquidCollector", "parse_candle", "parse_trade"]
