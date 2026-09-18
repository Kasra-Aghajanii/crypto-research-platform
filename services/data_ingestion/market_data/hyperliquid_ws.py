"""Hyperliquid WebSocket transport shared by the market data collectors.

Wraps ``wss://api.hyperliquid.xyz/ws`` with the plumbing every subscriber needs:
subscription management, keepalive pings, dead-socket detection and exponential
reconnect with re-subscription.  Consumers just iterate decoded messages.

Wire format (public market data, no authentication required)::

    -> {"method": "subscribe", "subscription": {"type": "candle",
                                                "coin": "BTC", "interval": "1m"}}
    <- {"channel": "subscriptionResponse", "data": {...}}
    <- {"channel": "candle", "data": {...}}

Reference: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self

import websockets
from websockets.asyncio.client import ClientConnection

from libs.config import HyperliquidSettings, settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Subscription:
    """One Hyperliquid WebSocket subscription request.

    Attributes:
        type: Channel type, e.g. ``"candle"``, ``"trades"`` or ``"l2Book"``.
        coin: Hyperliquid coin symbol, e.g. ``"BTC"``.
        interval: Candle interval; only meaningful for the ``candle`` channel.
    """

    type: str
    coin: str
    interval: str | None = None

    def to_payload(self) -> dict[str, Any]:
        """Render the subscription as the exchange's request payload."""
        payload: dict[str, Any] = {"type": self.type, "coin": self.coin}
        if self.interval is not None:
            payload["interval"] = self.interval
        return payload


class HyperliquidWebSocketClient:
    """Resilient WebSocket client for Hyperliquid public market data.

    The client reconnects with exponential backoff and replays its
    subscriptions on every reconnect, so a dropped socket is invisible to the
    caller apart from a gap in the message stream.

    Args:
        subscriptions: Subscriptions to install on connect and after reconnects.
        config: Hyperliquid settings override, mainly for tests.

    Example:
        >>> client = HyperliquidWebSocketClient(
        ...     subscriptions=[Subscription(type="l2Book", coin="BTC")]
        ... )
        >>> async with client:
        ...     async for channel, data in client.messages():
        ...         ...
    """

    def __init__(
        self,
        *,
        subscriptions: Sequence[Subscription],
        config: HyperliquidSettings | None = None,
    ) -> None:
        """Initialise the client with its subscription set."""
        self._config = config or settings.hyperliquid
        self._subscriptions = tuple(subscriptions)
        self._connection: ClientConnection | None = None
        self._ping_task: asyncio.Task[None] | None = None
        self._closing = asyncio.Event()
        self.reconnect_count = 0

    @property
    def subscriptions(self) -> tuple[Subscription, ...]:
        """Return the configured subscriptions."""
        return self._subscriptions

    async def close(self) -> None:
        """Stop the client and close the underlying socket."""
        self._closing.set()
        await self._teardown()

    async def _teardown(self) -> None:
        """Cancel the keepalive task and close the socket, ignoring errors."""
        if self._ping_task is not None:
            self._ping_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._ping_task
            self._ping_task = None
        if self._connection is not None:
            try:
                await self._connection.close()
            except Exception:
                logger.debug("Error while closing WebSocket", exc_info=True)
            self._connection = None

    async def _subscribe_all(self, connection: ClientConnection) -> None:
        """Send every subscription request on a freshly opened connection."""
        for subscription in self._subscriptions:
            request = {"method": "subscribe", "subscription": subscription.to_payload()}
            await connection.send(json.dumps(request))
            logger.info("Subscribed", extra={"subscription": subscription.to_payload()})

    async def _keepalive(self, connection: ClientConnection) -> None:
        """Send application-level pings until the connection drops."""
        try:
            while True:
                await asyncio.sleep(self._config.ws_ping_interval_s)
                await connection.send(json.dumps({"method": "ping"}))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("Keepalive stopped", exc_info=True)

    async def messages(self) -> AsyncIterator[tuple[str, Any]]:
        """Yield ``(channel, data)`` pairs, reconnecting as needed.

        Control frames (``pong`` and ``subscriptionResponse``) are consumed
        internally and never yielded.

        Yields:
            The channel name and its decoded ``data`` payload.
        """
        delay = self._config.reconnect_initial_delay_s
        while not self._closing.is_set():
            try:
                async with websockets.connect(
                    self._config.ws_url,
                    open_timeout=15.0,
                    close_timeout=5.0,
                    max_queue=1024,
                ) as connection:
                    self._connection = connection
                    delay = self._config.reconnect_initial_delay_s
                    logger.info("WebSocket connected", extra={"url": self._config.ws_url})
                    await self._subscribe_all(connection)
                    self._ping_task = asyncio.create_task(self._keepalive(connection))

                    while not self._closing.is_set():
                        raw = await asyncio.wait_for(
                            connection.recv(), timeout=self._config.ws_receive_timeout_s
                        )
                        message = self._decode(raw)
                        if message is None:
                            continue
                        channel, data = message
                        if channel in {"pong", "subscriptionResponse"}:
                            continue
                        if channel == "error":
                            logger.error("Exchange reported an error", extra={"payload": data})
                            continue
                        yield channel, data
            except asyncio.CancelledError:
                raise
            except TimeoutError:
                logger.warning(
                    "No WebSocket traffic within the receive timeout; reconnecting",
                    extra={"timeout_s": self._config.ws_receive_timeout_s},
                )
            except Exception as exc:  # noqa: BLE001 - any failure means reconnect
                logger.warning(
                    "WebSocket connection lost; reconnecting",
                    extra={"error": f"{type(exc).__name__}: {exc}", "retry_in_s": delay},
                )
            finally:
                await self._teardown()

            if self._closing.is_set():
                break
            self.reconnect_count += 1
            await asyncio.sleep(delay)
            delay = min(delay * 2.0, self._config.reconnect_max_delay_s)

    @staticmethod
    def _decode(raw: str | bytes) -> tuple[str, Any] | None:
        """Decode one frame into ``(channel, data)``.

        Args:
            raw: The raw frame.

        Returns:
            The channel and payload, or ``None`` if the frame is unusable.
        """
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning("Discarding non-JSON WebSocket frame")
            return None
        if not isinstance(payload, dict):
            return None
        channel = payload.get("channel")
        if not isinstance(channel, str):
            return None
        return channel, payload.get("data")

    async def __aenter__(self) -> Self:
        """Enter the async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the client on context exit."""
        await self.close()


__all__ = ["HyperliquidWebSocketClient", "Subscription"]
