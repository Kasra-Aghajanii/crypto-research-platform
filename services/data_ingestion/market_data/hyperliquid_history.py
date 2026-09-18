"""Historical candle backfill from the Hyperliquid REST API.

Phase 3, item 4 (part one).  The replay engine needs real history; this fetches
it from the public ``/info`` endpoint::

    POST https://api.hyperliquid.xyz/info
    {"type": "candleSnapshot",
     "req": {"coin": "BTC", "interval": "1h",
             "startTime": <epoch ms>, "endTime": <epoch ms>}}

The endpoint caps how many candles it returns per call, so the fetcher walks
forward in windows and stops when a window returns nothing new -- which is also
what protects it from looping forever at the end of available history.

Results can be written to TimescaleDB, to a local JSONL cache, or both.  The
cache exists so historical validation is runnable without a database: the whole
point of this step is to test the analyst against real data, and that should not
be blocked on infrastructure being up.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import httpx

from libs.config import HyperliquidSettings, settings
from libs.kafka_client.serialization import serialize
from libs.schemas.market import Candle
from services.data_ingestion.market_data.parsing import PayloadError, to_datetime, to_decimal

logger = logging.getLogger(__name__)

SERVICE_NAME: Final[str] = "hyperliquid_history"

async def _request_with_backoff(
    client: httpx.AsyncClient, url: str, body: dict[str, Any], *, attempts: int = 6
) -> Any:
    """POST with exponential backoff on rate limiting.

    A long backfill across many symbols issues hundreds of requests, and the
    public endpoint rate-limits well before that finishes.  Failing the whole
    run on a 429 wastes every request already made, so transient throttling is
    retried rather than raised.

    Args:
        client: Open HTTP client.
        url: Endpoint URL.
        body: Request payload.
        attempts: Maximum tries before giving up.

    Returns:
        The decoded JSON response.

    Raises:
        httpx.HTTPStatusError: If every attempt is rate-limited, or on any
            non-retryable status.
    """
    delay = 1.0
    last_error: httpx.HTTPStatusError | None = None
    for _ in range(attempts):
        response = await client.post(url, json=body)
        if response.status_code == 429:
            last_error = httpx.HTTPStatusError(
                "rate limited", request=response.request, response=response
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2.0, 30.0)
            continue
        response.raise_for_status()
        return response.json()
    assert last_error is not None  # noqa: S101 - unreachable unless attempts is 0
    raise last_error


INTERVAL_SECONDS: Final[dict[str, int]] = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1_800,
    "1h": 3_600,
    "2h": 7_200,
    "4h": 14_400,
    "8h": 28_800,
    "12h": 43_200,
    "1d": 86_400,
}
"""Seconds per supported candle interval."""


def interval_seconds(interval: str) -> int:
    """Return the number of seconds in one candle of ``interval``.

    Args:
        interval: Interval identifier such as ``"15m"``.

    Returns:
        Length of the interval in seconds.

    Raises:
        ValueError: If the interval is not supported.
    """
    try:
        return INTERVAL_SECONDS[interval]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported interval {interval!r}; expected one of {sorted(INTERVAL_SECONDS)}."
        ) from exc


def parse_history_candle(payload: dict[str, Any], *, source: str = SERVICE_NAME) -> Candle:
    """Convert one historical candle payload into a :class:`Candle`.

    The REST payload uses the same field names as the WebSocket candle channel.
    Historical candles are always closed.

    Args:
        payload: One candle object from the ``candleSnapshot`` response.
        source: Event source to stamp on the candle.

    Returns:
        The parsed candle, marked closed.

    Raises:
        PayloadError: If a required field is missing or malformed.
    """
    symbol = payload.get("s")
    interval = payload.get("i")
    if not isinstance(symbol, str) or not isinstance(interval, str):
        raise PayloadError(f"History candle missing symbol/interval: {payload!r}")
    open_time = to_datetime(payload.get("t"), "t")
    return Candle(
        source=source,
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
        is_closed=True,
    )


class HyperliquidHistoryClient:
    """Fetches historical candles from the Hyperliquid REST endpoint.

    Args:
        config: Hyperliquid settings override, mainly for tests.
        client: Injected HTTP client, mainly for tests.
        request_pause_s: Delay between windows, to stay polite to the endpoint.
    """

    def __init__(
        self,
        *,
        config: HyperliquidSettings | None = None,
        client: httpx.AsyncClient | None = None,
        request_pause_s: float = 0.25,
    ) -> None:
        """Store configuration without opening a connection."""
        self._config = config or settings.hyperliquid
        self._client = client
        self._owns_client = client is None
        self._pause = max(0.0, request_pause_s)

    async def __aenter__(self) -> HyperliquidHistoryClient:
        """Open an HTTP client if one was not injected."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Close the HTTP client if this instance owns it."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _post(self, payload: dict[str, Any]) -> Any:
        """POST a request body to the info endpoint.

        Raises:
            RuntimeError: If used outside the async context manager.
        """
        if self._client is None:
            raise RuntimeError("HyperliquidHistoryClient used outside its context manager.")
        return await _request_with_backoff(self._client, self._config.rest_url, payload)

    async def fetch_window(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> tuple[Candle, ...]:
        """Fetch one window of candles.

        Args:
            symbol: Coin to fetch.
            interval: Candle interval.
            start: Inclusive window start.
            end: Exclusive window end.

        Returns:
            Parsed candles, oldest first. Malformed entries are skipped.
        """
        body = {
            "type": "candleSnapshot",
            "req": {
                "coin": symbol,
                "interval": interval,
                "startTime": int(start.timestamp() * 1000),
                "endTime": int(end.timestamp() * 1000),
            },
        }
        data = await self._post(body)
        if not isinstance(data, list):
            logger.warning("Unexpected history payload", extra={"symbol": symbol})
            return ()

        candles: list[Candle] = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            try:
                candles.append(parse_history_candle(entry))
            except PayloadError as exc:
                logger.warning("Skipping malformed history candle", extra={"error": str(exc)})
        candles.sort(key=lambda candle: candle.open_time)
        return tuple(candles)

    async def fetch_range(
        self,
        symbol: str,
        interval: str,
        *,
        start: datetime,
        end: datetime | None = None,
        max_windows: int = 200,
        max_empty_windows: int = 40,
    ) -> tuple[Candle, ...]:
        """Fetch a full range by walking forward one window at a time.

        Args:
            symbol: Coin to fetch.
            interval: Candle interval.
            start: Inclusive range start.
            end: Exclusive range end; defaults to now.
            max_windows: Safety bound on the number of requests.
            max_empty_windows: Consecutive empty windows tolerated before giving
                up. Fine intervals are retained only briefly, so the oldest part
                of a long request legitimately returns nothing.

        Returns:
            Deduplicated candles, oldest first.
        """
        step = interval_seconds(interval)
        finish = end or datetime.now(tz=UTC)
        window = timedelta(seconds=step * 4_000)

        collected: dict[datetime, Candle] = {}
        cursor = start
        empty_windows = 0
        for _ in range(max_windows):
            if cursor >= finish:
                break
            window_end = min(cursor + window, finish)
            batch = await self.fetch_window(symbol, interval, cursor, window_end)
            fresh = [candle for candle in batch if candle.open_time not in collected]
            for candle in fresh:
                collected[candle.open_time] = candle

            logger.info(
                "Fetched history window",
                extra={
                    "symbol": symbol,
                    "interval": interval,
                    "window_start": cursor.isoformat(),
                    "received": len(batch),
                    "new": len(fresh),
                    "total": len(collected),
                },
            )

            if not batch:
                # An empty window does not mean history has run out. Hyperliquid
                # retains fine intervals only briefly -- 1m candles go back about
                # two days -- so the *oldest* windows of a long request come back
                # empty while later ones are full. Skip forward and keep going,
                # giving up only after several consecutive empty windows.
                empty_windows += 1
                if empty_windows >= max_empty_windows:
                    logger.info(
                        "Stopping backfill after consecutive empty windows",
                        extra={
                            "symbol": symbol,
                            "interval": interval,
                            "empty_windows": empty_windows,
                        },
                    )
                    break
                cursor = window_end
                if self._pause:
                    await asyncio.sleep(self._pause)
                continue

            empty_windows = 0
            newest = max(candle.open_time for candle in batch)
            if newest <= cursor:
                break
            cursor = newest + timedelta(seconds=step)
            if self._pause:
                await asyncio.sleep(self._pause)

        return tuple(collected[key] for key in sorted(collected))


def write_cache(path: Path, candles: Sequence[Candle]) -> int:
    """Write candles to a JSONL cache file.

    Args:
        path: Destination file; parent directories are created.
        candles: Candles to write, oldest first.

    Returns:
        The number of candles written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for candle in candles:
            handle.write(serialize(candle).decode("utf-8"))
            handle.write("\n")
    return len(candles)


def read_cache(path: Path) -> tuple[Candle, ...]:
    """Read candles back from a JSONL cache file.

    Args:
        path: Cache file to read.

    Returns:
        Candles ordered oldest first.

    Raises:
        FileNotFoundError: If the cache file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"No candle cache at {path}.")
    candles: list[Candle] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                candles.append(Candle.model_validate(json.loads(stripped)))
    candles.sort(key=lambda candle: candle.open_time)
    return tuple(candles)


def cache_path(root: Path, symbol: str, interval: str) -> Path:
    """Return the conventional cache path for one series."""
    return root / f"{symbol.upper()}_{interval}.jsonl"


__all__ = [
    "INTERVAL_SECONDS",
    "HyperliquidHistoryClient",
    "cache_path",
    "interval_seconds",
    "parse_history_candle",
    "read_cache",
    "write_cache",
]
