"""Historical funding-rate backfill from the Hyperliquid REST API.

Funding is the one non-price series in this study that Hyperliquid exposes
historically::

    POST https://api.hyperliquid.xyz/info
    {"type": "fundingHistory", "coin": "BTC", "startTime": <epoch ms>}

Rates settle hourly.  Order book depth and open interest are **not** available
historically -- only live snapshots -- which is why the two signals that depend
on them cannot be backtested; see :mod:`services.research.signals`.
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
from libs.schemas.market import FundingRate
from services.data_ingestion.market_data.parsing import PayloadError, to_datetime, to_decimal

logger = logging.getLogger(__name__)

SERVICE_NAME: Final[str] = "hyperliquid_funding"

FUNDING_INTERVAL_S: Final[int] = 3_600
"""Hyperliquid settles funding once per hour."""

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



def parse_funding(payload: dict[str, Any], *, source: str = SERVICE_NAME) -> FundingRate:
    """Convert one ``fundingHistory`` entry into a :class:`FundingRate`.

    Payload shape::

        {"coin": "BTC", "fundingRate": "0.0000125",
         "premium": "-0.0003401568", "time": 1788627600018}

    Args:
        payload: One entry from the funding history response.
        source: Event source to stamp on the event.

    Returns:
        The parsed funding observation.

    Raises:
        PayloadError: If a required field is missing or malformed.
    """
    symbol = payload.get("coin")
    if not isinstance(symbol, str):
        raise PayloadError(f"Funding payload missing coin: {payload!r}")
    raw_premium = payload.get("premium")
    return FundingRate(
        source=source,
        symbol=symbol,
        funding_rate=to_decimal(payload.get("fundingRate"), "fundingRate"),
        premium=to_decimal(raw_premium, "premium") if raw_premium is not None else None,
        occurred_at=to_datetime(payload.get("time"), "time"),
    )


class HyperliquidFundingClient:
    """Fetches historical funding rates.

    Args:
        config: Hyperliquid settings override, mainly for tests.
        client: Injected HTTP client, mainly for tests.
        request_pause_s: Delay between windows.
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

    async def __aenter__(self) -> HyperliquidFundingClient:
        """Open an HTTP client if one was not injected."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Close the HTTP client if this instance owns it."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch_range(
        self,
        symbol: str,
        *,
        start: datetime,
        end: datetime | None = None,
        max_windows: int = 400,
    ) -> tuple[FundingRate, ...]:
        """Fetch funding history by walking forward in windows.

        Args:
            symbol: Coin to fetch.
            start: Inclusive range start.
            end: Exclusive range end; defaults to now.
            max_windows: Safety bound on the number of requests.

        Returns:
            Deduplicated funding observations, oldest first.

        Raises:
            RuntimeError: If used outside the async context manager.
        """
        if self._client is None:
            raise RuntimeError("HyperliquidFundingClient used outside its context manager.")

        finish = end or datetime.now(tz=UTC)
        collected: dict[datetime, FundingRate] = {}
        cursor = start

        # No endTime is sent. The endpoint returns the earliest available page
        # at or after startTime, so a request from before the contract listed
        # simply returns the first real data rather than an empty window. Pinning
        # an endTime instead forces empty responses for every window that
        # predates the listing, which previously made long backfills return
        # nothing at all.
        for _ in range(max_windows):
            if cursor >= finish:
                break
            body = {
                "type": "fundingHistory",
                "coin": symbol,
                "startTime": int(cursor.timestamp() * 1000),
            }
            data = await _request_with_backoff(self._client, self._config.rest_url, body)
            if not isinstance(data, list) or not data:
                break

            fresh = 0
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                try:
                    point = parse_funding(entry)
                except PayloadError as exc:
                    logger.warning("Skipping malformed funding entry", extra={"error": str(exc)})
                    continue
                if point.occurred_at <= finish and point.occurred_at not in collected:
                    collected[point.occurred_at] = point
                    fresh += 1

            if not collected:
                break
            newest = max(collected)
            logger.info(
                "Fetched funding window",
                extra={"symbol": symbol, "new": fresh, "total": len(collected)},
            )
            if fresh == 0 or newest <= cursor:
                break
            cursor = newest + timedelta(seconds=FUNDING_INTERVAL_S)
            if self._pause:
                await asyncio.sleep(self._pause)

        return tuple(collected[key] for key in sorted(collected))


def write_funding_cache(path: Path, points: Sequence[FundingRate]) -> int:
    """Write funding observations to a JSONL cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for point in points:
            handle.write(serialize(point).decode("utf-8"))
            handle.write("\n")
    return len(points)


def read_funding_cache(path: Path) -> tuple[FundingRate, ...]:
    """Read funding observations back from a JSONL cache.

    Raises:
        FileNotFoundError: If the cache does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"No funding cache at {path}.")
    points: list[FundingRate] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                points.append(FundingRate.model_validate(json.loads(stripped)))
    points.sort(key=lambda point: point.occurred_at)
    return tuple(points)


def funding_cache_path(root: Path, symbol: str) -> Path:
    """Return the conventional cache path for a symbol's funding history."""
    return root / f"{symbol.upper()}_funding.jsonl"


__all__ = [
    "FUNDING_INTERVAL_S",
    "HyperliquidFundingClient",
    "funding_cache_path",
    "parse_funding",
    "read_funding_cache",
    "write_funding_cache",
]
