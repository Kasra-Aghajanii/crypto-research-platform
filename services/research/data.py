"""Data loading for the research harness.

Candles and funding come from the same cached-JSONL path the Phase 3 validation
used, so a research run is offline and reproducible after the first fetch.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

from libs.schemas.market import Candle, FundingRate
from services.data_ingestion.market_data.hyperliquid_funding import (
    HyperliquidFundingClient,
    funding_cache_path,
    read_funding_cache,
    write_funding_cache,
)
from services.data_ingestion.market_data.hyperliquid_history import (
    HyperliquidHistoryClient,
    cache_path,
    read_cache,
    write_cache,
)

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR: Final[Path] = Path("data/history")


async def load_candles(
    symbol: str,
    interval: str,
    *,
    days: int,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    refresh: bool = False,
) -> tuple[Candle, ...]:
    """Load candles from cache, fetching and caching them if needed.

    Args:
        symbol: Coin to load.
        interval: Candle interval.
        days: How far back to fetch on a miss.
        cache_dir: Cache directory.
        refresh: Ignore any existing cache.

    Returns:
        Candles oldest first.
    """
    start = datetime.now(tz=UTC) - timedelta(days=days)
    path = cache_path(cache_dir, symbol, interval)
    if path.exists() and not refresh:
        cached = read_cache(path)
        # A cache built for a shorter window must not silently answer a longer
        # request: the report would name a date range the study never covered.
        if cached and cached[0].open_time <= start + timedelta(days=1):
            return cached
        logger.info(
            "Cache does not span the requested window; refetching",
            extra={
                "symbol": symbol,
                "requested_start": start.isoformat(),
                "cached_start": cached[0].open_time.isoformat() if cached else None,
            },
        )

    async with HyperliquidHistoryClient() as client:
        candles = await client.fetch_range(symbol, interval, start=start)
    write_cache(path, candles)
    logger.info(
        "Cached candles", extra={"symbol": symbol, "interval": interval, "count": len(candles)}
    )
    return candles


async def load_funding(
    symbol: str,
    *,
    days: int,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    refresh: bool = False,
) -> tuple[FundingRate, ...]:
    """Load funding history from cache, fetching and caching it if needed.

    Args:
        symbol: Coin to load.
        days: How far back to fetch on a miss.
        cache_dir: Cache directory.
        refresh: Ignore any existing cache.

    Returns:
        Funding observations oldest first; empty if the fetch fails.
    """
    start = datetime.now(tz=UTC) - timedelta(days=days)
    path = funding_cache_path(cache_dir, symbol)
    if path.exists() and not refresh:
        cached = read_funding_cache(path)
        if cached and cached[0].occurred_at <= start + timedelta(days=1):
            return cached

    try:
        async with HyperliquidFundingClient() as client:
            points = await client.fetch_range(symbol, start=start)
    except Exception as exc:  # noqa: BLE001 - funding is optional for most signals
        logger.warning(
            "Funding history unavailable",
            extra={"symbol": symbol, "error": f"{type(exc).__name__}: {exc}"},
        )
        return ()
    write_funding_cache(path, points)
    logger.info("Cached funding", extra={"symbol": symbol, "count": len(points)})
    return points


__all__ = ["DEFAULT_CACHE_DIR", "load_candles", "load_funding"]
