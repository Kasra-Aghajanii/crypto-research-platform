"""Open interest collector.

Phase 5, item 1b.  Hyperliquid reports open interest only as a live value in
``metaAndAssetCtxs`` -- there is no historical endpoint, so open interest that is
not polled and stored is lost permanently.  This service polls it on a fixed
interval and publishes a :class:`~libs.schemas.market.PerpMetrics` event per
symbol per poll.

The same payload carries mark price, oracle price, premium and the current
funding rate.  They are recorded too: the row costs the same, and premium
against oracle is the live perp-versus-index basis that any carry analysis
needs.

Run with::

    python -m services.data_ingestion.market_data.hyperliquid_perp_metrics
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Sequence
from typing import Any, Final

import httpx

from libs.config import Settings, settings
from libs.kafka_client import KafkaProducer, Topics
from libs.logging_config import configure_logging
from libs.schemas.base import utc_now
from libs.schemas.market import PerpMetrics
from services.data_ingestion.market_data.parsing import PayloadError, to_decimal

logger = logging.getLogger(__name__)

SERVICE_NAME: Final[str] = "hyperliquid_perp_metrics"


def parse_contexts(
    payload: Any, symbols: Sequence[str], *, source: str = SERVICE_NAME
) -> tuple[PerpMetrics, ...]:
    """Convert a ``metaAndAssetCtxs`` response into per-symbol metrics.

    The response is a two-element array: universe metadata, and a parallel array
    of asset contexts.  Position in the universe is the only thing linking a
    context to its coin.

    Args:
        payload: The decoded response.
        symbols: Symbols to extract; others are ignored.
        source: Event source to stamp.

    Returns:
        One :class:`PerpMetrics` per requested symbol that was found.

    Raises:
        PayloadError: If the response is not the expected two-part structure.
    """
    if not isinstance(payload, list) or len(payload) < 2:
        raise PayloadError(f"metaAndAssetCtxs payload is not a two-element array: {payload!r}")
    meta, contexts = payload[0], payload[1]
    universe = meta.get("universe") if isinstance(meta, dict) else None
    if not isinstance(universe, list) or not isinstance(contexts, list):
        raise PayloadError("metaAndAssetCtxs payload missing universe or contexts.")
    if len(universe) != len(contexts):
        raise PayloadError(
            f"universe ({len(universe)}) and contexts ({len(contexts)}) lengths differ."
        )

    wanted = {symbol.upper() for symbol in symbols}
    now = utc_now()
    results: list[PerpMetrics] = []

    for entry, context in zip(universe, contexts, strict=True):
        if not isinstance(entry, dict) or not isinstance(context, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str) or name.upper() not in wanted:
            continue
        try:
            results.append(
                PerpMetrics(
                    source=source,
                    symbol=name,
                    occurred_at=now,
                    open_interest=to_decimal(context.get("openInterest"), "openInterest"),
                    mark_price=to_decimal(context.get("markPx"), "markPx"),
                    oracle_price=_optional(context, "oraclePx"),
                    mid_price=_optional(context, "midPx"),
                    funding_rate=_optional(context, "funding"),
                    premium=_optional(context, "premium"),
                    day_notional_volume=_optional(context, "dayNtlVlm"),
                    day_base_volume=_optional(context, "dayBaseVlm"),
                )
            )
        except PayloadError as exc:
            logger.warning(
                "Skipping malformed asset context", extra={"symbol": name, "error": str(exc)}
            )
    return tuple(results)


def _optional(context: dict[str, Any], field: str) -> Any:
    """Return a decimal field when present and parseable, else ``None``."""
    raw = context.get(field)
    if raw is None:
        return None
    try:
        return to_decimal(raw, field)
    except PayloadError:
        return None


class PerpMetricsCollector:
    """Polls open interest and contract state, publishing each snapshot.

    Args:
        config: Settings override, mainly for tests.
        producer: Kafka producer override, mainly for tests.
        client: HTTP client override, mainly for tests.
    """

    name = SERVICE_NAME

    def __init__(
        self,
        *,
        config: Settings | None = None,
        producer: KafkaProducer | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Build the collector from configuration."""
        self.settings = config or settings
        self._hl = self.settings.hyperliquid
        self._interval = self.settings.recording.perp_metrics_interval_s
        self._producer = producer or KafkaProducer(client_id=SERVICE_NAME)
        self._client = client
        self._owns_client = client is None
        self._stopping = asyncio.Event()
        self.polls = 0
        self.published = 0
        self.failures = 0

    async def poll_once(self) -> tuple[PerpMetrics, ...]:
        """Fetch one snapshot for every configured symbol.

        Returns:
            The parsed metrics.

        Raises:
            RuntimeError: If called before the HTTP client is open.
        """
        if self._client is None:
            raise RuntimeError("PerpMetricsCollector used before start().")
        response = await self._client.post(
            self._hl.rest_url, json={"type": "metaAndAssetCtxs"}
        )
        response.raise_for_status()
        self.polls += 1
        return parse_contexts(response.json(), self._hl.symbols)

    async def start(self) -> None:
        """Open the HTTP client and the Kafka producer."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=20.0)
        await self._producer.start()

    async def stop(self) -> None:
        """Close everything and log a summary."""
        self._stopping.set()
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None
        await self._producer.stop()
        logger.info(
            "Perp metrics collector stopped",
            extra={"polls": self.polls, "published": self.published, "failures": self.failures},
        )

    async def run(self) -> None:
        """Poll on the configured interval until stopped.

        A failed poll is logged and retried on the next tick: this service is
        meant to run for months, so a transient HTTP error must never end it.
        """
        await self.start()
        logger.info(
            "Perp metrics collector starting",
            extra={
                "symbols": self._hl.symbols,
                "interval_s": self._interval,
                "rest_url": self._hl.rest_url,
            },
        )
        try:
            while not self._stopping.is_set():
                try:
                    for item in await self.poll_once():
                        await self._producer.publish(
                            Topics.MARKET_PERP_METRICS, item, key=item.symbol
                        )
                        self.published += 1
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - keep polling across failures
                    self.failures += 1
                    logger.warning(
                        "Perp metrics poll failed",
                        extra={"error": f"{type(exc).__name__}: {exc}"},
                    )
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._stopping.wait(), timeout=self._interval)
        finally:
            await self.stop()


async def main() -> None:
    """Service entrypoint."""
    configure_logging(SERVICE_NAME)
    collector = PerpMetricsCollector()
    try:
        await collector.run()
    except KeyboardInterrupt:  # pragma: no cover - interactive shutdown
        await collector.stop()


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    asyncio.run(main())


__all__ = ["PerpMetricsCollector", "parse_contexts"]
