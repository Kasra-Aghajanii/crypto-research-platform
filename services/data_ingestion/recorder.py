"""Market data recorder -- writes the irreplaceable streams to TimescaleDB.

Phase 5, item 1.  Order book depth and open interest have no historical endpoint
on Hyperliquid.  Everything else in this platform can be rebuilt from the API
after the fact; these two cannot.  A gap in this service is a permanent gap in
the research data.

Why a separate service rather than writing from the collectors
-------------------------------------------------------------
The collectors already publish to Kafka, and this consumes those topics as a
committed consumer group.  Two different failures matter, and they are handled
differently:

*Recorder restart.*  Offsets are committed as messages are consumed, so a
restarted recorder resumes from where it left off and Redpanda's retention
covers the gap.  A collector writing straight to the database would instead
drop everything produced while the recorder was down.

*Database outage.*  Offsets have already advanced, so Kafka will not replay
those messages.  Rows that fail to write are therefore retained in memory and
retried on the next flush (see :meth:`MarketRecorder.flush`).  That buffer is
bounded, so a long outage still loses the oldest rows -- the honest guarantee is
"survives a transient outage", not "survives any outage".

It also keeps the architecture rule intact: services communicate through Kafka,
not by calling each other.

Batching
--------
Rows are buffered and flushed on whichever comes first, a full batch or the
flush interval.  The buffer is drained on shutdown, so an orderly stop loses
nothing.

Run with::

    python -m services.data_ingestion.recorder
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Mapping, Sequence
from typing import Any, Final

from libs.config import Settings
from libs.kafka_client import Topics
from libs.logging_config import configure_logging
from libs.persistence import CandleRepository, try_connect
from libs.persistence.database import PostgresDatabase
from libs.persistence.recording import (
    FundingRepository,
    OrderBookRepository,
    PerpMetricsRepository,
)
from libs.schemas.base import BaseEvent
from libs.schemas.market import Candle, FundingRate, OrderBookSnapshot, PerpMetrics
from services.agents.common.base_agent import BaseAgent, Publication

logger = logging.getLogger(__name__)

SERVICE_NAME: Final[str] = "market_recorder"


class MarketRecorder(BaseAgent):
    """Persists order book, open interest, funding and candle streams.

    Args:
        config: Settings override, mainly for tests.
        database: Database override; when given, persistence is not auto-connected.
    """

    name = SERVICE_NAME
    version = "1.0.0"

    def __init__(
        self,
        *,
        config: Settings | None = None,
        database: PostgresDatabase | None = None,
    ) -> None:
        """Initialise the recorder with empty buffers."""
        super().__init__(config=config)
        self.params = self.settings.recording
        self._database = database
        self._owns_database = database is None
        self._books: OrderBookRepository | None = None
        self._metrics: PerpMetricsRepository | None = None
        self._funding: FundingRepository | None = None
        self._candles: CandleRepository | None = None

        self._book_buffer: list[OrderBookSnapshot] = []
        self._metric_buffer: list[PerpMetrics] = []
        self._funding_buffer: list[FundingRate] = []
        self._candle_buffer: list[Candle] = []
        self._last_book_at: dict[str, float] = {}

        self._flush_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self.rows_written = 0
        self.books_throttled = 0

    @property
    def input_topics(self) -> Mapping[str, type[BaseEvent]]:
        """Consume every stream worth keeping."""
        return {
            Topics.MARKET_ORDERBOOK: OrderBookSnapshot,
            Topics.MARKET_PERP_METRICS: PerpMetrics,
            Topics.MARKET_CANDLES: Candle,
        }

    async def on_start(self) -> None:
        """Connect the database and start the periodic flush.

        Raises:
            RuntimeError: If persistence is unavailable. Unlike the trading
                services, a recorder with nowhere to write has no degraded mode
                worth running -- it would silently discard irreplaceable data.
        """
        if self._database is None:
            self._database = await try_connect()
        if self._database is None:
            raise RuntimeError(
                "MarketRecorder cannot start without TimescaleDB: its entire purpose is "
                "to persist streams that cannot be backfilled. Start the database first."
            )
        self._books = OrderBookRepository(
            self._database, stored_levels=self.params.stored_book_levels
        )
        self._metrics = PerpMetricsRepository(self._database)
        self._funding = FundingRepository(self._database)
        self._candles = CandleRepository(self._database)
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info(
            "Recorder started",
            extra={
                "book_interval_s": self.params.orderbook_interval_s,
                "batch_size": self.params.batch_size,
                "flush_interval_s": self.params.flush_interval_s,
                "stored_levels": self.params.stored_book_levels,
            },
        )

    async def on_stop(self) -> None:
        """Drain the buffers and close the database."""
        if self._flush_task is not None:
            self._flush_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._flush_task
            self._flush_task = None
        await self.flush()
        if self._owns_database and self._database is not None:
            await self._database.close()
            self._database = None
        logger.info(
            "Recorder stopped",
            extra={"rows_written": self.rows_written, "books_throttled": self.books_throttled},
        )

    async def handle(self, topic: str, event: BaseEvent) -> Sequence[Publication]:
        """Buffer one event for persistence.

        Args:
            topic: Topic the event arrived on.
            event: The decoded event.

        Returns:
            Nothing: the recorder is a sink and publishes no events.
        """
        if isinstance(event, OrderBookSnapshot):
            if self._should_store_book(event):
                self._book_buffer.append(event)
            else:
                self.books_throttled += 1
        elif isinstance(event, PerpMetrics):
            self._metric_buffer.append(event)
        elif isinstance(event, FundingRate):
            self._funding_buffer.append(event)
        elif isinstance(event, Candle) and event.is_closed:
            self._candle_buffer.append(event)

        if self._buffered() >= self.params.batch_size:
            await self.flush()
        return ()

    def _buffered(self) -> int:
        """Return the total number of buffered rows."""
        return (
            len(self._book_buffer)
            + len(self._metric_buffer)
            + len(self._funding_buffer)
            + len(self._candle_buffer)
        )

    def _should_store_book(self, snapshot: OrderBookSnapshot) -> bool:
        """Apply the per-symbol recording throttle.

        The collector publishes far more often than the research needs.  Storing
        every snapshot would grow the table by gigabytes a day for resolution
        nothing consumes.

        Args:
            snapshot: The candidate snapshot.

        Returns:
            Whether to persist it.
        """
        interval = self.params.orderbook_interval_s
        if interval <= 0:
            return True
        stamp = snapshot.occurred_at.timestamp()
        previous = self._last_book_at.get(snapshot.symbol)
        if previous is not None and (stamp - previous) < interval:
            return False
        self._last_book_at[snapshot.symbol] = stamp
        return True

    async def flush(self) -> int:
        """Write every buffered row.

        Each stream is written independently so that a failure in one cannot
        discard the others, and rows that fail to write are put **back** in
        their buffer to be retried on the next flush.  Kafka offsets are
        auto-committed as messages are consumed, so a dropped buffer would be a
        permanent hole in data that cannot be re-fetched -- retrying in memory
        is what actually protects it across a transient database outage.

        The retry buffer is bounded: past ``max_retry_buffer`` rows the oldest
        are dropped, because an unbounded buffer would take the process down
        with it and lose everything rather than the excess.

        Returns:
            The number of rows written.
        """
        async with self._lock:
            books, self._book_buffer = self._book_buffer, []
            metrics, self._metric_buffer = self._metric_buffer, []
            funding, self._funding_buffer = self._funding_buffer, []
            candles, self._candle_buffer = self._candle_buffer, []

        written = 0
        failures: list[str] = []

        async def write(name: str, rows: list[Any], sink: Any, buffer: list[Any]) -> int:
            """Write one stream, returning rows written and re-buffering on failure."""
            if not rows or sink is None:
                return 0
            try:
                return int(await sink(rows))
            except Exception as exc:  # noqa: BLE001 - one stream must not sink the others
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
                buffer[:0] = rows
                del buffer[self.params.max_retry_buffer :]
                return 0

        written += await write(
            "orderbook",
            books,
            self._books.record_many if self._books else None,
            self._book_buffer,
        )
        written += await write(
            "perp_metrics",
            metrics,
            self._metrics.record_many if self._metrics else None,
            self._metric_buffer,
        )
        written += await write(
            "funding",
            funding,
            self._funding.record_many if self._funding else None,
            self._funding_buffer,
        )
        written += await write(
            "candles",
            candles,
            self._candles.upsert_many if self._candles else None,
            self._candle_buffer,
        )

        if failures:
            logger.error(
                "Flush partially failed; rows retained for retry",
                extra={"failures": failures, "retained": self._buffered()},
            )

        if written:
            self.rows_written += written
            logger.info(
                "Flushed rows",
                extra={
                    "written": written,
                    "books": len(books),
                    "metrics": len(metrics),
                    "candles": len(candles),
                    "total": self.rows_written,
                },
            )
        return written

    async def _flush_loop(self) -> None:
        """Flush on a timer so a quiet stream still reaches the database."""
        while True:
            await asyncio.sleep(self.params.flush_interval_s)
            try:
                await self.flush()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Periodic flush failed")


async def main() -> None:
    """Service entrypoint for ``python -m services.data_ingestion.recorder``."""
    configure_logging(SERVICE_NAME)
    await MarketRecorder.main()


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    asyncio.run(main())


__all__ = ["MarketRecorder"]
