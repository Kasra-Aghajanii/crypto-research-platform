"""TimescaleDB connection management.

Exposes a narrow :class:`Database` protocol rather than passing ``asyncpg``
objects around, so repositories depend on four methods instead of a driver.
That keeps them unit-testable without a running server -- which matters here,
because the DB is optional: every service degrades to in-memory operation when
``persistence_enabled`` is off or the server is unreachable.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from types import TracebackType
from typing import Any, Protocol, Self, runtime_checkable

import asyncpg

from libs.config import StorageSettings, settings

logger = logging.getLogger(__name__)


@runtime_checkable
class Database(Protocol):
    """The database surface repositories are allowed to use."""

    async def execute(self, query: str, *args: Any) -> str:
        """Run a statement that returns no rows."""
        ...

    async def executemany(self, query: str, args: Sequence[Sequence[Any]]) -> None:
        """Run a statement once per argument tuple."""
        ...

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        """Run a query and return every row."""
        ...

    async def fetchrow(self, query: str, *args: Any) -> Any | None:
        """Run a query and return the first row, or ``None``."""
        ...


class PostgresDatabase:
    """Async connection pool over TimescaleDB.

    Args:
        dsn: Connection string; defaults to the configured Timescale DSN.
        min_size: Minimum pooled connections.
        max_size: Maximum pooled connections.
        config: Storage settings override, mainly for tests.
    """

    def __init__(
        self,
        *,
        dsn: str | None = None,
        min_size: int = 1,
        max_size: int = 8,
        config: StorageSettings | None = None,
    ) -> None:
        """Store connection parameters without connecting."""
        self._dsn = dsn or (config or settings.storage).timescale_dsn
        self._min_size = min_size
        self._max_size = max_size
        self._pool: asyncpg.Pool[Any] | None = None

    @property
    def is_connected(self) -> bool:
        """Return whether the pool is open."""
        return self._pool is not None

    async def connect(self) -> None:
        """Open the pool. Safe to call more than once."""
        if self._pool is not None:
            return
        self._pool = await asyncpg.create_pool(
            dsn=self._dsn, min_size=self._min_size, max_size=self._max_size
        )
        logger.info("Database pool opened", extra={"min_size": self._min_size})

    async def close(self) -> None:
        """Close the pool."""
        if self._pool is None:
            return
        await self._pool.close()
        self._pool = None
        logger.info("Database pool closed")

    def _require_pool(self) -> asyncpg.Pool[Any]:
        """Return the open pool.

        Raises:
            RuntimeError: If called before :meth:`connect`.
        """
        if self._pool is None:
            raise RuntimeError("Database used before connect().")
        return self._pool

    async def execute(self, query: str, *args: Any) -> str:
        """Run a statement that returns no rows."""
        status: str = await self._require_pool().execute(query, *args)
        return status

    async def executemany(self, query: str, args: Sequence[Sequence[Any]]) -> None:
        """Run a statement once per argument tuple, in one round trip."""
        await self._require_pool().executemany(query, args)

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        """Run a query and return every row."""
        return list(await self._require_pool().fetch(query, *args))

    async def fetchrow(self, query: str, *args: Any) -> Any | None:
        """Run a query and return the first row, or ``None``."""
        return await self._require_pool().fetchrow(query, *args)

    async def __aenter__(self) -> Self:
        """Open the pool for use as an async context manager."""
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the pool on context exit."""
        await self.close()


async def try_connect(dsn: str | None = None) -> PostgresDatabase | None:
    """Connect to TimescaleDB, returning ``None`` if it is unavailable.

    Persistence is optional: a service that cannot reach the database logs a
    warning and continues in memory rather than refusing to trade on paper.

    Args:
        dsn: Connection string override.

    Returns:
        A connected database, or ``None``.
    """
    database = PostgresDatabase(dsn=dsn)
    try:
        await database.connect()
    except (OSError, asyncpg.PostgresError) as exc:
        logger.warning(
            "TimescaleDB unavailable; continuing without persistence",
            extra={"error": f"{type(exc).__name__}: {exc}"},
        )
        return None
    return database


__all__ = ["Database", "PostgresDatabase", "try_connect"]
