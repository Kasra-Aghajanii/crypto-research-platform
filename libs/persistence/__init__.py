"""Persistence layer: TimescaleDB connection management and repositories.

Persistence is optional by design.  Services call
:func:`~libs.persistence.database.try_connect`, and when the database is
unreachable they log a warning and run in memory instead of refusing to start.
Everything that must survive a restart -- open positions, account cash,
trailing-stop state -- is written through a repository here.
"""

from __future__ import annotations

from libs.persistence.database import Database, PostgresDatabase, try_connect
from libs.persistence.recording import (
    FundingRepository,
    OrderBookRepository,
    PerpMetricsRepository,
)
from libs.persistence.repositories import (
    CandleRepository,
    ExecutionRepository,
    PerformanceRepository,
    PortfolioRepository,
    ProtectionRepository,
    SignalRepository,
)

__all__ = [
    "CandleRepository",
    "Database",
    "ExecutionRepository",
    "FundingRepository",
    "OrderBookRepository",
    "PerformanceRepository",
    "PerpMetricsRepository",
    "PortfolioRepository",
    "PostgresDatabase",
    "ProtectionRepository",
    "SignalRepository",
    "try_connect",
]
