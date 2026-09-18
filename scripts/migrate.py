"""Apply database migrations.

Run after ``docker compose up -d``::

    python -m scripts.migrate

Each migration is applied at most once; applied ids are recorded in
``schema_migrations``.  Statements are idempotent, so re-running is harmless.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass

from libs.logging_config import configure_logging
from libs.persistence.database import PostgresDatabase
from scripts.migration import initial, market_recording

logger = logging.getLogger("migrate")


@dataclass(frozen=True, slots=True)
class Migration:
    """One migration: an id, a description and its ordered statements."""

    migration_id: str
    description: str
    statements: tuple[str, ...]


MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        migration_id=initial.MIGRATION_ID,
        description=initial.DESCRIPTION,
        statements=initial.STATEMENTS,
    ),
    Migration(
        migration_id=market_recording.MIGRATION_ID,
        description=market_recording.DESCRIPTION,
        statements=market_recording.STATEMENTS,
    ),
)
"""Every migration, in application order."""


async def apply(database: PostgresDatabase, migrations: Sequence[Migration]) -> list[str]:
    """Apply any migrations that have not run yet.

    Args:
        database: An open database connection.
        migrations: Migrations in order.

    Returns:
        The ids that were applied by this run.
    """
    await database.execute(initial.CREATE_MIGRATIONS_TABLE)
    rows = await database.fetch("SELECT migration_id FROM schema_migrations")
    already = {row["migration_id"] for row in rows}

    applied: list[str] = []
    for migration in migrations:
        if migration.migration_id in already:
            logger.info("Migration already applied", extra={"migration": migration.migration_id})
            continue
        logger.info("Applying migration", extra={"migration": migration.migration_id})
        for statement in migration.statements:
            await database.execute(statement)
        await database.execute(
            "INSERT INTO schema_migrations (migration_id, description) VALUES ($1, $2) "
            "ON CONFLICT (migration_id) DO NOTHING",
            migration.migration_id,
            migration.description,
        )
        applied.append(migration.migration_id)
    return applied


async def main(dsn: str | None) -> None:
    """Connect and apply migrations.

    Args:
        dsn: Connection string override.
    """
    async with PostgresDatabase(dsn=dsn) as database:
        applied = await apply(database, MIGRATIONS)
    if applied:
        logger.info("Migrations applied", extra={"migrations": applied})
    else:
        logger.info("Schema already up to date")


def cli() -> None:
    """Parse arguments and run the migrator."""
    parser = argparse.ArgumentParser(description="Apply TimescaleDB migrations.")
    parser.add_argument("--dsn", default=None, help="Override the configured Timescale DSN.")
    args = parser.parse_args()

    configure_logging("migrate")
    asyncio.run(main(args.dsn))


if __name__ == "__main__":
    cli()
