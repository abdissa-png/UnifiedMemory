from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from unified_memory.storage.sql.models import Base

# ---------------------------------------------------------------------------
# Alembic Config
# ---------------------------------------------------------------------------

config = context.config

if config.config_file_name is not None:
    # Do not disable app loggers (e.g. unified_memory.*) when Alembic runs
    # from the API lifespan — the default breaks console logging after migrate.
    fileConfig(config.config_file_name, disable_existing_loggers=False)

# ---------------------------------------------------------------------------
# Database URL (REQUIRED)
# ---------------------------------------------------------------------------

# Prefer URL already on the Config (e.g. ``init_db`` passes the same URL as
# the running app). Otherwise use the environment variable so CLI invocations
# still work without duplicating defaults in two places.
database_url = config.get_main_option("sqlalchemy.url") or ""
if not database_url:
    database_url = os.environ.get("UMS_DATABASE_URL") or ""
if not database_url:
    raise RuntimeError(
        "Database URL is not set: configure sqlalchemy.url on the Alembic "
        "Config or set UMS_DATABASE_URL in the environment."
    )
config.set_main_option("sqlalchemy.url", database_url)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

target_metadata = Base.metadata

# ---------------------------------------------------------------------------
# Offline migrations
# ---------------------------------------------------------------------------


def run_migrations_offline() -> None:
    """Run migrations without a DB connection."""
    url = config.get_main_option("sqlalchemy.url")

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------------
# Online migrations
# ---------------------------------------------------------------------------


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations with an async DB engine."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if context.is_offline_mode():
    run_migrations_offline()
else:
    import asyncio

    asyncio.run(run_migrations_online())