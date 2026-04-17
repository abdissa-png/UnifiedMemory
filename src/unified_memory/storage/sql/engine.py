"""
SQLAlchemy async engine and session factory helpers.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Sequence

from .models import Base

logger = logging.getLogger(__name__)


def create_sql_engine(
    database_url: str,
    echo: bool = False,
):
    """Create an async SQLAlchemy engine."""
    from sqlalchemy.ext.asyncio import create_async_engine

    return create_async_engine(database_url, echo=echo)


def create_session_factory(engine):
    """Create an async session factory bound to *engine*."""
    from sqlalchemy.ext.asyncio import async_sessionmaker

    return async_sessionmaker(engine, expire_on_commit=False)

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _alembic_ini_path() -> Path:
    return _repo_root() / "alembic.ini"


def _create_all_sync(engine) -> None:
    import asyncio as _asyncio

    async def _create_all() -> None:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    _asyncio.run(_create_all())


def _run_alembic_upgrade(database_url: str) -> None:
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(_alembic_ini_path()))
    cfg.set_main_option("script_location", str(_repo_root() / "alembic"))
    cfg.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(cfg, "head")


async def verify_expected_tables(
    engine,
    table_names: Sequence[str] = ("users",),
) -> None:
    """Fail fast if core tables are missing (migration wrong DB or stamp drift).

    Uses PostgreSQL ``to_regclass``; other dialects are skipped.
    """
    if engine.dialect.name != "postgresql":
        logger.debug(
            "db.init.table_verify_skipped dialect=%s", engine.dialect.name
        )
        return

    from sqlalchemy import text

    async with engine.connect() as conn:
        for name in table_names:
            # ``name`` is fixed literals from callers only.
            reg = f"public.{name}"
            result = await conn.execute(
                text("SELECT to_regclass(:reg)"),
                {"reg": reg},
            )
            if result.scalar() is None:
                raise RuntimeError(
                    f"Database schema is missing required table '{name}' after "
                    "migrations. Confirm UMS_DATABASE_URL points at this database "
                    "and run `alembic upgrade head`."
                )


async def init_db(
    engine,
    database_url: str,
    *,
    allow_create_all_fallback: bool = False,
) -> None:
    """Apply Alembic migrations, optionally falling back to ``create_all``."""
    alembic_ini = _alembic_ini_path()
    if alembic_ini.exists():
        try:
            start = time.perf_counter()
            logger.info("db.init.alembic.start")
            await asyncio.to_thread(_run_alembic_upgrade, database_url)
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info("db.init.alembic.complete duration_ms=%.2f", duration_ms)
            await verify_expected_tables(engine)
            return
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.exception("db.init.alembic.failed duration_ms=%.2f", duration_ms)
            if not allow_create_all_fallback:
                raise

    if not allow_create_all_fallback:
        raise RuntimeError(
            "Alembic migrations are required. Set "
            "UMS_SQL_CREATE_ALL_FALLBACK=true only for local/dev fallback."
        )

    await asyncio.to_thread(_create_all_sync, engine)
    await verify_expected_tables(engine)
