"""
SQLAlchemy async engine and session factory helpers.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from .models import Base


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
            await asyncio.to_thread(_run_alembic_upgrade, database_url)
            return
        except Exception:
            if not allow_create_all_fallback:
                raise

    if not allow_create_all_fallback:
        raise RuntimeError(
            "Alembic migrations are required. Set "
            "UMS_SQL_CREATE_ALL_FALLBACK=true only for local/dev fallback."
        )

    await asyncio.to_thread(_create_all_sync, engine)
