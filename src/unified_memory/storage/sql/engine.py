"""
SQLAlchemy async engine and session factory helpers.
"""

from __future__ import annotations

from typing import Optional

from .models import Base


def create_sql_engine(
    database_url: str = "sqlite+aiosqlite:///./memory_system.db",
    echo: bool = False,
):
    """Create an async SQLAlchemy engine."""
    from sqlalchemy.ext.asyncio import create_async_engine

    return create_async_engine(database_url, echo=echo)


def create_session_factory(engine):
    """Create an async session factory bound to *engine*."""
    from sqlalchemy.ext.asyncio import async_sessionmaker

    return async_sessionmaker(engine, expire_on_commit=False)


async def init_db(engine) -> None:
    """Create all tables (for development / first-run bootstrap).
    
    WARNING: For this to work correctly, all SQLAlchemy models must be defined 
    in (or imported by) `storage.sql.models` before this function is called.
    `Base.metadata.create_all` only creates tables for models it knows about.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
