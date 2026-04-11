import asyncio

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from unified_memory.storage.sql.models import Base
from unified_memory.storage.sql.session_manager import ChatSessionManager


@pytest.mark.asyncio
async def test_session_updated_at_changes_on_message_and_document_association():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        sf = async_sessionmaker(engine, expire_on_commit=False)
        manager = ChatSessionManager(sf)

        session = await manager.create_session(
            tenant_id="tenant-1",
            user_id="user-1",
            namespace="tenant:tenant-1/user:user-1",
            title="Initial title",
        )
        initial_updated_at = session.updated_at

        await asyncio.sleep(0.01)
        await manager.add_message(session.id, "user", "hello world")
        refreshed = await manager.get_session(session.id)
        assert refreshed is not None
        assert refreshed.updated_at >= initial_updated_at

        updated_after_message = refreshed.updated_at
        await asyncio.sleep(0.01)
        await manager.associate_document(session.id, "doc-1")
        refreshed = await manager.get_session(session.id)
        assert refreshed is not None
        assert refreshed.updated_at >= updated_after_message
    finally:
        await engine.dispose()
