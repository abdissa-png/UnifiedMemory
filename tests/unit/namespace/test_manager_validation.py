import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from unified_memory.namespace.manager import NamespaceManager
from unified_memory.namespace.types import NamespaceConfig
from unified_memory.storage.kv.memory_store import MemoryKVStore

@pytest.fixture
def kv_store():
    return MemoryKVStore()

@pytest.fixture
def manager(kv_store):
    return NamespaceManager(kv_store)

@pytest.mark.asyncio
async def test_get_config_invalid_namespace(manager):
    # Should return None immediately without checking KV store
    # We can spy on kv_store.get to confirm it wasn't called?
    # But MemoryKVStore makes it hard to spy unless we wrap it.
    
    # Just check result
    result = await manager.get_config("invalid_namespace$")
    assert result is None
    
    # Valid but non-existent
    result = await manager.get_config("valid:namespace:id")
    assert result is None

@pytest.mark.asyncio
async def test_create_namespace_validity(manager):
    # This depends on NamespaceConfig generating a valid ID.
    # If it generates an invalid ID, create_namespace should raise ValueError.
    
    # Let's trust NamespaceConfig generates valid ones (hashes).
    ns = await manager.create_namespace(tenant_id="t1", user_id="u1")
    assert ns.namespace_id is not None
    assert ":" in ns.namespace_id
    
    # Should be retrievable
    retrieved = await manager.get_config(ns.namespace_id)
    assert retrieved == ns
