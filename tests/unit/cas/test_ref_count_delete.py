
import pytest
import asyncio
from unified_memory.cas.registry import CASRegistry
from unified_memory.storage.kv.memory_store import MemoryKVStore

@pytest.fixture
def kv_store():
    return MemoryKVStore()

@pytest.fixture
def registry(kv_store):
    return CASRegistry(kv_store)

@pytest.mark.asyncio
async def test_remove_reference_simple(registry):
    chash = "hash-1"
    # Register and add reference
    await registry.register(chash, "cid-1")
    await registry.add_reference(chash, "ns-1", "doc-1", 0)
    
    entry = await registry.get_entry(chash)
    assert len(entry.refs) == 1
    
    # Remove
    success = await registry.remove_reference(chash, "ns-1", "doc-1", 0)
    assert success is True
    
    entry = await registry.get_entry(chash)
    assert len(entry.refs) == 0

@pytest.mark.asyncio
async def test_remove_reference_partial(registry):
    chash = "hash-1"
    await registry.register(chash, "cid-1")
    # Two refs: same doc, different chunks
    await registry.add_reference(chash, "ns-1", "doc-1", 0)
    await registry.add_reference(chash, "ns-1", "doc-1", 1)
    
    # Remove chunk 0
    success = await registry.remove_reference(chash, "ns-1", "doc-1", 0)
    assert success is True
    
    entry = await registry.get_entry(chash)
    assert len(entry.refs) == 1
    assert entry.refs[0].chunk_index == 1

@pytest.mark.asyncio
async def test_remove_reference_document(registry):
    chash = "hash-1"
    await registry.register(chash, "cid-1")
    await registry.add_reference(chash, "ns-1", "doc-1", 0)
    await registry.add_reference(chash, "ns-1", "doc-1", 1)
    
    # Remove entire document (chunk_index=None)
    success = await registry.remove_reference(chash, "ns-1", "doc-1", None)
    assert success is True
    
    entry = await registry.get_entry(chash)
    assert len(entry.refs) == 0

@pytest.mark.asyncio
async def test_remove_concurrent(registry):
    chash = "hash-race"
    await registry.register(chash, "cid-1")
    # Add many refs
    await registry.add_reference(chash, "ns-1", "doc-1", 0)
    await registry.add_reference(chash, "ns-1", "doc-2", 0)
    
    # Concurrently remove both
    t1 = registry.remove_reference(chash, "ns-1", "doc-1", 0)
    t2 = registry.remove_reference(chash, "ns-1", "doc-2", 0)
    
    results = await asyncio.gather(t1, t2)
    
    assert all(results) # Both should succeed
    
    entry = await registry.get_entry(chash)
    assert len(entry.refs) == 0

@pytest.mark.asyncio
async def test_remove_nonexistent(registry):
    chash = "hash-1"
    await registry.register(chash, "cid-1")
    
    # Try to remove something not there
    success = await registry.remove_reference(chash, "ns-1", "doc-1", 0)
    assert success is False
