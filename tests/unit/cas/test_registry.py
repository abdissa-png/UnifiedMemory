import pytest
import asyncio
import dataclasses
from unified_memory.storage.kv.memory_store import MemoryKVStore
from unified_memory.cas.registry import CASRegistry, CASEntry

@pytest.fixture
def kv_store():
    return MemoryKVStore()

@pytest.fixture
def registry(kv_store):
    return CASRegistry(kv_store)

@pytest.mark.asyncio
async def test_register_and_get(registry):
    content_hash = "hash123"
    content_id = "content:abc"
    vector_id = "vec:xyz"
    
    # First registration
    # In reality, IngestionPipeline uses:
    # content_id = f"content:{content_hash}"
    # vector_id = f"text:{content_hash}"
    c_id = f"content:{content_hash}"
    v_id = f"text:{content_hash}"
    
    entry = await registry.register(content_hash, c_id, v_id)
    assert entry.content_hash == content_hash
    assert entry.content_id == c_id
    assert entry.vector_id == v_id
    assert len(entry.refs) == 0
    
    # Retrieve
    retrieved = await registry.get_entry(content_hash)
    assert retrieved is not None
    assert retrieved.content_id == c_id
    
    # Register duplicate (should return existing)
    entry2 = await registry.register(content_hash, "different_id")
    assert entry2.content_id == c_id  # Should match original
    assert entry2.content_id != "different_id"

@pytest.mark.asyncio
async def test_add_remove_refs(registry):
    content_hash = "hash_refs"
    await registry.register(content_hash, "cid")
    
    # Add references
    added = await registry.add_reference(content_hash, "ns1", "doc1", 0)
    assert added is True
    
    added2 = await registry.add_reference(content_hash, "ns2", "doc2", 1)
    assert added2 is True
    
    # Add duplicate ref
    added3 = await registry.add_reference(content_hash, "ns1", "doc1", 0)
    assert added3 is True  # Returns True because it "exists" or was successfully handled
    
    entry = await registry.get_entry(content_hash)
    assert len(entry.refs) == 2  # Should filter duplicates
    
    # Remove reference
    removed = await registry.remove_reference(content_hash, "ns1", "doc1", 0)
    assert removed is True
    
    entry = await registry.get_entry(content_hash)
    assert len(entry.refs) == 1
    assert entry.refs[0].namespace == "ns2"

@pytest.mark.asyncio
async def test_detect_orphans(registry):
    # Create orphan
    await registry.register("orphan1", "cid1")
    
    # Create used
    await registry.register("used1", "cid2")
    await registry.add_reference("used1", "ns", "doc", 0)
    
    orphans = await registry.get_orphans()
    assert "orphan1" in orphans
    assert "used1" not in orphans

@pytest.mark.asyncio
async def test_concurrent_remove_reference(registry):
    """Verify that multiple concurrent removals are handled correctly."""
    content_hash = "shared_hash"
    await registry.register(content_hash, "cid")
    
    # Add many references
    count = 20
    for i in range(count):
        await registry.add_reference(content_hash, "ns", f"doc{i}", 0)
        
    # Remove them concurrently
    tasks = [
        registry.remove_reference(content_hash, "ns", f"doc{i}", 0)
        for i in range(count)
    ]
    results = await asyncio.gather(*tasks)
    
    assert all(results)
    entry = await registry.get_entry(content_hash)
    assert len(entry.refs) == 0
