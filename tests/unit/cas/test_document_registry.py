
import pytest
import asyncio
from unified_memory.cas.document_registry import DocumentRegistry, DocumentEntry
from unified_memory.storage.kv.memory_store import MemoryKVStore

@pytest.fixture
def kv_store():
    return MemoryKVStore()

@pytest.fixture
def registry(kv_store):
    return DocumentRegistry(kv_store)

@pytest.mark.asyncio
async def test_register_new_document(registry):
    tenant = "tenant-1"
    doc_hash = "hash-123"
    namespace = "ns-1"
    doc_id = "doc-uuid-1"
    
    is_new = await registry.register_document(tenant, doc_hash, namespace, doc_id)
    assert is_new is True
    
    entry = await registry.get_document(tenant, doc_hash)
    assert entry is not None
    assert entry.tenant_id == tenant
    assert entry.doc_hash == doc_hash
    assert entry.document_id == doc_id
    assert entry.namespaces == [namespace]

@pytest.mark.asyncio
async def test_register_duplicate_document(registry):
    tenant = "tenant-1"
    doc_hash = "hash-abc"
    doc_id_1 = "doc-uuid-1"
    doc_id_2 = "doc-uuid-2" # User tries to upload same content with new ID
    
    # First upload
    is_new = await registry.register_document(tenant, doc_hash, "ns-1", doc_id_1)
    assert is_new is True
    
    # Second upload by different user
    is_new = await registry.register_document(tenant, doc_hash, "ns-2", doc_id_2)
    assert is_new is False
    
    entry = await registry.get_document(tenant, doc_hash)
    assert entry.document_id == doc_id_1 # Should keep original ID
    assert "ns-1" in entry.namespaces
    assert "ns-2" in entry.namespaces
    assert len(entry.namespaces) == 2

@pytest.mark.asyncio
async def test_add_remove_namespace(registry):
    tenant = "tenant-1"
    doc_hash = "hash-xyz"
    
    await registry.register_document(tenant, doc_hash, "ns-1", "doc-1")
    
    # Add existing
    await registry.add_namespace(tenant, doc_hash, "ns-1")
    entry = await registry.get_document(tenant, doc_hash)
    assert len(entry.namespaces) == 1
    
    # Add new
    await registry.add_namespace(tenant, doc_hash, "ns-2")
    entry = await registry.get_document(tenant, doc_hash)
    assert len(entry.namespaces) == 2
    
    # Remove
    await registry.remove_namespace(tenant, doc_hash, "ns-1")
    entry = await registry.get_document(tenant, doc_hash)
    assert entry.namespaces == ["ns-2"]
    
    # Remove last
    deleted = await registry.remove_namespace(tenant, doc_hash, "ns-2")
    assert deleted is True
    
    # Should be gone or empty
    entry = await registry.get_document(tenant, doc_hash)
    assert entry is None

@pytest.mark.asyncio
async def test_concurrent_registration(registry):
    tenant = "tenant-race"
    doc_hash = "hash-race"
    
    # Simulate concurrent registration
    # Both try to register same doc at same time
    # One should verify True, one False
    
    t1 = registry.register_document(tenant, doc_hash, "ns-A", "doc-A")
    t2 = registry.register_document(tenant, doc_hash, "ns-B", "doc-B")
    
    results = await asyncio.gather(t1, t2)
    
    assert True in results
    assert False in results
    
    entry = await registry.get_document(tenant, doc_hash)
    assert "ns-A" in entry.namespaces
    assert "ns-B" in entry.namespaces
