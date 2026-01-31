import pytest
import numpy as np
from unified_memory.storage.vector.memory_store import MemoryVectorStore

@pytest.fixture
def vector_store():
    return MemoryVectorStore()

@pytest.mark.asyncio
async def test_upsert_and_get(vector_store):
    # Upsert
    vecs = [
        {"id": "v1", "embedding": [0.1, 0.2, 0.3], "metadata": {"tag": "a"}},
        {"id": "v2", "embedding": [0.4, 0.5, 0.6], "metadata": {"tag": "b"}}
    ]
    count = await vector_store.upsert(vecs, namespace="ns1", collection="col1")
    assert count == 2
    
    # Get by ID
    res = await vector_store.get_by_id("v1", collection="col1")
    assert res.id == "v1"
    assert res.embedding == [0.1, 0.2, 0.3]
    assert res.metadata == {"tag": "a"} # Metadata stored as-is
    
    # Get non-existent
    res = await vector_store.get_by_id("none", collection="col1")
    assert res is None

@pytest.mark.asyncio
async def test_search(vector_store):
    # [1, 0] orthogonal to [0, 1]
    # [1, 0] parallel to [1, 0]
    vecs = [
        {"id": "v1", "embedding": [1.0, 0.0], "metadata": {}},
        {"id": "v2", "embedding": [0.0, 1.0], "metadata": {}},
        {"id": "v3", "embedding": [0.707, 0.707], "metadata": {}} # 45 degrees
    ]
    await vector_store.upsert(vecs, namespace="ns1", collection="col1")
    
    # Query with [1, 0] should get v1 (1.0), v3 (0.707), v2 (0.0)
    results = await vector_store.search(
        query_embedding=[1.0, 0.0],
        top_k=3,
        namespace="ns1", 
        collection="col1"
    )
    
    assert len(results) == 3
    assert results[0].id == "v1"
    assert results[0].score > 0.99
    
    assert results[1].id == "v3"
    assert 0.7 < results[1].score < 0.72

    assert results[2].id == "v2"
    assert results[2].score < 0.001

@pytest.mark.asyncio
async def test_filtered_search(vector_store):
    vecs = [
        {"id": "v1", "embedding": [1.0, 0.0], "metadata": {"type": "A"}},
        {"id": "v2", "embedding": [1.0, 0.0], "metadata": {"type": "B"}}
    ]
    await vector_store.upsert(vecs, namespace="ns1", collection="col1")
    
    # Search with filter
    results = await vector_store.search(
        query_embedding=[1.0, 0.0],
        filters={"type": "B"},
        namespace="ns1",
        collection="col1"
    )
    
    assert len(results) == 1
    assert results[0].id == "v2"

@pytest.mark.asyncio
async def test_batch_get(vector_store):
    vecs = [{"id": f"v{i}", "embedding": [float(i)], "metadata": {}} for i in range(5)]
    await vector_store.upsert(vecs, namespace="ns1", collection="col1")
    
    # Batch get
    results = await vector_store.get_by_ids(["v1", "v3", "v99"], collection="col1")
    assert len(results) == 2
    ids = {r.id for r in results}
    assert "v1" in ids
    assert "v3" in ids

@pytest.mark.asyncio
async def test_delete(vector_store):
    vecs = [{"id": "v1", "embedding": [], "metadata": {}}]
    await vector_store.upsert(vecs, namespace="ns1", collection="col1")
    
    count = await vector_store.delete(["v1"], collection="col1")
    assert count == 1
    
    res = await vector_store.get_by_id("v1", collection="col1")
    assert res is None
