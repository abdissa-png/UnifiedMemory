import pytest
import os
import uuid
from unified_memory.storage.kv.memory_store import MemoryKVStore

@pytest.fixture
def kv_store():
    store = MemoryKVStore()
    return store

@pytest.mark.asyncio
async def test_basic_crud(kv_store):
    # Set
    await kv_store.set("key1", {"value": "test"})
    
    # Get
    val = await kv_store.get("key1")
    assert val.data == {"value": "test"}
    assert val.version == 1
    
    # Update
    await kv_store.set("key1", {"value": "updated"})
    val = await kv_store.get("key1")
    assert val.data == {"value": "updated"}
    assert val.version == 2
    
    # Delete
    deleted = await kv_store.delete("key1")
    assert deleted is True
    val = await kv_store.get("key1")
    assert val is None
    
    # Delete non-existent
    deleted = await kv_store.delete("key1")
    assert deleted is False

@pytest.mark.asyncio
async def test_set_if_not_exists(kv_store):
    # Success
    success = await kv_store.set_if_not_exists("key1", {"value": "1"})
    assert success is True
    val = await kv_store.get("key1")
    assert val.data == {"value": "1"}
    
    # Failure (already exists)
    success = await kv_store.set_if_not_exists("key1", {"value": "2"})
    assert success is False
    val = await kv_store.get("key1")
    assert val.data == {"value": "1"}

@pytest.mark.asyncio
async def test_compare_and_swap(kv_store):
    await kv_store.set("key1", {"value": "1", "version": 1})
    
    # Success
    success = await kv_store.compare_and_swap(
        "key1", 
        expected_version=1, 
        new_value={"value": "2", "version": 2}
    )
    assert success is True
    val = await kv_store.get("key1")
    assert val.data == {"value": "2", "version": 2}
    
    # Failure (version mismatch)
    success = await kv_store.compare_and_swap(
        "key1", 
        expected_version=1, 
        new_value={"value": "3", "version": 3}
    )
    assert success is False
    val = await kv_store.get("key1")
    assert val.data == {"value": "2", "version": 2}
    
    # Failure (key not found)
    success = await kv_store.compare_and_swap(
        "key2", 
        expected_version=1, 
        new_value={}
    )
    assert success is False

@pytest.mark.asyncio
async def test_delete_if_version(kv_store):
    await kv_store.set("key1", {"value": "1", "version": 1})
    
    # Failure (version mismatch)
    success = await kv_store.delete_if_version("key1", expected_version=2)
    assert success is False
    val = await kv_store.get("key1")
    assert val is not None
    
    # Success
    success = await kv_store.delete_if_version("key1", expected_version=1)
    assert success is True
    val = await kv_store.get("key1")
    assert val is None

@pytest.mark.asyncio
async def test_scan_and_delete_pattern(kv_store):
    await kv_store.set("user:1:config", {"data": 1})
    await kv_store.set("user:1:profile", {"data": 2})
    await kv_store.set("user:2:config", {"data": 3})
    
    # Scan
    keys = await kv_store.scan("user:1:*")
    assert len(keys) == 2
    assert "user:1:config" in keys
    assert "user:1:profile" in keys
    
    # Delete pattern
    count = await kv_store.delete_pattern("user:1:*")
    assert count == 2
    
    keys = await kv_store.scan("user:1:*")
    assert len(keys) == 0
    
    keys = await kv_store.scan("user:2:*")
    assert len(keys) == 1


@pytest.fixture
def redis_url():
    return os.getenv("REDIS_URL","redis://localhost:6379/0")


@pytest.fixture
def redis_kv_store(redis_url):
    if not redis_url:
        pytest.skip("REDIS_URL not set; skipping RedisKVStore parity tests.")
    from unified_memory.storage.kv.redis_store import RedisKVStore
    return RedisKVStore(url=redis_url)


@pytest.mark.asyncio
async def test_redis_kvstore_parity_basic_crud(redis_kv_store):
    prefix = f"testkv:{uuid.uuid4()}:"
    k = f"{prefix}key1"
    await redis_kv_store.delete_pattern(f"{prefix}*")

    await redis_kv_store.set(k, {"value": "test"})
    val = await redis_kv_store.get(k)
    assert val is not None
    assert val.data == {"value": "test"}
    assert val.version == 1

    await redis_kv_store.set(k, {"value": "updated"})
    val = await redis_kv_store.get(k)
    assert val is not None
    assert val.data == {"value": "updated"}
    assert val.version == 2

    assert await redis_kv_store.delete(k) is True
    assert await redis_kv_store.get(k) is None
    assert await redis_kv_store.delete(k) is False


@pytest.mark.asyncio
async def test_redis_kvstore_parity_scan_delete_pattern(redis_kv_store):
    prefix = f"testkv:{uuid.uuid4()}:"
    await redis_kv_store.set(f"{prefix}user:1:config", {"data": 1})
    await redis_kv_store.set(f"{prefix}user:1:profile", {"data": 2})
    await redis_kv_store.set(f"{prefix}user:2:config", {"data": 3})

    keys = await redis_kv_store.scan(f"{prefix}user:1:*")
    assert len(keys) == 2

    count = await redis_kv_store.delete_pattern(f"{prefix}user:1:*")
    assert count == 2

    keys = await redis_kv_store.scan(f"{prefix}user:1:*")
    assert len(keys) == 0
