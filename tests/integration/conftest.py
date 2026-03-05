
import pytest
import os
import uuid
import asyncio
from typing import AsyncGenerator
import pytest_asyncio

# Allow overriding via env vars, default to docker-compose ports
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_AUTH = ("neo4j", "password")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")

@pytest_asyncio.fixture(scope="function")
async def real_vector_store():
    # Ping to check connection?
    try:
        from unified_memory.storage.vector.qdrant import QdrantVectorStore
        store = QdrantVectorStore(url=QDRANT_URL)
        await store.list_collections()
    except (ImportError, Exception) as e:
        pytest.skip(f"Qdrant not available or dependency missing: {e}")
        
    yield store
    await store.disconnect()

@pytest_asyncio.fixture(scope="function")
async def real_graph_store():
    try:
        from unified_memory.storage.graph.neo4j import Neo4jGraphStore
        store = Neo4jGraphStore(uri=NEO4J_URI, auth=NEO4J_AUTH)
        # Verify actual connection by running a simple query
        async with store.driver.session() as session:
            result = await session.run("RETURN 1 as test")
            await result.consume()
    except (ImportError, Exception) as e:
        pytest.skip(f"Neo4j not available or dependency missing: {e}")
        
    yield store
    await store.close()

@pytest.fixture(scope="function")
async def cleanup_collections(real_vector_store):
    """Cleanup collections after test."""
    yield
    collections = await real_vector_store.list_collections("test_")
    for c in collections:
        await real_vector_store.delete_collection(c)


@pytest_asyncio.fixture(scope="function")
async def real_redis_kv_store():
    """Redis KV store; skips if Redis is not available."""
    try:
        from unified_memory.storage.kv.redis_store import RedisKVStore
        store = RedisKVStore(url=REDIS_URL)
        ping_key = "_ping:integration"
        await store.set(ping_key, {"p": 1})
        await store.delete(ping_key)
    except (ImportError, Exception) as e:
        pytest.skip(f"Redis not available or dependency missing: {e}")
    yield store


@pytest_asyncio.fixture(scope="function")
async def real_elasticsearch_store():
    """Elasticsearch store with a unique test index; skips if ES not available."""
    store = None
    try:
        from unified_memory.storage.search.elasticsearch_store import ElasticSearchStore
        index_name = f"unified_memory_test_{uuid.uuid4().hex[:12]}"
        store = ElasticSearchStore(url=ELASTICSEARCH_URL, index_name=index_name)
        await store.ensure_index()
    except (ImportError, Exception) as e:
        if store is not None:
            try:
                await store.close()
            except Exception:
                pass
        pytest.skip(f"Elasticsearch not available or dependency missing: {e}")
    yield store
    try:
        await store._es.indices.delete(index=store._index)
    except Exception:
        pass
    await store.close()
