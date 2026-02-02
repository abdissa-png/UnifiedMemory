
import pytest
import os
import asyncio
from typing import AsyncGenerator
import pytest_asyncio
# Allow overriding via env vars, default to docker-compose ports
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_AUTH = ("neo4j", "password")

@pytest_asyncio.fixture(scope="function")
async def real_vector_store():
    from unified_memory.storage.vector.qdrant import QdrantVectorStore
    store = QdrantVectorStore(url=QDRANT_URL)
    # Ping to check connection?
    try:
        await store.list_collections()
    except Exception as e:
        pytest.skip(f"Qdrant not available at {QDRANT_URL}: {e}")
        
    yield store
    await store.disconnect()

@pytest_asyncio.fixture(scope="function")
async def real_graph_store():
    from unified_memory.storage.graph.neo4j import Neo4jGraphStore
    try:
        store = Neo4jGraphStore(uri=NEO4J_URI, auth=NEO4J_AUTH)
        # Verify actual connection by running a simple query
        async with store.driver.session() as session:
            result = await session.run("RETURN 1 as test")
            await result.consume()
    except Exception as e:
        pytest.skip(f"Neo4j not available at {NEO4J_URI}: {e}")
        
    yield store
    await store.close()

@pytest.fixture(scope="function")
async def cleanup_collections(real_vector_store):
    """Cleanup collections after test."""
    yield
    collections = await real_vector_store.list_collections("test_")
    for c in collections:
        await real_vector_store.delete_collection(c)
