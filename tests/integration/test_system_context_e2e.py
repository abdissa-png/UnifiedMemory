"""
E2E tests for SystemContext bootstrap with all real backends
(Redis KV, Qdrant, Neo4j, optional Elasticsearch).
"""

import os
import pytest
import uuid
from pathlib import Path

from unified_memory.bootstrap import SystemContext
from unified_memory.core.types import CollectionType
from unified_memory.namespace.tenant_manager import TenantManager
from unified_memory.namespace.types import TenantConfig, EmbeddingModelConfig


def _integration_config():
    """Config pointing at docker-compose.test.yml services."""
    return {
        "kv_store": "redis",
        "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        "vector_store": "qdrant",
        "qdrant_url": os.getenv("QDRANT_URL", "http://localhost:6333"),
        "graph_store": "neo4j",
        "neo4j_uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "neo4j_auth": ("neo4j", "password"),
        "sparse_retriever": "elasticsearch",
        "elasticsearch_url": os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"),
        "elasticsearch_index": f"unified_memory_ctx_{uuid.uuid4().hex[:12]}",
        "embedding_providers": {
            "mock:mock-model": {
                "provider": "mock",
                "model": "mock-model",
                "dimension": 128,
            }
        },
    }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_system_context_real_backends(tmp_path: Path):
    """
    Bootstrap SystemContext with Redis, Qdrant, Neo4j, Elasticsearch;
    build services, create tenant/namespace, ingest file, search (dense + graph).
    """
    config = _integration_config()
    try:
        ctx = SystemContext(config=config)
    except (ImportError, ValueError) as e:
        pytest.skip(
            f"SystemContext with real backends not available (e.g. aiohttp for ES): {e}"
        )
    ctx.build_services(default_embedding_key="mock:mock-model")

    assert ctx.ingestion_pipeline is not None
    assert ctx.search_service is not None
    assert ctx.kv_store is not None
    assert ctx.elasticsearch_store is not None

    # Skip if any required service is unreachable (e.g. Docker not running)
    try:
        await ctx.vector_store.list_collections()
        _ = await ctx.kv_store.get("_ping:ctx_e2e")
        async with ctx.graph_store.driver.session() as session:
            result = await session.run("RETURN 1")
            await result.consume()
        await ctx.elasticsearch_store.ensure_index()
    except Exception as e:
        try:
            await ctx.elasticsearch_store.close()
        except Exception:
            pass
        pytest.skip(
            f"One or more real backends unreachable (start with: docker compose -f docker-compose.test.yml up -d): {e}"
        )

    tenant_id = f"ctx_e2e_tenant_{uuid.uuid4().hex[:8]}"
    user_id = "ctx_user"

    # Tenant config is already satisfied by embedding_providers; ensure tenant exists in KV
    tenant_manager = TenantManager(ctx.kv_store)
    tenant_config = TenantConfig(
        tenant_id=tenant_id,
        text_embedding=EmbeddingModelConfig(
            provider="mock",
            model="mock-model",
            dimension=128,
        ),
    )
    await tenant_manager.set_tenant_config(tenant_id, tenant_config)

    ns = await ctx.namespace_manager.create_namespace(tenant_id, user_id)
    namespace_id = ns.namespace_id

    # Create Qdrant collections required by the pipeline (pipeline does not create them)
    def _sanitize(name: str) -> str:
        return name.replace("/", "_").replace(":", "_")

    existing = await ctx.vector_store.list_collections()
    for ctype in (
        CollectionType.TEXTS,
        CollectionType.ENTITIES,
        CollectionType.RELATIONS,
        CollectionType.PAGE_IMAGES,
    ):
        coll = await ctx.namespace_manager.get_collection_name(namespace_id, ctype)
        safe = _sanitize(coll)
        if safe not in existing:
            await ctx.vector_store.create_collection(name=safe, dimension=128)
            existing.append(safe)

    # Ingest
    content = "SystemContext bootstrap E2E test with Redis Qdrant Neo4j."
    test_file = tmp_path / "ctx_e2e.txt"
    test_file.write_text(content)

    result = await ctx.ingestion_pipeline.ingest_file(
        test_file,
        namespace=namespace_id,
    )
    assert result.success, result.errors
    assert result.chunk_count > 0

    # Search (dense + graph; sparse would need pipeline to index to ES)
    from unified_memory.namespace.types import RetrievalConfig

    config_retrieval = RetrievalConfig(
        paths=["dense", "graph"],
        rerank=False,
    )
    results = await ctx.search_service.search(
        query="SystemContext bootstrap",
        user_id=user_id,
        namespace=namespace_id,
        config=config_retrieval,
    )
    assert len(results) > 0
    assert any("SystemContext" in r.content for r in results)

    # Cleanup Qdrant collections
    for ctype in (
        CollectionType.TEXTS,
        CollectionType.ENTITIES,
        CollectionType.RELATIONS,
        CollectionType.PAGE_IMAGES,
    ):
        coll = await ctx.namespace_manager.get_collection_name(namespace_id, ctype)
        safe = _sanitize(coll)
        try:
            await ctx.vector_store.delete_collection(safe)
        except Exception:
            pass
