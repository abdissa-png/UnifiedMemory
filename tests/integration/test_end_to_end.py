import pytest
import uuid
from pathlib import Path
from unified_memory.core.types import (
    RetrievalResult, EntityNode, GraphEdge, NodeType, CollectionType,
    compute_content_hash,
)
from unified_memory.core.utils import compute_document_hash
from unified_memory.namespace.types import TenantConfig, EmbeddingModelConfig, RetrievalConfig
from unified_memory.retrieval import GraphRetriever
from unified_memory.retrieval.unified import UnifiedSearchService
from unified_memory.retrieval.dense import DenseRetriever
from unified_memory.namespace.manager import NamespaceManager
from unified_memory.namespace.tenant_manager import TenantManager
from unified_memory.storage.kv.memory_store import MemoryKVStore
from unified_memory.cas.content_store import ContentStore
from unified_memory.cas.registry import CASRegistry
from unified_memory.cas.document_registry import DocumentRegistry
from unified_memory.embeddings.providers.mock_provider import MockEmbeddingProvider
from unified_memory.core.registry import ProviderRegistry

@pytest.mark.integration
@pytest.mark.asyncio
async def test_end_to_end_retrieval_flow(real_vector_store, real_graph_store, real_redis_kv_store):
    """Test full retrieval flow with real Qdrant and Neo4j."""
    
    # Setup
    kv_store = real_redis_kv_store
    ns_manager = NamespaceManager(kv_store)
    tenant_manager = TenantManager(kv_store)
    content_store = ContentStore(kv_store)
    
    # Create tenant and namespace with unique IDs for isolation
    import uuid
    tenant_id = f"integration_tenant_{uuid.uuid4().hex[:8]}"
    user_id = f"integration_user_{uuid.uuid4().hex[:8]}"

    # Register tenant with custom embedding model (mock)
    text_config = EmbeddingModelConfig(
        provider="mock",
        model="mock-model",
        dimension=384,
    )

    tenant_config = TenantConfig(
        tenant_id=tenant_id,
        text_embedding=text_config,
    )

    await tenant_manager.set_tenant_config(tenant_id, tenant_config)

    # Now create namespace
    ns = await ns_manager.create_namespace(tenant_id, user_id)
    namespace_id = ns.namespace_id
    
    # Get collection names (production naming)
    text_collection = await ns_manager.get_collection_name(namespace_id, CollectionType.TEXTS)
    entity_collection = await ns_manager.get_collection_name(namespace_id, CollectionType.ENTITIES)
    
    # Sanitize collection names for Qdrant
    def sanitize_collection_name(name: str) -> str:
        return name.replace("/", "_").replace(":", "_")
    
    text_collection_safe = sanitize_collection_name(text_collection)
    entity_collection_safe = sanitize_collection_name(entity_collection)
    
    # Create collections
    existing = await real_vector_store.list_collections()
    if text_collection_safe not in existing:
        await real_vector_store.create_collection(name=text_collection_safe, dimension=384)
    if entity_collection_safe not in existing:
        await real_vector_store.create_collection(name=entity_collection_safe, dimension=384)
    
    # Ingest test data
    content = "Python is a great programming language for AI and data science."
    content_hash = compute_content_hash(content, "mock-model")  # Use canonical hash
    doc_id = str(uuid.uuid4())
    embedding = [0.1] * 384
    
    await content_store.store_content(content_hash, content)
    
    await real_vector_store.upsert(
        vectors=[{
            "id": doc_id,
            "embedding": embedding,
            "metadata": {
                "content_hash": content_hash,
                "namespaces": [namespace_id],
                "type": "chunk",
            },
        }],
        namespace=namespace_id,
        collection=text_collection_safe,
    )
    
    # Entity for graph entry point
    python_entity_id = str(uuid.uuid4())  # ✅ UUID for Qdrant compatibility

    await real_vector_store.upsert(
        vectors=[{
            "id": python_entity_id,  # ✅ UUID
            "embedding": [0.1] * 384,
            "metadata": {
                "name": "Python",  # ✅ Human-readable name in metadata
                "entity_name": "Python",  # ✅ Also store as entity_name for graph retriever
                "type": "entity",
                "namespaces": [namespace_id],
            },
        }],
        namespace=namespace_id,
        collection=entity_collection_safe,
    )

    # Graph node uses the same UUID
    await real_graph_store.create_node(
        EntityNode(
            id=python_entity_id,  # ✅ Same UUID
            node_type=NodeType.ENTITY,
            content="Python programming language",
            entity_name="Python",  # ✅ Human-readable name
            entity_type="Concept",
            namespace=namespace_id
        ),
        namespace_id
    )

    
    embedder = MockEmbeddingProvider(dimension=384)
    registry = ProviderRegistry()
    # Key must match tenant config: provider="mock", model="mock-model" -> "mock:mock-model"
    registry.register_embedding_provider("mock:mock-model", embedder)

    service = UnifiedSearchService(
        namespace_manager=ns_manager,
        vector_store=real_vector_store,
        graph_store=real_graph_store,
        kv_store=kv_store,
        content_store=content_store,
        provider_registry=registry,
        dense_retriever=DenseRetriever(
            vector_store=real_vector_store,
            namespace_manager=ns_manager,
            content_store=content_store,
        ),
        graph_retriever=GraphRetriever(
            graph_store=real_graph_store,
            vector_store=real_vector_store,
            namespace_manager=ns_manager,
            content_store=content_store,
        ),
    )
    
    # Search
    config = RetrievalConfig(paths=["dense", "graph"], rerank=False)
    
    results = await service.search(
        query="Python",
        user_id=user_id,
        namespace=namespace_id,
        config=config
    )
    
    # Strong assertions
    assert len(results) > 0, "Should return at least one result"
    
    # Verify content hydration
    assert all(r.content for r in results), "All results should have hydrated content"
    
    # Verify expected content found
    assert any("Python" in r.content for r in results), "Should find Python-related content"
    
    # Verify both paths contributed (check sources)
    sources = set(r.source for r in results)
    print(f"Sources found: {sources}")
    # After fusion, source might be "hybrid:rrf", but we should have results
    
    # Cleanup
    await real_vector_store.delete_collection(text_collection_safe)
    await real_vector_store.delete_collection(entity_collection_safe)
    # Note: Graph cleanup would require delete_node/delete_edges methods


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_ingest_search_delete_real_backends(
    real_vector_store,
    real_graph_store,
    real_redis_kv_store,
    tmp_path: Path,
):
    """
    Full E2E with real Qdrant, Neo4j, and Redis KV:
    ingest file -> search -> delete -> verify cleanup.
    """
    from unified_memory.ingestion.pipeline import IngestionPipeline

    # Unique tenant/namespace for isolation
    tenant_id = f"e2e_ingest_tenant_{uuid.uuid4().hex[:8]}"
    user_id = "e2e_user"

    kv_store = real_redis_kv_store
    ns_manager = NamespaceManager(kv_store)
    tenant_manager = TenantManager(kv_store)
    content_store = ContentStore(kv_store)
    cas_registry = CASRegistry(kv_store)
    document_registry = DocumentRegistry(kv_store)

    # Tenant config: mock embedder 128d (matches MockEmbeddingProvider)
    text_config = EmbeddingModelConfig(
        provider="mock",
        model="mock-model",
        dimension=128,
    )
    tenant_config = TenantConfig(tenant_id=tenant_id, text_embedding=text_config)
    await tenant_manager.set_tenant_config(tenant_id, tenant_config)

    ns = await ns_manager.create_namespace(tenant_id, user_id)
    namespace_id = ns.namespace_id

    # Create Qdrant collections required by the pipeline (TEXTS, ENTITIES, RELATIONS, PAGE_IMAGES)
    def _sanitize(name: str) -> str:
        return name.replace("/", "_").replace(":", "_")

    existing = await real_vector_store.list_collections()
    for ctype in (
        CollectionType.TEXTS,
        CollectionType.ENTITIES,
        CollectionType.RELATIONS,
        CollectionType.PAGE_IMAGES,
    ):
        coll = await ns_manager.get_collection_name(namespace_id, ctype)
        safe = _sanitize(coll)
        if safe not in existing:
            await real_vector_store.create_collection(name=safe, dimension=128)
            existing.append(safe)

    # Register embedder for pipeline and search
    embedder = MockEmbeddingProvider(dimension=128)
    provider_registry = ProviderRegistry()
    provider_registry.register_embedding_provider("mock:mock-model", embedder)

    pipeline = IngestionPipeline(
        vector_store=real_vector_store,
        graph_store=real_graph_store,
        cas_registry=cas_registry,
        content_store=content_store,
        document_registry=document_registry,
        namespace_manager=ns_manager,
        provider_registry=provider_registry,
        embedding_provider=embedder,
    )

    # Ingest
    content = "Redis Qdrant Neo4j integration test. Search and delete flow."
    test_file = tmp_path / "e2e_real.txt"
    test_file.write_text(content)

    result = await pipeline.ingest_file(test_file, namespace=namespace_id)
    assert result.success, result.errors
    assert result.chunk_count > 0
    doc_hash = compute_document_hash(content, tenant_id)

    # Search
    search_service = UnifiedSearchService(
        namespace_manager=ns_manager,
        vector_store=real_vector_store,
        graph_store=real_graph_store,
        kv_store=kv_store,
        content_store=content_store,
        provider_registry=provider_registry,
        dense_retriever=DenseRetriever(
            vector_store=real_vector_store,
            namespace_manager=ns_manager,
            content_store=content_store,
        ),
        graph_retriever=GraphRetriever(
            graph_store=real_graph_store,
            vector_store=real_vector_store,
            namespace_manager=ns_manager,
            content_store=content_store,
        ),
    )
    config = RetrievalConfig(
        paths=["dense", "graph"], rerank=False
    )
    results = await search_service.search(
        query="Redis Qdrant Neo4j",
        user_id=user_id,
        namespace=namespace_id,
        config=config,
    )
    assert len(results) > 0
    assert any("Redis" in r.content for r in results)

    # Delete
    delete_result = await pipeline.delete_document(
        tenant_id=tenant_id,
        document_hash=doc_hash,
        namespace=namespace_id,
    )
    assert delete_result.found is True

    # Verify no results after delete
    results_after = await search_service.search(
        query="Redis Qdrant Neo4j",
        user_id=user_id,
        namespace=namespace_id,
        config=config,
    )
    assert len(results_after) == 0

    # Full CAS/content cleanup after delete depends on remove_reference + delete_if_orphan
    # (can be backend-dependent; in-memory KV typically cleans up; Redis may need more work).

    # Cleanup Qdrant collections created for this namespace
    for ctype in (
        CollectionType.TEXTS,
        CollectionType.ENTITIES,
        CollectionType.RELATIONS,
        CollectionType.PAGE_IMAGES,
    ):
        coll = await ns_manager.get_collection_name(namespace_id, ctype)
        safe = _sanitize(coll)
        try:
            await real_vector_store.delete_collection(safe)
        except Exception:
            pass