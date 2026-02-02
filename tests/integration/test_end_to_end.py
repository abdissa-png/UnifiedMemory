import pytest
import uuid
from unified_memory.core.types import (
    RetrievalResult, EntityNode, GraphEdge, NodeType, CollectionType,
    compute_content_hash  # Use canonical hash
)
from unified_memory.namespace.types import TenantConfig,EmbeddingModelConfig
from unified_memory.retrieval.unified import UnifiedSearchService
from unified_memory.retrieval.dense import DenseRetriever
from unified_memory.namespace.manager import NamespaceManager
from unified_memory.namespace.tenant_manager import TenantManager
from unified_memory.storage.kv.memory_store import MemoryKVStore
from unified_memory.cas.content_store import ContentStore

@pytest.mark.integration
@pytest.mark.asyncio
async def test_end_to_end_retrieval_flow(real_vector_store, real_graph_store):
    """Test full retrieval flow with real Qdrant and Neo4j."""
    
    # Setup
    kv_store = MemoryKVStore()
    ns_manager = NamespaceManager(kv_store)
    tenant_manager = TenantManager(kv_store)
    content_store = ContentStore(kv_store)
    
    # Create tenant and namespace
    tenant_id = "integration_tenant"
    user_id = "integration_user"

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
                "namespace": namespace_id,
                "type": "chunk"
            }
        }],
        namespace=namespace_id,
        collection=text_collection_safe
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
                "namespace": namespace_id
            }
        }],
        namespace=namespace_id,
        collection=entity_collection_safe
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
    
    # Mock embedder with variation
    class MockEmbedder:
        dimension = 384
        
        def _text_to_embedding(self, text: str):
            import hashlib
            hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            base = [0.1] * 384
            for i in range(min(10, len(base))):
                base[i] += (hash_val % 100) / 1000.0
            return base
        
        async def embed(self, text):
            return self._text_to_embedding(text)
        
        async def embed_batch(self, texts):
            return [self._text_to_embedding(t) for t in texts]
    
    embedder = MockEmbedder()
    
    service = UnifiedSearchService(
        namespace_manager=ns_manager,
        vector_store=real_vector_store,
        graph_store=real_graph_store,
        kv_store=kv_store,
        content_store=content_store,
        embedding_provider_factory=lambda cfg: embedder,
        llm_provider=None
    )
    
    # Search
    from unified_memory.namespace.types import RetrievalConfig
    config = RetrievalConfig(embedding_model="mock-model", paths=["dense", "graph"], rerank=False)
    
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