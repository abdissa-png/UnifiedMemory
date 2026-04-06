import pytest
import asyncio
from pathlib import Path
from unified_memory.core.types import Modality, CollectionType
from unified_memory.namespace.types import EmbeddingModelConfig

@pytest.mark.integration
@pytest.mark.asyncio
async def test_two_user_dedup_flow(ingestion_pipeline, namespace_manager, tenant_manager, provider_registry, tmp_path):
    """
    Scenario:
    1. Tenant A exists.
    2. User 1 (Namespace 1) ingests Document X.
    3. User 2 (Namespace 2) ingests the SAME Document X.
    4. Verify both can retrieve it, but storage IDs are shared (deduplicated).
    """
    tenant_id = "tenant_dedup"
    config = await tenant_manager.register_tenant(
        tenant_id,
        admin_user_id="admin-123",
        text_embedding=EmbeddingModelConfig(
            provider="mock", model="mock:mock-model", dimension=128
        ),
    )
    user1 = "user1"
    user2 = "user2"
    
    # Setup namespaces
    ns1_config = await namespace_manager.create_namespace(tenant_id, user1)
    ns2_config = await namespace_manager.create_namespace(tenant_id, user2)
    
    ns1 = ns1_config.namespace_id
    ns2 = ns2_config.namespace_id
    
    # Create a test file
    test_file = tmp_path / "test.txt"
    content = "The quick brown fox jumps over the lazy dog. This is a unique document for testing deduplication."
    test_file.write_text(content)
    
    # 1. User 1 Ingests
    result1 = await ingestion_pipeline.ingest_file(test_file, namespace=ns1)
    assert result1.success
    assert result1.chunk_count > 0
    
    # Capture first vector IDs
    vec_ids1 = [c.content_hash for c in result1.chunks]
    
    # 2. User 2 Ingests same file
    result2 = await ingestion_pipeline.ingest_file(test_file, namespace=ns2)
    assert result2.success
    assert result2.chunk_count == result1.chunk_count
    
    # 3. Verify IDs are the same (Deterministic IDs)
    vec_ids2 = [c.content_hash for c in result2.chunks]
    assert vec_ids1 == vec_ids2
    
    # 4. Verify Vector Store reflects both namespaces
    vector_store = ingestion_pipeline.vector_store
    text_collection = await namespace_manager.get_collection_name(ns1, CollectionType.TEXTS)
    
    # Compute deterministic vector ID
    from unified_memory.core.types import compute_vector_id
    first_vec_id = compute_vector_id(vec_ids1[0], config.text_embedding.model)
    
    # Check the first chunk (pass ns1 - vector has both ns1 and ns2)
    stored_vec = await vector_store.get_by_id(first_vec_id, collection=text_collection, namespace=ns1)
    assert stored_vec is not None
    
    # It should have BOTH namespaces
    namespaces = stored_vec.metadata.get("namespaces", [])
    assert ns1 in namespaces
    assert ns2 in namespaces
    
    # 5. Verify both can retrieve
    from unified_memory.retrieval.dense import DenseRetriever
    retriever = DenseRetriever(
        vector_store=vector_store,
        namespace_manager=namespace_manager,
        content_store=ingestion_pipeline.content_store,
    )

    embedder = provider_registry.get_embedding_provider(config.text_embedding.model)
    results1 = await retriever.retrieve(
        "fox jumps", namespaces=[ns1], embedding_provider=embedder
    )
    assert len(results1) > 0
    assert "fox" in results1[0].content
    
    results2 = await retriever.retrieve(
        "fox jumps", namespaces=[ns2], embedding_provider=embedder
    )
    assert len(results2) > 0
    assert results2[0].id == results1[0].id
