import pytest
import asyncio
from pathlib import Path
from unified_memory.core.types import Modality, CollectionType
from unified_memory.core.utils import compute_document_hash
from unified_memory.embeddings.providers.mock_provider import MockEmbeddingProvider
from unified_memory.namespace.types import EmbeddingModelConfig

@pytest.mark.integration
@pytest.mark.asyncio
async def test_multi_user_delete_flow(ingestion_pipeline, provider_registry, namespace_manager, tenant_manager, tmp_path):
    """
    Scenario:
    1. Tenant A exists.
    2. User 1 (NS1) and User 2 (NS2) both ingest the same Document X.
    3. User 1 deletes Document X for their namespace.
    4. Verify User 1 cannot retrieve it, but User 2 STILL CAN.
    5. User 2 deletes Document X.
    6. Verify Document is fully removed from storage (CAS Registry clean).
    """
    tenant_id = "tenant_delete"
    config = await tenant_manager.register_tenant(tenant_id, admin_user_id="admin-123", text_embedding=EmbeddingModelConfig(provider="mock", model="mock:mock-model", dimension=128))
    provider_registry.register_embedding_provider("mock:mock-model", MockEmbeddingProvider(model_id="mock:mock-model", dimension=128))
    ns1_config = await namespace_manager.create_namespace(tenant_id, "user1")
    ns2_config = await namespace_manager.create_namespace(tenant_id, "user2")
    
    ns1 = ns1_config.namespace_id
    ns2 = ns2_config.namespace_id
    
    test_file = tmp_path / "shared.txt"
    content = "Shared content for deletion test."
    test_file.write_text(content)
    
    # Both ingest
    await ingestion_pipeline.ingest_file(test_file, namespace=ns1)
    await ingestion_pipeline.ingest_file(test_file, namespace=ns2)
    
    # 1. Verify both have it
    from unified_memory.retrieval.dense import DenseRetriever
    retriever = DenseRetriever(
        vector_store=ingestion_pipeline.vector_store,
        namespace_manager=namespace_manager,
        content_store=ingestion_pipeline.content_store
    )
    
    embedder = provider_registry.get_embedding_provider(config.text_embedding.model)
    res1 = await retriever.retrieve("Shared", namespaces=[ns1], embedding_provider=embedder)
    res2 = await retriever.retrieve("Shared", namespaces=[ns2], embedding_provider=embedder)
    assert len(res1) > 0
    assert len(res2) > 0

    # Capture the chunk-level content hash for CAS/ContentStore verification
    content_hash = res1[0].metadata["content_hash"]
    
    # Compute tenant-scoped document hash (used by registry/delete API)
    doc_hash = compute_document_hash(content, tenant_id)
    
    # 2. User 1 deletes (Cascade Delete) via unified API
    delete_result_1 = await ingestion_pipeline.delete_document(
        tenant_id=tenant_id,
        document_hash=doc_hash,
        namespace=ns1,
    )
    assert delete_result_1.found is True
    
    # 3. Verify User 1 cannot retrieve
    res1_after = await retriever.retrieve("Shared", namespaces=[ns1], embedding_provider=embedder)
    assert len(res1_after) == 0
    
    # 4. Verify User 2 STILL CAN
    res2_after = await retriever.retrieve("Shared", namespaces=[ns2], embedding_provider=embedder)
    assert len(res2_after) > 0
    assert res2_after[0].metadata.get("source_doc_ids"), "shared delete should preserve source_doc_ids"
    assert res2_after[0].metadata.get("source_locations"), "shared delete should preserve source_locations"
    
    # 5. User 2 deletes (second/last namespace)
    delete_result_2 = await ingestion_pipeline.delete_document(
        tenant_id=tenant_id,
        document_hash=doc_hash,
        namespace=ns2,
    )
    assert delete_result_2.found is True
    
    # 6. Verify fully removed
    # After both deletes, neither namespace should be able to retrieve the document
    res1_final = await retriever.retrieve("Shared", namespaces=[ns1], embedding_provider=embedder)
    res2_final = await retriever.retrieve("Shared", namespaces=[ns2], embedding_provider=embedder)
    assert len(res1_final) == 0
    assert len(res2_final) == 0

    # And the underlying content and CAS entry should be gone as well
    cas_entry = await ingestion_pipeline.cas_registry.get_entry(content_hash)
    assert cas_entry is None
    payload = await ingestion_pipeline.content_store.get_content(content_hash)
    assert payload is None
