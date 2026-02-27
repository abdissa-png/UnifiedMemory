import pytest

from unified_memory.core.registry import ProviderRegistry
from unified_memory.namespace.tenant_manager import TenantManager
from unified_memory.retrieval.dense import DenseRetriever
from unified_memory.embeddings.providers.mock_provider import MockEmbeddingProvider
from unified_memory.storage.vector.memory_store import MemoryVectorStore
from unified_memory.storage.kv.memory_store import MemoryKVStore
from unified_memory.cas.content_store import ContentStore
from unified_memory.namespace.manager import NamespaceManager
from unified_memory.namespace.types import EmbeddingModelConfig, NamespaceConfig
from unified_memory.core.types import CollectionType
from dataclasses import asdict


@pytest.mark.asyncio
async def test_dense_respects_namespaces_array():
    """
    Ensure DenseRetriever uses the vector store's namespaces array filter correctly:
    - Same vector shared across two namespaces via namespaces metadata.
    - Each namespace search only sees results it has access to.
    """
    kv = MemoryKVStore()
    ns_manager = NamespaceManager(kv)
    tenant_manager = TenantManager(kv)
    vec_store = MemoryVectorStore()
    content_store = ContentStore(kv)
    provider_registry = ProviderRegistry()
    provider_registry.register_embedding_provider("mock:mock-model", MockEmbeddingProvider(model_id="mock:mock-model", dimension=8))
    # Two namespaces in same tenant
    tenant_id = "tenant-dense"
    config = await tenant_manager.register_tenant(tenant_id, text_embedding=EmbeddingModelConfig(provider="mock", model="mock:mock-model", dimension=8))
    ns1_cfg = await ns_manager.create_namespace(tenant_id, "alice")
    ns2_cfg = await ns_manager.create_namespace(tenant_id, "bob")

    ns1 = ns1_cfg.namespace_id
    ns2 = ns2_cfg.namespace_id

    # Text collection is tenant-level; reuse for both namespaces
    text_collection = await ns_manager.get_collection_name(ns1, CollectionType.TEXTS)

    # Single shared vector with namespaces [ns1, ns2]
    content_hash = "hash-dense-shared"
    await content_store.store_content(content_hash, "Shared dense content.")

    await vec_store.upsert(
        vectors=[
            {
                "id": "vec-shared",
                "embedding": [0.1] * 8,
                "metadata": {
                    "content_hash": content_hash,
                },
            }
        ],
        namespace=ns1,
        collection=text_collection,
    )
    # Grant access to ns2 via add_namespace, mimicking doc-level dedup fast path
    await vec_store.add_namespace("vec-shared", ns2, collection=text_collection)

    retriever = DenseRetriever(
        vector_store=vec_store,
        namespace_manager=ns_manager,
        content_store=content_store,
    )

    # Each namespace should see the shared vector via namespaces array filter
    res1 = await retriever.retrieve("Shared", namespaces=[ns1], embedding_provider=provider_registry.get_embedding_provider(config.text_embedding.model))
    res2 = await retriever.retrieve("Shared", namespaces=[ns2], embedding_provider=provider_registry.get_embedding_provider(config.text_embedding.model))

    assert len(res1) == 1
    assert len(res2) == 1
    assert res1[0].id == "vec-shared"
    assert res2[0].id == "vec-shared"
