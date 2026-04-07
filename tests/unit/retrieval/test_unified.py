
import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, ANY
from typing import List

from unified_memory.retrieval import DenseRetriever, GraphRetriever
from unified_memory.retrieval.unified import UnifiedSearchService, RetrievalConfig
from unified_memory.core.types import RetrievalResult, MemoryStatus, Modality
from unified_memory.namespace.types import NamespaceConfig, TenantConfig, EmbeddingModelConfig
from unified_memory.core.registry import ProviderRegistry

@pytest.mark.asyncio
async def test_unified_search_flow():
    mock_ns_manager = AsyncMock()
    mock_vector_store = AsyncMock()
    mock_graph_store = AsyncMock()
    mock_kv_store = AsyncMock()
    mock_content_store = AsyncMock()
    mock_embedder = AsyncMock()
    mock_reranker = AsyncMock()

    mock_ns_manager.get_config.return_value = NamespaceConfig(
        tenant_id="tenant1",
        user_id="user1",
        namespace_id="ns1"
    )
    mock_ns_manager.get_tenant_config.return_value = TenantConfig(
        tenant_id="tenant1",
        text_embedding=EmbeddingModelConfig("openai", "text-embedding-3-small", 1536)
    )

    mock_embedder.embed.return_value = [0.1] * 1536

    from unified_memory.storage.base import VectorSearchResult
    mock_vector_store.search.return_value = [
        VectorSearchResult(
            id="vec1",
            score=0.9,
            metadata={"document_id": "doc-123", "content_hash": "h1", "namespace": "ns1"},
            content=None,
        )
    ]
    mock_content_store.get_content.return_value = "Dense Content"

    mock_graph_store.personalized_pagerank.return_value = [("node1", 0.8)]
    mock_graph_store.get_nodes_batch.return_value = [
        Mock(id="node1", content="Graph Content", type="entity", properties={})
    ]

    registry = ProviderRegistry()
    registry.register_embedding_provider("openai:text-embedding-3-small", mock_embedder)
    registry.register_reranker("default", mock_reranker)

    service = UnifiedSearchService(
        namespace_manager=mock_ns_manager,
        vector_store=mock_vector_store,
        graph_store=mock_graph_store,
        kv_store=mock_kv_store,
        content_store=mock_content_store,
        provider_registry=registry,
        dense_retriever=DenseRetriever(
            vector_store=mock_vector_store,
            namespace_manager=mock_ns_manager,
            content_store=mock_content_store,
        ),
        graph_retriever=GraphRetriever(
            graph_store=mock_graph_store,
            vector_store=mock_vector_store,
            namespace_manager=mock_ns_manager,
            content_store=mock_content_store,
        ),
    )

    config = RetrievalConfig(
        paths=["dense", "graph"],
        rerank=True,
    )

    mock_reranker.rerank.side_effect = lambda q, res, top_k: res

    results = await service.search(
        query="test query",
        user_id="user1",
        namespace="ns1",
        config=config,
        filters={"dense": {"document_id": "doc-123"}},
    )

    assert len(results) >= 1
    assert mock_vector_store.search.called
    assert mock_graph_store.personalized_pagerank.called
    assert mock_reranker.rerank.called

    # Dense path receives allowlisted filters only (content_hash is dropped)
    assert any(
        call.kwargs.get("filters") == {"document_id": "doc-123"}
        for call in mock_vector_store.search.call_args_list
        if "filters" in call.kwargs
    )

@pytest.mark.asyncio
async def test_memory_filtering():
    """Verify that INVALID memories are filtered out."""
    mock_ns_manager = AsyncMock()
    mock_vector_store = AsyncMock()
    mock_content_store = AsyncMock()

    mock_ns_manager.get_config.return_value = NamespaceConfig(tenant_id="t", user_id="u")
    mock_ns_manager.get_tenant_config.return_value = TenantConfig(tenant_id="t")

    mock_embedder = AsyncMock(embed=AsyncMock(return_value=[0.1] * 1536))

    registry = ProviderRegistry()
    registry.register_embedding_provider("openai:text-embedding-3-small", mock_embedder)

    from unified_memory.storage.base import VectorSearchResult
    mock_vector_store.search.return_value = [
        VectorSearchResult(id="valid", score=0.9, metadata={"status": "valid"}, content=None),
        VectorSearchResult(id="invalid", score=0.8, metadata={"status": "invalid_superseded"}, content=None)
    ]
    mock_content_store.get_content.return_value = "Content"

    service = UnifiedSearchService(
        namespace_manager=mock_ns_manager,
        vector_store=mock_vector_store,
        graph_store=AsyncMock(),
        kv_store=AsyncMock(),
        content_store=mock_content_store,
        provider_registry=registry,
        dense_retriever=DenseRetriever(
            vector_store=mock_vector_store,
            namespace_manager=mock_ns_manager,
            content_store=mock_content_store,
        ),
        graph_retriever=GraphRetriever(
            graph_store=AsyncMock(),
            vector_store=mock_vector_store,
            namespace_manager=mock_ns_manager,
            content_store=mock_content_store,
        ),
    )

    config = RetrievalConfig(paths=["dense"], rerank=False)

    results = await service.search("q", "u", "ns", config=config, filters={"status": "valid"})

    assert len(results) == 1
    assert results[0].id == "valid"

