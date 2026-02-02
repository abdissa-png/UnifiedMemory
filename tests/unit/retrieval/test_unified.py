
import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, ANY
from typing import List

from unified_memory.retrieval.unified import UnifiedSearchService, RetrievalConfig
from unified_memory.core.types import RetrievalResult, MemoryStatus, Modality
from unified_memory.namespace.types import NamespaceConfig, TenantConfig, EmbeddingModelConfig

@pytest.mark.asyncio
async def test_unified_search_flow():
    # Mocks
    mock_ns_manager = AsyncMock()
    mock_vector_store = AsyncMock()
    mock_graph_store = AsyncMock()
    mock_kv_store = AsyncMock()
    mock_content_store = AsyncMock()
    mock_embedder = AsyncMock()
    mock_llm = AsyncMock()
    mock_reranker = AsyncMock()
    
    # Factory mock
    def embedder_factory(config):
        return mock_embedder
        
    # Setup Data
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
    
    # Mock retrieval results from paths
    # Dense results (will be hydrated inside the flow? No, DenseRetriever does hydration)
    # But wait, UnifiedSearchService instantiates DenseRetriever using REAL class.
    # So we need to mock what DenseRetriever calls: vector_store.search and content_store.get_content
    
    # Mock Vector Store Search Result (Dense)
    from unified_memory.storage.base import VectorSearchResult
    mock_vector_store.search.return_value = [
        VectorSearchResult(id="vec1", score=0.9, metadata={"content_hash": "h1", "namespace": "ns1"}, content=None)
    ]
    mock_content_store.get_content.return_value = "Dense Content"
    
    # Mock Graph Store PPR (Graph) (Assuming GraphRetriever uses vector store + graph store)
    mock_graph_store.personalized_pagerank.return_value = [("node1", 0.8)]
    mock_graph_store.get_nodes_batch.return_value = [
        Mock(id="node1", content="Graph Content", type="entity", properties={})
    ]
    # GraphRetriever also searches vector store for entities.
    # We can mock side effects for vector_store.search to handle different calls (text vs entities)
    # But for simplicity, let's assume it returns something usable for both or differentiate by collection arg
    
    # Initialize Service
    service = UnifiedSearchService(
        namespace_manager=mock_ns_manager,
        vector_store=mock_vector_store,
        graph_store=mock_graph_store,
        kv_store=mock_kv_store,
        content_store=mock_content_store,
        embedding_provider_factory=embedder_factory,
        llm_provider=mock_llm,
        reranker=mock_reranker
    )
    
    # Config
    config = RetrievalConfig(
        embedding_model="test",
        paths=["dense", "graph"], # Enable Dense and Graph
        rerank=True
    )
    
    # Mock Reranker
    mock_reranker.rerank.side_effect = lambda q, res, top_k: res # Identity
    
    # Execution
    results = await service.search(
        query="test query",
        user_id="user1",
        namespace="ns1",
        config=config
    )
    
    # Verification
    assert len(results) >= 1
    # Check that DenseRetriever was active (vector search called)
    assert mock_vector_store.search.called
    
    # Check that GraphRetriever was active (ppr called)
    assert mock_graph_store.personalized_pagerank.called
    
    # Check that Reranker was called
    assert mock_reranker.rerank.called

@pytest.mark.asyncio
async def test_memory_filtering():
    """Verify that INVALID memories are filtered out."""
     # Mocks
    mock_ns_manager = AsyncMock()
    mock_vector_store = AsyncMock()
    mock_content_store = AsyncMock()
    # ... other mocks minimal ...
    
    mock_ns_manager.get_config.return_value = NamespaceConfig(tenant_id="t", user_id="u")
    mock_ns_manager.get_tenant_config.return_value = TenantConfig(tenant_id="t")
    
    # Mock factories
    embedder_factory = lambda c: AsyncMock(embed=AsyncMock(return_value=[0.1]*1536))
    
    # Setup Vector Store to return results with status
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
        embedding_provider_factory=embedder_factory,
        llm_provider=AsyncMock(),
    )
    
    config = RetrievalConfig(embedding_model="test", paths=["dense"], rerank=False)
    
    results = await service.search("q", "u", "ns", config=config)
    
    # Only valid result should remain
    assert len(results) == 1
    assert results[0].id == "valid"

