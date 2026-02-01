import pytest
from unified_memory.core.types import Chunk, SourceType
from unified_memory.ingestion.extractors.mock import MockExtractor
from unified_memory.ingestion.pipeline import IngestionPipeline
from unified_memory.embeddings.providers.mock_provider import MockEmbeddingProvider
from unified_memory.storage.kv.memory_store import MemoryKVStore
from unified_memory.storage.vector.memory_store import MemoryVectorStore
from unified_memory.storage.graph.networkx_store import NetworkXGraphStore
from unified_memory.cas.registry import CASRegistry
from unified_memory.cas.content_store import ContentStore
from unified_memory.namespace.manager import NamespaceManager
from unified_memory.namespace.types import NamespaceConfig

@pytest.fixture
def extraction_deps():
    kv = MemoryKVStore()
    namespace_manager = NamespaceManager(kv)
    return {
        "embedding_provider": MockEmbeddingProvider(dimension=16),
        "vector_store": MemoryVectorStore(),
        "cas_registry": CASRegistry(kv),
        "content_store": ContentStore(kv),
        "graph_store": NetworkXGraphStore(),
        "namespace_manager": namespace_manager,
        "kv": kv
    }

@pytest.fixture
def pipeline_with_extraction(extraction_deps):
    # Remove 'kv' before splatting
    deps = {k: v for k, v in extraction_deps.items() if k != "kv"}
    pipeline = IngestionPipeline(**deps)
    pipeline.extractors.append(MockExtractor())
    return pipeline

@pytest.mark.asyncio
async def test_extraction_flow(pipeline_with_extraction, extraction_deps):
    from dataclasses import asdict
    ns_id = "tenant:graph-tenant/user:graph-user"
    ns_cfg = NamespaceConfig(tenant_id="graph-tenant", user_id="graph-user")
    await extraction_deps["kv"].set(f"ns_config:{ns_id}", asdict(ns_cfg))
    
    text = "Alice works at BobCorp. It is a big Company."
    # Capitalized: Alice, BobCorp, Company
    
    result = await pipeline_with_extraction.ingest_text(
        text=text,
        namespace=ns_id
    )
    
    assert result.success
    
    # Verify Graph Store has the entities
    graph = extraction_deps["graph_store"]
    
    # 1. Verify Page Node
    # ingest_text creates a doc with 1 page by default
    page_id = f"page:{result.document_id}:1"
    page_node = await graph.get_node(page_id, namespace=ns_id)
    assert page_node is not None
    assert page_node.node_type.value == "page"
    
    # 2. Verify Passage Node
    passage_id = f"chunk:{result.document_id}:0"
    passage_node = await graph.get_node(passage_id, namespace=ns_id)
    assert passage_node is not None
    assert passage_node.node_type.value == "passage"
    
    # 3. Verify Structural Relation: Page -> Chunk
    page_neighbors = await graph.get_neighbors(page_id, namespace=ns_id)
    assert any(n.id == passage_id for n in page_neighbors)
    
    # 4. Verify Extraction: Chunk -> Entity
    passage_neighbors = await graph.get_neighbors(passage_id, namespace=ns_id)
    assert any(n.id == "Alice" for n in passage_neighbors)
    
    # Verify MockExtractor should have extracted "Alice", "BobCorp", "Company"
    alice = await graph.get_node("Alice", namespace=ns_id)
    assert alice is not None
    assert alice.entity_type == "Concept"
    assert alice.namespace == ns_id


