import pytest
from unified_memory.core.types import Chunk, SourceType, make_entity_id, compute_content_hash
from unified_memory.ingestion.extractors.mock import MockExtractor
from unified_memory.ingestion.extractors.schema import ExtractionResult, ExtractedEntity, ExtractedRelation
from unified_memory.ingestion.pipeline import IngestionPipeline
from unified_memory.embeddings.providers.mock_provider import MockEmbeddingProvider
from unified_memory.storage.kv.memory_store import MemoryKVStore
from unified_memory.storage.vector.memory_store import MemoryVectorStore
from unified_memory.storage.graph.networkx_store import NetworkXGraphStore
from unified_memory.cas.registry import CASRegistry
from unified_memory.cas.content_store import ContentStore
from unified_memory.cas.document_registry import DocumentRegistry
from unified_memory.namespace.manager import NamespaceManager
from unified_memory.namespace.types import EmbeddingModelConfig, NamespaceConfig, TenantConfig, ExtractionConfig

@pytest.fixture
def extraction_deps():
    kv = MemoryKVStore()
    namespace_manager = NamespaceManager(kv)
    return {
        "embedding_provider": MockEmbeddingProvider(dimension=16),
        "vector_store": MemoryVectorStore(),
        "cas_registry": CASRegistry(kv),
        "content_store": ContentStore(kv),
        "document_registry": DocumentRegistry(kv),
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
    tenant_id = "graph-tenant"
    ns_id = f"tenant:{tenant_id}/user:graph-user"
    ns_cfg = NamespaceConfig(tenant_id=tenant_id, user_id="graph-user")
    await extraction_deps["kv"].set(f"tenant_config:{tenant_id}", asdict(TenantConfig(tenant_id=tenant_id, text_embedding=EmbeddingModelConfig(provider="mock", model="mock-model", dimension=128))))
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
    
    # 2. Verify Passage Node (now uses tenant-scoped content hash ID)
    content_hash = compute_content_hash(text, "graph-tenant")
    passage_id = f"passage:graph-tenant:{content_hash}"
    passage_node = await graph.get_node(passage_id, namespace=ns_id)
    assert passage_node is not None
    assert passage_node.node_type.value == "passage"
    
    # 3. Verify Structural Relation: Page -> Chunk
    page_neighbors = await graph.get_neighbors(page_id, namespace=ns_id)
    assert any(n.id == passage_id for n in page_neighbors)
    
    # 4. Verify Extraction: Chunk -> Entity (uses stable entity IDs)
    alice_id = make_entity_id("Alice", "graph-tenant")
    passage_neighbors = await graph.get_neighbors(passage_id, namespace=ns_id)
    assert any(n.id == alice_id for n in passage_neighbors)

    alice = await graph.get_node(alice_id, namespace=ns_id)
    assert alice is not None
    assert alice.entity_type == "Concept"
    assert alice.namespace == ns_id


# ---------------------------------------------------------------------------
# B5: strict_type_filtering
# ---------------------------------------------------------------------------

class _TypedExtractor(MockExtractor):
    """Returns one 'Person' entity and one 'Location' entity."""

    async def extract(self, chunk):
        return ExtractionResult(
            entities=[
                ExtractedEntity(name="Alice", type="Person"),
                ExtractedEntity(name="London", type="Location"),
            ],
            relations=[
                ExtractedRelation(
                    source_entity="Alice",
                    target_entity="London",
                    relation_type="LIVES_IN",
                ),
                ExtractedRelation(
                    source_entity="Alice",
                    target_entity="London",
                    relation_type="VISITED",
                ),
            ],
        )


@pytest.mark.asyncio
async def test_strict_type_filtering_drops_disallowed(extraction_deps):
    """strict_type_filtering=True drops entities/relations not in the allow-list."""
    from dataclasses import asdict

    ns_id = "tenant:filter-tenant/user:filter-user"
    tenant = TenantConfig(
        tenant_id="filter-tenant",
        extraction=ExtractionConfig(
            extractor_type="mock",
            entity_types=["Person"],  # Location is NOT allowed
            relation_types=["LIVES_IN"],  # VISITED is NOT allowed
            strict_type_filtering=True,
        ),
    )
    ns_cfg = NamespaceConfig(tenant_id="filter-tenant", user_id="filter-user")

    kv = extraction_deps["kv"]
    await kv.set(f"ns_config:{ns_id}", asdict(ns_cfg))
    await kv.set(f"tenant_config:filter-tenant", asdict(tenant))

    # Build a pipeline with our typed extractor
    deps = {k: v for k, v in extraction_deps.items() if k != "kv"}
    pipeline = IngestionPipeline(**deps)
    pipeline.extractors.append(_TypedExtractor())

    result = await pipeline.ingest_text(text="Alice lives in London.", namespace=ns_id)
    assert result.success

    graph = extraction_deps["graph_store"]

    # Alice (Person) must be present
    alice_id = make_entity_id("Alice", "filter-tenant")
    alice_node = await graph.get_node(alice_id, namespace=ns_id)
    assert alice_node is not None, "Person entity should pass the filter"

    # London (Location) must be absent
    london_id = make_entity_id("London", "filter-tenant")
    london_node = await graph.get_node(london_id, namespace=ns_id)
    assert london_node is None, "Location entity should be dropped by strict filter"


@pytest.mark.asyncio
async def test_no_strict_filtering_keeps_all(extraction_deps):
    """strict_type_filtering=False (default) keeps entities regardless of type lists."""
    from dataclasses import asdict

    ns_id = "tenant:nofilter-tenant/user:nofilter-user"
    tenant = TenantConfig(
        tenant_id="nofilter-tenant",
        extraction=ExtractionConfig(
            extractor_type="mock",
            entity_types=["Person"],  # has a list, but strict=False
            strict_type_filtering=False,
        ),
    )
    ns_cfg = NamespaceConfig(tenant_id="nofilter-tenant", user_id="nofilter-user")

    kv = extraction_deps["kv"]
    await kv.set(f"ns_config:{ns_id}", asdict(ns_cfg))
    await kv.set(f"tenant_config:nofilter-tenant", asdict(tenant))

    deps = {k: v for k, v in extraction_deps.items() if k != "kv"}
    pipeline = IngestionPipeline(**deps)
    pipeline.extractors.append(_TypedExtractor())

    result = await pipeline.ingest_text(text="Alice lives in London.", namespace=ns_id)
    assert result.success

    graph = extraction_deps["graph_store"]

    # Both entities must be present (no filtering)
    alice_id = make_entity_id("Alice", "nofilter-tenant")
    london_id = make_entity_id("London", "nofilter-tenant")
    assert await graph.get_node(alice_id, namespace=ns_id) is not None
    assert await graph.get_node(london_id, namespace=ns_id) is not None
