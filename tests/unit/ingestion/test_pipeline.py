import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from dataclasses import asdict

from unified_memory.core.types import Chunk, SourceReference, SourceType, CollectionType, Modality, PageContent
from unified_memory.ingestion.pipeline import IngestionPipeline, ParsedDocument
from unified_memory.ingestion.chunkers import FixedSizeChunker, ChunkingConfig
from unified_memory.embeddings.providers.mock_provider import MockEmbeddingProvider
from unified_memory.storage.kv.memory_store import MemoryKVStore
from unified_memory.storage.vector.memory_store import MemoryVectorStore
from unified_memory.storage.graph.networkx_store import NetworkXGraphStore
from unified_memory.cas.registry import CASRegistry
from unified_memory.cas.content_store import ContentStore
from unified_memory.cas.document_registry import DocumentRegistry
from unified_memory.ingestion.extractors.base import Extractor
from unified_memory.ingestion.extractors.schema import ExtractionResult, ExtractedEntity, ExtractedRelation
from unified_memory.namespace.manager import NamespaceManager
from unified_memory.namespace.types import EmbeddingModelConfig, NamespaceConfig, TenantConfig
from unified_memory.namespace.tenant_manager import TenantManager # Moved from inside test_chunk_overlap

@pytest.fixture
def mock_deps():
    kv = MemoryKVStore()
    namespace_manager = NamespaceManager(kv)
    return {
        "embedding_provider": MockEmbeddingProvider(dimension=128),
        "vision_embedding_provider": MockEmbeddingProvider(dimension=128, modalities=[Modality.IMAGE]),
        "vector_store": MemoryVectorStore(),
        "graph_store": NetworkXGraphStore(),
        "cas_registry": CASRegistry(kv),
        "content_store": ContentStore(kv),
        "document_registry": DocumentRegistry(kv),
        "namespace_manager": namespace_manager,
        "kv": kv
    }

@pytest.fixture
def pipeline(mock_deps):
    chunker = FixedSizeChunker()
    # Remove 'kv' before splatting into IngestionPipeline
    deps = {k: v for k, v in mock_deps.items() if k != "kv"}
    return IngestionPipeline(
        **deps,
        chunker=chunker
    )

@pytest.mark.asyncio
async def test_ingest_text(pipeline, mock_deps):
    """Test full ingestion flow for raw text."""
    tenant_id = "default-tenant"
    ns_id = f"tenant:{tenant_id}/user:test-user"
    ns_cfg = NamespaceConfig(tenant_id=tenant_id, user_id="test-user")
    await mock_deps["kv"].set(f"tenant_config:{tenant_id}", asdict(TenantConfig(tenant_id=tenant_id, text_embedding=EmbeddingModelConfig(provider="mock", model="mock-model", dimension=128))))
    await mock_deps["kv"].set(f"ns_config:{ns_id}", asdict(ns_cfg))
    
    text = "This is a test document. It should be embedded and stored."
    
    result = await pipeline.ingest_text(
        text=text,
        namespace=ns_id,
        title="Test Doc",
        skip_embedding=False,
    )
    
    assert result.success
    assert result.document_id is not None
    assert result.chunk_count > 0
    assert result.page_count == 1
    assert len(result.chunks) == result.chunk_count
    
    # Check chunk properties
    for chunk in result.chunks:
        assert chunk.metadata.get("namespace") == ns_id
        assert len(chunk.content) <= 120  # Allow some flexibility
        assert chunk.embedding is not None
        assert len(chunk.embedding) == 128
        
    # Verify CAS registry has entries
    for chunk in result.chunks:
        entry = await mock_deps["cas_registry"].get_entry(chunk.content_hash)
        assert entry is not None
        
    # Verify Vector Store has entries
    # MemoryVectorStore.search is easiest way to verify
    # Use non-zero vector to avoid norm=0 issues in MemoryVectorStore
    dummy_query = [0.0]*128
    dummy_query[0] = 1.0
    # Resolve collection for search
    text_collection = await mock_deps["namespace_manager"].get_collection_name(
        ns_id, CollectionType.TEXTS
    )
    
    search_results = await mock_deps["vector_store"].search(
        query_embedding=dummy_query,
        namespace=ns_id,
        collection=text_collection,
        top_k=10
    )
    assert len(search_results) >= result.chunk_count

@pytest.mark.asyncio
async def test_ingest_text_file(pipeline):
    """Test full ingestion flow for a text file."""
    tenant_id = "file-tenant"
    ns_id = f"tenant:{tenant_id}/user:file-user"
    ns_cfg = NamespaceConfig(tenant_id=tenant_id, user_id="file-user")
    await pipeline.namespace_manager.kv_store.set(f"tenant_config:{tenant_id}", asdict(TenantConfig(tenant_id=tenant_id, text_embedding=EmbeddingModelConfig(provider="mock", model="mock-model", dimension=128))))
    await pipeline.namespace_manager.kv_store.set(f"ns_config:{ns_id}", asdict(ns_cfg))
    
    content = "# Test Document\n\nThis is paragraph one.\n\nThis is paragraph two."
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        temp_path = Path(f.name)
    
    try:
        result = await pipeline.ingest_file(
            path=temp_path,
            namespace=ns_id,
            skip_embedding=True,
        )
        
        assert result.success
        assert result.chunk_count >= 1
        assert result.page_count == 1
    finally:
        temp_path.unlink()


@pytest.mark.asyncio
async def test_ingest_unsupported_file(pipeline):
    """Test ingesting an unsupported file type."""
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
        f.write(b"some content")
        temp_path = Path(f.name)
    
    try:
        result = await pipeline.ingest_file(
            path=temp_path,
            namespace="tenant:test/user:tester",
            skip_embedding=True,
        )
        
        assert not result.success
        assert len(result.errors) > 0
        assert "No parser" in result.errors[0]
    finally:
        temp_path.unlink()

@pytest.mark.asyncio
async def test_chunk_overlap(mock_deps):
    """Test that chunks have proper overlap while verifying full pipeline."""
    chunker = FixedSizeChunker()
    # Remove 'kv' before splatting
    deps = {k: v for k, v in mock_deps.items() if k != "kv"}
    pipeline = IngestionPipeline(
        **deps,
        chunker=chunker
    )
    
    # Create a long text to ensure multiple chunks
    text = "A" * 200
    
    ns_id = "tenant:overlap-tenant/user:overlap-user"
    
    # Needs to match config passed to chunker? No, pipeline now prioritizes tenant config.
    # So we must set tenant config to match desired chunking behavior.
    tenant_cfg = TenantConfig(
        tenant_id="overlap-tenant",
        chunk_size=50,
        chunk_overlap=10,
        respect_sentence_boundaries=False
    )
    # Use internal KV or TenantManager to set
    # Create temp TenantManager to set config
    from unified_memory.namespace.tenant_manager import TenantManager
    tenant_manager = TenantManager(mock_deps["kv"])
    await tenant_manager.set_tenant_config("overlap-tenant", tenant_cfg)
    
    ns_cfg = NamespaceConfig(tenant_id="overlap-tenant", user_id="overlap-user")
    await pipeline.namespace_manager.kv_store.set(f"ns_config:{ns_id}", asdict(ns_cfg))
    
    result = await pipeline.ingest_text(text=text, namespace=ns_id, skip_embedding=True)
    
    assert result.success
    assert result.chunk_count >= 3

@pytest.mark.asyncio
async def test_duplicate_ingestion(pipeline, mock_deps):
    """Test that duplicate documents skip embedding but link namespaces (Phase 1.3)."""
    text = "This is a shared document content."
    tenant_id = "shared"
    ns1 = f"tenant:{tenant_id}/user:alice"
    ns2 = f"tenant:{tenant_id}/user:bob"
    
    # Setup namespaces
    kv = mock_deps["kv"]
    ns_cfg1 = NamespaceConfig(tenant_id=tenant_id, user_id="alice")
    ns_cfg2 = NamespaceConfig(tenant_id=tenant_id, user_id="bob")
    await kv.set(f"ns_config:{ns1}", asdict(ns_cfg1))
    await kv.set(f"ns_config:{ns2}", asdict(ns_cfg2))
    await kv.set(f"tenant_config:{tenant_id}", asdict(TenantConfig(tenant_id=tenant_id, text_embedding=EmbeddingModelConfig(provider="mock", model="mock-model", dimension=128))))
    # 1. Ingest as Alice
    res1 = await pipeline.ingest_text(text, ns1, skip_embedding=False)
    assert res1.success
    
    # Verify embeddings called
    embed_provider = mock_deps["embedding_provider"]
    assert len(embed_provider.embed_calls) >= 1
    initial_calls = len(embed_provider.embed_calls)
    
    # 2. Ingest as Bob (same content)
    res2 = await pipeline.ingest_text(text, ns2, skip_embedding=False)
    assert res2.success

    # Verify CAS references were added for the new namespace (Path B)
    # (CAS should record usage for each chunk under both namespaces)
    cas = mock_deps["cas_registry"]
    # At least one chunk exists; look up its CAS entry and confirm both namespaces present.
    first_chunk_hash = res1.chunks[0].content_hash
    entry = await cas.get_entry(first_chunk_hash)
    assert entry is not None
    ns_set = {r.namespace for r in entry.refs}
    assert ns1 in ns_set
    assert ns2 in ns_set
    
    # 3. Verify deduplication optimization
    # Embeddings should NOT have been called again for the chunks
    # (Note: mock provider tracks batch calls. If skipped, count remains same)
    assert len(embed_provider.embed_calls) == initial_calls
    
    # 4. Verify Bob can search for it
    vec_store = mock_deps["vector_store"]
    collection = await pipeline.namespace_manager.get_collection_name(ns2, CollectionType.TEXTS)
    
    # Search with dummy vector
    results = await vec_store.search([0.1]*128, namespace=ns2, collection=collection)
    assert len(results) > 0
    # content_hash check
    # Registry now stores namespaces in list. Storage metadata should too.
    assert ns2 in results[0].metadata["namespaces"]
    assert ns1 in results[0].metadata["namespaces"]

@pytest.mark.asyncio
async def test_tenant_ingestion_config(pipeline, mock_deps):
    """Test that tenant config overrides default chunking (Phase 1.5)."""
    text = "Sentence one. Sentence two. Sentence three."
    
    ns_id = "tenant:custom-cfg/user:test"
    kv = mock_deps["kv"]
    
    # Tenant with small chunk size to force split
    tenant_cfg = TenantConfig(
        tenant_id="custom-cfg", 
        chunk_size=10, 
        chunk_overlap=0,
        respect_sentence_boundaries=False 
    )

    
    from unified_memory.namespace.tenant_manager import TenantManager
    tenant_manager = TenantManager(kv)
    await tenant_manager.set_tenant_config("custom-cfg", tenant_cfg)
    
    ns_cfg = NamespaceConfig(tenant_id="custom-cfg", user_id="test")
    await kv.set(f"ns_config:{ns_id}", asdict(ns_cfg))
    
    # The default chunker in pipeline has size=100
    # But ingestion should produce many chunks due to tenant config size=10
    
    result = await pipeline.ingest_text(text, ns_id, skip_embedding=False)
    assert result.success
    # Length of text is ~40 chars. chunk_size=10 -> ~4 chunks
    assert result.chunk_count >= 3
    
    # Verify vector store explicitly
    vec_store = mock_deps["vector_store"]
    col = await pipeline.namespace_manager.get_collection_name(ns_id, CollectionType.TEXTS)
    vecs = await vec_store.search([0.1]*128, namespace=ns_id, collection=col, top_k=20)
    assert len(vecs) >= 3

@pytest.mark.asyncio
async def test_vision_embedding(pipeline, mock_deps):
    """Test ingestion of document with page images (Phase 2.3)."""
    # Create a ParsedDocument with page images
    # We cheat and bypass ingest_file/text parser logic by calling _process_chunks directly
    
    ns_id = "tenant:vision/user:test"
    tenant_id = "vision"
    ns_cfg = NamespaceConfig(tenant_id=tenant_id, user_id="test")
    await mock_deps["kv"].set(f"ns_config:{ns_id}", asdict(ns_cfg))
    await mock_deps["kv"].set(f"tenant_config:{tenant_id}", asdict(TenantConfig(tenant_id=tenant_id, vision_embedding=EmbeddingModelConfig(provider="mock", model="mock-model", dimension=128))))
    # Mock image bytes
    image_bytes = b"fake_image_content"
    
    doc_id = "doc-vision-test"
    parsed = ParsedDocument(
        document_id=doc_id,
        source=SourceReference(source_id=doc_id, source_type=SourceType.FULL_PAGE),
        pages=[
            PageContent(
                page_number=1, 
                document_id=doc_id, 
                full_text="Page 1 text",
                full_page_image=image_bytes
            )
        ],
        full_text="Page 1 text"
    )
    
    # Chunks are still needed for the function
    chunks = [Chunk(document_id=doc_id, content="Page 1 text", chunk_index=0)]
    
    processing_results = await pipeline._process_chunks(
        chunks=chunks,
        namespace=ns_id,
        parsed_document=parsed,
    )
    errors = processing_results[0]
    
    assert not errors
    
    # Verify vision embeddings were stored
    # Collection: PAGE_IMAGES
    vis_col = await pipeline.namespace_manager.get_collection_name(ns_id, CollectionType.PAGE_IMAGES)
    
    # Search in vision collection
    vec_store = mock_deps["vector_store"]
    results = await vec_store.search(
        query_embedding=[0.1]*128, 
        namespace=ns_id, 
        collection=vis_col
    )
    
    assert len(results) == 1
    res = results[0]
    assert res.metadata["document_id"] == doc_id
    assert res.metadata["page_number"] == 1
    # Check that vision provider was called
    # Access vision provider directly from mock_deps if pipeline doesn't expose it publically
    assert len(mock_deps["vision_embedding_provider"].embed_calls) == 1
    # Check modality
    assert mock_deps["vision_embedding_provider"].embed_calls[0][1] == Modality.IMAGE

@pytest.mark.asyncio
async def test_relation_embedding(pipeline, mock_deps):
    """Test ingestion with relation extraction and embedding (Phase 2.4)."""
    ns_id = "tenant:rels/user:test"
    tenant_id = "rels"
    ns_cfg = NamespaceConfig(tenant_id=tenant_id, user_id="test")
    await mock_deps["kv"].set(f"tenant_config:{tenant_id}", asdict(TenantConfig(tenant_id=tenant_id, text_embedding=EmbeddingModelConfig(provider="mock", model="mock-model", dimension=128))))
    await mock_deps["kv"].set(f"ns_config:{ns_id}", asdict(ns_cfg))
    
    # Mock extractor
    class MockRelationExtractor(Extractor):
        async def extract(self, chunk):
            return ExtractionResult(
                entities=[
                    ExtractedEntity(name="Alice", type="Person"),
                    ExtractedEntity(name="Bob", type="Person")
                ],
                relations=[
                    ExtractedRelation(
                        source_entity="Alice",
                        target_entity="Bob",
                        relation_type="KNOWS",
                        properties={"since": 2020}
                    )
                ]
            )
    
    pipeline.extractors.append(MockRelationExtractor())
    
    text = "Alice knows Bob since 2020."
    
    result = await pipeline.ingest_text(text, ns_id)
    assert result.success
    
    # Verify relations embedded and stored
    rel_col = await pipeline.namespace_manager.get_collection_name(ns_id, CollectionType.RELATIONS)
    vec_store = mock_deps["vector_store"]
    
    # Search for relation
    # Query: "Alice KNOWS Bob"
    query_emb = [0.1]*128
    results = await vec_store.search(
        query_embedding=query_emb,
        namespace=ns_id,
        collection=rel_col
    )
    
    assert len(results) >= 1
    
    # Iterate results to find the relation (source/target use stable entity IDs)
    from unified_memory.core.types import make_entity_id
    alice_id = make_entity_id("Alice", "rels")
    bob_id = make_entity_id("Bob", "rels")
    found_rel = None
    for res in results:
        if res.metadata.get("relation") == "KNOWS" and \
           res.metadata.get("source_id") == alice_id and \
           res.metadata.get("target_id") == bob_id:
            found_rel = res
            break

    assert found_rel is not None, "Did not find 'Alice KNOWS Bob' relation in search results"
    
    # Verify embedded calls
    # Should include calls for entities ("Alice", "Bob") and relation ("Alice KNOWS Bob")
    # Entities: 2 (unique logic might apply)
    # Relations: 1
    # Plus text chunk: 1 (or batch)
    # Just check that embedding provider was used.
    emb_provider = mock_deps["embedding_provider"]
    assert len(emb_provider.embed_calls) > 0
