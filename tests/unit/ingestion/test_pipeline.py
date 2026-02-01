import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from dataclasses import asdict

from unified_memory.core.types import Chunk, SourceReference, CollectionType
from unified_memory.ingestion.pipeline import IngestionPipeline
from unified_memory.ingestion.chunkers import FixedSizeChunker, ChunkingConfig
from unified_memory.embeddings.providers.mock_provider import MockEmbeddingProvider
from unified_memory.storage.kv.memory_store import MemoryKVStore
from unified_memory.storage.vector.memory_store import MemoryVectorStore
from unified_memory.cas.registry import CASRegistry
from unified_memory.cas.content_store import ContentStore
from unified_memory.namespace.manager import NamespaceManager
from unified_memory.namespace.types import NamespaceConfig

@pytest.fixture
def mock_deps():
    kv = MemoryKVStore()
    namespace_manager = NamespaceManager(kv)
    return {
        "embedding_provider": MockEmbeddingProvider(dimension=128),
        "vector_store": MemoryVectorStore(),
        "cas_registry": CASRegistry(kv),
        "content_store": ContentStore(kv),
        "namespace_manager": namespace_manager,
        "kv": kv
    }

@pytest.fixture
def pipeline(mock_deps):
    config = ChunkingConfig(chunk_size=100, chunk_overlap=20)
    chunker = FixedSizeChunker(config)
    # Remove 'kv' before splatting into IngestionPipeline
    deps = {k: v for k, v in mock_deps.items() if k != "kv"}
    return IngestionPipeline(
        **deps,
        chunker=chunker
    )

@pytest.mark.asyncio
async def test_ingest_text(pipeline, mock_deps):
    """Test full ingestion flow for raw text."""
    ns_id = "tenant:default-tenant/user:test-user"
    ns_cfg = NamespaceConfig(tenant_id="default-tenant", user_id="test-user")
    await mock_deps["kv"].set(f"ns_config:{ns_id}", asdict(ns_cfg))
    
    text = "This is a test document. It should be embedded and stored."
    
    result = await pipeline.ingest_text(
        text=text,
        namespace=ns_id,
        title="Test Doc",
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
    ns_id = "tenant:file-tenant/user:file-user"
    ns_cfg = NamespaceConfig(tenant_id="file-tenant", user_id="file-user")
    await pipeline.namespace_manager.kv_store.set(f"ns_config:{ns_id}", asdict(ns_cfg))
    
    content = "# Test Document\n\nThis is paragraph one.\n\nThis is paragraph two."
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        temp_path = Path(f.name)
    
    try:
        result = await pipeline.ingest_file(
            path=temp_path,
            namespace=ns_id,
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
        )
        
        assert not result.success
        assert len(result.errors) > 0
        assert "No parser" in result.errors[0]
    finally:
        temp_path.unlink()

@pytest.mark.asyncio
async def test_chunk_overlap(mock_deps):
    """Test that chunks have proper overlap while verifying full pipeline."""
    config = ChunkingConfig(
        chunk_size=50,
        chunk_overlap=10,
        respect_sentence_boundaries=False,
    )
    chunker = FixedSizeChunker(config)
    # Remove 'kv' before splatting
    deps = {k: v for k, v in mock_deps.items() if k != "kv"}
    pipeline = IngestionPipeline(
        **deps,
        chunker=chunker
    )
    
    # Create a long text to ensure multiple chunks
    text = "A" * 200
    
    ns_id = "tenant:overlap-tenant/user:overlap-user"
    ns_cfg = NamespaceConfig(tenant_id="overlap-tenant", user_id="overlap-user")
    await pipeline.namespace_manager.kv_store.set(f"ns_config:{ns_id}", asdict(ns_cfg))
    
    result = await pipeline.ingest_text(text=text, namespace=ns_id)
    
    assert result.success
    assert result.chunk_count >= 3
