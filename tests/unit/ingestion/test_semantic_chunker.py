import pytest
from unittest.mock import AsyncMock
from unified_memory.core.types import PageContent
from unified_memory.ingestion.parsers.base import ParsedDocument
from unified_memory.ingestion.chunkers.semantic import SemanticChunker, ChunkingConfig
from unified_memory.embeddings.providers.mock_provider import MockEmbeddingProvider

@pytest.fixture
def mock_embedding_provider():
    return MockEmbeddingProvider(dimension=16)

@pytest.fixture
def document():
    text = (
        "Dogs are loyal animals. They love to play fetch. "
        "Cats are independent. They like to sleep all day. "
        "Space exploration is expensive. Rockets consume lots of fuel."
    )
    # The sentences are naturally grouped: 2 about dogs, 2 about cats, 2 about space.
    # The mock embedding provider uses content hash, so embeddings will be random but deterministic.
    # We can't guarantee semantic similarity structure with mock provider unless we force it.
    
    # Actually, MockEmbeddingProvider is just hash-based. So similarity will be random.
    # SemanticChunker logic: 1 - dot(v1, v2) > threshold.
    # Random vectors will have low similarity, so likely > threshold distance.
    # So it will likely split every sentence.
    
    return ParsedDocument(
        document_id="doc1",
        source=None, # type: ignore
        pages=[PageContent(
            page_number=1,
            document_id="doc1",
            text_blocks=[{"text": text}],
            full_text=text
        )]
    )

@pytest.mark.asyncio
async def test_semantic_chunking_with_mock(mock_embedding_provider, document):
    config = ChunkingConfig(similarity_threshold=0.99) # Very high threshold -> Force splits
    chunker = SemanticChunker(mock_embedding_provider, config)
    
    chunks = await chunker.chunk(document, "ns", "model-1")
    
    # Ideally, with hash embeddings, similarities are low (<0.99), so distances are high (>0.01).
    # Since config.threshold=0.99 (distance threshold 0.01), almost any distance triggers split.
    # So we expect many chunks.
    assert len(chunks) > 1
    
    # Check structure
    assert chunks[0].document_id == "doc1"
    assert chunks[0].content_hash is not None

@pytest.mark.asyncio
async def test_semantic_chunking_single_block(mock_embedding_provider):
    # Test with very low threshold (high distance tolerance) -> Should keep together
    # Threshold -1.0 implies we tolerate distance up to 2.0 (max possible)
    config = ChunkingConfig(similarity_threshold=-1.0)
    chunker = SemanticChunker(mock_embedding_provider, config)
    
    text = "Sentence one. Sentence two."
    doc = ParsedDocument(
        document_id="doc2", source=None, # type: ignore
        pages=[PageContent(
            page_number=1, document_id="doc2", 
            text_blocks=[{"text": text}], full_text=text
        )]
    )
    
    # Mock vectors are random, dot product of normalized random vectors in 16D ~ 0.
    # Distance ~ 1.0.
    # Threshold 0.01 distance means we split if dist > 0.99.
    # Wait: config_threshold = 1 - 0.01 = 0.99.
    # If distance (1.0) > 0.99 -> Split.
    
    # Wait, my logic: 
    # similarity_threshold = 0.5.
    # config_threshold = 1 - 0.5 = 0.5.
    # if distance > 0.5 (sim < 0.5) -> split.
    
    # If I want to KEEP together, I need similarity threshold very LOW (-1.0).
    # so config_threshold = 1 - (-1.0) = 2.0.
    # distance (0..2) > 2.0 is impossible. -> Never split.
    
    config.similarity_threshold = -1.0 
    
    chunks = await chunker.chunk(doc, "ns", "model-1")
    assert len(chunks) == 1
    assert "Sentence one. Sentence two." in chunks[0].content
