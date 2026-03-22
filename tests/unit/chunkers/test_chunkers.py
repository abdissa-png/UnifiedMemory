import pytest
from unified_memory.ingestion.chunkers import FixedSizeChunker, RecursiveChunker, SemanticChunker, ChunkingConfig
from unified_memory.ingestion.parsers.base import ParsedDocument, PageContent
from unified_memory.embeddings.providers.mock_provider import MockEmbeddingProvider

from unified_memory.core.types import SourceReference, SourceType

@pytest.fixture
def mock_embedding_provider():
    return MockEmbeddingProvider(dimension=128)

@pytest.fixture
def sample_document():
    source = SourceReference(source_id="doc1", source_type=SourceType.TEXT_BLOCK)
    pages = [
        PageContent(page_number=1, document_id="doc1", full_text="Sentence one. Sentence two."),
        PageContent(page_number=2, document_id="doc1", full_text="Sentence three. Sentence four.")
    ]
    return ParsedDocument(document_id="doc1", source=source, pages=pages)

def test_fixed_size_chunker_validation():
    """Test that FixedSizeChunker validates overlap < chunk_size."""
    with pytest.raises(ValueError, match="chunk_overlap"):
        ChunkingConfig(chunk_size=100, chunk_overlap=100)
    
    with pytest.raises(ValueError, match="chunk_overlap"):
        ChunkingConfig(chunk_size=100, chunk_overlap=150)

@pytest.mark.asyncio
async def test_recursive_chunker_base_case():
    """Test that RecursiveChunker handles text with no separators left."""
    config = ChunkingConfig(chunk_size=10, chunk_overlap=0)
    # separators = [" "]
    chunker = RecursiveChunker(separators=[" "])
    
    pages = [PageContent(page_number=1, document_id="doc1", full_text="1234567890ABCDE")] # No space
    source = SourceReference(source_id="doc1", source_type=SourceType.TEXT_BLOCK)
    doc = ParsedDocument(document_id="doc1", source=source, pages=pages)
    
    chunks = await chunker.chunk(doc, "ns", "model", config=config)
    
    # Should be split into "1234567890" and "ABCDE"
    assert len(chunks) == 2
    assert chunks[0].content == "1234567890"
    assert chunks[1].content == "ABCDE"

@pytest.mark.asyncio
async def test_recursive_chunker_overlap():
    """Test that RecursiveChunker correctly implements overlap."""
    # Chunk size 20, overlap 10
    config = ChunkingConfig(chunk_size=20, chunk_overlap=10)
    chunker = RecursiveChunker(separators=[" "])
    
    text = "word1 word2 word3 word4 word5 word6" 
    # Lengths: word1(5), word2(5), word3(5), word4(5), word5(5), word6(5)
    # Total with spaces: "word1 word2 word3" = 5+1+5+1+5 = 17 chars (Next " word4" would be 17+6=23 > 20)
    # Chunk 1: "word1 word2 word3"
    # Overlap 10: "word2 word3" is 11 chars? No, let's check:
    # "word2" (5) + " " (1) + "word3" (5) = 11 chars. > 10.
    # So overlap should take "word3" (5) only? Or "word2 word3" if it fits.
    # Let's see the logic: overlap_len + len(s) + (len(separator) if overlap_chunk else 0) <= chunk_overlap
    # "word3" (5) <= 10 -> overlap_chunk = ["word3"], overlap_len = 5
    # "word2" (5) + 1 + 5 = 11 > 10 -> break.
    # So next chunk starts with "word3".
    # Chunk 2 text starts with: "word3" + " " + "word4" + " " + "word5"...
    
    pages = [PageContent(page_number=1, document_id="doc1", full_text=text)]
    source = SourceReference(source_id="doc1", source_type=SourceType.TEXT_BLOCK)
    doc = ParsedDocument(document_id="doc1", source=source, pages=pages)
    
    chunks = await chunker.chunk(doc, "ns", "model", config=config)
    
    assert len(chunks) >= 2
    assert "word3" in chunks[0].content
    assert "word3" in chunks[1].content # Overlap check

@pytest.mark.asyncio
async def test_fixed_size_chunker_character_overlap():
    """Verify simple character-based overlap."""
    config = ChunkingConfig(chunk_size=10, chunk_overlap=5, respect_sentence_boundaries=False)
    chunker = FixedSizeChunker()
    
    text = "abcdefghij12345" # 15 chars
    # Chunk 1: "abcdefghij" (0-10)
    # Next start: 10 - 5 = 5
    # Chunk 2: "fghij12345" (5-15)
    
    pages = [PageContent(page_number=1, document_id="doc1", full_text=text)]
    source = SourceReference(source_id="doc1", source_type=SourceType.TEXT_BLOCK)
    doc = ParsedDocument(document_id="doc1", source=source, pages=pages)
    
    chunks = await chunker.chunk(doc, "ns", "model", config=config)
    assert len(chunks) == 2
    assert chunks[0].content == "abcdefghij"
    assert chunks[1].content == "fghij12345"

@pytest.mark.asyncio
async def test_semantic_chunker_cross_page(mock_embedding_provider, sample_document):
    """Test that SemanticChunker concatenates pages and maps correctly."""
    config = ChunkingConfig(similarity_threshold=0.9)  # High threshold to force splits
    chunker = SemanticChunker(mock_embedding_provider)
    
    # Mock embeddings to force a split between page 1 and page 2
    # Sentence 1: "Sentence one."
    # Sentence 2: "Sentence two."
    # Sentence 3: "Sentence three."
    # Sentence 4: "Sentence four."
    
    # Mocking embed_batch is tricky, let's just run it with MockEmbeddingProvider
    # which returns random vectors (mostly dissimilar)
    
    chunks = await chunker.chunk(sample_document, "ns", "model", config=config)
    
    assert len(chunks) > 0
    
    # Check that chunks have page numbers
    for chunk in chunks:
        assert chunk.page_number in [1, 2]
        assert "start_char" in chunk.metadata
        assert "end_char" in chunk.metadata
        assert chunk.metadata["is_semantic"] is True

@pytest.mark.asyncio
async def test_semantic_chunker_offset_mapping(mock_embedding_provider):
    """Verify offset mapping to pages."""
    pages = [
        PageContent(page_number=1, document_id="doc1", full_text="AAAA"), # len 4 + 1 (\n) = 5
        PageContent(page_number=10, document_id="doc1", full_text="BBBB") # len 4 + 1 = 5
    ]
    source = SourceReference(source_id="doc1", source_type=SourceType.TEXT_BLOCK)
    doc = ParsedDocument(document_id="doc1", source=source, pages=pages)
    
    chunker = SemanticChunker(mock_embedding_provider)
    chunks = await chunker.chunk(doc, "ns", "model")
    
    # Chunks starting at offset 0-4 should be page 1
    # Chunks starting at offset 5-9 should be page 10
    
    for chunk in chunks:
        start = chunk.metadata["start_char"]
        if start < 5:
            assert chunk.page_number == 1
        else:
            assert chunk.page_number == 10
