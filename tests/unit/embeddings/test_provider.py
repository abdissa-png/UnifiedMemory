import pytest
from unified_memory.core.types import Modality
from unified_memory.embeddings.providers.mock_provider import MockEmbeddingProvider

@pytest.fixture
def provider():
    return MockEmbeddingProvider(dimension=128)

@pytest.mark.asyncio
async def test_embed_single(provider):
    embedding = await provider.embed("test content", Modality.TEXT)
    assert len(embedding) == 128
    assert isinstance(embedding, list)
    assert isinstance(embedding[0], float)

@pytest.mark.asyncio
async def test_embed_batch(provider):
    contents = ["content 1", "content 2", "content 3"]
    embeddings = await provider.embed_batch(contents, Modality.TEXT)
    
    assert len(embeddings) == 3
    for emb in embeddings:
        assert len(emb) == 128
        
@pytest.mark.asyncio
async def test_deterministic_embedding(provider):
    """Test that same content produces same embedding."""
    emb1 = await provider.embed("same content", Modality.TEXT)
    emb2 = await provider.embed("same content", Modality.TEXT)
    
    assert emb1 == emb2
    
@pytest.mark.asyncio
async def test_different_embedding(provider):
    """Test that different content produces different embeddings."""
    emb1 = await provider.embed("content A", Modality.TEXT)
    emb2 = await provider.embed("content B", Modality.TEXT)
    
    assert emb1 != emb2

@pytest.mark.asyncio
async def test_unsupported_modality(provider):
    with pytest.raises(ValueError):
        await provider.embed("image", Modality.IMAGE)
