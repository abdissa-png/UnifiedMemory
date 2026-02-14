import pytest
from unified_memory.core.types import Modality
from unified_memory.embeddings.providers.mock_provider import MockEmbeddingProvider
from unified_memory.embeddings.cache import CachedEmbeddingProvider

@pytest.fixture
def mock_provider():
    return MockEmbeddingProvider(dimension=64)

@pytest.fixture
def cached_provider(mock_provider):
    return CachedEmbeddingProvider(base_provider=mock_provider)

@pytest.mark.asyncio
async def test_cache_hit(cached_provider, mock_provider):
    content = "cached content"
    
    # First call - miss
    await cached_provider.embed(content)
    assert cached_provider.cache_stats["misses"] == 1
    assert cached_provider.cache_stats["hits"] == 0
    assert len(mock_provider.embed_calls) == 1
    
    # Second call - hit
    await cached_provider.embed(content)
    assert cached_provider.cache_stats["misses"] == 1
    assert cached_provider.cache_stats["hits"] == 1
    assert len(mock_provider.embed_calls) == 1  # Base provider not called again

@pytest.mark.asyncio
async def test_batch_cache(cached_provider, mock_provider):
    contents = ["A", "B", "A", "C", "B"]
    
    # Batch embed: A (miss), B (miss), A (hit), C (miss), B (hit)
    await cached_provider.embed_batch(contents)
    
    # Should have called base provider 3 times (A, B, C)
    # But note: mock_provider.embed_batch calls embed() sequentially in this mock implementation
    # CachedEmbeddingProvider.embed_batch only calls base.embed_batch for misses
    
    # Unique contents: A, B, C
    # All 5 items were misses because cache was empty at start
    assert cached_provider.cache_stats["misses"] == 5
    assert cached_provider.cache_stats["hits"] == 0
    
    # Verify base provider received only unique items (uncached)
    # The implementation sends a batch of uncached items
    # uncached were: A (idx 0), B (idx 1), C (idx 3)
    # So base.embed_batch should be called once with [A, B, C]
    
    # Let's check the number of calls to base provider
    # MockEmbeddingProvider tracks calls in embed(), but embed_batch() calls embed() 
    # The Mock implementation of embed_batch just loops.
    # So we simply check total embed calls on the mock.
    assert len(mock_provider.embed_calls) == 3

@pytest.mark.asyncio
async def test_different_modalities_cache(cached_provider, mock_provider):
    # Mock provider needs to support image for this test
    # Access private _modalities list (MockEmbeddingProvider implementation detail)
    mock_provider._modalities.append(Modality.IMAGE)
    
    content = "same bytes"
    
    # 1. Text embedding
    emb1 = await cached_provider.embed(content, Modality.TEXT)
    # Cache miss
    assert cached_provider.cache_stats["misses"] == 1
    
    # 2. Image embedding (mocked)
    emb2 = await cached_provider.embed(content, Modality.IMAGE)
    # Should be another cache miss because key differs
    assert cached_provider.cache_stats["misses"] == 2
    
    # Verify keys are different in internal cache
    # Implementation detail: cache keys are hashes of "model_id:modality:content"
    # We can inspect _cache to verify there are 2 entries
    assert len(cached_provider._cache) == 2
    
    # Verify calling text again hits cache
    emb1_again = await cached_provider.embed(content, Modality.TEXT)
    assert cached_provider.cache_stats["misses"] == 2
    assert cached_provider.cache_stats["hits"] == 1
    assert emb1 == emb1_again
