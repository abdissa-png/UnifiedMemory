"""
Cached Embedding Provider.

Location: embeddings/cache.py
Design Reference: UNIFIED_MEMORY_SYSTEM_DESIGN.md Section 3.3

Wraps any EmbeddingProvider with caching using content hashes.
This eliminates redundant embedding calls for identical content.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from unified_memory.core.types import Modality, compute_content_hash
from unified_memory.embeddings.base import EmbeddingProvider


class CachedEmbeddingProvider(EmbeddingProvider):
    """
    Caching wrapper for embedding providers.
    
    Design Reference: UNIFIED_MEMORY_SYSTEM_DESIGN.md Section 3.3
    "CachedEmbeddingProvider wrapper"
    
    Uses content hash as cache key to ensure:
    - Same content + same model = cache hit
    - Same content + different model = cache miss (different hash)
    
    Cache can be:
    - In-memory dict (for testing/short-lived processes)
    - Redis/KV store (for production persistence)
    """
    
    def __init__(
        self,
        base_provider: EmbeddingProvider,
        cache: Optional[Dict[str, List[float]]] = None,
        kv_store: Optional[Any] = None,  # KVStoreBackend
    ) -> None:
        """
        Initialize cached provider.
        
        Args:
            base_provider: The underlying embedding provider
            cache: Optional in-memory cache dict
            kv_store: Optional KV store for persistent caching
        """
        self._base = base_provider
        self._cache: Dict[str, List[float]] = cache if cache is not None else {}
        self._kv_store = kv_store
        
        # Stats
        self._hits = 0
        self._misses = 0
    
    @property
    def model_id(self) -> str:
        return self._base.model_id
    
    @property
    def dimension(self) -> int:
        return self._base.dimension
    
    @property
    def supported_modalities(self) -> List[Modality]:
        return self._base.supported_modalities
    
    @property
    def cache_stats(self) -> Dict[str, int]:
        """Return cache hit/miss statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": total,
            "hit_rate": hit_rate,
        }
    
    def _compute_cache_key(self, content: Any, modality: Modality) -> str:
        """
        Compute cache key from content.
        
        Uses canonical hash: SHA256("{model_id}:{content}")
        """
        content_str = content if isinstance(content, str) else str(content)
        return compute_content_hash(content_str, self.model_id)
    
    async def embed(
        self,
        content: Any,
        modality: Modality = Modality.TEXT,
    ) -> List[float]:
        """
        Get embedding with caching.
        
        1. Compute content hash
        2. Check cache
        3. If miss, call base provider and cache result
        """
        self.validate_modality(modality)
        
        cache_key = self._compute_cache_key(content, modality)
        
        # Check in-memory cache first
        if cache_key in self._cache:
            self._hits += 1
            return self._cache[cache_key]
        
        # Check KV store if available
        if self._kv_store:
            cached = await self._kv_store.get(f"emb_cache:{cache_key}")
            if cached and "embedding" in cached:
                self._hits += 1
                embedding = cached["embedding"]
                self._cache[cache_key] = embedding  # Populate local cache
                return embedding
        
        # Cache miss - generate embedding
        self._misses += 1
        embedding = await self._base.embed(content, modality)
        
        # Store in cache
        self._cache[cache_key] = embedding
        
        # Store in KV if available
        if self._kv_store:
            await self._kv_store.set(
                f"emb_cache:{cache_key}",
                {"embedding": embedding, "model_id": self.model_id},
            )
        
        return embedding
    
    async def embed_batch(
        self,
        contents: List[Any],
        modality: Modality = Modality.TEXT,
    ) -> List[List[float]]:
        """
        Batch embedding with caching.
        
        1. Check cache for each item
        2. Deduplicate uncached items
        3. Batch-embed unique cache misses
        4. update cache and populate results
        """
        self.validate_modality(modality)
        
        results: List[Optional[List[float]]] = [None] * len(contents)
        
        # specific uncached items to their original indices in `contents`
        # Map: cache_key -> list of indices
        pending_misses: Dict[str, List[int]] = {}
        # Map: cache_key -> content
        failed_key_content_map: Dict[str, Any] = {}
        
        for i, content in enumerate(contents):
            cache_key = self._compute_cache_key(content, modality)
            
            if cache_key in self._cache:
                self._hits += 1
                results[i] = self._cache[cache_key]
            else:
                self._misses += 1
                if cache_key not in pending_misses:
                    pending_misses[cache_key] = []
                    failed_key_content_map[cache_key] = content
                pending_misses[cache_key].append(i)
        
        # If we have unique misses, process them
        if pending_misses:
            unique_keys = list(pending_misses.keys())
            unique_contents = [failed_key_content_map[k] for k in unique_keys]
            
            new_embeddings = await self._base.embed_batch(unique_contents, modality)
            
            # Distribute results
            for key, embedding in zip(unique_keys, new_embeddings):
                # Update cache
                self._cache[key] = embedding
                
                # Update KV if available (fire and forget or wait?)
                # For this implementation we await to ensure consistency
                if self._kv_store:
                    await self._kv_store.set(
                        f"emb_cache:{key}",
                        {"embedding": embedding, "model_id": self.model_id},
                    )
                
                # Fill all indices that requested this content
                for idx in pending_misses[key]:
                    results[idx] = embedding
        
        return results  # type: ignore
    
    def clear_cache(self) -> None:
        """Clear the in-memory cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
