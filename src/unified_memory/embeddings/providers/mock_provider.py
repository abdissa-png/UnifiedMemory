"""
Mock Embedding Provider for Testing.

Location: embeddings/providers/mock_provider.py

Generates deterministic embeddings for testing purposes.
"""

from __future__ import annotations

import hashlib
from typing import Any, List

from unified_memory.core.types import Modality
from unified_memory.embeddings.base import EmbeddingProvider


class MockEmbeddingProvider(EmbeddingProvider):
    """
    Mock embedding provider for testing.
    
    Generates deterministic embeddings based on content hash.
    This ensures:
    - Same content = same embedding (for test assertions)
    - Different content = different embedding
    - No external API calls
    """
    
    def __init__(
        self,
        model_id: str = "mock-embedding-model",
        dimension: int = 128,
        modalities: List[Modality] = None,
    ) -> None:
        self._model_id = model_id
        self._dimension = dimension
        self._modalities = modalities or [Modality.TEXT]
        
        # Track calls for test assertions
        self.embed_calls: List[tuple] = []
    
    @property
    def model_id(self) -> str:
        return self._model_id
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def supported_modalities(self) -> List[Modality]:
        return self._modalities
    
    async def embed(
        self,
        content: Any,
        modality: Modality = Modality.TEXT,
    ) -> List[float]:
        """
        Generate deterministic embedding from content hash.
        """
        self.validate_modality(modality)
        self.embed_calls.append((content, modality))
        
        # Create deterministic embedding from content hash
        content_str = content if isinstance(content, str) else str(content)
        hash_bytes = hashlib.sha256(content_str.encode()).digest()
        
        # Convert hash bytes to floats in [-1, 1] range
        embedding = []
        for i in range(self._dimension):
            # Use modulo to cycle through hash bytes
            byte_val = hash_bytes[i % len(hash_bytes)]
            # Normalize to [-1, 1]
            normalized = (byte_val / 127.5) - 1.0
            embedding.append(normalized)
        
        return embedding
    
    async def embed_batch(
        self,
        contents: List[Any],
        modality: Modality = Modality.TEXT,
    ) -> List[List[float]]:
        """
        Batch embed - just calls embed() for each (mock doesn't need optimization).
        """
        return [await self.embed(c, modality) for c in contents]
    
    def reset_calls(self) -> None:
        """Reset the call tracking for test assertions."""
        self.embed_calls.clear()
