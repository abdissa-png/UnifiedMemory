"""
Embedding Provider Base Classes.

Location: embeddings/base.py
Design Reference: UNIFIED_MEMORY_SYSTEM_DESIGN.md Section 3

This module provides the abstract base class for all embedding providers,
supporting multiple modalities (text, image, shared space).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Union

from unified_memory.core.types import Modality


@dataclass
class EmbeddingResult:
    """Result of an embedding operation."""
    embedding: List[float]
    model_id: str
    dimension: int
    tokens_used: Optional[int] = None


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.
    
    Design Reference: UNIFIED_MEMORY_SYSTEM_DESIGN.md Section 3.2
    
    Implementations:
    - OpenAIEmbeddingProvider
    - CohereEmbeddingProvider  
    - SentenceTransformerProvider (local)
    - MockEmbeddingProvider (testing)
    
    Key design decisions:
    - Single class hierarchy for text AND vision embeddings
    - Modality parameter determines which encoder to use
    - Batch operations are first-class citizens (not N+1 loops)
    - Model ID is part of content hash for deduplication
    """
    
    @property
    @abstractmethod
    def model_id(self) -> str:
        """
        Unique identifier for this embedding model.
        
        Used in content hash computation: SHA256("{model_id}:{content}")
        Examples: "text-embedding-3-small", "clip-vit-base-patch32"
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Embedding dimension.
        
        Examples: 1536 for text-embedding-3-small, 512 for CLIP
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def supported_modalities(self) -> List[Modality]:
        """
        List of modalities this provider supports.
        
        Most providers support only TEXT or only IMAGE.
        DualEmbeddingProvider supports both via composition.
        """
        raise NotImplementedError
    
    @abstractmethod
    async def embed(
        self,
        content: Any,
        modality: Modality = Modality.TEXT,
    ) -> List[float]:
        """
        Generate embedding for single content.
        
        Args:
            content: Text string or image bytes depending on modality
            modality: Which modality to embed as
            
        Returns:
            Embedding vector as list of floats
            
        Raises:
            ValueError: If modality not supported
        """
        raise NotImplementedError
    
    async def embed_batch(
        self,
        contents: List[Any],
        modality: Modality = Modality.TEXT,
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple contents in batch.
        
        Design Reference: INITIAL_PLAN.md L206-207
        "Replace N+1 await embed_text(...) loops with batch embedding"
        
        Default implementation uses asyncio.gather for concurrency.
        Subclasses should override with native batch API if available.
        
        Args:
            contents: List of content items (text strings or image bytes)
            modality: Which modality to embed as
            
        Returns:
            List of embedding vectors
        """
        import asyncio
        
        tasks = [self.embed(content, modality) for content in contents]
        return await asyncio.gather(*tasks)
    
    def supports_modality(self, modality: Modality) -> bool:
        """Check if this provider supports the given modality."""
        return modality in self.supported_modalities
    
    def validate_modality(self, modality: Modality) -> None:
        """Raise ValueError if modality not supported."""
        if not self.supports_modality(modality):
            supported = [m.value for m in self.supported_modalities]
            raise ValueError(
                f"Modality {modality.value} not supported. "
                f"Supported: {supported}"
            )
