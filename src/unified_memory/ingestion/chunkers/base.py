"""
Base chunker interface.

Chunkers split parsed documents into smaller chunks for embedding and retrieval.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from unified_memory.core.types import Chunk, SourceReference
from unified_memory.ingestion.parsers.base import ParsedDocument


@dataclass
class ChunkingConfig:
    """
    Configuration for chunking behavior.
    
    Adjust these parameters based on:
    - Embedding model context window
    - Retrieval granularity requirements
    - Document characteristics
    """
    
    # Size limits
    chunk_size: int = 512  # Target tokens/characters per chunk
    chunk_overlap: int = 64  # Overlap between consecutive chunks
    
    # Boundaries
    respect_sentence_boundaries: bool = True
    respect_paragraph_boundaries: bool = True
    
    # Metadata
    include_page_numbers: bool = True
    include_section_headers: bool = True
    
    # Optional semantic parameters (for semantic chunkers)
    similarity_threshold: float = 0.5
    min_chunk_size: int = 100
    max_chunk_size: int = 2048

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.chunk_overlap >= self.chunk_size:
            # P1 fix #17 - Prevent infinite loops in chunkers
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size})."
            )


class Chunker(ABC):
    """
    Abstract base class for document chunkers.
    
    Implementations handle different chunking strategies:
    - FixedSizeChunker: Fixed token/character count
    - RecursiveChunker: Hierarchical splitting by separators
    - SemanticChunker: Embedding-based similarity grouping
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this chunker."""
        raise NotImplementedError
    
    @abstractmethod
    async def chunk(
        self,
        document: ParsedDocument,
        namespace: str,
        tenant_id: str,
        config: Optional[ChunkingConfig] = None,
    ) -> List[Chunk]:
        """
        Split a parsed document into chunks.
        
        Args:
            document: Parsed document to chunk
            namespace: Namespace for the resulting chunks
            embedding_model: Model ID used for content hashing
            
        Returns:
            List of Chunk objects ready for embedding
        """
        raise NotImplementedError
    
    def _create_chunk(
        self,
        text: str,
        document: ParsedDocument,
        chunk_index: int,
        namespace: str,
        tenant_id: str,
        page_number: Optional[int] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        config: Optional[ChunkingConfig] = None,
    ) -> Chunk:
        """
        Helper to create a Chunk with proper metadata.
        """
        from unified_memory.core.types import compute_content_hash, Modality

        # Canonical content hash (TEXT modality by default for document chunks)
        content_hash = compute_content_hash(text, tenant_id, Modality.TEXT)
        
        cfg = config or ChunkingConfig()

        metadata = {
            "chunker": self.name,
            "namespace": namespace,
            "document_id": document.document_id,
        }
        
        if document.title and cfg.include_section_headers:
            metadata["document_title"] = document.title
            
        if extra_metadata:
            metadata.update(extra_metadata)
        
        return Chunk(
            document_id=document.document_id,
            content=text,
            chunk_index=chunk_index,
            page_number=page_number,
            content_hash=content_hash,
            metadata=metadata,
        )

