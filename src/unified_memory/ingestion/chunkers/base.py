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


class Chunker(ABC):
    """
    Abstract base class for document chunkers.
    
    Implementations handle different chunking strategies:
    - FixedSizeChunker: Fixed token/character count
    - RecursiveChunker: Hierarchical splitting by separators
    - SemanticChunker: Embedding-based similarity grouping
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None) -> None:
        self.config = config or ChunkingConfig()
    
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
        embedding_model: str,
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
        embedding_model: str,
        page_number: Optional[int] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Chunk:
        """
        Helper to create a Chunk with proper metadata.
        """
        from unified_memory.core.types import compute_content_hash
        
        # Canonical content hash
        content_hash = compute_content_hash(text, embedding_model)
        
        metadata = {
            "chunker": self.name,
            "namespace": namespace,
        }
        
        if document.title and self.config.include_section_headers:
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

