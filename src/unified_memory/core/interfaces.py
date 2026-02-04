"""
Core Interface Protocols - REQUIRED DEFINITIONS.

These Protocol classes define contracts that must be implemented.
Using Protocol (structural subtyping) allows flexibility without inheritance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Callable, runtime_checkable, Tuple

from .types import (
    Chunk,
    PageContent,
    RetrievalResult,
)



@runtime_checkable
class EmbeddingProvider(Protocol):
    """
    Protocol for embedding providers.
    """
    
    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        ...
        
    async def embed(
        self,
        text: str,
    ) -> List[float]:
        """Embed a single string."""
        ...
        
    async def embed_batch(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """Embed a batch of strings."""
        ...


@runtime_checkable
class LLMProvider(Protocol):
    """
    Protocol for LLM providers (answer generation, extraction).

    Implementations: OpenAI, Anthropic, local models, etc.
    """

    @property
    def model_id(self) -> str:
        """Unique identifier for the model."""
        ...

    @property
    def max_tokens(self) -> int:
        """Maximum context window size."""
        ...

    async def generate(
        self,
        prompt: str,
        max_output_tokens: int = 1024,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """Generate text completion."""
        ...

    async def generate_with_images(
        self,
        prompt: str,
        images: List[bytes],
        max_output_tokens: int = 1024,
    ) -> str:
        """Generate with multimodal input (for MLLM)."""
        ...


@runtime_checkable
class Reranker(Protocol):
    """
    Protocol for cross-encoder rerankers.

    Implementations: Cohere Rerank, BGE Reranker, etc.
    """

    @property
    def model_id(self) -> str:
        """Unique identifier for the reranker model."""
        ...

    async def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """
        Rerank results using cross-encoder.

        Returns results sorted by relevance with updated scores.
        """
        ...


@runtime_checkable
class SparseRetriever(Protocol):
    """
    Protocol for sparse retrieval (BM25, SPLADE).

    Design: Separate from vector store because sparse indexes
    have different storage requirements (inverted index).
    """

    async def add_namespace(
        self,
        content_id: str,
        namespace: str,
    ) -> bool:
        """Add a namespace to an existing document."""
        ...

    async def index(
        self,
        documents: List[Dict[str, Any]],
        namespace: str,
    ) -> int:
        """Index documents for sparse retrieval. Returns count indexed."""
        ...

    async def retrieve(
        self,
        query: str,
        namespaces: List[str],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Retrieve documents using sparse matching."""
        ...

    async def delete(
        self,
        doc_ids: List[str],
        namespace: str,
        document_id: Optional[str] = None,
    ) -> int:
        """Delete documents from sparse index. Returns count deleted."""
        ...


@runtime_checkable
class DocumentParser(Protocol):
    """
    Protocol for document parsing (PDF, DOCX, HTML).

    Implementations: DocTR, PyMuPDF, Unstructured, etc.
    """

    @property
    def supported_formats(self) -> List[str]:
        """List of supported file extensions."""
        ...

    async def parse(
        self,
        content: bytes,
        file_type: str,
        extract_images: bool = True,
        extract_tables: bool = True,
    ) -> List[PageContent]:
        """
        Parse document into pages with text, images, tables.

        Returns list of PageContent, one per page.
        """
        ...


@runtime_checkable
class Chunker(Protocol):
    """
    Protocol for text chunking strategies.

    Implementations: SemanticChunker, RecursiveChunker, FixedChunker
    """

    @property
    def strategy(self) -> str:
        """Chunking strategy name."""
        ...

    @property
    def chunk_size(self) -> int:
        """Target chunk size in characters."""
        ...

    @property
    def overlap(self) -> int:
        """Overlap between chunks in characters."""
        ...

    async def chunk_text(
        self,
        text: str,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Chunk a single text into Chunk objects."""
        ...

    async def chunk_pages(
        self,
        pages: List[PageContent],
        document_id: str,
    ) -> List[Chunk]:
        """Chunk multiple pages, preserving page references."""
        ...





# Type alias for chunker factory
ChunkerFactory = Callable[[str, int, int], Chunker]

