"""
Core Interface Protocols - REQUIRED DEFINITIONS.

These Protocol classes define contracts that must be implemented.
Using Protocol (structural subtyping) allows flexibility without inheritance.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from .types import (
    RetrievalResult,
)


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


