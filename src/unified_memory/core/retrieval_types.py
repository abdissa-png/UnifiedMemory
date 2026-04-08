"""
Retrieval type exports extracted from ``core.types``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .enums import Modality

@dataclass
class VectorSearchResult:
    """
    Raw result from vector store search operations.

    This is the low-level type returned directly by VectorStoreBackend.
    It gets transformed into RetrievalResult by the retrieval pipeline.
    """

    id: str
    score: float
    embedding: Optional[List[float]] = None  # Optional: only if requested

    # In updated design, chunk text is NOT stored in the vector DB.
    # Retrieval should hydrate content from ContentStore using content_hash/content_id.
    content: str = ""

    # All vector store metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Collection it came from (for multi-collection queries)
    collection: Optional[str] = None


@dataclass
class RetrievalResult:
    """Result from retrieval operations - ENHANCED."""

    id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"  # Which retrieval path: "dense", "sparse", "graph", "visual"

    # For MMKG-style results
    entity_ids: List[str] = field(default_factory=list)
    relation_ids: List[str] = field(default_factory=list)
    page_number: Optional[int] = None

    # Cross-modal info
    modality: Modality = Modality.TEXT

    # Evidence for answer generation
    evidence_type: Optional[str] = None  # "text", "visual", "graph"


@dataclass
class QueryResult:
    """Full query result with provenance (MMKG-style)."""

    answer: str
    results: List[RetrievalResult]

    # Evidence breakdown
    visual_evidence: List[Dict[str, Any]] = field(default_factory=list)
    graph_evidence: Dict[str, Any] = field(default_factory=dict)

    # Intermediate answers for fusion
    intermediate_answers: Dict[str, str] = field(default_factory=dict)

    confidence: float = 0.0

    # Metrics
    retrieval_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

__all__ = ["QueryResult", "RetrievalResult", "VectorSearchResult"]
