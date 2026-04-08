"""
Ingestion type exports extracted from ``core.types``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .enums import SourceType
from .type_helpers import compute_content_hash

@dataclass
class PageContent:
    """All content from a single page for unified extraction."""

    page_number: int
    document_id: str

    # Text blocks with spatial info
    text_blocks: List[Dict[str, Any]] = field(default_factory=list)

    # Figures with spatial info
    figures: List[Dict[str, Any]] = field(default_factory=list)

    # Tables with spatial info
    tables: List[Dict[str, Any]] = field(default_factory=list)

    # Full page image for overall context
    full_page_image: Optional[bytes] = None

    # Extracted text (concatenated)
    full_text: str = ""


@dataclass
class Chunk:
    """Document chunk - ENHANCED with content hash."""

    document_id: str
    content: str
    chunk_index: int = 0
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    token_count: Optional[int] = None
    embedding: Optional[List[float]] = None

    # Content hash for deduplication
    content_hash: Optional[str] = None

    # Source info
    page_number: Optional[int] = None
    source_type: SourceType = SourceType.TEXT_BLOCK

    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_content_hash(self, tenant_id: str) -> str:
        """
        Get content hash using canonical function.

        Delegates to compute_content_hash().
        """

        return compute_content_hash(self.content, tenant_id)


# Type alias for chunker factory used in ingestion/chunkers/base.py
ChunkerFactory = Callable[[str, int, int], "Chunker"]  # defined in ingestion/chunkers/base.py

__all__ = ["Chunk", "ChunkerFactory", "PageContent"]
