"""
Serialization helpers for workflow step boundaries.

Converts domain objects (``ParsedDocument``, ``Chunk``, ``PageContent``, etc.)
to/from plain dicts that are JSON-safe and free of large binary payloads.

Binary data (page images, figure crops) are **not** included in the
serialized form — the caller is responsible for externalising those to an
``ArtifactStore`` *before* calling ``parsed_doc_to_dict`` and injecting
the resulting URIs into the page dicts.
"""

from __future__ import annotations

from dataclasses import asdict, fields
from typing import Any, Dict, List, Optional

from unified_memory.core.types import (
    PageContent,
    SourceReference,
    SourceType,
    BoundingBox,
    Chunk,
    Modality,
)
from unified_memory.ingestion.parsers.base import ParsedDocument


# ======================================================================
# SourceReference
# ======================================================================


def source_ref_to_dict(ref: SourceReference) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "source_id": ref.source_id,
        "source_type": ref.source_type.value if isinstance(ref.source_type, SourceType) else ref.source_type,
    }
    if ref.page_number is not None:
        d["page_number"] = ref.page_number
    if ref.bounding_box is not None:
        d["bounding_box"] = asdict(ref.bounding_box)
    if ref.caption:
        d["caption"] = ref.caption
    if ref.figure_ref:
        d["figure_ref"] = ref.figure_ref
    if ref.message_id:
        d["message_id"] = ref.message_id
    if ref.session_id:
        d["session_id"] = ref.session_id
    if ref.adjacent_elements:
        d["adjacent_elements"] = ref.adjacent_elements
    return d


def source_ref_from_dict(d: Dict[str, Any]) -> SourceReference:
    bb = d.get("bounding_box")
    if bb and isinstance(bb, dict):
        bb = BoundingBox(**bb)
    return SourceReference(
        source_id=d["source_id"],
        source_type=SourceType(d["source_type"]),
        page_number=d.get("page_number"),
        bounding_box=bb,
        caption=d.get("caption"),
        figure_ref=d.get("figure_ref"),
        message_id=d.get("message_id"),
        session_id=d.get("session_id"),
        adjacent_elements=d.get("adjacent_elements", []),
    )


# ======================================================================
# PageContent
# ======================================================================


def page_to_dict(page: PageContent) -> Dict[str, Any]:
    """Serialize a ``PageContent`` to a JSON-safe dict.

    ``full_page_image`` bytes are **dropped** — the caller must store
    images in an ``ArtifactStore`` and set ``image_uri`` in the returned
    dict before passing it between workflow steps.

    The same applies to ``image_bytes`` inside figure and table dicts.
    """
    figures_clean: List[Dict[str, Any]] = []
    for fig in page.figures:
        fig_copy = {k: v for k, v in fig.items() if k != "image_bytes"}
        figures_clean.append(fig_copy)

    tables_clean: List[Dict[str, Any]] = []
    for tbl in page.tables:
        tbl_copy = {k: v for k, v in tbl.items() if k != "image_bytes"}
        tables_clean.append(tbl_copy)

    return {
        "page_number": page.page_number,
        "document_id": page.document_id,
        "text_blocks": page.text_blocks,
        "figures": figures_clean,
        "tables": tables_clean,
        "full_text": page.full_text,
    }


def page_from_dict(d: Dict[str, Any]) -> PageContent:
    """Reconstruct a ``PageContent`` from a dict.

    Image bytes are not restored (they live in the artifact store).
    """
    return PageContent(
        page_number=d["page_number"],
        document_id=d["document_id"],
        text_blocks=d.get("text_blocks", []),
        figures=d.get("figures", []),
        tables=d.get("tables", []),
        full_page_image=None,
        full_text=d.get("full_text", ""),
    )


# ======================================================================
# ParsedDocument
# ======================================================================


def parsed_doc_to_dict(doc: ParsedDocument) -> Dict[str, Any]:
    """Serialize a ``ParsedDocument`` to a JSON-safe dict.

    All binary payloads (page images, figure/table image bytes) are
    stripped.  The caller must externalise them beforehand.
    """
    return {
        "document_id": doc.document_id,
        "source": source_ref_to_dict(doc.source),
        "title": doc.title,
        "pages": [page_to_dict(p) for p in doc.pages],
        "full_text": doc.full_text,
        "metadata": doc.metadata,
        "parse_errors": doc.parse_errors,
    }


def parsed_doc_from_dict(d: Dict[str, Any]) -> ParsedDocument:
    """Reconstruct a ``ParsedDocument`` from a dict."""
    return ParsedDocument(
        document_id=d["document_id"],
        source=source_ref_from_dict(d["source"]),
        title=d.get("title"),
        pages=[page_from_dict(p) for p in d.get("pages", [])],
        full_text=d.get("full_text", ""),
        metadata=d.get("metadata", {}),
        parse_errors=d.get("parse_errors", []),
    )


# ======================================================================
# Chunk (lightweight reference form for manifests)
# ======================================================================


def chunk_ref_to_dict(chunk: Chunk) -> Dict[str, Any]:
    """Convert a ``Chunk`` to a lightweight reference dict.

    Only metadata fields are kept — ``content`` and ``embedding`` are
    omitted because content lives in ``ContentStore`` and embeddings are
    written directly to the vector store.
    """
    return {
        "chunk_index": chunk.chunk_index,
        "document_id": chunk.document_id,
        "content_hash": chunk.content_hash,
        "page_number": chunk.page_number,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
        "token_count": chunk.token_count,
        "source_type": chunk.source_type.value if isinstance(chunk.source_type, SourceType) else chunk.source_type,
        "metadata": chunk.metadata,
    }


def chunk_ref_from_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Passthrough — chunk refs stay as plain dicts in manifests.

    Callers that need a full ``Chunk`` object should load content from
    ``ContentStore`` separately.
    """
    return d


def rebuild_chunk_from_ref(
    ref: Dict[str, Any],
    content: str,
) -> Chunk:
    """Reconstruct a ``Chunk`` from a manifest reference and stored content."""
    return Chunk(
        document_id=ref["document_id"],
        content=content,
        chunk_index=ref["chunk_index"],
        content_hash=ref.get("content_hash"),
        page_number=ref.get("page_number"),
        start_char=ref.get("start_char"),
        end_char=ref.get("end_char"),
        token_count=ref.get("token_count"),
        source_type=SourceType(ref["source_type"]) if ref.get("source_type") else SourceType.TEXT_BLOCK,
        metadata=ref.get("metadata", {}),
    )
