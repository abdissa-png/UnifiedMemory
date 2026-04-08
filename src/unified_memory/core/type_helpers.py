"""
Shared helper functions used by core type modules.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Dict, List

from unified_memory.core.source_types import SourceLocation


def compute_content_hash(
    content: str,
    tenant_id: str,
    modality: object | None = None,
) -> str:
    """Canonical tenant-scoped content hash."""
    modality_str = getattr(modality, "value", "text")
    hash_input = f"{tenant_id}:{modality_str}:{content}"
    return hashlib.sha256(hash_input.encode()).hexdigest()


def compute_vector_id(
    content_hash: str,
    embedding_model: str,
    prefix: str = "text",
) -> str:
    """Compute a deterministic vector ID from content and model."""
    model_hash = hashlib.sha256(embedding_model.encode()).hexdigest()[:16]
    return f"{prefix}:{content_hash}:{model_hash}"


def compute_document_hash(content: str, tenant_id: str) -> str:
    """Tenant-scoped document hash."""
    hash_input = f"{tenant_id}:{content}"
    return hashlib.sha256(hash_input.encode()).hexdigest()


def utc_now() -> datetime:
    """Get current UTC time with timezone info."""
    return datetime.now(timezone.utc)

def make_entity_id(name: str, tenant_id: str) -> str:
    """Deterministic entity ID from name and tenant.

    The ID is intentionally name-only (no ``entity_type``) so that the same
    real-world entity extracted from different documents, or with varying type
    labels across extraction runs, always maps to the **same** graph node.
    This is the desired merge-by-name behaviour for knowledge graphs.

    Collision risk
    --------------
    Two genuinely different entities that share a name (e.g. "Python" the
    snake vs "Python" the language) will share a node.  When that matters,
    differentiate at query time using the ``entity_type`` property stored on
    the ``EntityNode``, or use the surrounding relationship context.
    The extractor's ``source_type`` / ``target_type`` hints (stored as edge
    properties on ``GraphEdge``) can also aid disambiguation without
    requiring separate nodes.

    node_type vs entity_type
    ------------------------
    ``node_type`` (``NodeType`` enum) is a structural graph meta-field that
    classifies *graph node kinds* (PAGE, PASSAGE, ENTITY, …).  It must
    never be conflated with ``entity_type``, which is a domain label
    assigned by the knowledge-graph extractor (e.g. "Person", "Concept").
    ``EntityNode.node_type`` is always ``NodeType.ENTITY``; the extractor's
    label lives in ``EntityNode.entity_type``.
    """
    normalized = name.strip().lower()
    return f"entity:{tenant_id}:{normalized}"

def normalize_relation_type(rel: str) -> str:
    """Normalize a relation type for consistent storage across backends.

    Uppercases and replaces spaces with underscores so that the same string
    is used in both NetworkX and Neo4j (where relationship types are
    conventionally UPPER_SNAKE_CASE).
    """
    return rel.strip().upper().replace(" ", "_")


def source_locations_to_parallel_arrays(
    locs: List["SourceLocation"],
) -> Dict[str, List]:
    """Convert SourceLocation list to parallel arrays for graph-store storage.

    Neo4j only supports arrays of primitives, so we store provenance as two
    aligned lists rather than a list of objects.
    """
    return {
        "source_doc_ids": [loc.document_id for loc in locs],
        "source_chunk_indices": [loc.chunk_index for loc in locs],
    }


def parallel_arrays_to_source_locations(
    doc_ids: List[str], chunk_indices: List[int]
) -> List["SourceLocation"]:
    """Reconstruct SourceLocation list from parallel arrays."""
    return [
        SourceLocation(document_id=d, chunk_index=i)
        for d, i in zip(doc_ids, chunk_indices)
    ]