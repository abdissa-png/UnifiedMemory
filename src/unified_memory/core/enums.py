"""
Enums for the unified memory system.
"""

from __future__ import annotations
from enum import StrEnum

class MemoryType(StrEnum):
    """Types of memories."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    CORE = "core"
    PREFERENCE = "preference"


class MemoryStatus(StrEnum):
    """Memory validity status."""

    VALID = "valid"
    INVALID_SUPERSEDED = "invalid_superseded"
    INVALID_RETRACTED = "invalid_retracted"
    PENDING = "pending"


class MemoryLayer(StrEnum):
    """H-MEM memory layers."""

    L1_WORKING = "l1_working"
    L2_EPISODIC = "l2_episodic"
    L3_SEMANTIC = "l3_semantic"
    L4_ARCHIVAL = "l4_archival"


class Permission(StrEnum):
    """
    Namespace permission types.

    Explicit permission model instead of free-form strings.
    """

    READ = "read"  # Can search/retrieve from namespace
    WRITE = "write"  # Can add/modify content
    DELETE = "delete"  # Can delete content
    ADMIN = "admin"  # Can manage ACL, delete namespace
    SHARE = "share"  # Can share namespace with others


class ACLEffect(StrEnum):
    """ACL rule effect (allow or deny)."""

    ALLOW = "allow"
    DENY = "deny"


class SourceType(StrEnum):
    """Source types for entity provenance (from MMKG_DESIGN)."""

    TEXT_BLOCK = "text_block"
    FIGURE = "figure"
    TABLE = "table"
    CAPTION = "caption"
    FULL_PAGE = "full_page"
    CONVERSATION = "conversation"  # Added for Mem0-style memories


class CollectionType(StrEnum):
    """Vector collection types (tenant-level physical collections)."""

    TEXTS = "texts"  # Document chunks
    ENTITIES = "entities"  # Extracted entities (text embeddings)
    RELATIONS = "relations"  # Extracted relations (text embeddings)
    PAGE_IMAGES = "page_images"  # Visual retrieval (vision embeddings)
    MEMORIES = "memories"  # Consolidated memories (Mem0-style)


class Modality(StrEnum):
    """
    Content modalities - UNIFIED from both docs.

    INPUT MODALITIES (source content types):
    - TEXT: Plain text content
    - IMAGE: Image files (PNG, JPEG, etc.)
    - AUDIO: Audio files
    - VIDEO: Video files
    - DOCUMENT: Multi-modal documents (PDF with text + images)

    OUTPUT MODALITY (embedding space):
    - SHARED: The aligned embedding space where different modalities
      can be compared directly.
    """

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
    SHARED = "shared"  # aligned embedding space, not an input modality


class ConsolidationAction(StrEnum):
    """Memory consolidation actions."""

    ADD = "add"
    UPDATE = "update"
    MERGE = "merge"  # Combine two memories into one
    DELETE = "delete"
    SUPERSEDE = "supersede"  # Mark old as invalid, create new
    NOOP = "noop"


class NodeType(StrEnum):
    """Enumeration of valid node types."""

    ENTITY = "entity"
    PASSAGE = "passage"
    PAGE = "page"


__all__ = [
    "ACLEffect",
    "CollectionType",
    "MemoryLayer",
    "MemoryStatus",
    "MemoryType",
    "Modality",
    "ConsolidationAction",
    "NodeType",
    "Permission",
    "SourceType",
]
