"""
Core Type Definitions - UNIFIED.

This module is the single source of truth for core dataclasses and enums
used throughout the unified memory system.

It consolidates types from:
- Existing prototypes
- MULTIMODAL_GRAPHRAG_DESIGN.md (ExtractedEntity, SourceReference, etc.)
- ARCHITECTURAL_REFACTORING_PLAN.md (enhanced consolidation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Callable
import asyncio
import uuid
import hashlib


# ============================================================================
# CANONICAL HASH FUNCTION (Single Source of Truth)
# ============================================================================


def compute_content_hash(
    content: str,
    tenant_id: str,
    modality: "Modality" | None = None,
) -> str:
    """
    CANONICAL content hash computation for deduplication.

    ALL code that needs to compute content hashes MUST use this function.
    Do NOT duplicate this logic elsewhere.

    Format: SHA256("{tenant_id}:{modality.value}:{content}")

    Design decisions:
    - Tenant ID comes FIRST for explicit tenant scoping (GDPR/compliance).
    - Modality is included so TEXT vs IMAGE (etc.) are distinct.
    - Full SHA256 hash (64 hex chars) for collision resistance.
    - Same content + same tenant + same modality = same hash.
    """

    # Default to TEXT when modality is not provided.
    modality_str = getattr(modality, "value", "text")
    hash_input = f"{tenant_id}:{modality_str}:{content}"
    return hashlib.sha256(hash_input.encode()).hexdigest()


def compute_vector_id(
    content_hash: str,
    embedding_model: str,
    prefix: str = "text",
) -> str:
    """
    Compute a deterministic vector ID from content+model.

    Format: "{prefix}:{content_hash}:{model_hash}"
    The model_hash ensures that the same content embedded with different
    models gets distinct vector IDs.
    """
    model_hash = hashlib.sha256(embedding_model.encode()).hexdigest()[:16]
    return f"{prefix}:{content_hash}:{model_hash}"


def compute_document_hash(content: str, tenant_id: str) -> str:
    """
    Tenant-scoped DOCUMENT hash for document-level deduplication.

    This intentionally does *not* depend on the embedding model so that
    the same normalized document content within a tenant always maps to
    the same hash, regardless of embedding configuration.

    Format: SHA256("{tenant_id}:{content}")
    """

    hash_input = f"{tenant_id}:{content}"
    return hashlib.sha256(hash_input.encode()).hexdigest()


def utc_now() -> datetime:
    """
    Get current UTC time with timezone info.

    Use this instead of datetime.now() to avoid timezone issues.
    """

    return datetime.now(timezone.utc)


# ============================================================================
# ENUMERATIONS (Consolidated)
# ============================================================================


class MemoryType(Enum):
    """Types of memories."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    CORE = "core"
    PREFERENCE = "preference"


class MemoryStatus(Enum):
    """Memory validity status."""

    VALID = "valid"
    INVALID_SUPERSEDED = "invalid_superseded"
    INVALID_RETRACTED = "invalid_retracted"
    PENDING = "pending"


class MemoryLayer(Enum):
    """H-MEM memory layers."""

    L1_WORKING = "l1_working"
    L2_EPISODIC = "l2_episodic"
    L3_SEMANTIC = "l3_semantic"
    L4_ARCHIVAL = "l4_archival"


class Permission(Enum):
    """
    Namespace permission types.

    Explicit permission model instead of free-form strings.
    """

    READ = "read"  # Can search/retrieve from namespace
    WRITE = "write"  # Can add/modify content
    DELETE = "delete"  # Can delete content
    ADMIN = "admin"  # Can manage ACL, delete namespace
    SHARE = "share"  # Can share namespace with others


class ACLEffect(Enum):
    """ACL rule effect (allow or deny)."""

    ALLOW = "allow"
    DENY = "deny"


@dataclass
class ACLEntry:
    """
    Access Control List entry.

    - Explicit effect (ALLOW/DENY) rather than implied allow
    - DENY takes precedence over ALLOW (fail-secure)
    - Can target user_id, role, or "*" for public
    """

    principal: str  # user_id, role name, or "*" for public
    principal_type: str  # "user", "role", or "public"
    permissions: List[Permission]
    effect: ACLEffect = ACLEffect.ALLOW

    # Optional: inheritance from parent namespace
    inherited: bool = False
    source_namespace: Optional[str] = None  # Where the ACL was inherited from


@dataclass
class NamespaceACL:
    """
    Full ACL for a namespace.

    Resolution order:
    1. Explicit DENY rules (highest priority)
    2. Explicit ALLOW rules
    3. Inherited DENY rules
    4. Inherited ALLOW rules
    5. Default: DENY (fail-secure)
    """

    entries: List[ACLEntry] = field(default_factory=list)

    # Inheritance settings
    inherit_from_parent: bool = True
    inherit_from_tenant: bool = True

    def check_permission(
        self,
        principal: str,
        permission: Permission,
        roles: Optional[List[str]] = None,
        tenant_acl: Optional["NamespaceACL"] = None,
    ) -> bool:
        """
        Check if principal has permission, incorporating tenant inheritance.

        Args:
            principal: user_id to check
            permission: Permission to check
            roles: Optional list of roles the user has
            tenant_acl: Optional tenant-level default ACL to fall back to if inherit_from_tenant is True
        """

        roles = roles or []

        # 1. Check explicit namespace DENY first (highest priority)
        for entry in self.entries:
            if entry.effect == ACLEffect.DENY and self._matches_principal(
                entry, principal, roles
            ):
                if permission in entry.permissions:
                    return False

        # 2. Check explicit namespace ALLOW
        for entry in self.entries:
            if entry.effect == ACLEffect.ALLOW and self._matches_principal(
                entry, principal, roles
            ):
                if permission in entry.permissions:
                    return True
                    
        # 3. Check inherited ACL if permitted
        if self.inherit_from_tenant and tenant_acl is not None:
            # Check tenant DENY
            for entry in tenant_acl.entries:
                if entry.effect == ACLEffect.DENY and self._matches_principal(
                    entry, principal, roles
                ):
                    if permission in entry.permissions:
                        return False
            # Check tenant ALLOW
            for entry in tenant_acl.entries:
                if entry.effect == ACLEffect.ALLOW and self._matches_principal(
                    entry, principal, roles
                ):
                    if permission in entry.permissions:
                        return True

        # 4. Default: DENY
        return False

    def _matches_principal(
        self,
        entry: ACLEntry,
        principal: str,
        roles: List[str],
    ) -> bool:
        """Check if ACL entry matches the principal."""

        if entry.principal_type == "public" or entry.principal == "*":
            return True
        if entry.principal_type == "user" and entry.principal == principal:
            return True
        if entry.principal_type == "role" and entry.principal in roles:
            return True
        return False

    # ---- serialisation helpers ----

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict (Enum values → strings)."""
        return {
            "entries": [
                {
                    "principal": e.principal,
                    "principal_type": e.principal_type,
                    "permissions": [p.value for p in e.permissions],
                    "effect": e.effect.value,
                    "inherited": e.inherited,
                    "source_namespace": e.source_namespace,
                }
                for e in self.entries
            ],
            "inherit_from_parent": self.inherit_from_parent,
            "inherit_from_tenant": self.inherit_from_tenant,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NamespaceACL":
        """Reconstruct from a dict produced by ``to_dict``."""
        if not data:
            return cls()
        entries = []
        for e in data.get("entries", []):
            entries.append(
                ACLEntry(
                    principal=e["principal"],
                    principal_type=e["principal_type"],
                    permissions=[Permission(p) for p in e["permissions"]],
                    effect=ACLEffect(e["effect"]),
                    inherited=e.get("inherited", False),
                    source_namespace=e.get("source_namespace"),
                )
            )
        return cls(
            entries=entries,
            inherit_from_parent=data.get("inherit_from_parent", True),
            inherit_from_tenant=data.get("inherit_from_tenant", True),
        )


class Modality(Enum):
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


class ConsolidationAction(Enum):
    """Memory consolidation actions."""

    ADD = "add"
    UPDATE = "update"
    MERGE = "merge"  # Combine two memories into one
    DELETE = "delete"
    SUPERSEDE = "supersede"  # Mark old as invalid, create new
    NOOP = "noop"


@dataclass
class ConsolidationTrigger:
    """
    Defines when memory consolidation should be triggered.

    Triggers:
    - THRESHOLD: Memory count exceeds threshold
    - SCHEDULED: Periodic consolidation (e.g., daily)
    - ON_CONFLICT: When conflicting memories detected
    - MANUAL: User-initiated consolidation
    """

    trigger_type: str  # "threshold", "scheduled", "on_conflict", "manual"
    threshold_count: Optional[int] = 100  # For threshold trigger
    schedule_cron: Optional[str] = None  # For scheduled trigger
    namespace_pattern: str = "*"  # Which namespaces to consolidate


@dataclass
class ConsolidationRule:
    """
    Rule for determining consolidation action.

    Conflict resolution strategy:
    1. Exact duplicates (same content hash): MERGE, keep older timestamp
    2. Near duplicates (>0.95 similarity): MERGE, keep richer content
    3. Conflicting facts (contradiction detected): SUPERSEDE
    4. Complementary memories: NOOP, keep both
    """

    memory_type: MemoryType
    similarity_threshold: float = 0.95
    contradiction_detection: bool = True
    prefer_recent: bool = True  # On conflict, prefer recent memory


@dataclass
class ConsolidationResult:
    """Result of a consolidation operation."""

    action: ConsolidationAction
    source_ids: List[str]  # IDs of memories that were consolidated
    result_id: Optional[str]  # ID of resulting memory (if merged/updated)
    reason: str  # Why this action was taken
    confidence: float = 1.0  # Confidence in the consolidation decision


class SourceType(Enum):
    """Source types for entity provenance (from MMKG_DESIGN)."""

    TEXT_BLOCK = "text_block"
    FIGURE = "figure"
    TABLE = "table"
    CAPTION = "caption"
    FULL_PAGE = "full_page"
    CONVERSATION = "conversation"  # Added for Mem0-style memories


class CollectionType(Enum):
    """Vector collection types (tenant-level physical collections)."""

    TEXTS = "texts"  # Document chunks
    ENTITIES = "entities"  # Extracted entities (text embeddings)
    RELATIONS = "relations"  # Extracted relations (text embeddings)
    PAGE_IMAGES = "page_images"  # Visual retrieval (vision embeddings)
    MEMORIES = "memories"  # Consolidated memories (Mem0-style)


# ============================================================================
# SOURCE PROVENANCE (from MMKG_DESIGN, extended)
# ============================================================================


@dataclass
class BoundingBox:
    """
    Spatial location on a page (from OCR/layout analysis).

    Coordinates are normalized to 0-1 range for resolution independence.
    """

    x: float
    y: float
    width: float
    height: float
    page: int


@dataclass
class SourceLocation:
    """
    Canonical (document, chunk) location for provenance tracking.
 
    Keeps explicit (document_id, chunk_index) pairs for accurate provenance.
    """
 
    document_id: str
    chunk_index: int
 
    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
        }
 
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceLocation":
        return cls(
            document_id=data["document_id"],
            chunk_index=data["chunk_index"],
        )


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


@dataclass
class SourceReference:
    """
    Reference to the source where content was extracted.

    Supports both:
    - MMKG-style: page/figure/table sources
    - Mem0-style: conversation/message sources
    """

    source_id: str
    source_type: SourceType

    # For document sources
    page_number: Optional[int] = None
    bounding_box: Optional[BoundingBox] = None
    caption: Optional[str] = None
    figure_ref: Optional[str] = None  # e.g., "Figure 3"

    # For conversation sources
    message_id: Optional[str] = None
    session_id: Optional[str] = None

    # Spatial context
    adjacent_elements: List[str] = field(default_factory=list)


# ============================================================================
# ENTITY & RELATION TYPES (Consolidated)
# ============================================================================


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


@dataclass
class Entity:
    """
    Knowledge graph entity - UNIFIED.

    Consolidates:
    - Basic Entity from earlier prototypes
    - ExtractedEntity from MMKG_DESIGN
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    entity_type: str = "entity"
    description: str = ""

    # Embedding for retrieval
    embedding: Optional[List[float]] = None

    # Canonical provenance: explicit (document_id, chunk_index) pairs
    source_locations: List[SourceLocation] = field(default_factory=list)

    # Confidence from extraction (0-1)
    confidence: float = 1.0

    # Flexible properties
    properties: Dict[str, Any] = field(default_factory=dict)

    # Namespace for isolation
    namespace: str = "default"

    def add_source(self, document_id: str, chunk_id: str):
        """Add a source reference to source_locations."""
        try:
            idx = int(chunk_id)
        except (TypeError, ValueError):
            return

        loc = SourceLocation(document_id=document_id, chunk_index=idx)
        if loc not in self.source_locations:
            self.source_locations.append(loc)

    def get_embedding_text(self) -> str:
        """Text representation for embedding generation.

        Includes the entity_type when available so that embeddings capture
        lightweight type information without over-fragmenting IDs.
        """
        parts: List[str] = [self.name]
        # Avoid repeating the default generic label
        if self.entity_type and self.entity_type != "entity":
            parts.append(f"({self.entity_type})")
        if self.description:
            parts.append(f": {self.description}")
        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type,
            "description": self.description,
            "embedding": self.embedding,
            "source_locations": [
                {"document_id": loc.document_id, "chunk_index": loc.chunk_index}
                for loc in self.source_locations
            ],
            "confidence": self.confidence,
            "properties": self.properties,
            "namespace": self.namespace,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        """Deserialize from dictionary."""
        raw_locs = data.get("source_locations", [])
        source_locations: List[SourceLocation] = [
            SourceLocation(
                document_id=loc["document_id"],
                chunk_index=loc["chunk_index"],
            )
            for loc in raw_locs
            if "document_id" in loc and "chunk_index" in loc
        ]

        return cls(
            id=data["id"],
            name=data["name"],
            entity_type=data.get("entity_type", "entity"),
            description=data.get("description", ""),
            embedding=data.get("embedding"),
            source_locations=source_locations,
            confidence=data.get("confidence", 1.0),
            properties=data.get("properties", {}),
            namespace=data.get("namespace", "default"),
        )


@dataclass
class Relation:
    """
    Knowledge graph relation - UNIFIED.

    Consolidates:
    - Triple from earlier types (renamed for clarity)
    - ExtractedRelation from MMKG_DESIGN
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Authoritative entity references (use these for graph operations)
    subject_id: str = ""  # Entity ID - AUTHORITATIVE
    predicate: str = ""  # Relationship type
    object_id: str = ""  # Entity ID - AUTHORITATIVE

    # Denormalized names (for display/embedding - may be stale)
    subject: str = ""  # Entity name - DENORMALIZED, may be stale
    object: str = ""  # Entity name - DENORMALIZED, may be stale
    names_synced_at: Optional[datetime] = None  # When names were last verified

    # Rich description
    description: str = ""
    keywords: List[str] = field(default_factory=list)

    # Embedding for retrieval
    embedding: Optional[List[float]] = None

    # Canonical provenance: explicit (document_id, chunk_index) pairs
    source_locations: List[SourceLocation] = field(default_factory=list)

    # Edge semantics
    weight: float = 1.0
    confidence: float = 1.0
    is_bidirectional: bool = False
    inverse_relation: Optional[str] = None

    # Flexible properties
    properties: Dict[str, Any] = field(default_factory=dict)

    # Namespace for isolation
    namespace: str = "default"

    def add_source(self, document_id: str, chunk_id: str) -> None:
        """Add a source reference to source_locations."""
        try:
            idx = int(chunk_id)
        except (TypeError, ValueError):
            return
        loc = SourceLocation(document_id=document_id, chunk_index=idx)
        if loc not in self.source_locations:
            self.source_locations.append(loc)

    def get_embedding_text(self) -> str:
        """Text representation for embedding generation.

        Incorporates description, inverse_relation and keywords when available
        so that semantically richer embeddings are produced.
        """
        parts = [f"{self.subject} {self.predicate} {self.object}"]
        if self.description:
            parts.append(self.description)
        if self.inverse_relation:
            parts.append(f"(inverse: {self.object} {self.inverse_relation} {self.subject})")
        if self.keywords:
            parts.append(f"[{', '.join(self.keywords)}]")
        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.

        Use this instead of the removed to_triple() method.
        """

        return {
            "id": self.id,
            "subject": self.subject,
            "subject_id": self.subject_id,
            "predicate": self.predicate,
            "object": self.object,
            "object_id": self.object_id,
            "confidence": self.confidence,
            "source_locations": [
                {"document_id": loc.document_id, "chunk_index": loc.chunk_index}
                for loc in self.source_locations
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relation":
        """Deserialize from dictionary."""
        raw_locs = data.get("source_locations", [])
        source_locations: List[SourceLocation] = [
            SourceLocation(document_id=loc["document_id"], chunk_index=loc["chunk_index"])
            for loc in raw_locs
            if "document_id" in loc and "chunk_index" in loc
        ]

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            subject_id=data.get("subject_id", ""),
            predicate=data.get("predicate", ""),
            object_id=data.get("object_id", ""),
            subject=data.get("subject", ""),
            object=data.get("object", ""),
            description=data.get("description", ""),
            keywords=data.get("keywords", []),
            source_locations=source_locations,
            weight=data.get("weight", 1.0),
            confidence=data.get("confidence", 1.0),
            is_bidirectional=data.get("is_bidirectional", False),
            inverse_relation=data.get("inverse_relation"),
            properties=data.get("properties", {}),
            namespace=data.get("namespace", "default"),
        )


# ============================================================================
# GRAPH TYPES (Refactored from God Object)
# ============================================================================


class NodeType(Enum):
    """Enumeration of valid node types."""

    ENTITY = "entity"
    PASSAGE = "passage"
    PAGE = "page"


@dataclass
class BaseGraphNode:
    """
    Base class for all graph nodes.

    Contains only fields common to ALL node types.
    Specific node types extend this with their own fields.
    """

    id: str
    node_type: NodeType
    content: str

    # Canonical provenance
    source_locations: List[SourceLocation] = field(default_factory=list)

    # Namespace for isolation
    namespace: str = "default"

    # Flexible properties for extension
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EntityNode(BaseGraphNode):
    """Graph node representing an extracted entity."""

    # Entity-specific: edge counts for graph analytics
    incoming_edge_count: int = 0
    outgoing_edge_count: int = 0

    # Reference to entity type/name
    entity_type: str = "entity"
    entity_name: str = ""

    def __post_init__(self) -> None:
        self.node_type = NodeType.ENTITY


@dataclass
class PassageNode(BaseGraphNode):
    """Graph node representing a text passage/chunk."""

    page_number: Optional[int] = None

    # Entities mentioned in this passage
    mentioned_entity_ids: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.node_type = NodeType.PASSAGE

    @staticmethod
    def make_id(tenant_id: str, content_hash: str) -> str:
        return f"passage:{tenant_id}:{content_hash}"


@dataclass
class PageNode(BaseGraphNode):
    """
    Graph node representing a document page (MMKG style).

    NOTE: Visual embeddings live in the vector store (page_images collection),
    not in the graph DB.
    """

    # Page-specific fields
    page_number: int = 0
    document_id: str = ""
    text_summary: str = ""

    # Entities and relations extracted from this page
    entity_ids: List[str] = field(default_factory=list)
    relation_ids: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.node_type = NodeType.PAGE

    @staticmethod
    def make_id(document_id: str, page_number: int) -> str:
        return f"page:{document_id}:{page_number}"


# Type alias for any graph node (use for functions that accept any node type)
GraphNode = Union[EntityNode, PassageNode, PageNode]


@dataclass
class GraphEdge:
    """Graph edge - UNIFIED with direction semantics."""

    source_id: str  # FROM node
    target_id: str  # TO node
    relation: str
    id: Optional[str] = field(default_factory=lambda: str(uuid.uuid4()))
    weight: float = 1.0

    # Direction semantics
    is_bidirectional: bool = False
    inverse_relation: Optional[str] = None

    # Optional denormalised names for embedding / display
    source_entity_name: str = ""
    target_entity_name: str = ""

    # Canonical provenance
    source_locations: List[SourceLocation] = field(default_factory=list)

    # Namespace
    namespace: str = "default"

    properties: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# MEMORY TYPE (Enhanced with Validation)
# ============================================================================


# Valid combinations of MemoryType and MemoryLayer
VALID_MEMORY_LAYER_COMBINATIONS: Dict[MemoryType, Set[MemoryLayer]] = {
    MemoryType.EPISODIC: {
        MemoryLayer.L1_WORKING,
        MemoryLayer.L2_EPISODIC,
        MemoryLayer.L4_ARCHIVAL,
    },
    MemoryType.SEMANTIC: {
        MemoryLayer.L2_EPISODIC,
        MemoryLayer.L3_SEMANTIC,
        MemoryLayer.L4_ARCHIVAL,
    },
    MemoryType.PROCEDURAL: {
        MemoryLayer.L3_SEMANTIC,
        MemoryLayer.L4_ARCHIVAL,
    },
    MemoryType.CORE: {
        MemoryLayer.L3_SEMANTIC,
        MemoryLayer.L4_ARCHIVAL,
    },
    MemoryType.PREFERENCE: {
        MemoryLayer.L2_EPISODIC,
        MemoryLayer.L3_SEMANTIC,
    },
}


class MemoryValidationError(ValueError):
    """Raised when memory type/layer combination is invalid."""

    pass


@dataclass
class Memory:
    """
    Core memory record - ENHANCED with source tracking and validation.

    STORAGE MAPPING (UPDATED DESIGN):
    - Authoritative Memory record is stored in KV:
      key: "memory:{memory_id}" -> full Memory dict
    - Memory embedding is stored in the tenant-level vector store collection (memories)
    """

    content: str
    memory_type: MemoryType = MemoryType.SEMANTIC
    user_id: str = "default"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    namespace: str = "default"
    embedding: Optional[List[float]] = None
    status: MemoryStatus = MemoryStatus.VALID
    layer: MemoryLayer = MemoryLayer.L3_SEMANTIC
    importance: float = 0.5
    access_count: int = 0
    created_at: datetime = field(default_factory=utc_now)
    updated_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None
    supersedes_id: Optional[str] = None

    # Source tracking
    sources: List[SourceReference] = field(default_factory=list)
    source_chunk_ids: Set[str] = field(default_factory=set)

    # Content hash for deduplication
    content_hash: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate memory type/layer combination."""

        valid_layers = VALID_MEMORY_LAYER_COMBINATIONS.get(self.memory_type, set())
        if self.layer not in valid_layers:
            raise MemoryValidationError(
                f"Invalid combination: {self.memory_type.value} memory "
                f"cannot be in {self.layer.value} layer. "
                f"Valid layers for {self.memory_type.value}: "
                f"{[l.value for l in valid_layers]}"
            )

    def get_content_hash(self, tenant_id: str) -> str:
        """
        Get content hash using canonical function.

        Note: Renamed from compute_content_hash to clarify it delegates
        to the canonical compute_content_hash() function.
        """

        return compute_content_hash(self.content, tenant_id)


# ============================================================================
# RETRIEVAL TYPES (Consolidated)
# ============================================================================


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


# ============================================================================
# DOCUMENT & CHUNK TYPES
# ============================================================================


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


# Type alias for chunker factory used in interfaces
ChunkerFactory = Callable[[str, int, int], "Chunker"]  # defined in interfaces via Protocol

