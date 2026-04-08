"""
Source and provenance type exports extracted from ``core.types``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid

from .enums import ACLEffect, Permission, SourceType
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


__all__ = [
    "ACLEntry",
    "BoundingBox",
    "Entity",
    "NamespaceACL",
    "Relation",
    "SourceLocation",
    "SourceReference",
]
