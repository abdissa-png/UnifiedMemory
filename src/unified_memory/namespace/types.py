"""
Enhanced Namespace System.

Implements the namespace and tenant configuration model described in
UNIFIED_MEMORY_SYSTEM_DESIGN.md and INITIAL_PLAN.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import hashlib

from unified_memory.core.types import NamespaceACL, Modality
from unified_memory.core.utils import utc_now


@dataclass
class EmbeddingModelConfig:
    """Configuration for an embedding model."""

    provider: str  # "openai", "cohere", "local"
    model: str  # "text-embedding-3-small"
    dimension: int  # 1536
    api_key_ref: Optional[str] = None  # Reference to secret, not actual key


@dataclass
class TenantConfig:
    """
    Tenant-level configuration.

    CRITICAL DESIGN CHANGE:
    - Embedding models are TENANT-WIDE, not per-namespace.
    - Guarantees that any cross-namespace retrieval within a tenant operates
      in a single, compatible embedding space.
    """

    tenant_id: str
    text_embedding: EmbeddingModelConfig = field(
        default_factory=lambda: EmbeddingModelConfig(
            provider="openai",
            model="text-embedding-3-small",
            dimension=1536,
        )
    )
    vision_embedding: Optional[EmbeddingModelConfig] = field(
        default_factory=lambda: EmbeddingModelConfig(
            provider="openai",
            model="clip-vit-base-patch32",
            dimension=512,
        )
    )
    created_at: str = field(default_factory=lambda: utc_now().isoformat())
    updated_at: str = ""


@dataclass
class Namespace:
    """
    Enhanced namespace definition.

    Changes from simpler versions:
    - Added tenant_id for multi-tenant support
    - Maintained backward compatibility for user_id/agent/session
    """

    user_id: str
    tenant_id: str = "default"
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    scope: str = "private"

    def to_string(self) -> str:
        """Convert namespace to canonical string key."""

        parts = [f"tenant:{self.tenant_id}", f"user:{self.user_id}"]
        if self.agent_id:
            parts.append(f"agent:{self.agent_id}")
        if self.session_id:
            parts.append(f"session:{self.session_id}")
        return "/".join(parts)

    def to_hash(self, length: int = 32) -> str:
        """
        Get hash of namespace string.

        NOTE: No longer used for collection naming (collections are tenant-level).
        Only used for internal caching/indexing where collision is less critical.
        """

        full_hash = hashlib.sha256(self.to_string().encode()).hexdigest()
        return full_hash[: min(length, 64)]

    @classmethod
    def from_string(cls, namespace_str: str) -> "Namespace":
        """Parse namespace from canonical string representation."""

        parts = namespace_str.split("/")
        tenant_id = "default"
        user_id = "anonymous"
        agent_id: Optional[str] = None
        session_id: Optional[str] = None

        for part in parts:
            if part.startswith("tenant:"):
                tenant_id = part[7:]
            elif part.startswith("user:"):
                user_id = part[5:]
            elif part.startswith("agent:"):
                agent_id = part[6:]
            elif part.startswith("session:"):
                session_id = part[8:]

        return cls(
            tenant_id=tenant_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )


@dataclass
class NamespaceConfig:
    """
    Full configuration for a namespace.

    Stored in KV store for retrieval during operations.

    NAMESPACE IDENTIFIER DESIGN:
    - namespace_id: Full hierarchical string like "tenant:acme/user:alice/agent:x"
      This is the CANONICAL identifier used for:
        - ACL lookups
        - Graph filtering
        - Metadata filtering in vector stores

    COLLECTION STRATEGY:
    - Collections are at TENANT level, not namespace level.
    - Namespace isolation is achieved via metadata filtering on namespace_id.
    """

    # REQUIRED fields - must be provided at construction
    tenant_id: str
    user_id: str

    # Optional hierarchy components
    agent_id: Optional[str] = None
    session_id: Optional[str] = None

    # Derived field - computed in __post_init__ if not provided
    namespace_id: str = ""

    # TENANT-LEVEL collection names (shared by all users in tenant)
    text_collection: str = ""
    entity_collection: str = ""
    relation_collection: str = ""
    page_image_collection: str = ""
    memory_collection: str = ""

    # Graph isolation (property-based)
    graph_namespace_property: str = ""

    # Access control
    owner_id: str = ""
    acl: NamespaceACL = field(default_factory=NamespaceACL)
    scope: str = "private"  # "private", "shared", "public"

    # Timestamps
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        """
        Derive fields from components.

        Collections are tenant-level, isolation is via metadata.
        """

        ns = Namespace(
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            agent_id=self.agent_id,
            session_id=self.session_id,
        )

        if not self.namespace_id:
            self.namespace_id = ns.to_string()

        # Tenant-level collections
        if not self.text_collection:
            self.text_collection = f"{self.tenant_id}_texts"
        if not self.entity_collection:
            self.entity_collection = f"{self.tenant_id}_entities"
        if not self.relation_collection:
            self.relation_collection = f"{self.tenant_id}_relations"
        if not self.page_image_collection:
            self.page_image_collection = f"{self.tenant_id}_page_images"
        if not self.memory_collection:
            self.memory_collection = f"{self.tenant_id}_memories"

        # Graph namespace property uses full namespace string
        if not self.graph_namespace_property:
            self.graph_namespace_property = self.namespace_id

        # Owner defaults to user_id
        if not self.owner_id:
            self.owner_id = self.user_id

        # Timestamps
        if not self.created_at:
            self.created_at = utc_now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at

    def get_namespace_filter(self) -> Dict[str, str]:
        """
        Get the metadata filter for namespace isolation.

        This filter should be applied to ALL vector store queries.
        """

        return {"namespace": self.namespace_id}

