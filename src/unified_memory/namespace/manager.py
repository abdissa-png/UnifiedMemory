"""
NamespaceManager implementation.

Handles:
- NamespaceConfig lifecycle and caching
- TenantConfig lookup for embedding models
- Collection name resolution
- Accessible namespaces listing
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional
from contextvars import ContextVar

from unified_memory.core.types import CollectionType, Modality
from unified_memory.core.utils import utc_now
from unified_memory.namespace.types import (
    NamespaceConfig,
    TenantConfig,
    EmbeddingModelConfig,
)
from unified_memory.storage.base import KVStoreBackend
from unified_memory.namespace.validation import validate_namespace_id


_request_namespace_cache: ContextVar[Dict[str, NamespaceConfig]] = ContextVar(
    "namespace_config_cache"
)
_request_tenant_cache: ContextVar[Dict[str, TenantConfig]] = ContextVar(
    "tenant_config_cache"
)


class NamespaceManager:
    """
    Enhanced namespace manager with config storage.

    - Stores NamespaceConfig in KV store
    - Provides embedding model lookup via TenantConfig
    - Handles multi-tenant scenarios
    - Uses request-scoped cache to avoid repeated KV lookups
    """

    def __init__(
        self,
        kv_store: KVStoreBackend,
        default_scope: str = "private",
    ) -> None:
        self.kv_store = kv_store
        self.default_scope = default_scope

    async def get_config(self, namespace: str) -> Optional[NamespaceConfig]:
        """Get namespace configuration."""
        try:
             validate_namespace_id(namespace)
        except ValueError:
             return None

        try:
            cache = _request_namespace_cache.get()
        except LookupError:
            cache = {}
            _request_namespace_cache.set(cache)

        if namespace in cache:
            return cache[namespace]

        versioned = await self.kv_store.get(f"ns_config:{namespace}")
        if not versioned:
            return None

        config = NamespaceConfig(**versioned.data)
        cache[namespace] = config
        return config

    async def create_namespace(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> NamespaceConfig:
        """Create a new namespace with configuration."""

        config = NamespaceConfig(
            tenant_id=tenant_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            namespace_id="",  # derived in __post_init__
        )

        # Validate generated ID
        validate_namespace_id(config.namespace_id)
            
        # Check if namespace already exists
        existing_config = await self.get_config(config.namespace_id)
        if existing_config:
            raise ValueError(f"Namespace already exists: {config.namespace_id}")

        await self.kv_store.set(f"ns_config:{config.namespace_id}", asdict(config))

        # Update secondary indexes for fast lookup
        await self._update_user_namespace_index(
            tenant_id=tenant_id,
            user_id=user_id,
            namespace_id=config.namespace_id,
            add=True,
        )

        # If namespace is public, add to public index as well
        if config.scope == "public":
            await self._update_public_namespace_index(
                tenant_id=tenant_id,
                namespace_id=config.namespace_id,
                add=True,
            )

        try:
            cache = _request_namespace_cache.get()
        except LookupError:
            cache = {}
            _request_namespace_cache.set(cache)
        
        cache[config.namespace_id] = config

        return config

    async def get_embedding_model(
        self,
        namespace: str,
        modality: Modality = Modality.TEXT,
    ) -> EmbeddingModelConfig:
        """
        Get embedding model configuration for namespace.

        Models are TENANT-WIDE via TenantConfig.
        NamespaceConfig determines tenant_id; tenant config determines model.
        """

        config = await self.get_config(namespace)
        if not config:
            raise ValueError(f"Namespace {namespace} not found")

        tenant_cfg = await self.get_tenant_config(config.tenant_id)
        if modality in (Modality.TEXT, Modality.SHARED):
            return tenant_cfg.text_embedding
        if modality in (Modality.IMAGE, Modality.DOCUMENT):
            if not tenant_cfg.vision_embedding:
                raise ValueError(
                    f"Vision embedding not configured for tenant {config.tenant_id}"
                )
            return tenant_cfg.vision_embedding
        return tenant_cfg.text_embedding

    async def get_tenant_config(self, tenant_id: str) -> TenantConfig:
        """
        Get tenant-level configuration.

        Stored in KV:
          tenant_config:{tenant_id} -> TenantConfig dict
        """


        try:
            cache = _request_tenant_cache.get()
        except LookupError:
            cache = {}
            _request_tenant_cache.set(cache)

        if tenant_id in cache:
            return cache[tenant_id]

        versioned = await self.kv_store.get(f"tenant_config:{tenant_id}")
        if not versioned:
            # Derive default tenant config if missing
            cfg = TenantConfig(
                tenant_id=tenant_id,
                updated_at=utc_now().isoformat(),
            )
            await self.kv_store.set(f"tenant_config:{tenant_id}", asdict(cfg))
            cache[tenant_id] = cfg
            return cfg

        data = versioned.data
        # Handle nested dataclasses - ensure we convert dicts to objects
        if "text_embedding" in data:
            if isinstance(data["text_embedding"], dict):
                data["text_embedding"] = EmbeddingModelConfig(**data["text_embedding"])
            # If it's already an EmbeddingModelConfig, leave it as is
        
        if "vision_embedding" in data and data["vision_embedding"] is not None:
            if isinstance(data["vision_embedding"], dict):
                data["vision_embedding"] = EmbeddingModelConfig(**data["vision_embedding"])
        cfg = TenantConfig(**data)
        if not cfg.updated_at:
            cfg.updated_at = cfg.created_at
        
        cache[tenant_id] = cfg
        return cfg

    async def get_collection_name(
        self,
        namespace: str,
        collection_type: CollectionType,
    ) -> str:
        """Get collection name for namespace and type."""

        config = await self.get_config(namespace)
        if not config:
            raise ValueError(f"Namespace {namespace} not found")

        mapping = {
            CollectionType.TEXTS: config.text_collection,
            CollectionType.ENTITIES: config.entity_collection,
            CollectionType.RELATIONS: config.relation_collection,
            CollectionType.PAGE_IMAGES: config.page_image_collection,
            CollectionType.MEMORIES: config.memory_collection,
        }

        return mapping[collection_type]

    async def get_accessible_namespaces(
        self,
        user_id: str,
        tenant_id: str = "default",
        include_public: bool = True,
    ) -> List[str]:
        """
        Get all namespaces accessible to a user.

        Uses secondary indexes, not O(N) full scans.
        """

        namespaces: List[str] = []

        # 1. Own namespaces
        user_ns_key = f"idx:user_namespaces:{tenant_id}:{user_id}"
        versioned_own = await self.kv_store.get(user_ns_key)
        if versioned_own and "namespaces" in versioned_own.data:
            namespaces.extend(versioned_own.data["namespaces"])

        # 2. Shared namespaces via ACL
        acl_key = f"idx:acl_access:{tenant_id}:{user_id}"
        versioned_shared = await self.kv_store.get(acl_key)
        if versioned_shared and "namespaces" in versioned_shared.data:
            namespaces.extend(versioned_shared.data["namespaces"])

        # 3. Public namespaces
        if include_public:
            public_key = f"idx:public_namespaces:{tenant_id}"
            versioned_public = await self.kv_store.get(public_key)
            if versioned_public and "namespaces" in versioned_public.data:
                namespaces.extend(versioned_public.data["namespaces"])

        # De-duplicate
        return list(set(namespaces))

    async def _update_user_namespace_index(
        self,
        tenant_id: str,
        user_id: str,
        namespace_id: str,
        add: bool = True,
    ) -> None:
        """
        Update the user-to-namespaces index.

        Called when creating/deleting namespaces.
        """

        key = f"idx:user_namespaces:{tenant_id}:{user_id}"
        versioned = await self.kv_store.get(key)
        data = versioned.data if versioned else {"namespaces": []}
        ns_set = set(data.get("namespaces", []))
        if add:
            ns_set.add(namespace_id)
        else:
            ns_set.discard(namespace_id)
        data["namespaces"] = list(ns_set)
        await self.kv_store.set(key, data)

    async def _update_public_namespace_index(
        self,
        tenant_id: str,
        namespace_id: str,
        add: bool = True,
    ) -> None:
        """
        Update the public namespaces index.

        This mirrors the design in UNIFIED_MEMORY_SYSTEM_DESIGN.md, where
        public namespaces are discoverable via a dedicated index key.
        """

        key = f"idx:public_namespaces:{tenant_id}"
        versioned = await self.kv_store.get(key)
        data = versioned.data if versioned else {"namespaces": []}
        ns_set = set(data.get("namespaces", []))
        if add:
            ns_set.add(namespace_id)
        else:
            ns_set.discard(namespace_id)
        data["namespaces"] = list(ns_set)
        await self.kv_store.set(key, data)


