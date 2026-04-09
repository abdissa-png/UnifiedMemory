"""
NamespaceManager implementation.

Handles:
- NamespaceConfig lifecycle and caching
- TenantConfig lookup for embedding models
- Collection name resolution
- Accessible namespaces listing
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Any, Dict, List, Optional
from contextvars import ContextVar

from unified_memory.core.exceptions import (
    CASConflictError,
    ConfigurationError,
    NamespaceNotFoundError,
    TenantNotFoundError,
    TenantConfigNotFoundError,
    TenantConfigConflictError,
)
from unified_memory.core.types import (
    CollectionType,
    Modality,
    Permission,
    ACLEntry,
    ACLEffect,
    NamespaceACL,
)
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

    The namespace and tenant caches are scoped to the current async context,
    typically a single request. A ``LookupError`` means the cache has not yet
    been initialized for that context.
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

        data = versioned.data
        # Deserialise ACL from dict if stored as plain dict
        if "acl" in data and isinstance(data["acl"], dict):
            data["acl"] = NamespaceACL.from_dict(data["acl"])

        config = NamespaceConfig(**data)
        cache[namespace] = config
        return config

    def invalidate_cache(self, namespace_id: Optional[str] = None) -> None:
        """Clear the request-scoped namespace cache."""
        try:
            cache = _request_namespace_cache.get()
            if namespace_id:
                cache.pop(namespace_id, None)
            else:
                cache.clear()
        except LookupError:
            pass

    async def create_namespace(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        scope: str = "private",
    ) -> NamespaceConfig:
        """Create a new namespace with owner-full-permission ACL."""

        owner_acl = NamespaceACL(entries=[
            ACLEntry(
                principal=user_id,
                principal_type="user",
                permissions=[
                    Permission.READ,
                    Permission.WRITE,
                    Permission.DELETE,
                    Permission.ADMIN,
                    Permission.SHARE,
                ],
                effect=ACLEffect.ALLOW,
            ),
        ])

        config = NamespaceConfig(
            tenant_id=tenant_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            namespace_id="",  # derived in __post_init__
            acl=owner_acl,
            scope=scope,
        )

        # Validate generated ID
        validate_namespace_id(config.namespace_id)
            
        # Check if namespace already exists
        existing_config = await self.get_config(config.namespace_id)
        if existing_config:
            raise ConfigurationError(f"Namespace already exists: {config.namespace_id}")

        await self._save_config(config)

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
            
        return config

    # ---- sharing -----------------------------------------------------------

    async def share_namespace(
        self,
        namespace_id: str,
        target_user_id: str,
        permissions: List[Permission],
    ) -> NamespaceConfig:
        """Grant *target_user_id* the given permissions on *namespace_id*."""
        config = await self.get_config(namespace_id)
        if not config:
            raise NamespaceNotFoundError(f"Namespace not found: {namespace_id}")

        config.acl.entries.append(
            ACLEntry(
                principal=target_user_id,
                principal_type="user",
                permissions=permissions,
                effect=ACLEffect.ALLOW,
            )
        )
        if config.scope == "private":
            config.scope = "shared"
        config.updated_at = utc_now().isoformat()
        await self._save_config(config)

        await self._update_acl_access_index(
            tenant_id=config.tenant_id,
            user_id=target_user_id,
            namespace_id=namespace_id,
            add=True,
        )
        return config

    async def unshare_namespace(
        self,
        namespace_id: str,
        target_user_id: str,
    ) -> NamespaceConfig:
        """Revoke all permissions for *target_user_id* on *namespace_id*."""
        config = await self.get_config(namespace_id)
        if not config:
            raise NamespaceNotFoundError(f"Namespace not found: {namespace_id}")

        config.acl.entries = [
            e
            for e in config.acl.entries
            if not (e.principal_type == "user" and e.principal == target_user_id)
        ]
        config.updated_at = utc_now().isoformat()
        await self._save_config(config)

        await self._update_acl_access_index(
            tenant_id=config.tenant_id,
            user_id=target_user_id,
            namespace_id=namespace_id,
            add=False,
        )
        return config

    async def delete_namespace(self, namespace_id: str) -> bool:
        """Remove namespace config and all secondary indexes."""
        config = await self.get_config(namespace_id)
        if not config:
            return False

        await self.kv_store.delete(f"ns_config:{namespace_id}")

        await self._update_user_namespace_index(
            tenant_id=config.tenant_id,
            user_id=config.user_id,
            namespace_id=namespace_id,
            add=False,
        )
        if config.scope == "public":
            await self._update_public_namespace_index(
                tenant_id=config.tenant_id,
                namespace_id=namespace_id,
                add=False,
            )

        # Remove from acl_access indexes for every explicitly shared user
        for entry in config.acl.entries:
            if (
                entry.principal_type == "user"
                and entry.principal != config.user_id
            ):
                await self._update_acl_access_index(
                    tenant_id=config.tenant_id,
                    user_id=entry.principal,
                    namespace_id=namespace_id,
                    add=False,
                )

        self.invalidate_cache(namespace_id)

        return True

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
            raise NamespaceNotFoundError(f"Namespace {namespace} not found")

        tenant_cfg = await self.get_tenant_config(config.tenant_id)
        if not tenant_cfg:
            raise TenantConfigNotFoundError(f"Tenant config not found for tenant {config.tenant_id}")
        if modality in (Modality.TEXT, Modality.SHARED):
            return tenant_cfg.text_embedding
        if modality in (Modality.IMAGE, Modality.DOCUMENT):
            if not tenant_cfg.vision_embedding:
                raise ConfigurationError(
                    f"Vision embedding not configured for tenant {config.tenant_id}"
                )
            return tenant_cfg.vision_embedding
        return tenant_cfg.text_embedding

    async def get_tenant_config(self, tenant_id: str) -> Optional[TenantConfig]:
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
            return None

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

    async def get_or_create_tenant_config(self, tenant_id: str) -> TenantConfig:
        """Read tenant config, creating the default record explicitly if missing."""
        cfg = await self.get_tenant_config(tenant_id)
        if cfg is not None:
            return cfg

        cfg = TenantConfig(
            tenant_id=tenant_id,
            updated_at=utc_now().isoformat(),
        )

        for attempt in range(5):
            if await self.kv_store.set_if_not_exists(f"tenant_config:{tenant_id}", asdict(cfg)):
                try:
                    cache = _request_tenant_cache.get()
                except LookupError:
                    cache = {}
                    _request_tenant_cache.set(cache)
                cache[tenant_id] = cfg
                return cfg

            existing = await self.get_tenant_config(tenant_id)
            if existing is not None:
                return existing
            await asyncio.sleep(0.01 * (attempt + 1))

        raise CASConflictError(
            f"CAS failed after 5 attempts on tenant_config:{tenant_id}"
        )

    async def get_collection_name(
        self,
        namespace: str,
        collection_type: CollectionType,
    ) -> str:
        """Get collection name for namespace and type."""

        config = await self.get_config(namespace)
        if not config:
            raise NamespaceNotFoundError(f"Namespace {namespace} not found")

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
        await self._update_namespace_index(key, namespace_id, add=add)

    async def _save_config(self, config: NamespaceConfig) -> None:
        """Persist a NamespaceConfig with proper ACL serialisation."""
        data = asdict(config)
        # Replace the raw asdict ACL (which keeps Enum objects) with a safe dict
        data["acl"] = config.acl.to_dict()
        key = f"ns_config:{config.namespace_id}"

        for attempt in range(5):
            versioned = await self.kv_store.get(key)
            if versioned is None:
                if await self.kv_store.set_if_not_exists(key, data):
                    try:
                        cache = _request_namespace_cache.get()
                    except LookupError:
                        cache = {}
                        _request_namespace_cache.set(cache)
                    cache[config.namespace_id] = config
                    return
            elif await self.kv_store.compare_and_swap(key, versioned.version, data):
                try:
                    cache = _request_namespace_cache.get()
                except LookupError:
                    cache = {}
                    _request_namespace_cache.set(cache)
                cache[config.namespace_id] = config
                return

            await asyncio.sleep(0.01 * (attempt + 1))

        raise CASConflictError(f"CAS failed after 5 attempts on {key}")

    async def _update_acl_access_index(
        self,
        tenant_id: str,
        user_id: str,
        namespace_id: str,
        add: bool = True,
    ) -> None:
        """Maintain the idx:acl_access:{tenant}:{user} index."""
        key = f"idx:acl_access:{tenant_id}:{user_id}"
        await self._update_namespace_index(key, namespace_id, add=add)

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
        await self._update_namespace_index(key, namespace_id, add=add)

    async def _update_namespace_index(
        self,
        key: str,
        namespace_id: str,
        *,
        add: bool,
    ) -> None:
        """Update an index entry with CAS retries."""
        for attempt in range(5):
            versioned = await self.kv_store.get(key)
            if versioned is None:
                data = {"namespaces": [namespace_id] if add else []}
                if await self.kv_store.set_if_not_exists(key, data):
                    return
            else:
                data = dict(versioned.data)
                ns_set = set(data.get("namespaces", []))
                if add:
                    ns_set.add(namespace_id)
                else:
                    ns_set.discard(namespace_id)
                data["namespaces"] = list(ns_set)
                if await self.kv_store.compare_and_swap(key, versioned.version, data):
                    return

            await asyncio.sleep(0.01 * (attempt + 1))

        raise CASConflictError(f"CAS failed after 5 attempts on {key}")


