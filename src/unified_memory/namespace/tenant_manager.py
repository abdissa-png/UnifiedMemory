"""
Tenant Manager Implementation.

Manages tenant configurations, particularly embedding models.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from typing import Optional

from unified_memory.core.exceptions import CASConflictError
from unified_memory.storage.base import KVStoreBackend
from unified_memory.namespace.types import TenantConfig, EmbeddingModelConfig
from unified_memory.core.types import Modality

from unified_memory.core.logging import get_logger,log_event
logger = get_logger(__name__)


class TenantManager:
    """
    Manages tenant lifecycles and configurations.
    
    Responsibilities:
    - Create/Update tenant configs
    - Retrieve embedding configurations for ingestion
    - Default model management
    """
    
    def __init__(self, kv_store: KVStoreBackend):
        self.kv_store = kv_store
        
    def _key(self, tenant_id: str) -> str:
        return f"tenant_config:{tenant_id}"
        
    async def get_tenant_config(self, tenant_id: str) -> Optional[TenantConfig]:
        """Retrieve tenant configuration."""
        versioned = await self.kv_store.get(self._key(tenant_id))
        if not versioned:
            return None
            
        data = versioned.data
            
        # Handle nested dataclasses
        if "text_embedding" in data and isinstance(data["text_embedding"], dict):
            data["text_embedding"] = EmbeddingModelConfig(**data["text_embedding"])
            
        if "vision_embedding" in data and isinstance(data["vision_embedding"], dict):
            data["vision_embedding"] = EmbeddingModelConfig(**data["vision_embedding"])
            
        return TenantConfig(**data)
        
    async def set_tenant_config(self, tenant_id: str, config: TenantConfig) -> None:
        """Store tenant configuration."""
        key = self._key(tenant_id)
        payload = asdict(config)
        for attempt in range(5):
            versioned = await self.kv_store.get(key)
            if versioned is None:
                if await self.kv_store.set_if_not_exists(key, payload):
                    return
            elif await self.kv_store.compare_and_swap(key, versioned.version, payload):
                return
            await asyncio.sleep(0.01 * (attempt + 1))
        raise CASConflictError(f"CAS failed after 5 attempts on {key}")
        
    async def get_embedding_model_id(
        self, 
        tenant_id: str, 
        modality: Modality = Modality.TEXT
    ) -> str:
        """
        Get the active embedding model ID for a tenant and modality.
        
        If tenant config is missing or model not set, returns system default.
        """
        config = await self.get_tenant_config(tenant_id)
        
        system_default_text = "text-embedding-3-small"
        system_default_image = "clip-vit-base-patch32"
        
        if not config:
            log_event(logger, logging.WARNING, "tenant.embedding.model.not_found",
                tenant_id=tenant_id,
                modality=modality,
                system_default_text=system_default_text,
                system_default_image=system_default_image,
            )
            return system_default_text if modality == Modality.TEXT else system_default_image
            
        if modality == Modality.TEXT:
            return config.text_embedding.model
        elif modality == Modality.IMAGE:
             if config.vision_embedding:
                 return config.vision_embedding.model
             return system_default_image
        
        return system_default_text
        
    async def register_tenant(
        self,
        tenant_id: str,
        admin_user_id: str,
        text_embedding: Optional[EmbeddingModelConfig] = None,
        vision_embedding: Optional[EmbeddingModelConfig] = None,
        **kwargs
    ) -> TenantConfig:
        """Register a new tenant with default ACL granting admin full access.
        
        Optional kwargs allow overriding other TenantConfig defaults (e.g. chunk_size,
        chunker_type, enable_graph_storage).
        """
        from unified_memory.core.types import (
            Permission,
            ACLEntry,
            ACLEffect,
            NamespaceACL,
        )

        # Default text embedding model if not provided
        if text_embedding is None:
            text_embedding = EmbeddingModelConfig(
                provider="openai",
                model="text-embedding-3-small",
                dimension=1536,
            )

        # Build a tenant-level default ACL
        acl_entries = [
            ACLEntry(
                principal=admin_user_id,
                principal_type="user",
                permissions=list(Permission),
                effect=ACLEffect.ALLOW,
            ),
            # All tenant members get READ by default
            ACLEntry(
                principal="tenant_member",
                principal_type="role",
                permissions=[Permission.READ],
                effect=ACLEffect.ALLOW,
            )
        ]
        
        default_acl = NamespaceACL(entries=acl_entries).to_dict()

        kw_args: dict = {
            "tenant_id": tenant_id,
            "text_embedding": text_embedding,
            "default_acl": default_acl,
        }

        # Optional vision embedding override (otherwise TenantConfig default applies)
        if vision_embedding is not None:
            kw_args["vision_embedding"] = vision_embedding
            
        kw_args.update(kwargs)

        config = TenantConfig(**kw_args)

        await self.set_tenant_config(tenant_id, config)
        return config
