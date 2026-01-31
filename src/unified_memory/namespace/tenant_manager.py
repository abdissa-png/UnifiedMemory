"""
Tenant Manager Implementation.

Manages tenant configurations, particularly embedding models.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Optional

from unified_memory.storage.base import KVStoreBackend
from unified_memory.namespace.types import TenantConfig, EmbeddingModelConfig
from unified_memory.core.types import Modality

logger = logging.getLogger(__name__)


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
        data = await self.kv_store.get(self._key(tenant_id))
        if not data:
            return None
            
        # Handle nested dataclasses
        if "text_embedding" in data and isinstance(data["text_embedding"], dict):
            data["text_embedding"] = EmbeddingModelConfig(**data["text_embedding"])
            
        if "vision_embedding" in data and isinstance(data["vision_embedding"], dict):
            data["vision_embedding"] = EmbeddingModelConfig(**data["vision_embedding"])
            
        return TenantConfig(**data)
        
    async def set_tenant_config(self, tenant_id: str, config: TenantConfig) -> None:
        """Store tenant configuration."""
        await self.kv_store.set(self._key(tenant_id), asdict(config))
        
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
        text_model: str = "text-embedding-3-small",
        image_model: Optional[str] = None
    ) -> TenantConfig:
        """Helper to register a new tenant."""
        
        text_config = EmbeddingModelConfig(
            provider="openai", # Defaulting to OpenAI for now
            model=text_model,
            dimension=1536 if "small" in text_model else 3072,
        )
        
        vision_config = None
        if image_model:
             vision_config = EmbeddingModelConfig(
                provider="openai",
                model=image_model,
                dimension=512,
            )
            
        # Create config. Note: TenantConfig uses default_factory if not provided
        # We need to construct it carefully
        
        # If I use constructor:
        # TenantConfig(tenant_id=..., text_embedding=..., vision_embedding=...)
        
        kw_args = {
            "tenant_id": tenant_id,
            "text_embedding": text_config,
        }
        if vision_config:
            kw_args["vision_embedding"] = vision_config
            
        config = TenantConfig(**kw_args)
        
        await self.set_tenant_config(tenant_id, config)
        return config
