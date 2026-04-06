import pytest
import dataclasses
from unified_memory.core.types import Modality
from unified_memory.namespace.tenant_manager import TenantManager
from unified_memory.namespace.types import EmbeddingModelConfig
from unified_memory.storage.kv.memory_store import MemoryKVStore

@pytest.fixture
def kv_store():
    return MemoryKVStore()

@pytest.fixture
def manager(kv_store):
    return TenantManager(kv_store)

@pytest.mark.asyncio
async def test_register_and_get(manager):
    tenant_id = "test-tenant"
    text_cfg = EmbeddingModelConfig(
        provider="openai",
        model="my-text-model",
        dimension=1536,
    )
    await manager.register_tenant(tenant_id, admin_user_id="admin-123", text_embedding=text_cfg)
    
    config = await manager.get_tenant_config(tenant_id)
    assert config is not None
    assert config.tenant_id == tenant_id
    assert config.text_embedding.model == "my-text-model"

@pytest.mark.asyncio
async def test_get_embedding_model_id(manager):
    tenant_id = "test-tenant-2"
    
    # Before registration: returns default
    model_id = await manager.get_embedding_model_id(tenant_id, Modality.TEXT)
    assert model_id == "text-embedding-3-small"
    
    # Usage registration with explicit embedding config
    text_cfg = EmbeddingModelConfig(
        provider="openai",
        model="custom-model-v1",
        dimension=1536,
    )
    await manager.register_tenant(tenant_id, admin_user_id="admin-456", text_embedding=text_cfg)
    
    model_id = await manager.get_embedding_model_id(tenant_id, Modality.TEXT)
    assert model_id == "custom-model-v1"
    
    # Image fallback default
    img_model_id = await manager.get_embedding_model_id(tenant_id, Modality.IMAGE)
    assert "clip" in img_model_id
