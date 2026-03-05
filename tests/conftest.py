import pytest
import asyncio
from typing import AsyncGenerator, Dict, Any

from unified_memory.core.registry import ProviderRegistry
from unified_memory.namespace.tenant_manager import TenantManager
from unified_memory.storage.kv.memory_store import MemoryKVStore
from unified_memory.namespace.manager import NamespaceManager
from unified_memory.cas.registry import CASRegistry
from unified_memory.cas.content_store import ContentStore
from unified_memory.cas.document_registry import DocumentRegistry
from unified_memory.embeddings.providers.mock_provider import MockEmbeddingProvider
from unified_memory.storage.vector.memory_store import MemoryVectorStore
from unified_memory.storage.graph.networkx_store import NetworkXGraphStore
from unified_memory.core.types import Modality

@pytest.fixture
async def mock_embedding_model():
    """Mock embedding model config."""
    return {
        "provider": "test",
        "model": "test-model",
        "dimension": 4
    }

@pytest.fixture
def kv_store():
    """Shared in-memory KV store."""
    return MemoryKVStore()

@pytest.fixture
def namespace_manager(kv_store):
    """Shared namespace manager."""
    return NamespaceManager(kv_store)

@pytest.fixture
def tenant_manager(kv_store):
    """Shared tenant manager."""
    return TenantManager(kv_store)

@pytest.fixture
def provider_registry():
    """Shared provider registry."""
    return ProviderRegistry()

@pytest.fixture
def cas_registry(kv_store):
    """Shared CAS registry."""
    return CASRegistry(kv_store)

@pytest.fixture
def content_store(kv_store):
    """Shared content store."""
    return ContentStore(kv_store)

@pytest.fixture
def document_registry(kv_store):
    """Shared document registry."""
    return DocumentRegistry(kv_store)

@pytest.fixture
def pipeline_deps(namespace_manager, cas_registry, content_store, document_registry, kv_store, provider_registry):
    """Full set of dependencies for IngestionPipeline."""
    return {
        "provider_registry": provider_registry,
        "vector_store": MemoryVectorStore(),
        "graph_store": NetworkXGraphStore(),
        "cas_registry": cas_registry,
        "content_store": content_store,
        "document_registry": document_registry,
        "namespace_manager": namespace_manager,
        "kv": kv_store
    }

@pytest.fixture
async def tenant_with_config(namespace_manager):
    """Create a tenant with a default ingestion config."""
    tenant_id = "tenant_123"
    from unified_memory.namespace.types import TenantConfig
    
    # Check if create_tenant exists
    if hasattr(namespace_manager, "create_tenant"):
        config = await namespace_manager.create_tenant(
            tenant_id=tenant_id,
            name="Test Tenant"
        )
    else:
        # Fallback to direct KV set if manager doesn't have create_tenant yet
        config = TenantConfig(tenant_id=tenant_id)
        await namespace_manager.kv_store.set(f"tenant_config:{tenant_id}", config.__dict__)
    
    return config

@pytest.fixture
def ingestion_pipeline(pipeline_deps):
    """Ready-to-use IngestionPipeline."""
    from unified_memory.ingestion.pipeline import IngestionPipeline
    # Extract deps
    deps = pipeline_deps.copy()
    provider_registry = deps.pop("provider_registry")
    provider_registry.register_embedding_provider("mock:mock-model", MockEmbeddingProvider(model_id="mock-model", dimension=128))
    provider_registry.register_vision_embedding_provider("mock:mock-model", MockEmbeddingProvider(model_id="mock-model", dimension=128, modalities=[Modality.IMAGE]))
    kv = deps.pop("kv") # Not needed for init
    return IngestionPipeline(provider_registry=provider_registry, **deps)

