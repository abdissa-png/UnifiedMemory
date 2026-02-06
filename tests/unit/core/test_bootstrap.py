import pytest
from unified_memory.bootstrap import SystemContext


def test_default_bootstrap():
    """SystemContext with default config builds in-memory stores."""
    ctx = SystemContext()
    assert ctx.kv_store is not None
    assert ctx.vector_store is not None
    assert ctx.graph_store is not None
    assert ctx.content_store is not None
    assert ctx.cas_registry is not None
    assert ctx.document_registry is not None
    assert ctx.namespace_manager is not None
    assert ctx.provider_registry is not None


def test_build_services_with_mock_provider():
    """Build services after registering a mock embedding provider."""
    ctx = SystemContext(
        config={
            "embedding_providers": {
                "mock:test-model": {
                    "provider": "mock",
                    "model": "test-model",
                    "dimension": 32,
                }
            }
        }
    )
    ctx.build_services(default_embedding_key="mock:test-model")
    assert ctx.ingestion_pipeline is not None
    assert ctx.search_service is not None
