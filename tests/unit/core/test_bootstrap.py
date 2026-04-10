import pytest

from unified_memory.bootstrap import SystemContext
from unified_memory.core.exceptions import ConfigurationError


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
    ctx.build_services(default_text_embedding_key="mock:test-model")
    assert ctx.ingestion_pipeline is not None
    assert ctx.search_service is not None


def test_bootstrap_redis_kv_missing_url():
    """kv_store=redis with empty redis_url should raise ConfigurationError."""
    with pytest.raises(ConfigurationError):
        SystemContext(config={"kv_store": "redis", "redis_url": ""})


def test_build_services_no_embedding_provider():
    """build_services without any embedding provider raises ConfigurationError."""
    ctx = SystemContext()
    with pytest.raises(ConfigurationError):
        ctx.build_services()


def test_from_config_file_nonexistent():
    """from_config_file with bad path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        SystemContext.from_config_file("/nonexistent.yaml")


def test_hot_reload_refuses_infra_change(tmp_path):
    """hot_reload_from_file rejects changing infrastructure backends."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
infra:
  kv_store: memory
  vector_store: qdrant
embedding_providers:
  mock:test-model:
    provider: mock
    model: test-model
    dimension: 32
""".strip()
    )

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
    ctx._app_config = type(
        "DummyAppConfig",
        (),
        {
            "infra": type(
                "DummyInfra",
                (),
                {
                    "kv_store": "memory",
                    "vector_store": "memory",
                    "graph_store": "networkx",
                    "sparse_retriever": "bm25",
                },
            )()
        },
    )()

    with pytest.raises(ConfigurationError):
        ctx.hot_reload_from_file(config_path)
