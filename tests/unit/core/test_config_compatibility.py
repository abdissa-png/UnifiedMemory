import pytest

from unified_memory.core.config import (
    AppConfig,
    InfraConfig,
    DefaultsConfig,
    EmbeddingProviderConfig,
    ExtractorConfig,
    LLMProviderConfig,
    validate_config_compatibility,
)


def test_validate_config_compatibility_happy_path():
    infra = InfraConfig()
    defaults = DefaultsConfig(retrieval_paths=["dense", "sparse"], skip_embedding=False)
    app = AppConfig(
        infra=infra,
        embedding_providers={
            "mock:mock-model": EmbeddingProviderConfig(
                provider="mock",
                model="mock-model",
                dimension=128,
            )
        },
        defaults=defaults,
    )

    errors = validate_config_compatibility(app)
    assert errors == []


def test_graph_requires_graph_store():
    infra = InfraConfig(graph_store="none")
    defaults = DefaultsConfig(retrieval_paths=["graph"])
    app = AppConfig(infra=infra, defaults=defaults)

    errors = validate_config_compatibility(app)
    assert any("graph" in e for e in errors)


def test_elasticsearch_requires_url_and_index_when_sparse_enabled():
    infra = InfraConfig(
        sparse_retriever="elasticsearch",
        elasticsearch_url="",
        elasticsearch_index="",
    )
    defaults = DefaultsConfig(retrieval_paths=["sparse"])
    app = AppConfig(infra=infra, defaults=defaults)

    errors = validate_config_compatibility(app)
    # Missing URL
    assert any("elasticsearch_url is empty" in e for e in errors)

    # Now set URL but leave index empty and ensure index error still appears
    infra.elasticsearch_url = "http://localhost:9200"
    errors = validate_config_compatibility(
        AppConfig(infra=infra, defaults=defaults)
    )
    assert any("elasticsearch_index is empty" in e for e in errors)


def test_qdrant_requires_url():
    infra = InfraConfig(vector_store="qdrant", qdrant_url="")
    app = AppConfig(infra=infra)

    errors = validate_config_compatibility(app)
    assert any("qdrant_url is empty" in e for e in errors)


def test_redis_requires_url():
    infra = InfraConfig(kv_store="redis", redis_url="")
    app = AppConfig(infra=infra)

    errors = validate_config_compatibility(app)
    assert any("redis_url is empty" in e for e in errors)


def test_visual_indexing_requires_vision_provider():
    infra = InfraConfig()
    defaults = DefaultsConfig(enable_visual_indexing=True)
    # Only text providers configured
    app = AppConfig(
        infra=infra,
        defaults=defaults,
        embedding_providers={
            "mock:mock-model": EmbeddingProviderConfig(
                provider="mock",
                model="mock-model",
                dimension=128,
                modality="text",
            )
        },
    )

    errors = validate_config_compatibility(app)
    assert any("enable_visual_indexing" in e for e in errors)


def test_skip_embedding_incompatible_with_graph():
    infra = InfraConfig(graph_store="networkx")
    defaults = DefaultsConfig(retrieval_paths=["graph"], skip_embedding=True)
    app = AppConfig(infra=infra, defaults=defaults)

    errors = validate_config_compatibility(app)
    assert any("skip_embedding=True" in e for e in errors)


def test_dense_or_graph_require_embedding_providers():
    infra = InfraConfig()
    defaults = DefaultsConfig(retrieval_paths=["dense", "graph"])
    app = AppConfig(infra=infra, defaults=defaults, embedding_providers={})

    errors = validate_config_compatibility(app)
    assert any("embedding_providers is empty" in e for e in errors)


def test_llm_extractor_must_reference_existing_llm_provider():
    infra = InfraConfig()
    defaults = DefaultsConfig()
    extractors = {
        "llm-default": ExtractorConfig(type="llm", llm_provider="openai:gpt-4o-mini")
    }
    # Missing llm_providers entry -> should error
    app = AppConfig(
        infra=infra,
        defaults=defaults,
        extractors=extractors,
        llm_providers={},
    )

    errors = validate_config_compatibility(app)
    assert any(
        "Extractor 'llm-default' references llm_provider 'openai:gpt-4o-mini'"
        in e
        for e in errors
    )

    # When the provider exists, the error should disappear
    app_ok = AppConfig(
        infra=infra,
        defaults=defaults,
        extractors=extractors,
        llm_providers={
            "openai:gpt-4o-mini": LLMProviderConfig(
                provider="openai", model="gpt-4o-mini"
            )
        },
    )
    errors_ok = validate_config_compatibility(app_ok)
    assert not any(
        "Extractor 'llm-default' references llm_provider 'openai:gpt-4o-mini'"
        in e
        for e in errors_ok
    )

