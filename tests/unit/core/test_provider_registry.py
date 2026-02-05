import pytest
from unittest.mock import AsyncMock, Mock

from unified_memory.core.registry import ProviderRegistry


def test_register_and_get_embedding_provider():
    """Register and retrieve an embedding provider."""
    registry = ProviderRegistry()
    mock_provider = Mock()

    registry.register_embedding_provider("openai:text-embedding-3-small", mock_provider)
    assert registry.get_embedding_provider("openai:text-embedding-3-small") is mock_provider


def test_no_duplicate_registration():
    """Second registration with the same key is a no-op."""
    registry = ProviderRegistry()
    first = Mock()
    second = Mock()

    registry.register_embedding_provider("key", first)
    registry.register_embedding_provider("key", second)

    assert registry.get_embedding_provider("key") is first


def test_resolve_with_fallback():
    """resolve_embedding_provider uses fallback_key when primary not found."""
    registry = ProviderRegistry()
    fallback = Mock()
    registry.register_embedding_provider("fallback:model", fallback)

    result = registry.resolve_embedding_provider(
        "missing", "model", fallback_key="fallback:model"
    )
    assert result is fallback


def test_resolve_returns_none():
    """resolve_embedding_provider returns None if key and fallback missing."""
    registry = ProviderRegistry()
    assert registry.resolve_embedding_provider("x", "y") is None


def test_register_extractor():
    """Register and retrieve an extractor."""
    registry = ProviderRegistry()
    ext = Mock()
    registry.register_extractor("llm:gpt4", ext)
    assert registry.get_extractor("llm:gpt4") is ext
    assert registry.get_extractor("nope") is None


def test_register_reranker():
    """Register and retrieve a reranker."""
    registry = ProviderRegistry()
    rr = Mock()
    registry.register_reranker("bge:base", rr)
    assert registry.get_reranker("bge:base") is rr
    assert registry.get_reranker("nope") is None
