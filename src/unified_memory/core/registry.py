"""
Provider Registry — singleton-style provider management.

Providers (embedders, extractors, rerankers, chunkers) are registered once
at bootstrap time and looked up by key during ingestion / retrieval.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from pathlib import Path

from unified_memory.core.interfaces import EmbeddingProvider
from unified_memory.ingestion.extractors.base import Extractor
from unified_memory.ingestion.parsers.base import DocumentParser
from unified_memory.ingestion.parsers.registry import (
    get_parser_registry,
    ParserRegistry,
)

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """Thread-safe registry of provider instances.

    Text embedding providers are keyed by ``"provider:model"`` strings
    (e.g. ``"openai:text-embedding-3-small"``).

    Vision embedding providers are stored in a separate namespace keyed
    the same way but resolved via ``get_vision_embedding_provider`` /
    ``resolve_vision_embedding_provider``.  This keeps the two modalities
    independent and allows the same provider class to serve both roles
    under different keys.

    Once registered a provider is never replaced — callers receive the same
    instance throughout the process lifetime.
    """

    def __init__(self) -> None:
        self._embedding_providers: Dict[str, EmbeddingProvider] = {}
        self._vision_embedding_providers: Dict[str, EmbeddingProvider] = {}
        self._llm_providers: Dict[str, Any] = {}
        self._extractors: Dict[str, Extractor] = {}
        self._rerankers: Dict[str, Any] = {}
        self._parser_registry: ParserRegistry = get_parser_registry()

    # ------------------------------------------------------------------
    # Text embedding providers
    # ------------------------------------------------------------------

    def register_embedding_provider(
        self, key: str, provider: EmbeddingProvider
    ) -> None:
        if key in self._embedding_providers:
            logger.debug("Embedding provider '%s' already registered, skipping.", key)
            return
        self._embedding_providers[key] = provider

    def get_embedding_provider(self, key: str) -> Optional[EmbeddingProvider]:
        return self._embedding_providers.get(key)

    def resolve_embedding_provider(
        self,
        provider_name: str,
        model: str,
        fallback_key: Optional[str] = None,
    ) -> Optional[EmbeddingProvider]:
        """Resolve by ``provider:model`` key, with optional fallback."""
        key = f"{provider_name}:{model}"
        result = self._embedding_providers.get(key)
        if result is None and fallback_key:
            result = self._embedding_providers.get(fallback_key)
        return result

    # ------------------------------------------------------------------
    # Vision embedding providers
    # ------------------------------------------------------------------

    def register_vision_embedding_provider(
        self, key: str, provider: EmbeddingProvider
    ) -> None:
        """Register a provider for image/multimodal embeddings.

        Key convention is the same as for text providers:
        ``"provider:model"`` (e.g. ``"openai:clip-vit-base-patch32"``).
        """
        if key in self._vision_embedding_providers:
            logger.debug(
                "Vision embedding provider '%s' already registered, skipping.", key
            )
            return
        self._vision_embedding_providers[key] = provider

    def get_vision_embedding_provider(
        self, key: str
    ) -> Optional[EmbeddingProvider]:
        return self._vision_embedding_providers.get(key)

    def resolve_vision_embedding_provider(
        self,
        provider_name: str,
        model: str,
        fallback_key: Optional[str] = None,
    ) -> Optional[EmbeddingProvider]:
        """Resolve a vision provider by ``provider:model`` key."""
        key = f"{provider_name}:{model}"
        result = self._vision_embedding_providers.get(key)
        if result is None and fallback_key:
            result = self._vision_embedding_providers.get(fallback_key)
        return result

    # ------------------------------------------------------------------
    # LLM providers
    # ------------------------------------------------------------------

    def register_llm_provider(self, key: str, provider: Any) -> None:
        if key in self._llm_providers:
            logger.debug("LLM provider '%s' already registered, skipping.", key)
            return
        self._llm_providers[key] = provider

    def get_llm_provider(self, key: str) -> Optional[Any]:
        return self._llm_providers.get(key)

    # ------------------------------------------------------------------
    # Extractors
    # ------------------------------------------------------------------

    def register_extractor(self, key: str, extractor: Any) -> None:
        if key in self._extractors:
            logger.debug("Extractor '%s' already registered, skipping.", key)
            return
        self._extractors[key] = extractor

    def get_extractor(self, key: str) -> Optional[Any]:
        return self._extractors.get(key)

    # ------------------------------------------------------------------
    # Rerankers
    # ------------------------------------------------------------------

    def register_reranker(self, key: str, reranker: Any) -> None:
        if key in self._rerankers:
            logger.debug("Reranker '%s' already registered, skipping.", key)
            return
        self._rerankers[key] = reranker

    def get_reranker(self, key: str) -> Optional[Any]:
        return self._rerankers.get(key)

    # ------------------------------------------------------------------
    # Parsers (delegate to global ParserRegistry)
    # ------------------------------------------------------------------

    def get_parser_registry(self) -> ParserRegistry:
        """Expose the shared ParserRegistry instance."""
        return self._parser_registry

    def register_parser(self, parser: DocumentParser) -> None:
        """Register a parser in the shared ParserRegistry."""
        self._parser_registry.register(parser)

    def get_parser_for_file(
        self, path: Path, mime_type: Optional[str] = None
    ) -> Optional[DocumentParser]:
        """Lookup a parser by file path and optional MIME type."""
        return self._parser_registry.get_parser_for_file(path, mime_type)
