"""
System Bootstrap — initializes infrastructure and builds services.

Usage::

    ctx = SystemContext(config)
    ctx.build_services()
    result = await ctx.ingestion_pipeline.ingest_text(...)
    results = await ctx.search_service.search(...)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from unified_memory.core.registry import ProviderRegistry
from unified_memory.cas.registry import CASRegistry
from unified_memory.cas.content_store import ContentStore
from unified_memory.cas.document_registry import DocumentRegistry
from unified_memory.namespace.manager import NamespaceManager
from unified_memory.storage.kv.memory_store import MemoryKVStore
from unified_memory.storage.vector.memory_store import MemoryVectorStore
from unified_memory.storage.graph.networkx_store import NetworkXGraphStore
from unified_memory.ingestion.pipeline import IngestionPipeline
from unified_memory.retrieval.unified import UnifiedSearchService
from unified_memory.retrieval.dense import DenseRetriever
from unified_memory.retrieval.graph import GraphRetriever

logger = logging.getLogger(__name__)


class SystemContext:
    """Central dependency container, initialised once at startup.

    Parameters
    ----------
    config : dict
        A configuration dictionary with optional keys:

        - ``kv_store``: ``"memory"`` (default) or ``"redis"``
        - ``redis_url``: URL for Redis (default ``redis://localhost:6379/0``)
        - ``vector_store``: ``"memory"`` (default) or ``"qdrant"``
        - ``graph_store``: ``"networkx"`` (default) or ``"neo4j"``
        - ``sparse_retriever``: ``"bm25"`` (default) or ``"elasticsearch"``
        - ``qdrant_url``, ``qdrant_api_key``
        - ``neo4j_uri``, ``neo4j_auth``
        - ``elasticsearch_url``, ``elasticsearch_api_key``, ``elasticsearch_index``
        - ``embedding_providers``: mapping of ``key -> {provider, model, dimension, api_key, ...}``
        - ``extractors``: mapping of ``key -> {type, ...}``
        - ``llm_providers``: mapping of ``key -> {provider, model, api_key, ...}``
        - ``rerankers``: mapping of ``key -> {type, ...}``
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}

        self.kv_store = self._build_kv_store(config)
        self.vector_store = self._build_vector_store(config)
        self.graph_store = self._build_graph_store(config)
        self.content_store = ContentStore(self.kv_store)
        self.cas_registry = CASRegistry(self.kv_store)
        self.document_registry = DocumentRegistry(self.kv_store)
        self.namespace_manager = NamespaceManager(self.kv_store)

        self.provider_registry = ProviderRegistry()

        # Optional ElasticSearch store (used as both content store and sparse retriever)
        self.elasticsearch_store = None
        if config.get("sparse_retriever") == "elasticsearch":
            self.elasticsearch_store = self._build_elasticsearch_store(config)

        self._register_providers(config)

        self.ingestion_pipeline: Optional[IngestionPipeline] = None
        self.search_service: Optional[UnifiedSearchService] = None

    # ------------------------------------------------------------------
    # Service construction
    # ------------------------------------------------------------------

    def build_services(
        self,
        *,
        default_embedding_key: Optional[str] = None,
    ) -> "SystemContext":
        """Build the ingestion pipeline and search service.

        Call after registering all providers.
        """
        default_embedder = None
        if default_embedding_key:
            default_embedder = self.provider_registry.get_embedding_provider(
                default_embedding_key
            )

        sparse_retriever = self.elasticsearch_store
        if not sparse_retriever:
            from unified_memory.retrieval.sparse import SparseRetriever as _SparseBM25
            sparse_retriever = _SparseBM25()

        # Build retrievers and register them in the provider registry so they
        # can be reused by other orchestration layers if needed.
        dense_retriever = DenseRetriever(
            vector_store=self.vector_store,
            namespace_manager=self.namespace_manager,
            content_store=self.content_store,
        )
        graph_retriever = GraphRetriever(
            graph_store=self.graph_store,
            vector_store=self.vector_store,
            namespace_manager=self.namespace_manager,
            content_store=self.content_store,
        )


        self.ingestion_pipeline = IngestionPipeline(
            vector_store=self.vector_store,
            graph_store=self.graph_store,
            sparse_store=sparse_retriever,
            cas_registry=self.cas_registry,
            content_store=self.content_store,
            document_registry=self.document_registry,
            namespace_manager=self.namespace_manager,
            provider_registry=self.provider_registry,
            embedding_provider=default_embedder,
        )

        self.search_service = UnifiedSearchService(
            namespace_manager=self.namespace_manager,
            vector_store=self.vector_store,
            graph_store=self.graph_store,
            kv_store=self.kv_store,
            content_store=self.content_store,
            provider_registry=self.provider_registry,
            sparse_retriever=self.elasticsearch_store,
            dense_retriever=dense_retriever,
            graph_retriever=graph_retriever,
        )
        return self

    # ------------------------------------------------------------------
    # Infrastructure builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_kv_store(config: Dict[str, Any]):
        backend = config.get("kv_store", "memory")
        if backend == "redis":
            from unified_memory.storage.kv.redis_store import RedisKVStore

            return RedisKVStore(
                url=config.get("redis_url", "redis://localhost:6379/0"),
            )
        return MemoryKVStore()

    @staticmethod
    def _build_vector_store(config: Dict[str, Any]):
        backend = config.get("vector_store", "memory")
        if backend == "qdrant":
            from unified_memory.storage.vector.qdrant import QdrantVectorStore

            return QdrantVectorStore(
                url=config.get("qdrant_url", "http://localhost:6333"),
                api_key=config.get("qdrant_api_key"),
            )
        return MemoryVectorStore()

    @staticmethod
    def _build_graph_store(config: Dict[str, Any]):
        backend = config.get("graph_store", "networkx")
        if backend == "neo4j":
            from unified_memory.storage.graph.neo4j import Neo4jGraphStore

            return Neo4jGraphStore(
                uri=config.get("neo4j_uri", "bolt://localhost:7687"),
                auth=config.get("neo4j_auth", ("neo4j", "password")),
            )
        return NetworkXGraphStore()

    @staticmethod
    def _build_elasticsearch_store(config: Dict[str, Any]):
        from unified_memory.storage.search.elasticsearch_store import (
            ElasticSearchStore,
        )

        return ElasticSearchStore(
            url=config.get("elasticsearch_url", "http://localhost:9200"),
            index_name=config.get("elasticsearch_index", "unified_memory_content"),
            api_key=config.get("elasticsearch_api_key"),
        )

    # ------------------------------------------------------------------
    # Provider registration
    # ------------------------------------------------------------------

    def _register_providers(self, config: Dict[str, Any]) -> None:
        # Embedding providers — ``modality`` field routes to the right slot:
        #   "text"   (default) → text embedding registry
        #   "vision" or "image" → vision embedding registry
        for key, emb_cfg in config.get("embedding_providers", {}).items():
            provider = self._build_embedding_provider(emb_cfg)
            if not provider:
                continue
            modality = emb_cfg.get("modality", "text").lower()
            if modality in ("vision", "image"):
                self.provider_registry.register_vision_embedding_provider(key, provider)
            else:
                self.provider_registry.register_embedding_provider(key, provider)

        # LLM providers (stored in registry for extractor use)
        llm_instances: Dict[str, Any] = {}
        for key, llm_cfg in config.get("llm_providers", {}).items():
            llm = self._build_llm_provider(llm_cfg)
            if llm:
                llm_instances[key] = llm

        # Extractors
        for key, ext_cfg in config.get("extractors", {}).items():
            extractor = self._build_extractor(ext_cfg, llm_instances)
            if extractor:
                self.provider_registry.register_extractor(key, extractor)

        # Rerankers
        for key, rnk_cfg in config.get("rerankers", {}).items():
            reranker = self._build_reranker(rnk_cfg)
            if reranker:
                self.provider_registry.register_reranker(key, reranker)

    @staticmethod
    def _build_embedding_provider(emb_cfg: Dict[str, Any]):
        prov = emb_cfg.get("provider", "mock")
        if prov == "mock":
            from unified_memory.embeddings.providers.mock_provider import (
                MockEmbeddingProvider,
            )

            return MockEmbeddingProvider(
                dimension=emb_cfg.get("dimension", 128),
            )
        if prov == "openai":
            from unified_memory.embeddings.providers.openai_provider import (
                OpenAIEmbeddingProvider,
            )

            return OpenAIEmbeddingProvider(
                api_key=emb_cfg["api_key"],
                model=emb_cfg.get("model", "text-embedding-3-small"),
                dimension=emb_cfg.get("dimension", 1536),
                base_url=emb_cfg.get("base_url"),
            )
        logger.warning("Unsupported embedding provider '%s'", prov)
        return None

    @staticmethod
    def _build_llm_provider(llm_cfg: Dict[str, Any]):
        prov = llm_cfg.get("provider", "")
        if prov == "openai":
            from unified_memory.llm.openai_provider import OpenAILLMProvider

            return OpenAILLMProvider(
                api_key=llm_cfg["api_key"],
                model=llm_cfg.get("model", "gpt-4o-mini"),
                base_url=llm_cfg.get("base_url"),
                supports_images=llm_cfg.get("supports_images"),  # None → False
            )
        logger.warning("Unsupported LLM provider '%s'", prov)
        return None

    @staticmethod
    def _build_extractor(ext_cfg: Dict[str, Any], llm_instances: Dict[str, Any]):
        ext_type = ext_cfg.get("type", "mock")
        if ext_type == "mock":
            from unified_memory.ingestion.extractors.mock import MockExtractor

            return MockExtractor()
        if ext_type == "llm":
            llm_key = ext_cfg.get("llm_provider", "")
            llm = llm_instances.get(llm_key)
            if not llm:
                logger.warning(
                    "LLM extractor requires llm_provider '%s' but it was not found",
                    llm_key,
                )
                return None
            from unified_memory.ingestion.extractors.llm_extractor import (
                LLMExtractor,
            )

            return LLMExtractor(llm_provider=llm)
        logger.warning("Unsupported extractor type '%s'", ext_type)
        return None

    @staticmethod
    def _build_reranker(rnk_cfg: Dict[str, Any]):
        """Build a reranker instance from a config dictionary.

        Config keys
        -----------
        type : str
            ``"bge"`` — local BGE reranker via sentence-transformers.
            ``"cohere"`` — Cohere Rerank API.
        model : str, optional
            Model name / ID (provider-specific defaults apply when absent).
        api_key : str, optional
            Required for ``"cohere"``.
        """
        rnk_type = rnk_cfg.get("type", "")
        if rnk_type == "bge":
            try:
                from unified_memory.retrieval.rerankers.models import BGEReranker

                model = rnk_cfg.get("model", "BAAI/bge-reranker-base")
                return BGEReranker(model_name=model)
            except ImportError:
                logger.warning("BGEReranker not available (missing sentence-transformers)")
                return None
        if rnk_type == "cohere":
            api_key = rnk_cfg.get("api_key", "")
            if not api_key:
                logger.warning(
                    "Cohere reranker requires 'api_key' in the reranker config"
                )
                return None
            try:
                from unified_memory.retrieval.rerankers.models import CohereReranker

                model = rnk_cfg.get("model", "rerank-english-v3.0")
                return CohereReranker(api_key=api_key, model=model)
            except ImportError:
                logger.warning("CohereReranker not available (missing 'cohere' package)")
                return None
        logger.warning("Unsupported reranker type '%s'", rnk_type)
        return None
