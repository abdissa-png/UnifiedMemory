"""
Unified Search Service.

Orchestrates retrieval across Dense, Sparse, and Graph paths.
"""

from typing import Any, Dict, List, Optional
import asyncio
import logging

from unified_memory.core.types import (
    RetrievalResult, QueryResult, Modality, CollectionType, MemoryStatus
)
from unified_memory.namespace.manager import NamespaceManager
from unified_memory.namespace.types import RetrievalConfig, Namespace
from unified_memory.storage.base import VectorStoreBackend, GraphStoreBackend, KVStoreBackend
from unified_memory.cas.content_store import ContentStore
from unified_memory.core.interfaces import SparseRetriever
from unified_memory.core.registry import ProviderRegistry

from .dense import DenseRetriever
from .graph import GraphRetriever
from .fusion import normalize_scores, reciprocal_rank_fusion, linear_fusion

logger = logging.getLogger(__name__)


class UnifiedSearchService:
    """
    Unified entry point for retrieval operations.

    Orchestrates:
    1. Namespace resolution
    2. Parallel retrieval execution (Dense, Sparse, Graph)
    3. Result fusion
    4. Reranking
    """

    def __init__(
        self,
        namespace_manager: NamespaceManager,
        vector_store: VectorStoreBackend,
        graph_store: GraphStoreBackend,
        kv_store: KVStoreBackend,
        content_store: ContentStore,
        provider_registry: Optional[ProviderRegistry] = None,
        sparse_retriever: Optional[SparseRetriever] = None,
        dense_retriever: Optional["DenseRetriever"] = None,
        graph_retriever: Optional["GraphRetriever"] = None,
    ):
        self.namespace_manager = namespace_manager
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.kv_store = kv_store
        self.content_store = content_store
        self.provider_registry = provider_registry or ProviderRegistry()
        self.sparse_retriever = sparse_retriever
        self.dense_retriever: DenseRetriever = dense_retriever 
        self.graph_retriever: GraphRetriever = graph_retriever

    async def search(
        self,
        query: str,
        user_id: str,
        namespace: Optional[str] = None,
        config: Optional[RetrievalConfig] = None,
        request_options: Optional[Dict[str, Any]] = None,
        target_namespaces: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """
        Execute unified search.
        """
        # 1. Resolve Namespace & Config
        if not namespace:
            # Default to user's private namespace
            namespace = Namespace(user_id=user_id).to_string()

        # Resolve config (hierarchical, allowing request-level overrides)
        if not config:
            config = await RetrievalConfig.resolve(
                namespace, self.namespace_manager, request_options=request_options
            )
            
        # 2. Get Embedding Provider (Tenant-Wide)
        ns_config = await self.namespace_manager.get_config(namespace)
        if not ns_config:
            raise ValueError(f"Namespace not found: {namespace}")
            
        tenant_config = await self.namespace_manager.get_tenant_config(
            ns_config.tenant_id
        )

        model_cfg = tenant_config.text_embedding
        provider_key = f"{model_cfg.provider}:{model_cfg.model}"

        embedder = self.provider_registry.resolve_embedding_provider(
            model_cfg.provider, model_cfg.model,
        )
        if embedder is None:
            raise ValueError(
                f"No embedding provider found for '{provider_key}'. "
                "Register one in the ProviderRegistry."
            )
            
        # 3. Embed Query ONCE
        query_embedding = await embedder.embed(query)
        
        # 4. Resolve Target Namespaces
        # We use `namespace` as the primary/config namespace and optionally
        # search across additional target namespaces (e.g. shared/public).
        if target_namespaces is None:
            target_namespaces = [namespace]
        elif namespace not in target_namespaces:
            target_namespaces = [namespace] + list(target_namespaces)
        
        # 5. Execute Paths in Parallel
        tasks = []

        # Support either:
        # - `filters` applied to all paths, OR
        # - a dict with per-path keys: {"dense": {...}, "sparse": {...}, "graph": {...}}
        dense_filters = filters
        sparse_filters = filters
        graph_filters = filters
        if isinstance(filters, dict) and any(k in filters for k in ("dense", "sparse", "graph")):
            dense_filters = filters.get("dense")
            sparse_filters = filters.get("sparse")
            graph_filters = filters.get("graph")
        
        # Dense Path
        if "dense" in config.paths:
            if self.dense_retriever is None:
                raise ValueError("DenseRetriever not configured")
            tasks.append(
                self._safe_execute(
                    self.dense_retriever.retrieve(
                        query=query,
                        namespaces=target_namespaces,
                        limit=config.top_k * 2,  # Fetch more for fusion
                        filters=dense_filters,
                        query_embedding=query_embedding,
                        embedding_provider=embedder,
                    ),
                    "dense",
                )
            )

        # Graph Path
        if "graph" in config.paths:
            if self.graph_retriever is None:
                raise ValueError("GraphRetriever not configured")
            tasks.append(
                self._safe_execute(
                    self.graph_retriever.retrieve(
                        query=query,
                        query_embedding=query_embedding,  # Pass embedding
                        namespaces=target_namespaces,
                        limit=config.top_k * 2,
                        filters=graph_filters,
                    ),
                    "graph",
                )
            )

        # Sparse Path
        sparse_retriever = self.sparse_retriever
        if "sparse" in config.paths and sparse_retriever:
            tasks.append(
                self._safe_execute(
                    sparse_retriever.retrieve(
                        query=query,
                        namespaces=target_namespaces,
                        top_k=config.top_k * 2,
                        filters=sparse_filters,
                    ),
                    "sparse",
                )
            )

            
        # 6. Gather Results
        results_list = await asyncio.gather(*tasks)
        all_results = []
        for res in results_list:
            all_results.extend(res)

        # 6.5 Filter invalid memories (superseded/retracted) This will be useful when we are retrieving information from memory store
        valid_results = []
        for r in all_results:
            status = r.metadata.get("status")
            if status in (
                MemoryStatus.INVALID_SUPERSEDED.value,
                MemoryStatus.INVALID_RETRACTED.value,
            ):
                continue
            valid_results.append(r)
        all_results = valid_results
            
        # 7. Normalize Scores
        normalized = normalize_scores(all_results)

        # 8. Fuse
        if config.fusion_method == "rrf":
            fused = reciprocal_rank_fusion(normalized, k=60)
        elif config.fusion_method == "linear":
            # Default weights favouring dense but keeping sparse/graph contributions
            default_weights = {
                "dense": 0.4,
                "sparse": 0.3,
                "graph:ppr": 0.3,
            }
            weights = config.fusion_weights or default_weights
            fused = linear_fusion(normalized, weights=weights, normalize_first=True)
        else:
            raise ValueError(f"Unknown fusion method: {config.fusion_method}")
            
        # Before reranking, ensure all candidates have content where possible
        # by hydrating from content_store using content_hash if present.
        for r in fused:
            if r.content:
                continue
            content_hash = r.metadata.get("content_hash")
            if not content_hash:
                continue
            try:
                content = await self.content_store.get_content(content_hash)
                if content:
                    r.content = content
            except Exception as e:
                logger.warning(f"Failed to hydrate content for hash {content_hash}: {e}")

        if config.rerank:
            reranker_key = getattr(config, "reranker_key", "default")
            reranker = self.provider_registry.get_reranker(reranker_key)
            if reranker is None:
                reranker = self.provider_registry.get_reranker("default")
            if reranker:
                candidates = fused[:config.rerank_candidates_limit]
                fused = await reranker.rerank(query, candidates, top_k=config.top_k)

        return fused[:config.top_k]

    async def _safe_execute(self, coroutine, source_name: str) -> List[RetrievalResult]:
        """Execute a retrieval path safely, logging errors."""
        try:
            return await coroutine
        except Exception as e:
            logger.error(f"Error in {source_name} retrieval: {e}", exc_info=True)
            return []
