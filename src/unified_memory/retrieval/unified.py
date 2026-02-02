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
from unified_memory.core.interfaces import EmbeddingProvider, LLMProvider, Reranker, SparseRetriever

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
    5. Answer generation (optional)
    """
    
    def __init__(
        self,
        namespace_manager: NamespaceManager,
        vector_store: VectorStoreBackend, # Note: Should ideally be NamespaceAwareVectorStore wrapper if available
        graph_store: GraphStoreBackend,
        kv_store: KVStoreBackend,
        content_store: ContentStore,
        embedding_provider_factory: Any, # Callable[[EmbeddingModelConfig], EmbeddingProvider]
        llm_provider: LLMProvider,
        reranker: Optional[Reranker] = None,
        sparse_retriever: Optional[SparseRetriever] = None,
    ):
        self.namespace_manager = namespace_manager
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.kv_store = kv_store
        self.content_store = content_store
        self.embedding_provider_factory = embedding_provider_factory
        self.llm_provider = llm_provider
        self.reranker = reranker
        self.sparse_retriever = sparse_retriever
        
        # Initialize sub-retrievers
        # Note: DenseRetriever needs an embedding provider. 
        # Since provider depends on tenant config, we might need to instantate it PER REQUEST 
        # or cache providers.
        # For this design, we'll instantiate retrievers per request or pass providers dynamically.
        # But DenseRetriever takes a provider in __init__.
        # FIX: We will create "Retriever Factories" or instantiate them inside search method 
        # once we have the correct tenant provider.
        # OR: We make DenseRetriever accept a provider in retrieve() or be specific to a tenant.
        
        # Let's instantiate them on the fly in search() or keeping them as stateless helpers 
        # that accept dependencies.
        # The current DenseRetriever definition takes provider in __init__.
        # This implies DenseRetriever is bound to a specific provider (and thus Model).
        # But UnifiedSearchService serves multiple tenants.
        # So we should probably keep them as internal helpers or factories.
        
        self._embedding_cache: Dict[str, EmbeddingProvider] = {}

    async def search(
        self,
        query: str,
        user_id: str,
        namespace: Optional[str] = None,
        config: Optional[RetrievalConfig] = None,
    ) -> List[RetrievalResult]:
        """
        Execute unified search.
        """
        # 1. Resolve Namespace & Config
        if not namespace:
            # Default to user's private namespace
            namespace = Namespace(user_id=user_id).to_string()
            
        # Resolve config (hierarchical)
        if not config:
            config = await RetrievalConfig.resolve(namespace, self.namespace_manager)
            
        # 2. Get Embedding Provider (Tenant-Wide)
        ns_config = await self.namespace_manager.get_config(namespace)
        if not ns_config:
            raise ValueError(f"Namespace not found: {namespace}")
            
        tenant_config = await self.namespace_manager.get_tenant_config(ns_config.tenant_id)
        
        # Get provider from factory/cache
        provider_key = f"{tenant_config.text_embedding.provider}:{tenant_config.text_embedding.model}"
        if provider_key in self._embedding_cache:
            embedder = self._embedding_cache[provider_key]
        else:
            embedder = self.embedding_provider_factory(tenant_config.text_embedding)
            self._embedding_cache[provider_key] = embedder
            
        # 3. Embed Query ONCE
        query_embedding = await embedder.embed(query)
        
        # 4. Resolve Target Namespaces
        # For MVP, we search the primary namespace. 
        # Future: Expand to shared/public namespaces based on ACL.
        target_namespaces = [namespace] 
        
        # 5. Execute Paths in Parallel
        tasks = []
        
        # Dense Path
        if "dense" in config.paths:
            # Instantiate ephemeral retriever with correct provider
            dense_retriever = DenseRetriever(
                vector_store=self.vector_store,
                embedding_provider=embedder,
                namespace_manager=self.namespace_manager,
                content_store=self.content_store
            )
            tasks.append(
                self._safe_execute(
                    dense_retriever.retrieve(
                        query=query, # Passed but not used for embedding if we refactored, but DenseRetriever embeds internally currently. 
                        # WAIT: DenseRetriever.retrieve() embeds internally. 
                        # We should refactor DenseRetriever to accept embedding OR 
                        # we passed the same embedder so caching might handle it?
                        # CachedEmbeddingProvider would handle it.
                        namespaces=target_namespaces,
                        limit=config.top_k * 2, # Fetch more for fusion
                        filters=None
                    ),
                    "dense"
                )
            )

        # Graph Path
        if "graph" in config.paths:
            graph_retriever = GraphRetriever(
                graph_store=self.graph_store,
                vector_store=self.vector_store,
                embedding_provider=embedder,
                namespace_manager=self.namespace_manager
            )
            tasks.append(
                self._safe_execute(
                    graph_retriever.retrieve(
                        query=query,
                        query_embedding=query_embedding, # Pass embedding!
                        namespace=namespace, # Graph currently single-namespace in signature
                        limit=config.top_k * 2
                    ),
                    "graph"
                )
            )

        # Sparse Path
        if "sparse" in config.paths and self.sparse_retriever:
            tasks.append(
                self._safe_execute(
                    self.sparse_retriever.retrieve(
                        query=query,
                        namespace=namespace,
                        top_k=config.top_k * 2
                    ),
                    "sparse"
                )
            )
            
        # 6. Gather Results
        results_list = await asyncio.gather(*tasks)
        all_results = []
        for res in results_list:
            all_results.extend(res)
            
        # 7. Filter Invalid Memories
        valid_results = []
        for r in all_results:
            # Check metadata for memory status
            status = r.metadata.get("status")
            if status in (MemoryStatus.INVALID_SUPERSEDED.value, MemoryStatus.INVALID_RETRACTED.value):
                continue
            valid_results.append(r)
            
        # 8. Normalize Scores
        normalized = normalize_scores(valid_results)
        
        # 9. Fuse
        if config.fusion_method == "rrf":
            fused = reciprocal_rank_fusion(normalized, k=60)
        else:
            # Default to RRF
            fused = reciprocal_rank_fusion(normalized)
            
        if config.rerank:
            reranker = self.reranker
            # If reranker not injected but requested, try to instantiate default local (BGE)
            # This is a basic fallback for "out of box" experience
            if not reranker:
                 # TODO: Factor this out into a factory
                 try: 
                     from .rerankers import BGEReranker
                     # Cache this if possible or instantiate ephemeral
                     # Using ephemeral for now, optimized production would cache
                     reranker = BGEReranker() 
                 except ImportError:
                     logger.warning("Reranking requested but modules not found. Skipping.")
                     reranker = None

            if reranker:
                # Take top N for reranking
                candidates = fused[:50] 
                fused = await reranker.rerank(query, candidates, top_k=config.top_k)
            
        return fused[:config.top_k]

    async def _safe_execute(self, coroutine, source_name: str) -> List[RetrievalResult]:
        """Execute a retrieval path safely, logging errors."""
        try:
            return await coroutine
        except Exception as e:
            logger.error(f"Error in {source_name} retrieval: {e}", exc_info=True)
            return []
