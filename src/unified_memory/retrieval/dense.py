"""
Dense Retrieval Implementation.

Performs vector similarity search and hydrates content from ContentStore.
"""

from typing import Any, Dict, List, Optional
import logging

from unified_memory.core.types import RetrievalResult, Modality, CollectionType
from unified_memory.storage.base import VectorStoreBackend
from unified_memory.cas.content_store import ContentStore
from unified_memory.namespace.manager import NamespaceManager
from unified_memory.core.interfaces import EmbeddingProvider

logger = logging.getLogger(__name__)

class DenseRetriever:
    """
    Retrieves documents using dense vector similarity.
    
    Flow:
    1. Embed query (using tenant-wide model)
    2. Search vector store (text collection)
    3. Hydrate content from ContentStore
    """
    
    def __init__(
        self,
        vector_store: VectorStoreBackend,
        embedding_provider: EmbeddingProvider,
        namespace_manager: NamespaceManager,
        content_store: ContentStore,
    ):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.namespace_manager = namespace_manager
        self.content_store = content_store
    
    async def retrieve(
        self,
        query: str,
        namespaces: List[str],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks from specified namespaces.
        
        Args:
            query: The search query string.
            namespaces: List of namespace IDs to search in.
            limit: Maximum number of results to return.
            filters: Optional metadata filters.
            
        Returns:
            List of hydrated RetrievalResult objects.
        """
        if not namespaces:
            return []

        # 1. Embed query (single embedding for all namespaces)
        # We assume all namespaces in the list belong to the same tenant/embedding space.
        # This is enforced by the caller (UnifiedSearchService).
        query_embedding = await self.embedding_provider.embed(query)
        
        # 2. Search each namespace
        # TODO: If the vector store supports multi-namespace search natively, optimize this loop.
        # Current design isolates by metadata filter.
        
        all_results: List[RetrievalResult] = []
        
        for namespace in namespaces:
            try:
                ns_config = await self.namespace_manager.get_config(namespace)
                if not ns_config:
                    logger.warning(f"Skipping unknown namespace: {namespace}")
                    continue
                
                collection = await self.namespace_manager.get_collection_name(
                    namespace, CollectionType.TEXTS
                )
                
                # Build namespace filter
                # CRITICAL: We must filter by the canonical namespace_id to ensure isolation
                ns_filter = {"namespace": ns_config.namespace_id}
                if filters:
                    ns_filter.update(filters)
                
                # Search
                vector_results = await self.vector_store.search(
                    query_embedding=query_embedding,
                    top_k=limit,
                    namespace=namespace,
                    collection=collection,
                    filters=ns_filter,
                )
                
                # 3. Hydrate content from ContentStore
                for vec_result in vector_results:
                    content_hash = vec_result.metadata.get("content_hash")
                    content = ""
                    
                    if content_hash:
                        content = await self.content_store.get_content(content_hash)
                        if content is None:
                            # Fallback or degraded state
                            content = ""
                            logger.warning(f"Content missing for hash: {content_hash}")
                    
                    all_results.append(RetrievalResult(
                        id=vec_result.id,
                        content=content,
                        score=vec_result.score,
                        metadata=vec_result.metadata,
                        source="dense",
                        evidence_type="text",
                    ))
                    
            except Exception as e:
                logger.error(f"Error retrieving from namespace {namespace}: {e}")
                # Continue with other namespaces
                continue
        
        # 4. Sort by score and return top_k
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:limit]
