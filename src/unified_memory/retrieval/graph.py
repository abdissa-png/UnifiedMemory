"""
Graph Retrieval Implementation.

Performs graph-based retrieval using entity entry points and Personalized PageRank (PPR).
"""

from typing import Any, Dict, List, Optional, Tuple
import logging

from unified_memory.core.types import RetrievalResult, CollectionType
from unified_memory.storage.base import VectorStoreBackend, GraphStoreBackend
from unified_memory.namespace.manager import NamespaceManager
from unified_memory.core.interfaces import EmbeddingProvider

logger = logging.getLogger(__name__)

class GraphRetriever:
    """
    Retrieves documents using graph algorithms (PPR).
    
    Flow:
    1. Embed query
    2. Vector search 'entities' collection to find seed entities
    3. Run Personalized PageRank (PPR) starting from seed entities
    4. Return top-ranked graph nodes (passages/entities)
    """
    
    def __init__(
        self,
        graph_store: GraphStoreBackend,
        vector_store: VectorStoreBackend,
        embedding_provider: EmbeddingProvider,
        namespace_manager: NamespaceManager,
    ):
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.namespace_manager = namespace_manager
    
    async def retrieve(
        self,
        query: str,
        # Allow passing pre-computed embedding to avoid re-embedding
        query_embedding: Optional[List[float]], 
        namespace: str,
        limit: int = 10,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant info via graph traversal.
        """
        ns_config = await self.namespace_manager.get_config(namespace)
        if not ns_config:
            return []
            
        # 1. Embed query if not provided
        if not query_embedding:
            query_embedding = await self.embedding_provider.embed(query)
            
        # 2. Match entities (Vector Search on Entities Collection)
        # This finds the "entry points" into the graph
        entity_collection = await self.namespace_manager.get_collection_name(
            namespace, CollectionType.ENTITIES
        )
        
        # We fetch more candidates (2x limit) for seed nodes
        entity_results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=limit * 2,
            namespace=namespace,
            collection=entity_collection,
            filters={"namespace": ns_config.namespace_id},
        )
        
        if not entity_results:
            return []
            
        entity_ids = [r.id for r in entity_results]
        
        # 3. Run PPR from matched entities
        # Note: PPR implementation in GraphStore should handle non-existent nodes gracefully
        ppr_scores = await self.graph_store.personalized_pagerank(
            seed_nodes=entity_ids,
            namespace=namespace,
            top_k=limit * 2, # Fetch more to filter types if needed
        )
        
        if not ppr_scores:
            # Fallback: return the matched entities themselves if no graph expansion possible
            return [
                 RetrievalResult(
                    id=r.id,
                    content=r.metadata.get("name", r.id), # Entities might not have full content in vector
                    score=r.score,
                    metadata=r.metadata,
                    source="graph:entity_match",
                    evidence_type="graph_entity",
                )
                for r in entity_results[:limit]
            ]

        # 4. Get nodes from PPR results (Batch Fetch)
        node_ids = [node_id for node_id, _ in ppr_scores]
        nodes = await self.graph_store.get_nodes_batch(node_ids, namespace)
        
        results: List[RetrievalResult] = []
        
        # Map back to results
        # Use a map for O(1) score lookup
        score_map = {node_id: score for node_id, score in ppr_scores}
        
        for node in nodes:
            results.append(RetrievalResult(
                id=node.id,
                content=node.content or "", # Graph nodes should have content or summary
                score=score_map.get(node.id, 0.0),
                metadata={
                    "node_type": node.type,
                    "properties": node.properties
                },
                source="graph:ppr",
                evidence_type="graph_node" if node.type != "chunk" else "graph_chunk",
            ))
            
        # Add direct entity matches that might not have been returned by PPR 
        # (if PPR expansion went towards chunks/other entities)
        # Only add if we have space
        existing_ids = set(r.id for r in results)
        for r in entity_results:
            if len(results) >= limit:
                break
            if r.id not in existing_ids:
                results.append(RetrievalResult(
                    id=r.id,
                    content=r.metadata.get("name", r.id),
                    score=r.score,
                    metadata=r.metadata,
                    source="graph:entity_match",
                    evidence_type="graph_entity",
                ))
                
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]
