"""
Graph Retrieval Implementation.

Performs graph-based retrieval using entity entry points and Personalized PageRank (PPR).
"""

from typing import Any, Dict, List, Optional, Tuple
import logging

from unified_memory.core.types import RetrievalResult, CollectionType, NodeType
from unified_memory.storage.base import VectorStoreBackend, GraphStoreBackend
from unified_memory.cas.content_store import ContentStore
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
        namespace_manager: NamespaceManager,
        content_store: Optional["ContentStore"] = None,
    ):
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.namespace_manager = namespace_manager
        self.content_store = content_store
    
    async def retrieve(
        self,
        query: str,
        # Allow passing pre-computed embedding to avoid re-embedding
        query_embedding: Optional[List[float]],
        namespaces: List[str],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant info via graph traversal.
        """
        if not namespaces:
            return []

        # For now, use the first namespace as the primary config namespace.
        primary_ns = namespaces[0]
        ns_config = await self.namespace_manager.get_config(primary_ns)
        if not ns_config:
            return []

        # 1. Embed query if not provided
        # Caller is expected to compute the embedding once and pass it in,
        # but we keep a defensive fallback path for backwards-compatibility.
        if not query_embedding:
            raise ValueError(
                "query_embedding must be provided to GraphRetriever.retrieve(). "
                "Compute it once in the orchestration layer."
            )
        # 2. Match entities (Vector Search on Entities Collection) per namespace,
        # and accumulate seeds across namespaces.
        all_entity_results: List[RetrievalResult] = []
        seed_ids: List[str] = []

        for ns in namespaces:
            try:
                entity_collection = await self.namespace_manager.get_collection_name(
                    ns, CollectionType.ENTITIES
                )

                entity_results = await self.vector_store.search(
                    query_embedding=query_embedding,
                    top_k=limit * 2,
                    namespace=ns,
                    collection=entity_collection,
                    filters=filters,
                )

                if not entity_results:
                    continue

                seed_ids.extend([r.id for r in entity_results])
                all_entity_results.extend(
                    RetrievalResult(
                        id=r.id,
                        content=r.metadata.get("name", r.id),
                        score=r.score,
                        metadata=r.metadata,
                        source="graph:entity_match",
                        evidence_type="graph_entity",
                    )
                    for r in entity_results
                )
            except Exception as e:
                logger.error(f"Error retrieving graph entities from namespace {ns}: {e}")

        if not seed_ids:
            return []

        # Deduplicate seeds — same entity in multiple namespaces should appear once
        seed_ids = list(dict.fromkeys(seed_ids))

        # 3. Run PPR per target namespace and merge scores (Issue: was only primary_ns)
        merged_scores: Dict[str, float] = {}
        for ns in namespaces:
            try:
                ppr_scores_ns = await self.graph_store.personalized_pagerank(
                    seed_nodes=seed_ids,
                    namespace=ns,
                    top_k=limit * 4,
                )
                for node_id, score in ppr_scores_ns:
                    merged_scores[node_id] = max(merged_scores.get(node_id, 0.0), score)
            except Exception as e:
                logger.warning(f"PPR failed for namespace {ns}: {e}")

        ppr_scores = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)

        if not ppr_scores:
            # Fallback: return the matched entities themselves if no graph expansion possible
            all_entity_results.sort(key=lambda x: x.score, reverse=True)
            return all_entity_results[:limit]

        # 4. Get nodes from PPR results (Batch Fetch) - try each namespace
        node_ids = [node_id for node_id, _ in ppr_scores]
        nodes_seen: Dict[str, Any] = {}
        for ns in namespaces:
            try:
                batch = await self.graph_store.get_nodes_batch(node_ids, ns)
                for n in batch:
                    if n.id not in nodes_seen:
                        nodes_seen[n.id] = n
            except Exception as e:
                logger.warning(f"get_nodes_batch failed for {ns}: {e}")
        nodes = list(nodes_seen.values())

        # Prefer passage nodes as in HippoRAG
        score_map = {node_id: score for node_id, score in ppr_scores}
        passage_results: List[RetrievalResult] = []
        other_results: List[RetrievalResult] = []

        for node in nodes:
            node_type = getattr(node, "node_type", None)
            content = node.content or ""

            # Hydrate passage content from content_store (graph nodes no longer store it)
            if node_type == NodeType.PASSAGE and self.content_store:
                content_hash = node.properties.get("content_hash", "")
                if content_hash:
                    stored = await self.content_store.get_content(content_hash)
                    if stored:
                        content = stored

            metadata = {
                "node_type": node_type.value if node_type else None,
                "properties": node.properties,
            }
            result = RetrievalResult(
                id=node.id,
                content=content,
                score=score_map.get(node.id, 0.0),
                metadata=metadata,
                source="graph:ppr",
                evidence_type="graph_chunk"
                if node_type == NodeType.PASSAGE
                else "graph_node",
            )
            if node_type == NodeType.PASSAGE:
                passage_results.append(result)
            else:
                other_results.append(result)

        # Sort and take passages first
        passage_results.sort(key=lambda x: x.score, reverse=True)
        other_results.sort(key=lambda x: x.score, reverse=True)

        combined: List[RetrievalResult] = passage_results + other_results

        # Optionally add a few entity matches that weren't in the PPR results
        existing_ids = {r.id for r in combined}
        for r in sorted(all_entity_results, key=lambda x: x.score, reverse=True):
            if len(combined) >= limit:
                break
            if r.id not in existing_ids:
                combined.append(r)
                existing_ids.add(r.id)

        combined.sort(key=lambda x: x.score, reverse=True)
        return combined[:limit]
