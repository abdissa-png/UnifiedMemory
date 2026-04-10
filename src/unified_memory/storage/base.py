"""
Storage backend abstractions for the unified memory system.

Defines:
- KVStoreBackend: key-value store with atomic operations
- VectorStoreBackend: vector DB abstraction with transactional helper
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, AsyncIterator, Tuple
from dataclasses import dataclass

from unified_memory.core.types import VectorSearchResult, GraphNode, GraphEdge


@dataclass
class VersionedValue:
    """Value with version for optimistic locking."""
    data: Dict[str, Any]
    version: int


class KVStoreBackend(ABC):
    """
    Key-Value store backend with atomic operations.

    CONCURRENCY: Must support atomic compare-and-swap for optimistic locking.
    Implementations: Redis, DynamoDB, etcd, in-memory, etc.
    """

    @abstractmethod
    async def get(self, key: str) -> Optional[VersionedValue]:
        """
        Get value by key. Returns None if not found.
        
        Returns VersionedValue containing the data and version number.
        """
        raise NotImplementedError

    @abstractmethod
    async def set(
        self,
        key: str,
        value: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """Set value. Returns True on success."""

        raise NotImplementedError

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key. Returns True if key existed."""

        raise NotImplementedError

    @abstractmethod
    async def set_if_not_exists(
        self,
        key: str,
        value: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """
        Atomically set value only if key doesn't exist.

        Returns True if set succeeded (key didn't exist), False otherwise.
        """

        raise NotImplementedError

    @abstractmethod
    async def compare_and_swap(
        self,
        key: str,
        expected_version: int,
        new_value: Dict[str, Any],
    ) -> bool:
        """
        Atomically update value only if version matches.

        The value dict must contain a "version" field that is checked.
        """

        raise NotImplementedError

    @abstractmethod
    async def delete_if_version(
        self,
        key: str,
        expected_version: int,
    ) -> bool:
        """
        Atomically delete only if version matches.

        Returns:
            True if delete succeeded (version matched)
            False if version mismatch or key not found
        """

        raise NotImplementedError

    @abstractmethod
    async def scan(
        self,
        pattern: str,
        limit: int = 1000,
    ) -> List[str]:
        """Scan keys matching pattern. Returns list of keys."""

        raise NotImplementedError

    @abstractmethod
    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern. Returns count deleted."""

        raise NotImplementedError

    async def close(self) -> None:
        """Clean up backend resources."""
        return None


class VectorStoreBackend(ABC):
    """
    Extended vector store with namespace-aware collection routing.

    Implementations wrap concrete backends like Qdrant, Pinecone, etc.
    """

    async def __aenter__(self) -> "VectorStoreBackend":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()

    async def connect(self) -> None:
        """Initialize connection. Override in implementations if needed."""

        return None

    async def disconnect(self) -> None:
        """Clean up connections. Override in implementations if needed."""

        return None

    async def close(self) -> None:
        """Clean up backend resources."""
        await self.disconnect()

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator["VectorStoreTransaction"]:
        """
        Context manager for transactional-style operations.

        Note: Most vector stores don't provide true ACID transactions.
        This helper implements compensating transactions (best-effort rollback).
        """

        tx = VectorStoreTransaction(self)
        try:
            yield tx
            await tx.commit()
        except Exception:
            await tx.rollback()
            raise

    @abstractmethod
    async def create_collection(
        self,
        name: str,
        dimension: int,
        distance_metric: str = "cosine",
        index_type: str = "hnsw",
        **kwargs: Any,
    ) -> bool:
        """Create a new collection.

        Accepts backend-specific keyword arguments (e.g. ``hnsw_m``,
        ``hnsw_ef_construct``, ``ivf_sq_quantile``) forwarded by callers
        that are aware of the concrete backend in use.
        """

        raise NotImplementedError

    @abstractmethod
    async def delete_collection(self, name: str) -> bool:
        """Delete a collection."""

        raise NotImplementedError

    @abstractmethod
    async def list_collections(self, prefix: Optional[str] = None) -> List[str]:
        """List collections, optionally filtered by prefix."""

        raise NotImplementedError

    @abstractmethod
    async def upsert(
        self,
        vectors: List[Dict[str, Any]],
        namespace: str = "default",
        collection: Optional[str] = None,
    ) -> int:
        """
        Upsert vectors.

        If collection is None, uses namespace as collection name.
        """

        raise NotImplementedError

    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        namespace: str = "default",
        collection: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[VectorSearchResult]:
        """Search by vector similarity."""

        raise NotImplementedError

    @abstractmethod
    async def get_by_id(
        self,
        id: str,
        collection: Optional[str] = None,
        namespace: str = "default",
    ) -> Optional[VectorSearchResult]:
        """Get vector by ID."""

        raise NotImplementedError

    @abstractmethod
    async def get_by_ids(
        self,
        ids: List[str],
        collection: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        """
        Batch get vectors by IDs.
        
        If namespace is provided, only return vectors that have that namespace.
        Returns list of VectorSearchResult (may be shorter than input if some IDs don't exist).
        """

        raise NotImplementedError

    @abstractmethod
    async def query_by_filter(
        self,
        filters: Dict[str, Any],
        collection: str,
        namespace: str = "default",
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Query vectors by metadata filter ONLY (no similarity search).
        """

        raise NotImplementedError

    @abstractmethod
    async def delete(
        self,
        ids: List[str],
        namespace: str = "default",
        collection: Optional[str] = None,
    ) -> int:
        """Delete vectors by ID."""

        raise NotImplementedError

    @abstractmethod
    async def delete_by_filter(
        self,
        filters: Dict[str, Any],
        namespace: str = "default",
        collection: Optional[str] = None,
    ) -> int:
        """Delete vectors matching filter."""

        raise NotImplementedError

    @abstractmethod
    async def add_namespace(
        self,
        id: str,
        namespace: str,
        collection: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> bool:
        """Add a namespace to an existing vector's metadata."""
        raise NotImplementedError

    @abstractmethod
    async def remove_namespace(
        self,
        id: str,
        namespace: str,
        collection: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> Tuple[bool, bool]:
        """
        Remove a namespace from a vector.
        
        Returns (success, was_last_namespace).
        """
        raise NotImplementedError

    @abstractmethod
    async def remove_document_reference(
        self,
        id: str,
        document_id: str,
        collection: Optional[str] = None,
    ) -> List[str]:
        """Remove a document from a vector's ``source_doc_ids`` /
        ``source_locations`` without touching the namespaces list.

        Returns the **remaining** ``source_doc_ids`` after removal so the
        caller can decide whether the namespace should also be removed.
        """
        raise NotImplementedError


class VectorStoreTransaction:
    """
    Transaction wrapper for vector store operations.

    Note: This is a best-effort abstraction, not a true ACID transaction.
    It keeps enough state to undo simple upserts/deletes on failure.
    """

    def __init__(self, store: VectorStoreBackend) -> None:
        self._store = store
        self._operations: List[Dict[str, Any]] = []
        self._committed = False

    async def upsert(
        self,
        vectors: List[Dict[str, Any]],
        namespace: str = "default",
        collection: Optional[str] = None,
    ) -> int:
        count = await self._store.upsert(vectors, namespace, collection)
        self._operations.append(
            {
                "type": "upsert",
                "ids": [v["id"] for v in vectors],
                "namespace": namespace,
                "collection": collection,
            }
        )
        return count

    async def delete(
        self,
        ids: List[str],
        namespace: str = "default",
        collection: Optional[str] = None,
    ) -> int:
        # Save current state for rollback (using batch method for efficiency)
        saved_vectors_raw = await self._store.get_by_ids(ids, collection, namespace=namespace)
        saved_vectors: List[Dict[str, Any]] = []
        for vec in saved_vectors_raw:
            saved_vectors.append(
                {
                    "id": vec.id,
                    "embedding": vec.embedding,
                    "metadata": vec.metadata,
                }
            )

        count = await self._store.delete(ids, namespace, collection)
        self._operations.append(
            {
                "type": "delete",
                "saved_vectors": saved_vectors,
                "namespace": namespace,
                "collection": collection,
            }
        )
        return count

    async def commit(self) -> None:
        self._committed = True
        self._operations.clear()

    async def rollback(self) -> None:
        if self._committed:
            return

        # Undo in reverse order
        for op in reversed(self._operations):
            if op["type"] == "upsert":
                await self._store.delete(
                    op["ids"],
                    op["namespace"],
                    op["collection"],
                )
            elif op["type"] == "delete":
                if op["saved_vectors"]:
                    await self._store.upsert(
                        op["saved_vectors"],
                        op["namespace"],
                        op["collection"],
                    )

        self._operations.clear()




class GraphStoreBackend(ABC):
    """
    Graph store interface with PPR methods.

    Implementations: Neo4j, NetworkX, TigerGraph, etc.
    """

    async def create_node(
        self,
        node: GraphNode,
        namespace: str,
    ) -> str:
        """Create a node. Returns node ID."""
        ...

    async def create_edge(
        self,
        edge: GraphEdge,
        namespace: str,
    ) -> str:
        """Create an edge. Returns edge ID."""
        ...

    async def create_nodes_batch(
        self,
        nodes: List[GraphNode],
        namespace: str,
    ) -> List[str]:
        """
        Batch create multiple nodes.
        
        FIX for batch operations: Avoids N+1 queries during ingestion.
        Returns list of node IDs in the same order as input nodes.
        """
        ...

    async def create_edges_batch(
        self,
        edges: List[GraphEdge],
        namespace: str,
    ) -> List[str]:
        """
        Batch create multiple edges.
        
        FIX for batch operations: Avoids N+1 queries during ingestion.
        Returns list of edge IDs in the same order as input edges.
        """
        ...

    async def get_node(
        self,
        node_id: str,
        namespace: str,
    ) -> Optional[GraphNode]:
        """Get node by ID."""
        ...

    async def get_nodes_batch(
        self,
        node_ids: List[str],
        namespace: str,
    ) -> List[GraphNode]:
        """
        Batch get multiple nodes by IDs.
        
        FIX for batch operations: Avoids N+1 queries for PPR expansion.
        Returns list of GraphNode (may be shorter than input if some IDs don't exist).
        """
        ...

    async def get_neighbors(
        self,
        node_id: str,
        namespace: str,
        direction: str = "both",  # "in", "out", "both"
        edge_types: Optional[List[str]] = None,
    ) -> List[GraphNode]:
        """Get neighboring nodes."""
        ...

    async def query_nodes(
        self,
        filters: Dict[str, Any],
        namespace: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query nodes by filter."""
        ...

    async def delete_node(
        self,
        node_id: str,
        namespace: str,
    ) -> bool:
        """Delete node. Returns True if deleted."""
        ...

    async def delete_edges(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> int:
        """Delete edges matching criteria. Returns count deleted."""
        ...

    async def personalized_pagerank(
        self,
        seed_nodes: List[str],
        namespace: str,
        damping: float = 0.85,
        max_iterations: int = 100,
        top_k: int = 20,
    ) -> List[Tuple[str, float]]:
        """
        Run Personalized PageRank from seed nodes.

        Returns list of (node_id, ppr_score) tuples, sorted by score descending.
        """
        ...

    async def get_subgraph(
        self,
        node_ids: List[str],
        namespace: str,
        max_hops: int = 2,
    ) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """
        Get subgraph around specified nodes.

        Returns:
            (nodes, edges) in the subgraph
        """
        ...

    async def add_namespace(
        self,
        id: str,
        namespace: str,
        document_id: Optional[str] = None,
    ) -> bool:
        """Add a namespace to an existing node or edge (convenience: tries node, then edge)."""
        ...

    async def add_namespace_to_node(
        self,
        node_id: str,
        namespace: str,
        document_id: Optional[str] = None,
    ) -> bool:
        """Add a namespace to an existing node."""
        ...

    async def add_namespace_to_edge(
        self,
        edge_id: str,
        namespace: str,
        document_id: Optional[str] = None,
    ) -> bool:
        """Add a namespace to an existing edge."""
        ...

    async def remove_namespace(
        self,
        id: str,
        namespace: str,
    ) -> Tuple[bool, bool]:
        """Remove a namespace from a node or edge (convenience: tries node, then edge).

        Returns (success, was_last_namespace).
        """
        ...

    async def remove_namespace_from_node(
        self,
        node_id: str,
        namespace: str,
    ) -> Tuple[bool, bool]:
        """Remove a namespace from a node.

        Returns (success, was_last_namespace).
        If the node has no remaining namespaces it is deleted.
        """
        ...

    async def remove_namespace_from_edge(
        self,
        edge_id: str,
        namespace: str,
    ) -> Tuple[bool, bool]:
        """Remove a namespace from an edge.

        Returns (success, was_last_namespace).
        If the edge has no remaining namespaces it is deleted.
        """
        ...

    async def remove_document_reference(
        self,
        id: str,
        document_id: str,
    ) -> List[str]:
        """Remove *document_id* from ``source_doc_ids`` /
        ``source_locations`` on a node or edge without touching
        namespaces.

        Returns the remaining ``source_doc_ids``.
        """
        ...

    async def close(self) -> None:
        """Clean up backend resources."""
        ...
