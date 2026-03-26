"""
In-memory implementation of VectorStoreBackend.
Useful for testing and local development.
"""

from __future__ import annotations

import asyncio
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from unified_memory.storage.base import VectorStoreBackend
from unified_memory.core.types import VectorSearchResult


class MemoryVectorStore(VectorStoreBackend):
    """
    In-memory Vector store implementation.
    
    Uses exact cosine similarity search (brute force) with numpy.
    Collections are simulated as separate dictionaries.
    
    NAMESPACE ARRAY MIGRATION:
    - Vectors now store `namespaces: List[str]` instead of `namespace: str`.
    - Backward compat: reads both `namespaces` (new) and `namespace` (old).
    - Search matches if query namespace is IN the namespaces list.
    """

    def __init__(self) -> None:
        # Structure: collections[collection_name][vector_id] = vector_dict
        self._collections: Dict[str, Dict[str, Dict[str, Any]]] = {}
        
        # Metadata configuration per collection
        self._collection_config: Dict[str, Dict[str, Any]] = {}
        
        self._lock = asyncio.Lock()

    def _get_namespaces(self, vec: Dict[str, Any]) -> List[str]:
        """Extract namespaces list from a stored vector (backward compat)."""
        if "namespaces" in vec:
            return vec["namespaces"]
        if "namespace" in vec:
            return [vec["namespace"]]
        return []

    async def create_collection(
        self,
        name: str,
        dimension: int,
        distance_metric: str = "cosine",
        index_type: str = "hnsw",
        **kwargs,
    ) -> bool:
        """Create a new collection (in-memory; backend-specific kwargs are ignored)."""
        async with self._lock:
            if name in self._collections:
                return False
            
            self._collections[name] = {}
            self._collection_config[name] = {
                "dimension": dimension,
                "distance_metric": distance_metric,
            }
            return True

    async def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        async with self._lock:
            if name not in self._collections:
                return False
            
            del self._collections[name]
            del self._collection_config[name]
            return True

    async def list_collections(self, prefix: Optional[str] = None) -> List[str]:
        """List collections."""
        async with self._lock:
            collections = list(self._collections.keys())
            if prefix:
                return [c for c in collections if c.startswith(prefix)]
            return collections

    async def upsert(
        self,
        vectors: List[Dict[str, Any]],
        namespace: str = "default",
        collection: Optional[str] = None,
    ) -> int:
        """
        Upsert vectors. Stores namespace as namespaces list.

        NOTE: `collection` is required; using `namespace` as a fallback collection
        name is no longer supported to avoid ambiguity between logical namespaces
        and physical collections.
        """
        if collection is None:
            raise ValueError("collection must be specified for MemoryVectorStore.upsert")

        target_collection = collection
        
        async with self._lock:
            if target_collection not in self._collections:
                # Auto-create for convenience in testing
                self._collections[target_collection] = {}
            
            store = self._collections[target_collection]
            count = 0
            
            for vec in vectors:
                vec_id = vec["id"]
                
                # If vector already exists, merge namespaces
                existing = store.get(vec_id)
                if existing:
                    existing_ns = self._get_namespaces(existing)
                    if namespace not in existing_ns:
                        existing_ns.append(namespace)
                    new_namespaces = existing_ns
                else:
                    new_namespaces = [namespace]
                
                # Deep copy to prevent mutation
                stored_vec = {
                    "id": vec_id,
                    "embedding": vec.get("embedding"),  # Can be None for metadata-only updates
                    "metadata": vec.get("metadata", {}).copy(),
                    "namespaces": new_namespaces,
                }
                
                # If updating existing, preserve embedding if new one is None
                if existing and stored_vec["embedding"] is None:
                    stored_vec["embedding"] = existing["embedding"]

                # Extract and ensure source_doc_ids / source_locations exist in payload
                metadata = stored_vec["metadata"]
                doc_ids = set(metadata.get("source_doc_ids") or [])
                if metadata.get("document_id"):
                    doc_ids.add(metadata["document_id"])
                
                # Update existing doc_ids if merging
                if existing:
                    payload = existing.get("metadata", {})
                    doc_ids.update(payload.get("source_doc_ids") or [])
                    if payload.get("document_id"):
                        doc_ids.add(payload["document_id"])
                        
                metadata["source_doc_ids"] = list(doc_ids)
                
                # Mirror namespaces to metadata for visibility
                metadata["namespaces"] = new_namespaces
                
                store[vec_id] = stored_vec
                count += 1
                
            return count

    async def add_namespace(
        self,
        id: str,
        namespace: str,
        collection: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> bool:
        """Add a namespace to an existing vector's namespace list."""
        target_collections = [collection] if collection else list(self._collections.keys())
        async with self._lock:
            for col_name in target_collections:
                if col_name in self._collections:
                    store = self._collections[col_name]
                    if id in store:
                        ns_list = self._get_namespaces(store[id])
                        if namespace not in ns_list:
                            ns_list.append(namespace)
                            store[id]["namespaces"] = ns_list
                            # Sync to metadata
                            store[id]["metadata"]["namespaces"] = ns_list
                            
                        # Handle document_id merging
                        if document_id:
                            metadata = store[id]["metadata"]
                            doc_ids = set(metadata.get("source_doc_ids") or [])
                            doc_ids.add(document_id)
                            metadata["source_doc_ids"] = list(doc_ids)
                            
                        return True
            return False

    async def remove_namespace(
        self,
        id: str,
        namespace: str,
        collection: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> Tuple[bool, bool]:
        """Remove a namespace from a vector's namespace list.

        Returns ``(success, was_last_namespace)``.  When
        ``was_last_namespace`` is ``True`` the vector was hard-deleted.

        This method **unconditionally** removes the namespace.  Callers
        that need shared-document awareness should call
        ``remove_document_reference`` first and only invoke this method
        when the namespace is no longer needed.
        """
        target_collections = [collection] if collection else list(self._collections.keys())
        async with self._lock:
            for col_name in target_collections:
                if col_name in self._collections:
                    store = self._collections[col_name]
                    if id in store:
                        ns_list = self._get_namespaces(store[id])
                        if namespace in ns_list:
                            ns_list.remove(namespace)

                            if not ns_list:
                                del store[id]
                                return True, True

                            store[id]["namespaces"] = ns_list
                            store[id]["metadata"]["namespaces"] = ns_list
                            return True, False
            return False, False

    async def remove_document_reference(
        self,
        id: str,
        document_id: str,
        collection: Optional[str] = None,
    ) -> List[str]:
        """Remove *document_id* from ``source_doc_ids`` / ``source_locations``
        without touching namespaces.

        Returns the remaining ``source_doc_ids`` so the caller can decide
        whether the namespace should also be removed.
        """
        target_collections = [collection] if collection else list(self._collections.keys())
        async with self._lock:
            for col_name in target_collections:
                if col_name in self._collections:
                    store = self._collections[col_name]
                    if id in store:
                        payload = store[id]["metadata"]

                        doc_ids = list(payload.get("source_doc_ids") or [])
                        if document_id in doc_ids:
                            doc_ids.remove(document_id)
                            payload["source_doc_ids"] = doc_ids

                        locations = list(payload.get("source_locations") or [])
                        locations = [
                            loc for loc in locations
                            if loc.get("document_id") != document_id
                        ]
                        payload["source_locations"] = locations

                        return doc_ids
            return []

    async def get_by_id(
        self,
        id: str,
        collection: Optional[str] = None,
        namespace: str = "default",
    ) -> Optional[VectorSearchResult]:
        """Get vector by ID. Returns None if namespace not in vector's namespaces."""
        target_collections = [collection] if collection else list(self._collections.keys())
        
        async with self._lock:
            for col_name in target_collections:
                if col_name not in self._collections:
                    continue
                    
                store = self._collections[col_name]
                if id in store:
                    vec = store[id]
                    vec_ns = self._get_namespaces(vec)
                    if namespace not in vec_ns:
                        return None
                    return VectorSearchResult(
                        id=vec["id"],
                        score=1.0, # exact match
                        embedding=vec["embedding"],
                        metadata=vec["metadata"],
                        collection=col_name
                    )
            return None

    async def get_by_ids(
        self,
        ids: List[str],
        collection: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        """Batch get vectors by IDs. If namespace given, filter by namespace in namespaces."""
        target_collections = [collection] if collection else list(self._collections.keys())
        results = []
        
        async with self._lock:
            for vec_id in ids:
                for col_name in target_collections:
                    if col_name not in self._collections:
                        continue
                    store = self._collections[col_name]
                    if vec_id in store:
                        vec = store[vec_id]
                        if namespace is not None and namespace not in self._get_namespaces(vec):
                            break
                        results.append(VectorSearchResult(
                            id=vec["id"],
                            score=1.0,
                            embedding=vec["embedding"],
                            metadata=vec["metadata"],
                            collection=col_name
                        ))
                        break
        return results

    async def delete(
        self,
        ids: List[str],
        namespace: str = "default",
        collection: Optional[str] = None,
    ) -> int:
        """
        Delete vectors by ID for a specific namespace.

        Implements ref-count semantics:
        - Removes the given namespace from each vector's `namespaces` list.
        - Hard-deletes the vector only when no namespaces remain.
        """
        if collection is None:
            raise ValueError("collection must be specified for MemoryVectorStore.delete")

        target_collection = collection

        async with self._lock:
            if target_collection not in self._collections:
                return 0

            store = self._collections[target_collection]
            count = 0
            for id in ids:
                if id not in store:
                    continue
                ns_list = self._get_namespaces(store[id])
                if namespace in ns_list:
                    ns_list.remove(namespace)
                    if not ns_list:
                        # Last namespace -> hard delete
                        del store[id]
                    else:
                        store[id]["namespaces"] = ns_list
                        store[id]["metadata"]["namespaces"] = ns_list
                    count += 1
            return count

    async def delete_by_filter(
        self,
        filters: Dict[str, Any],
        namespace: str = "default",
        collection: Optional[str] = None,
    ) -> int:
        """
        Delete vectors matching filter for a specific namespace.

        Ref-count semantics:
        - Remove the namespace from matched vectors' `namespaces` list.
        - Hard-delete only when no namespaces remain.
        """
        if collection is None:
            raise ValueError("collection must be specified for MemoryVectorStore.delete_by_filter")

        target_collection = collection
        
        async with self._lock:
            if target_collection not in self._collections:
                return 0

            store = self._collections[target_collection]
            affected = 0

            for vec_id, vec in list(store.items()):
                vec_namespaces = self._get_namespaces(vec)
                if namespace not in vec_namespaces:
                    continue

                if not self._matches_filter(vec["metadata"], filters):
                    continue

                vec_namespaces.remove(namespace)
                if not vec_namespaces:
                    del store[vec_id]
                else:
                    vec["namespaces"] = vec_namespaces
                    vec["metadata"]["namespaces"] = vec_namespaces
                affected += 1

            return affected

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
        target_collection = collection or namespace
        
        async with self._lock:
            if target_collection not in self._collections:
                return []
            
            store = self._collections[target_collection]
            candidates = []
            
            query_vec = np.array(query_embedding)
            norm_query = np.linalg.norm(query_vec)
            
            if norm_query == 0:
                return []
                
            for vec in store.values():
                # 1. Namespace array check
                vec_namespaces = self._get_namespaces(vec)
                if namespace not in vec_namespaces:
                    continue
                    
                if filters:
                    matches = self._matches_filter(vec["metadata"], filters)
                    if not matches:
                        continue
                
                # 2. Embedding Check
                doc_embedding = vec.get("embedding")
                if doc_embedding is None:
                    continue
                    
                doc_vec = np.array(doc_embedding)
                norm_doc = np.linalg.norm(doc_vec)
                
                if norm_doc == 0:
                    score = 0.0
                else:
                    score = float(np.dot(query_vec, doc_vec) / (norm_query * norm_doc))
                
                if score_threshold and score < score_threshold:
                    continue
                    
                candidates.append((score, vec))
            
            # Sort by score descending
            candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Take top_k
            results = []
            for score, vec in candidates[:top_k]:
                results.append(VectorSearchResult(
                    id=vec["id"],
                    score=score,
                    embedding=vec["embedding"], 
                    metadata=vec["metadata"],
                    collection=target_collection
                ))
                
            return results

    async def query_by_filter(
        self,
        filters: Dict[str, Any],
        collection: str,
        namespace: str = "default",
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Query vectors by metadata filter ONLY."""
        async with self._lock:
            if collection not in self._collections:
                return []
                
            store = self._collections[collection]
            results = []
            
            for vec in store.values():
                # Namespace array check
                vec_namespaces = self._get_namespaces(vec)
                if namespace not in vec_namespaces:
                    continue
                    
                if self._matches_filter(vec["metadata"], filters):
                    results.append({
                        "id": vec["id"],
                        "metadata": vec["metadata"],
                        "namespaces": vec_namespaces,
                    })
                    
                    if len(results) >= limit:
                        break
                        
            return results

    def _matches_filter(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Recursive metadata filter matching."""
        for key, value in filters.items():
            # Handle $ operators if needed, keeping it simple for now: exact match
            # TODO: Add $in, $gt, etc. support if needed for complex tests
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True
