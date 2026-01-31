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
    """

    def __init__(self) -> None:
        # Structure: collections[collection_name][vector_id] = vector_dict
        self._collections: Dict[str, Dict[str, Dict[str, Any]]] = {}
        
        # Metadata configuration per collection
        self._collection_config: Dict[str, Dict[str, Any]] = {}
        
        self._lock = asyncio.Lock()

    async def create_collection(
        self,
        name: str,
        dimension: int,
        distance_metric: str = "cosine",
        index_type: str = "hnsw",
    ) -> bool:
        """Create a new collection."""
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
        """Upsert vectors."""
        target_collection = collection or namespace
        
        async with self._lock:
            if target_collection not in self._collections:
                # Auto-create for convenience in testing
                self._collections[target_collection] = {}
            
            store = self._collections[target_collection]
            count = 0
            
            for vec in vectors:
                vec_id = vec["id"]
                # Deep copy to prevent mutation
                stored_vec = {
                    "id": vec_id,
                    "embedding": vec.get("embedding"), # Can be None for metadata-only updates
                    "metadata": vec.get("metadata", {}).copy(),
                    "namespace": namespace # Store namespace in metadata for filtering
                }
                
                # If updating existing, preserve embedding if new one is None
                if vec_id in store and stored_vec["embedding"] is None:
                     stored_vec["embedding"] = store[vec_id]["embedding"]
                
                store[vec_id] = stored_vec
                count += 1
                
            return count

    async def get_by_id(
        self,
        id: str,
        collection: Optional[str] = None,
    ) -> Optional[VectorSearchResult]:
        """Get vector by ID."""
        # Note: In real systems, we might search across collections if not specified,
        # but here we require strict collection targeting or default to namespace behavior
        # For simplicity in this mock, we search all if collection is None (inefficient)
        
        target_collections = [collection] if collection else list(self._collections.keys())
        
        async with self._lock:
            for col_name in target_collections:
                if col_name not in self._collections:
                    continue
                    
                store = self._collections[col_name]
                if id in store:
                    vec = store[id]
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
    ) -> List[VectorSearchResult]:
        """Batch get vectors by IDs."""
        target_collections = [collection] if collection else list(self._collections.keys())
        results = []
        
        async with self._lock:
            for id in ids:
                found = False
                for col_name in target_collections:
                    if col_name not in self._collections:
                        continue
                        
                    store = self._collections[col_name]
                    if id in store:
                        vec = store[id]
                        results.append(VectorSearchResult(
                            id=vec["id"],
                            score=1.0,
                            embedding=vec["embedding"],
                            metadata=vec["metadata"],
                            collection=col_name
                        ))
                        found = True
                        break # Stop checking other collections for this ID
        return results

    async def delete(
        self,
        ids: List[str],
        namespace: str = "default",
        collection: Optional[str] = None,
    ) -> int:
        """Delete vectors by ID."""
        target_collection = collection or namespace
        
        async with self._lock:
            if target_collection not in self._collections:
                return 0
            
            store = self._collections[target_collection]
            count = 0
            for id in ids:
                if id in store:
                    # Verify namespace matches if we are being strict
                    # But typically delete by ID is absolute
                    del store[id]
                    count += 1
            return count

    async def delete_by_filter(
        self,
        filters: Dict[str, Any],
        namespace: str = "default",
        collection: Optional[str] = None,
    ) -> int:
        """Delete vectors matching filter."""
        target_collection = collection or namespace
        
        async with self._lock:
            if target_collection not in self._collections:
                return 0
                
            store = self._collections[target_collection]
            to_delete = []
            
            for vec_id, vec in store.items():
                # Implicit namespace filter
                if vec.get("namespace") != namespace:
                    continue
                    
                if self._matches_filter(vec["metadata"], filters):
                    to_delete.append(vec_id)
            
            for vec_id in to_delete:
                del store[vec_id]
                
            return len(to_delete)

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
                # 1. Filter Check
                # Namespace check (stored in metadata or explicit field)
                if vec.get("namespace") != namespace:
                    continue
                    
                if filters and not self._matches_filter(vec["metadata"], filters):
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
                if vec.get("namespace") != namespace:
                    continue
                    
                if self._matches_filter(vec["metadata"], filters):
                    results.append({
                        "id": vec["id"],
                        "metadata": vec["metadata"],
                        "namespace": vec["namespace"]
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
