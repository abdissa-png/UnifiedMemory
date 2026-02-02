"""
Qdrant Vector Store Backend.

Implements VectorStoreBackend for Qdrant.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
import uuid
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from unified_memory.storage.base import VectorStoreBackend, VectorSearchResult
from unified_memory.core.types import RetrievalResult

logger = logging.getLogger(__name__)

class QdrantVectorStore(VectorStoreBackend):
    """
    Qdrant implementation of VectorStoreBackend.
    """
    
    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        prefer_grpc: bool = False,
    ):
        self.client = AsyncQdrantClient(
            url=url,
            api_key=api_key,
            prefer_grpc=prefer_grpc,
        )

    def _sanitize_collection_name(self, name: str) -> str:
        """
        Sanitize collection name to likely satisfy Qdrant requirements.
        Replaces chars invalid in URLs or paths (like / or :) with _.
        """
        # Qdrant allows alphanumeric, _, -.
        # We replace / and : with _ which are common in our IDs (tenant:x/user:y)
        safe_name = name.replace("/", "_").replace(":", "_").replace(" ", "_")
        return safe_name
        
    async def connect(self) -> None:
        # Client handles connection lazily usually, but we can verify
        # await self.client.get_collections()
        pass
        
    async def disconnect(self) -> None:
        await self.client.close()
        
    async def create_collection(
        self,
        name: str,
        dimension: int,
        distance_metric: str = "cosine",
        index_type: str = "hnsw", # Qdrant uses HNSW by default
    ) -> bool:
        metric_map = {
            "cosine": models.Distance.COSINE,
            "euclidean": models.Distance.EUCLID,
            "dot": models.Distance.DOT,
        }
        
        try:
            safe_name = self._sanitize_collection_name(name)
            await self.client.create_collection(
                collection_name=safe_name,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=metric_map.get(distance_metric, models.Distance.COSINE),
                )
            )
            return True
        except Exception as e:
            # Check if exists error or other
            logger.error(f"Failed to create collection {name}: {e}")
            return False

    async def delete_collection(self, name: str) -> bool:
        try:
            safe_name = self._sanitize_collection_name(name)
            await self.client.delete_collection(safe_name)
            return True
        except Exception:
            return False

    async def list_collections(self, prefix: Optional[str] = None) -> List[str]:
        response = await self.client.get_collections()
        names = [c.name for c in response.collections]
        if prefix:
            names = [n for n in names if n.startswith(prefix)]
        return names

    async def upsert(
        self,
        vectors: List[Dict[str, Any]],
        namespace: str = "default",
        collection: Optional[str] = None,
    ) -> int:
        """
        Upsert vectors.
        vectors: list of dicts with keys: id, embedding, metadata
        """
        if not collection:
            collection = namespace # Default strategy if no collection mapping
            
        collection = self._sanitize_collection_name(collection)
            
        points = []
        for v in vectors:
            # Qdrant IDs can be int or uuid. We assume string from system.
            # If system uses string IDs that are NOT UUIDs, we might need to hash them or ensure they are UUIDs.
            # Qdrant supports string UUIDs.
            
            # Ensure metadata includes namespace for filtering
            payload = v.get("metadata", {}).copy()
            payload["namespace"] = namespace
            
            points.append(models.PointStruct(
                id=v["id"],
                vector=v["embedding"],
                payload=payload
            ))
            
        if not points:
            return 0
            
        await self.client.upsert(
            collection_name=collection,
            points=points
        )
        return len(points)

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        namespace: str = "default",
        collection: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[VectorSearchResult]:
        if not collection:
            collection = namespace
            
        collection = self._sanitize_collection_name(collection)
            
        # Build Qdrant Filter
        must_conditions = []
        
        # Always filter by namespace
        must_conditions.append(
            models.FieldCondition(
                key="namespace",
                match=models.MatchValue(value=namespace)
            )
        )
        
        if filters:
            for k, v in filters.items():
                if k == "namespace": continue # Already added
                must_conditions.append(
                    models.FieldCondition(
                        key=k,
                        match=models.MatchValue(value=v)
                    )
                )
        
        search_filter = models.Filter(must=must_conditions)
        
        results = await self.client.query_points(
            collection_name=collection,
            query=query_embedding,
            query_filter=search_filter,
            limit=top_k,
            score_threshold=score_threshold
        )
        
        return [
            VectorSearchResult(
                id=str(r.id),
                score=r.score,
                metadata=r.payload or {},
                embedding=r.vector if hasattr(r, 'vector') else None
            )
            for r in results.points
        ]

    async def get_by_id(
        self,
        id: str,
        collection: Optional[str] = None,
        namespace: str = "default",
    ) -> Optional[VectorSearchResult]:
        if not collection:
            collection = namespace
            
        collection = self._sanitize_collection_name(collection)
            
        results = await self.client.retrieve(
            collection_name=collection,
            ids=[id],
            with_vectors=True
        )
        
        if not results:
            return None
            
        r = results[0]
        return VectorSearchResult(
            id=str(r.id),
            score=1.0, # Exact match
            metadata=r.payload or {},
            embedding=r.vector
        )

    async def get_by_ids(
        self,
        ids: List[str],
        collection: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        if not collection:
            return [] # Ambiguous
        
        collection = self._sanitize_collection_name(collection)
            
        results = await self.client.retrieve(
            collection_name=collection,
            ids=ids,
            with_vectors=True
        )
        
        return [
            VectorSearchResult(
                id=str(r.id),
                score=1.0,
                metadata=r.payload or {},
                embedding=r.vector
            )
            for r in results
        ]

    async def query_by_filter(
        self,
        filters: Dict[str, Any],
        collection: str,
        namespace: str = "default",
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        # This is scroll/match w/o vector
        must_conditions = [
            models.FieldCondition(key="namespace", match=models.MatchValue(value=namespace))
        ]
        for k, v in filters.items():
             must_conditions.append(models.FieldCondition(key=k, match=models.MatchValue(value=v)))
             
        filter_ = models.Filter(must=must_conditions)
        
        # Scroll
        response = await self.client.scroll(
            collection_name=self._sanitize_collection_name(collection),
            scroll_filter=filter_,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        
        points, _ = response
        
        return [
            {
                "id": str(p.id),
                "metadata": p.payload
            }
            for p in points
        ]

    async def delete(
        self,
        ids: List[str],
        namespace: str = "default",
        collection: Optional[str] = None,
    ) -> int:
        if not collection:
            collection = namespace
            
        collection = self._sanitize_collection_name(collection)
            
        # We must respect namespace!
        # Delete by IDs only looks at IDs, but standard Qdrant IDs are unique per collection.
        # If we share collection, we assume ID collision probability is zero (UUIDs)
        
        await self.client.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(points=ids)
        )
        return len(ids)

    async def delete_by_filter(
        self,
        filters: Dict[str, Any],
        namespace: str = "default",
        collection: Optional[str] = None,
    ) -> int:
        if not collection:
            collection = namespace
        
        collection = self._sanitize_collection_name(collection)
            
        must_conditions = [
            models.FieldCondition(key="namespace", match=models.MatchValue(value=namespace))
        ]
        for k, v in filters.items():
             must_conditions.append(models.FieldCondition(key=k, match=models.MatchValue(value=v)))
             
        filter_ = models.Filter(must=must_conditions)
        
        # Delete API doesn't return count directly easily without wrapper
        # We just fire and forget or check before?
        # Qdrant delete returns UpdateResult which has status.
        
        await self.client.delete(
            collection_name=collection,
            points_selector=models.FilterSelector(filter=filter_)
        )
        return 0 # Count unknown without pre-query
