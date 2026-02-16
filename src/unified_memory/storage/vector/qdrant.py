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

# Qdrant accepts point IDs as UUID or uint64. Pipeline uses strings like "text:{content_hash}".
# Convert non-UUID strings to a deterministic UUID so Qdrant accepts them; store original in payload.
_NAMESPACE_QDRANT_ID = uuid.UUID("7c9e6669-7375-4b6e-9460-4b4a6f2a1c3d")


def _to_qdrant_id(s: str):
    """Convert system ID to Qdrant-acceptable UUID. Preserves valid UUIDs; hashes others."""
    try:
        u = uuid.UUID(s)
        return u
    except (ValueError, TypeError, AttributeError):
        return uuid.uuid5(_NAMESPACE_QDRANT_ID, s)


def _from_point_id(point_id, payload: Optional[Dict[str, Any]]) -> str:
    """Return the system ID for a point (original id from payload if stored)."""
    if payload and "_sid" in payload:
        return payload["_sid"]
    return str(point_id)


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
        index_type: str = "hnsw",
        hnsw_m: Optional[int] = None,
        hnsw_ef_construct: Optional[int] = None,
        ivf_sq_quantile: float = 0.99,
    ) -> bool:
        """Create a Qdrant collection.

        Parameters
        ----------
        index_type:
            ``"hnsw"`` (default), ``"flat"`` (brute-force exact search), or
            ``"ivf_sq"`` (scalar quantisation — smaller memory footprint, fast
            queries at large scale, minor accuracy trade-off).
        ivf_sq_quantile:
            Clipping percentile for scalar quantisation (``index_type="ivf_sq"``
            only).  Passed directly to Qdrant's ``ScalarQuantizationConfig``.
        """
        metric_map = {
            "cosine": models.Distance.COSINE,
            "euclidean": models.Distance.EUCLID,
            "dot": models.Distance.DOT,
        }

        hnsw_config = None
        optimizers_config = None
        quantization_config = None

        if index_type == "flat":
            # Disable HNSW graph — Qdrant performs exact brute-force search.
            hnsw_config = models.HnswConfigDiff(m=0)
            optimizers_config = models.OptimizersConfigDiff(indexing_threshold=0)
        elif index_type == "ivf_sq":
            # Scalar quantisation: compress vectors to int8 to reduce memory.
            # Qdrant uses HNSW on top of quantised vectors by default.
            quantization_config = models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=ivf_sq_quantile,
                    always_ram=True,
                )
            )
        elif hnsw_m is not None or hnsw_ef_construct is not None:
            hnsw_config = models.HnswConfigDiff(
                m=hnsw_m or 16,
                ef_construct=hnsw_ef_construct or 100,
            )

        try:
            safe_name = self._sanitize_collection_name(name)
            await self.client.create_collection(
                collection_name=safe_name,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=metric_map.get(distance_metric, models.Distance.COSINE),
                ),
                hnsw_config=hnsw_config,
                optimizers_config=optimizers_config,
                quantization_config=quantization_config,
            )
            return True
        except Exception as e:
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
        Upsert vectors with deduplication and metadata merging.
        
        vectors: list of dicts with keys: id, embedding, metadata
        """
        if not collection:
            collection = namespace # Default strategy if no collection mapping
            
        collection = self._sanitize_collection_name(collection)
        if not vectors:
            return 0

        # 1. Fetch existing points to merge metadata (deduplication support)
        qids = [_to_qdrant_id(v["id"]) for v in vectors]
        try:
            existing_points = await self.client.retrieve(
                collection_name=collection,
                ids=qids,
                with_payload=True,
                with_vectors=False
            )
            existing_map = {str(p.id): p.payload for p in existing_points}
        except Exception:
            # Collection might not exist yet
            existing_map = {}
            
        points: List[models.PointStruct] = []
        for v in vectors:
            sid = v["id"]
            qid = _to_qdrant_id(sid)
            qid_str = str(qid)

            # Metadata merging logic
            new_metadata = v.get("metadata", {}).copy()
            existing_payload = existing_map.get(qid_str, {}) or {}
            
            # Merge namespaces
            namespaces = set(existing_payload.get("namespaces") or [])
            namespaces.add(namespace)
            if "namespaces" in new_metadata:
                ns_val = new_metadata["namespaces"]
                namespaces.update(ns_val if isinstance(ns_val, list) else [ns_val])
            new_metadata["namespaces"] = list(namespaces)

            # Merge source_doc_ids
            doc_ids = set(existing_payload.get("source_doc_ids") or [])
            if "document_id" in new_metadata:
                doc_ids.add(new_metadata.pop("document_id"))
            if "source_doc_ids" in new_metadata:
                sd_val = new_metadata["source_doc_ids"]
                doc_ids.update(sd_val if isinstance(sd_val, list) else [sd_val])
            new_metadata["source_doc_ids"] = list(doc_ids)

            # Merge source_locations (list of dicts)
            locations = existing_payload.get("source_locations") or []
            if not isinstance(locations, list): locations = []
            
            def loc_to_key(loc: Dict[str, Any]) -> str:
                return f"{loc.get('document_id')}:{loc.get('chunk_index')}"

            seen_locs = {loc_to_key(l) for l in locations}
            
            new_locs = new_metadata.get("source_locations") or []
            if not isinstance(new_locs, list): new_locs = [new_locs]
            
            for loc in new_locs:
                if loc_to_key(loc) not in seen_locs:
                    locations.append(loc)
                    seen_locs.add(loc_to_key(loc))
            
            new_metadata["source_locations"] = locations

            if new_metadata.get("_sid") is None:
                new_metadata["_sid"] = sid

            points.append(models.PointStruct(
                id=qid,
                vector=v["embedding"],
                payload=new_metadata,
            ))
            
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
        
        # Always filter by namespace (array-contains on `namespaces`)
        must_conditions.append(
            models.FieldCondition(
                key="namespaces",
                match=models.MatchAny(any=[namespace]),
            )
        )
        
        if filters:
            for k, v in filters.items():
                # Namespace is handled via the namespaces array condition above.
                if k in ("namespace", "namespaces"):
                    continue
                
                # Handle source_doc_id as a MatchAny on source_doc_ids array
                if k in ("document_id", "source_doc_id", "source_doc_ids"):
                    target_key = "source_doc_ids"
                    val_list = v if isinstance(v, list) else [v]
                    must_conditions.append(
                        models.FieldCondition(
                            key=target_key,
                            match=models.MatchAny(any=val_list),
                        )
                    )
                else:
                    must_conditions.append(
                        models.FieldCondition(
                            key=k,
                            match=models.MatchValue(value=v),
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
                id=_from_point_id(r.id, r.payload or {}),
                score=r.score,
                metadata=r.payload or {},
                embedding=r.vector if hasattr(r, 'vector') else None
            )
            for r in results.points
        ]

    async def add_namespace(
        self,
        id: str,
        namespace: str,
        collection: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> bool:
        """
        Add a namespace (and optional document) to an existing point.
        """
        if not collection:
            # Collection is required to disambiguate
            return False
 
        collection = self._sanitize_collection_name(collection)
 
        qid = _to_qdrant_id(id)
        points = await self.client.retrieve(
            collection_name=collection,
            ids=[qid],
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            return False
 
        p = points[0]
        payload = p.payload or {}
        
        # Merge namespaces
        namespaces = set(payload.get("namespaces") or [])
        namespaces.add(namespace)
        
        # Merge source_doc_ids
        doc_ids = set(payload.get("source_doc_ids") or [])
        if document_id:
            doc_ids.add(document_id)
            
        await self.client.set_payload(
            collection_name=collection,
            payload={
                "namespaces": list(namespaces),
                "source_doc_ids": list(doc_ids),
            },
            points=[p.id],
        )
        return True

    async def remove_namespace(
        self,
        id: str,
        namespace: str,
        collection: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> Tuple[bool, bool]:
        """
        Remove a namespace from a point. 
        If document_id is provided, also remove the document reference.
        
        Returns:
            (success, was_last_namespace)
        """
        if not collection:
            # Collection is required to disambiguate
            return False, False
 
        collection = self._sanitize_collection_name(collection)
 
        qid = _to_qdrant_id(id)
        points = await self.client.retrieve(
            collection_name=collection,
            ids=[qid],
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            return False, False
 
        p = points[0]
        payload = p.payload or {}
        
        # 1. Update namespaces
        namespaces = list(payload.get("namespaces") or [])
        if namespace in namespaces:
            namespaces.remove(namespace)
            
        # 2. Update source_doc_ids if requested
        doc_ids = list(payload.get("source_doc_ids") or [])
        if document_id and document_id in doc_ids:
            doc_ids.remove(document_id)
            
        # 3. Update source_locations (remove all for this doc)
        locations = list(payload.get("source_locations") or [])
        if document_id:
            locations = [loc for loc in locations if loc.get("document_id") != document_id]

        if not namespaces:
            # Hard delete if NO namespaces remain
            await self.client.delete(
                collection_name=collection,
                points_selector=models.PointIdsList(points=[p.id]),
            )
            return True, True
 
        # Update payload
        await self.client.set_payload(
            collection_name=collection,
            payload={
                "namespaces": namespaces,
                "source_doc_ids": doc_ids,
                "source_locations": locations,
            },
            points=[p.id],
        )
        return True, False

    async def get_by_id(
        self,
        id: str,
        collection: Optional[str] = None,
        namespace: str = "default",
    ) -> Optional[VectorSearchResult]:
        if not collection:
            collection = namespace
            
        collection = self._sanitize_collection_name(collection)

        qid = _to_qdrant_id(id)
        results = await self.client.retrieve(
            collection_name=collection,
            ids=[qid],
            with_payload=True,
            with_vectors=True,
        )

        if not results:
            return None

        r = results[0]
        payload = r.payload or {}
        namespaces = payload.get("namespaces") or []
        if namespace not in namespaces:
            # ID exists but not visible in this namespace
            return None

        return VectorSearchResult(
            id=_from_point_id(r.id, payload),
            score=1.0,  # Exact match
            metadata=payload,
            embedding=r.vector,
        )

    async def get_by_ids(
        self,
        ids: List[str],
        collection: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        if not collection:
            return []
        collection = self._sanitize_collection_name(collection)

        qids = [_to_qdrant_id(i) for i in ids]
        results = await self.client.retrieve(
            collection_name=collection,
            ids=qids,
            with_payload=True,
            with_vectors=True
        )
        out = []
        for r in results:
            payload = r.payload or {}
            if namespace is not None:
                ns_list = payload.get("namespaces", [])
                if namespace not in ns_list:
                    continue
            out.append(VectorSearchResult(
                id=_from_point_id(r.id, payload),
                score=1.0,
                metadata=payload,
                embedding=r.vector
            ))
        return out

    async def query_by_filter(
        self,
        filters: Dict[str, Any],
        collection: str,
        namespace: str = "default",
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        # This is scroll/match w/o vector
        must_conditions: List[models.FieldCondition] = [
            models.FieldCondition(
                key="namespaces",
                match=models.MatchValue(value=namespace),
            )
        ]
        for k, v in filters.items():
            # Namespace is handled via namespaces array above
            if k in ("namespace", "namespaces"):
                continue
            must_conditions.append(
                models.FieldCondition(
                    key=k,
                    match=models.MatchValue(value=v),
                )
            )
             
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
                "id": _from_point_id(p.id, p.payload or {}),
                "metadata": p.payload or {}
            }
            for p in points
        ]

    async def delete(
        self,
        ids: List[str],
        namespace: str = "default",
        collection: Optional[str] = None,
    ) -> int:
        """
        Namespace-aware delete with ref-count semantics.

        For each id:
        - If the point has the given namespace in its `namespaces` payload field,
          remove that namespace.
        - If that was the last namespace, hard-delete the point.
        """
        if not collection:
            collection = namespace

        collection = self._sanitize_collection_name(collection)

        qids = [_to_qdrant_id(i) for i in ids]
        existing = await self.client.retrieve(
            collection_name=collection,
            ids=qids,
            with_payload=True,
            with_vectors=False,
        )

        to_delete: List[Any] = []
        to_update: List[Tuple[str, List[str]]] = []

        for p in existing:
            payload = p.payload or {}
            namespaces = list(payload.get("namespaces") or [])
            if namespace not in namespaces:
                continue
            namespaces.remove(namespace)
            if not namespaces:
                to_delete.append(p.id)
            else:
                to_update.append((p.id, namespaces))

        # Apply updates
        for pid, namespaces in to_update:
            try:
                await self.client.set_payload(
                    collection_name=collection,
                    payload={"namespaces": namespaces},
                    points=[pid],
                )
            except Exception as e:
                logger.error(f"Failed to update namespaces for point {pid}: {e}")

        # Hard delete orphaned points
        if to_delete:
            await self.client.delete(
                collection_name=collection,
                points_selector=models.PointIdsList(points=to_delete),
            )

        return len(to_update) + len(to_delete)

    async def delete_by_filter(
        self,
        filters: Dict[str, Any],
        namespace: str = "default",
        collection: Optional[str] = None,
    ) -> int:
        if not collection:
            collection = namespace
        
        collection = self._sanitize_collection_name(collection)
            
        must_conditions: List[models.FieldCondition] = [
            models.FieldCondition(
                key="namespaces",
                match=models.MatchAny(any=[namespace]),
            )
        ]
        for k, v in filters.items():
            # Namespace handled via namespaces array above
            if k in ("namespace", "namespaces"):
                continue
            
            if k in ("document_id", "source_doc_id", "source_doc_ids"):
                val_list = v if isinstance(v, list) else [v]
                must_conditions.append(
                    models.FieldCondition(
                        key="source_doc_ids",
                        match=models.MatchAny(any=val_list),
                    )
                )
            else:
                must_conditions.append(
                    models.FieldCondition(
                        key=k,
                        match=models.MatchValue(value=v),
                    )
                )
             
        filter_ = models.Filter(must=must_conditions)
        
        # First scroll to find matching points, then apply namespace-aware deletion.
        scroll_resp = await self.client.scroll(
            collection_name=collection,
            scroll_filter=filter_,
            with_payload=True,
            with_vectors=False,
        )
        points, _ = scroll_resp
        if not points:
            return 0
 
        affected = 0
        document_id = filters.get("document_id") or filters.get("source_doc_id")
        
        for p in points:
            success, _ = await self.remove_namespace(
                id=_from_point_id(p.id, p.payload),
                namespace=namespace,
                collection=collection,
                document_id=document_id
            )
            if success:
                affected += 1
 
        return affected
