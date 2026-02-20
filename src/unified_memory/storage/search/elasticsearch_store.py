"""
ElasticSearch sparse retrieval backend (BM25).

This backend is intended to be a **search index**:
- It stores enough text to compute BM25 scores (typically chunk text).
- Canonical content still lives in CAS/ContentStore keyed by ``content_id``.
- Namespace visibility is enforced via a ``namespaces`` keyword array.

Requires the ``elasticsearch`` package (async client).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from unified_memory.core.types import RetrievalResult

logger = logging.getLogger(__name__)


class ElasticSearchStore:
    """
    ElasticSearch backend that serves as a sparse retriever (BM25).

    This class is designed to satisfy the ``SparseRetriever`` protocol defined
    in ``unified_memory.core.interfaces`` (structural typing). It can therefore
    be passed anywhere a ``SparseRetriever`` is expected.

    Each document is stored with fields:
    - ``content_id``: the content hash used as the unique key
    - ``content``: text indexed for BM25 scoring (copy; not canonical)
    - ``namespaces``: list of namespaces this content belongs to
    - ``metadata``: filterable attributes (stored as `flattened`)
    """

    _INDEX_SETTINGS = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "mappings": {
            "properties": {
                "content_id": {"type": "keyword"},
                "content": {"type": "text"},
                "namespaces": {"type": "keyword"},
                "source_doc_ids": {"type": "keyword"},
                "metadata": {"type": "flattened"},
            }
        },
    }

    def __init__(
        self,
        url: str = "http://localhost:9200",
        index_name: str = "unified_memory_content",
        api_key: Optional[str] = None,
        basic_auth: Optional[tuple] = None,
    ) -> None:
        try:
            from elasticsearch import AsyncElasticsearch
        except ImportError as exc:
            raise ImportError(
                "The 'elasticsearch' package is required for ElasticSearchStore. "
                "Install it with: pip install elasticsearch"
            ) from exc

        kwargs: Dict[str, Any] = {"hosts": [url]}
        if api_key:
            kwargs["api_key"] = api_key
        if basic_auth:
            kwargs["basic_auth"] = basic_auth
        self._es = AsyncElasticsearch(**kwargs)
        self._index = index_name

    async def ensure_index(self) -> None:
        """Create the index if it does not exist."""
        exists = await self._es.indices.exists(index=self._index)
        if not exists:
            # elasticsearch-py 8.x: pass settings/mappings as top-level kwargs, not body=
            await self._es.indices.create(
                index=self._index,
                **self._INDEX_SETTINGS,
            )

    # ------------------------------------------------------------------
    # SparseRetriever-compatible API
    # ------------------------------------------------------------------

    async def add_namespace(self, content_id: str, namespace: str) -> bool:
        """Add a namespace to an existing indexed document."""
        try:
            await self._es.update(
                index=self._index,
                id=content_id,
                body={
                    "script": {
                        "source": (
                            "if (ctx._source.namespaces == null) { ctx._source.namespaces = [] } "
                            "if (!ctx._source.namespaces.contains(params.ns)) { ctx._source.namespaces.add(params.ns) }"
                        ),
                        "params": {"ns": namespace},
                    }
                },
            )
            return True
        except Exception:
            return False

    async def add_namespace_batch(self, content_ids: List[str], namespace: str) -> int:
        """Add a namespace to many documents (best-effort)."""
        updated = 0
        for cid in content_ids:
            if await self.add_namespace(cid, namespace):
                updated += 1
        return updated

    async def index(
        self,
        documents: List[Dict[str, Any]],
        namespace: str,
    ) -> int:
        """
        Bulk-index documents for sparse retrieval with merging.
        """
        # We use 'update' with 'upsert' to merge namespaces and source_doc_ids
        from elasticsearch.helpers import async_bulk
 
        actions = []
        for doc in documents:
            doc_id = doc["id"]
            metadata = doc.get("metadata", {})
            
            # Extract document_id from metadata or use provided source_doc_ids
            doc_ids = set()
            if "document_id" in metadata:
                doc_ids.add(metadata.get("document_id"))
            if "source_doc_ids" in metadata:
                sd_val = metadata["source_doc_ids"]
                doc_ids.update(sd_val if isinstance(sd_val, list) else [sd_val])
            
            doc_ids_list = list(doc_ids)
 
            actions.append({
                "_op_type": "update",
                "_index": self._index,
                "_id": doc_id,
                "script": {
                   "source": """
                      if (ctx._source.namespaces == null) { ctx._source.namespaces = [] }
                      if (!ctx._source.namespaces.contains(params.ns)) { ctx._source.namespaces.add(params.ns) }
                      if (ctx._source.source_doc_ids == null) { ctx._source.source_doc_ids = [] }
                      for (id in params.doc_ids) {
                        if (!ctx._source.source_doc_ids.contains(id)) {
                           ctx._source.source_doc_ids.add(id)
                        }
                      }
                   """,
                   "params": {
                       "ns": namespace,
                       "doc_ids": doc_ids_list
                   }
                },
                "upsert": {
                    "content_id": doc_id,
                    "content": doc["content"],
                    "namespaces": [namespace],
                    "source_doc_ids": doc_ids_list,
                    "metadata": metadata,
                }
            })
            
        if not actions:
            return 0
            
        success, _ = await async_bulk(
            client=self._es,
            actions=actions,
            raise_on_error=False,
        )
        return success

    async def retrieve(
        self,
        query: str,
        namespaces: List[str],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """BM25-based retrieval over stored content."""
        must = [{"match": {"content": query}}]
        filter_clauses = [{"terms": {"namespaces": namespaces}}]

        if filters:
            for k, v in filters.items():
                if k in ("namespace", "namespaces"):
                    continue
                
                # Handle source_doc_id
                if k in ("document_id", "source_doc_id", "source_doc_ids"):
                    val_list = v if isinstance(v, list) else [v]
                    filter_clauses.append({"terms": {"source_doc_ids": val_list}})
                else:
                    # metadata is stored as `flattened`, so dot-access is supported
                    filter_clauses.append({"term": {f"metadata.{k}": v}})

        body = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": must,
                    "filter": filter_clauses,
                }
            },
        }

        response = await self._es.search(index=self._index, body=body)
        results: List[RetrievalResult] = []
        for hit in response["hits"]["hits"]:
            src = hit["_source"]
            results.append(
                RetrievalResult(
                    id=hit["_id"],
                    score=hit["_score"],
                    content=src.get("content", ""),
                    metadata=src.get("metadata", {}),
                    source="sparse",
                )
            )
        return results

    async def delete(
        self,
        doc_ids: List[str],
        namespace: str,
        document_id: Optional[str] = None,
    ) -> int:
        """
        Namespace-aware delete with source_doc_ids cleanup.
        """
        affected = 0
        for content_id in doc_ids:
            try:
                doc = await self._es.get(index=self._index, id=content_id)
                src = doc.get("_source", {}) or {}
                
                namespaces = list(src.get("namespaces") or [])
                if namespace in namespaces:
                    namespaces.remove(namespace)
                
                source_doc_ids = list(src.get("source_doc_ids") or [])
                if document_id and document_id in source_doc_ids:
                    source_doc_ids.remove(document_id)
                
                if not namespaces:
                    await self._es.delete(index=self._index, id=content_id)
                else:
                    await self._es.update(
                        index=self._index,
                        id=content_id,
                        body={
                            "doc": {
                                "namespaces": namespaces,
                                "source_doc_ids": source_doc_ids
                            }
                        },
                    )
                affected += 1
            except Exception:
                continue
        return affected

    async def close(self) -> None:
        await self._es.close()
