"""
Sparse Retrieval Implementation using rank-bm25.

In-memory implementation for MVP and testing.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import re
from rank_bm25 import BM25Plus

from unified_memory.core.types import RetrievalResult
from unified_memory.core.interfaces import SparseRetriever

logger = logging.getLogger(__name__)

class BM25SparseRetriever:
    """
    In-memory BM25 sparse retriever.
    
    Maintains a simple in-memory index per namespace.
    Suitable for small-to-medium scale (thousands of docs).
    For production scale, use Elasticsearch/Redis.
    """
    
    def __init__(self):
        # Namespace -> BM25 Index Object
        self._indices: Dict[str, BM25Plus] = {}
        # Namespace -> List of (doc_id, content, metadata) tuples, aligned with index
        self._documents: Dict[str, List[Tuple[str, str, Dict[str, Any]]]] = {}
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text, preserving words and individual symbols (emojis)."""
        # Lowercase and match alphanumeric chunks OR individual non-whitespace characters
        return re.findall(r'\w+|[^\w\s]', text.lower())
    
    async def add_namespace(
        self,
        content_id: str,
        namespace: str,
    ) -> bool:
        """Add a namespace to an existing document."""
        # Find the document in any other namespace
        found_doc = None
        for ns, docs in self._documents.items():
            for doc in docs:
                if doc[0] == content_id:
                    found_doc = doc
                    break
            if found_doc:
                break
        
        if not found_doc:
            return False
            
        # Add to the new namespace
        if namespace not in self._documents:
            self._documents[namespace] = []
            
        # Check if already there
        if any(d[0] == content_id for d in self._documents[namespace]):
            return True
            
        self._documents[namespace].append(found_doc)
        
        # Rebuild index for this namespace
        corpus_tokens = [self._tokenize(d[1]) for d in self._documents[namespace]]
        self._indices[namespace] = BM25Plus(corpus_tokens, delta=0)
        return True

    async def index(
        self,
        documents: List[Dict[str, Any]],
        namespace: str,
    ) -> int:
        """
        Index documents for sparse retrieval.
        
        Args:
            documents: List of dicts with 'id', 'content', 'metadata' keys.
            namespace: Namespace ID to index into.
        """
        if namespace not in self._documents:
            self._documents[namespace] = []
            
        current_docs = self._documents[namespace]
        
        # Process new docs
        count = 0
        for doc in documents:
            doc_id = doc.get("id")
            content = doc.get("content", "")
            metadata = doc.get("metadata", {}).copy()
            
            if not doc_id:
                continue
                
            # Ensure document_id is in source_doc_ids list if present
            doc_ids = set(metadata.get("source_doc_ids") or [])
            if metadata.get("document_id"):
                doc_ids.add(metadata["document_id"])
            metadata["source_doc_ids"] = list(doc_ids)

            # Check if doc_id already exists in this namespace
            existing_idx = next((i for i, d in enumerate(current_docs) if d[0] == doc_id), None)
            if existing_idx is not None:
                # Update existing (merge metadata/content?)
                # For BM25 we usually just replace
                current_docs[existing_idx] = (doc_id, content, metadata)
            else:
                current_docs.append((doc_id, content, metadata))
            count += 1
            
        # Rebuild index
        if current_docs:
            corpus_tokens = [self._tokenize(d[1]) for d in current_docs]
            self._indices[namespace] = BM25Plus(corpus_tokens, delta=0)
            
        return count

    async def retrieve(
        self,
        query: str,
        namespaces: Union[str, List[str]],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Retrieve using BM25 scoring across namespaces."""
        if isinstance(namespaces, str):
            namespaces = [namespaces]
            
        all_results: List[RetrievalResult] = []
        
        for namespace in namespaces:
            if namespace not in self._indices or namespace not in self._documents:
                continue
                
            bm25 = self._indices[namespace]
            docs_list = self._documents[namespace]
            
            # 1. Tokenize Query
            tokenized_query = self._tokenize(query)
            
            # 2. Get Scores
            scores = bm25.get_scores(tokenized_query)
            
            for i, score in enumerate(scores):
                if score <= 0:
                    continue
                    
                doc_id, content, metadata = docs_list[i]
                
                # 3. Apply Filters
                match = True
                if filters:
                    for k, v in filters.items():
                        if k in ("namespace", "namespaces"):
                            continue
                        
                        # Handle source_doc_id
                        if k in ("document_id", "source_doc_id", "source_doc_ids"):
                            target_ids = v if isinstance(v, list) else [v]
                            doc_ids = metadata.get("source_doc_ids") or []
                            if not any(tid in doc_ids for tid in target_ids):
                                match = False
                                break
                        else:
                            if metadata.get(k) != v:
                                match = False
                                break
                
                if match:
                    all_results.append(RetrievalResult(
                        id=doc_id,
                        content=content,
                        score=float(score),
                        metadata=metadata,
                        source="sparse:bm25",
                        evidence_type="text"
                    ))
        
        # Sort and limit
        all_results.sort(key=lambda x: x.score, reverse=True)
        # Unique by ID? In multiple namespaces same content might appear.
        seen = set()
        unique_results = []
        for r in all_results:
            if r.id not in seen:
                unique_results.append(r)
                seen.add(r.id)
                if len(unique_results) >= top_k:
                    break
                    
        return unique_results

    async def delete(
        self,
        doc_ids: List[str],
        namespace: str,
        document_id: Optional[str] = None,
    ) -> int:
        """Delete documents from index (unconditional namespace-level removal).

        When *document_id* is supplied the entry's ``source_doc_ids``
        metadata is trimmed, but the entry is only removed from the
        namespace when no ``source_doc_ids`` remain.  For the
        smart-deletion flow, prefer calling
        ``remove_document_reference`` first and only invoking this
        method when the namespace should truly be removed.
        """
        if namespace not in self._documents:
            return 0

        current_docs = self._documents[namespace]
        to_delete_ids = set(doc_ids)
        new_docs = []
        affected = 0

        for doc_id, content, metadata in current_docs:
            if doc_id in to_delete_ids:
                if document_id:
                    doc_ids_list = list(metadata.get("source_doc_ids") or [])
                    if document_id in doc_ids_list:
                        doc_ids_list.remove(document_id)
                        metadata["source_doc_ids"] = doc_ids_list
                        affected += 1

                    if doc_ids_list:
                        new_docs.append((doc_id, content, metadata))
                    else:
                        affected += 1
                else:
                    affected += 1
            else:
                new_docs.append((doc_id, content, metadata))

        if affected == 0:
            return 0

        self._documents[namespace] = new_docs

        corpus_tokens = [self._tokenize(d[1]) for d in new_docs]
        if corpus_tokens:
            self._indices[namespace] = BM25Plus(corpus_tokens, delta=0)
        else:
            if namespace in self._indices:
                del self._indices[namespace]

        return affected

    async def remove_document_reference(
        self,
        doc_ids: List[str],
        namespace: str,
        document_id: str,
    ) -> Dict[str, List[str]]:
        """Remove *document_id* from ``source_doc_ids`` on matching
        entries **without** removing the entry from the namespace.

        Returns ``{content_hash: remaining_doc_ids}`` for each matched
        entry so the caller can decide whether to hard-delete.
        """
        remaining_map: Dict[str, List[str]] = {}
        if namespace not in self._documents:
            return remaining_map

        for doc_id, content, metadata in self._documents[namespace]:
            if doc_id in doc_ids:
                doc_ids_list = list(metadata.get("source_doc_ids") or [])
                if document_id in doc_ids_list:
                    doc_ids_list.remove(document_id)
                    metadata["source_doc_ids"] = doc_ids_list
                remaining_map[doc_id] = doc_ids_list

        return remaining_map
