"""
Sparse Retrieval Implementation using rank-bm25.

In-memory implementation for MVP and testing.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
import re
from rank_bm25 import BM25Okapi

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
        self._indices: Dict[str, BM25Okapi] = {}
        # Namespace -> List of (doc_id, content, metadata) tuples, aligned with index
        self._documents: Dict[str, List[Tuple[str, str, Dict[str, Any]]]] = {}
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace and punctuation tokenization."""
        # Lowercase and split by non-alphanumeric
        return re.findall(r'\w+', text.lower())
    
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
            
        # 1. Tokenize and Store
        corpus_tokens = []
        
        # If we already have docs, we need to rebuild the whole index for this namespace
        # because rank_bm25 is immutable/batch-oriented.
        # So we combine existing docs + new docs.
        # TODO: Optimize to avoid full rebuild on every small batch if possible, 
        # or accept this trade-off for MVP.
        
        current_docs = self._documents[namespace]
        
        # Map existing docs to tokens for rebuilding
        for _, content, _ in current_docs:
            corpus_tokens.append(self._tokenize(content))
            
        # Process new docs
        count = 0
        for doc in documents:
            doc_id = doc.get("id")
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            if not doc_id:
                continue
                
            tokens = self._tokenize(content)
            corpus_tokens.append(tokens)
            current_docs.append((doc_id, content, metadata))
            count += 1
            
        # 2. Build Index (Rebuild)
        if corpus_tokens:
            self._indices[namespace] = BM25Okapi(corpus_tokens)
            
        return count

    async def retrieve(
        self,
        query: str,
        namespace: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Retrieve using BM25 scoring."""
        if namespace not in self._indices or namespace not in self._documents:
            return []
            
        bm25 = self._indices[namespace]
        docs_list = self._documents[namespace]
        
        # 1. Tokenize Query
        tokenized_query = self._tokenize(query)
        
        # 2. Get Scores
        scores = bm25.get_scores(tokenized_query)
        
        # 3. Rank and Filter
        # Pair docs with scores
        doc_scores = []
        for i, score in enumerate(scores):
            if score <= 0:
                continue
            doc_scores.append((docs_list[i], score))
            
        # Sort desc by score
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        results: List[RetrievalResult] = []
        
        for (doc_id, content, metadata), score in doc_scores:
            # Apply Metadata Filters
            if filters:
                match = True
                for key, value in filters.items():
                    # Simple equality check for MVP
                    if metadata.get(key) != value:
                        match = False
                        break
                if not match:
                    continue
            
            results.append(RetrievalResult(
                id=doc_id,
                content=content,
                score=float(score),
                metadata=metadata,
                source="sparse:bm25",
                evidence_type="text"
            ))
            
            if len(results) >= top_k:
                break
                
        return results

    async def delete(
        self,
        doc_ids: List[str],
        namespace: str,
    ) -> int:
        """
        Delete documents from index.
        Requires rebuild of index.
        """
        if namespace not in self._documents:
            return 0
            
        original_len = len(self._documents[namespace])
        
        # Filter out deleted docs
        to_delete = set(doc_ids)
        new_docs = [
            d for d in self._documents[namespace] 
            if d[0] not in to_delete
        ]
        
        if len(new_docs) == original_len:
            return 0
            
        self._documents[namespace] = new_docs
        
        # Rebuild index
        corpus_tokens = [self._tokenize(d[1]) for d in new_docs]
        if corpus_tokens:
            self._indices[namespace] = BM25Okapi(corpus_tokens)
        else:
            if namespace in self._indices:
                del self._indices[namespace]
                
        return original_len - len(new_docs)
