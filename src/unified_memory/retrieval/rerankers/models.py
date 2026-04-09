"""
Reranker implementations.

Supports:
- Cohere Rerank API
- BGE Reranker (Local/HuggingFace)
"""

from typing import List, Optional
import logging
import asyncio

from unified_memory.core.interfaces import Reranker
from unified_memory.core.types import RetrievalResult

from unified_memory.core.logging import get_logger,log_event
logger = get_logger(__name__)

class CohereReranker:
    """
    Reranker using Cohere API.
    """
    
    def __init__(self, api_key: str, model: str = "rerank-english-v3.0"):
        try:
            import cohere
        except ImportError:
            raise ImportError("Cohere support requires 'cohere' package: pip install cohere")
            
        self.client = cohere.AsyncClient(api_key=api_key)
        self._model_id = model
        
    @property
    def model_id(self) -> str:
        return self._model_id
        
    async def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        if not results:
            return []
            
        # Extract contents
        # Handling different evidence types: text vs graph
        documents = []
        for r in results:
            content = r.content
            if not content:
                # Fallback for graph nodes without content properties
                content = str(r.metadata)
            documents.append(content)
            
        try:
            response = await self.client.rerank(
                    model=self._model_id,
                    query=query,
                    documents=documents,
                    top_n=top_k,
                )
            self._record_rerank_usage(response)
            # Map back
            reranked = []
            for item in response.results:
                original = results[item.index]
                # item.relevance_score is float
                
                # Clone result with new score
                new_result = RetrievalResult(
                    id=original.id,
                    content=original.content,
                    score=item.relevance_score,
                    metadata=original.metadata,
                    source=original.source,
                    evidence_type=original.evidence_type,
                    entity_ids=original.entity_ids,
                    relation_ids=original.relation_ids,
                    page_number=original.page_number,
                    modality=original.modality,
                )
                reranked.append(new_result)
                
            return reranked
            
        except Exception as e:
            log_event(logger, logging.ERROR, "cohere.reranking.failed", error=str(e))
            fallback_results: List[RetrievalResult] = []
            for original in results[:top_k]:
                metadata = dict(original.metadata)
                metadata["rerank_skipped"] = True
                fallback_results.append(
                    RetrievalResult(
                        id=original.id,
                        content=original.content,
                        score=original.score,
                        metadata=metadata,
                        source=original.source,
                        evidence_type=original.evidence_type,
                        entity_ids=original.entity_ids,
                        relation_ids=original.relation_ids,
                        page_number=original.page_number,
                        modality=original.modality,
                    )
                )
            return fallback_results

    def _record_rerank_usage(self, response) -> None:
        try:
            from unified_memory.observability.tracing import record_usage, UsageRecord

            meta = getattr(response, "meta", None)
            if not meta:
                return
            billed = getattr(meta, "billed_units", None)
            if not billed:
                return
            record_usage(
                UsageRecord(
                    service="cohere",
                    model=self._model_id,
                    operation="rerank",
                    input_tokens=int(getattr(billed, "input_tokens", 0) or 0),
                    search_units=int(getattr(billed, "search_units", 0) or 0),
                )
            )
        except Exception as exc:
            log_event(logger, logging.DEBUG, "cohere.reranking.usage.failed", error=str(exc))


class BGEReranker:
    """
    Local Reranker using BAAI/bge-reranker models.
    """
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError("BGE support requires 'sentence-transformers': pip install sentence-transformers")
            
        log_event(logger, logging.INFO, "bge.reranker.loading", model_name=model_name)
        self.model = CrossEncoder(model_name)
        self._model_id = model_name
        
    @property
    def model_id(self) -> str:
        return self._model_id
        
    async def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        if not results:
            return []
            
        # Prepare pairs
        pairs = []
        for r in results:
            content = r.content or str(r.metadata)
            pairs.append([query, content])
            
        # Inference
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            self.model.predict,
            pairs
        )
        
        # Sort
        # scores is numpy array or list of floats
        scored_results = []
        for i, score in enumerate(scores):
            original = results[i]
            scored_results.append((original, float(score)))
            
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k
        reranked = []
        for r, score in scored_results[:top_k]:
            new_result = RetrievalResult(
                id=r.id,
                content=r.content,
                score=score,
                metadata=r.metadata,
                source=r.source,
                evidence_type=r.evidence_type,
                entity_ids=r.entity_ids,
                relation_ids=r.relation_ids,
                page_number=r.page_number,
                modality=r.modality,
            )
            reranked.append(new_result)
            
        return reranked
