"""
Search and answer endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from unified_memory.api.deps import ACLChecker, get_current_user, get_system_context
from unified_memory.api.schemas import (
    AnswerRequest,
    AnswerResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)
from unified_memory.auth.jwt_handler import AuthenticatedUser
from unified_memory.core.types import Permission
from unified_memory.observability.tracing import set_request_context

router = APIRouter(prefix="/v1", tags=["search"])

@router.post("/search/answer/{namespace:path}", response_model=AnswerResponse)
async def search_answer(
    namespace: str,
    body: AnswerRequest,
    ns_config=Depends(ACLChecker(Permission.READ)),
    user: AuthenticatedUser = Depends(get_current_user),
    ctx=Depends(get_system_context),
):
    set_request_context(
        tenant_id=user.tenant_id,
        namespace=namespace,
        user_id=user.user_id,
    )
    qa_agent = getattr(ctx, "qa_agent", None)
    if not qa_agent:
        raise HTTPException(501, "QA agent not configured")

    result = await qa_agent.answer(
        question=body.query,
        namespace=namespace,
        user_id=user.user_id,
    )
    return AnswerResponse(
        answer=result["answer"],
        sources=[
            SearchResultItem(
                id=s.get("id", ""),
                content=s.get("snippet", ""),
                score=s.get("score", 0.0),
            )
            for s in result.get("sources", [])
        ],
        reasoning_trace=result.get("reasoning_trace", []),
        token_usage=result.get("token_usage"),
    )

@router.post("/search/{namespace:path}", response_model=SearchResponse)
async def search(
    namespace: str,
    body: SearchRequest,
    ns_config=Depends(ACLChecker(Permission.READ)),
    user: AuthenticatedUser = Depends(get_current_user),
    ctx=Depends(get_system_context),
):
    set_request_context(
        tenant_id=user.tenant_id,
        namespace=namespace,
        user_id=user.user_id,
    )
    results = await ctx.search_service.search(
        query=body.query,
        user_id=user.user_id,
        namespace=namespace,
        request_options={
            "paths": body.paths,
            "top_k": body.top_k,
            "rerank": body.rerank,
            "fusion_method": body.fusion_method,
            "fusion_weights": body.fusion_weights,
            "score_threshold": body.score_threshold,
            "rerank_candidates_limit": body.rerank_candidates_limit,
            "reranker_key": body.reranker_key,
        },
        target_namespaces=body.target_namespaces,
        filters=body.filters,
    )
    items = [
        SearchResultItem(
            id=r.id,
            content=r.content or "",
            score=r.score,
            metadata=r.metadata,
            source=r.source or "",
            document_id=r.metadata.get("document_id"),
            chunk_index=r.metadata.get("chunk_index"),
            evidence_type=r.metadata.get("evidence_type"),
        )
        for r in results
    ]
    return SearchResponse(
        results=items,
        query=body.query,
        namespace=namespace,
        total_results=len(items),
        fusion_method=body.fusion_method,
        paths_used=body.paths,
    )



