"""
Chat session and message endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from unified_memory.api.deps import ACLChecker, get_current_user, get_system_context
from unified_memory.api.schemas import (
    AssociateDocumentRequest,
    CreateSessionRequest,
    MessageResponse,
    SearchResultItem,
    SendMessageRequest,
    SessionResponse,
)
from unified_memory.auth.jwt_handler import AuthenticatedUser
from unified_memory.core.types import Permission
from unified_memory.observability.tracing import set_request_context

router = APIRouter(prefix="/v1/chat", tags=["chat"])


def _require_session_manager(ctx):
    sm = getattr(ctx, "chat_session_manager", None)
    if sm is None:
        raise HTTPException(501, "Chat sessions not configured (SQL not enabled)")
    return sm


@router.post("/sessions/{namespace}", response_model=SessionResponse)
async def create_session(
    namespace: str,
    body: CreateSessionRequest,
    ns_config=Depends(ACLChecker(Permission.READ)),
    user: AuthenticatedUser = Depends(get_current_user),
    ctx=Depends(get_system_context),
):
    sm = _require_session_manager(ctx)
    session = await sm.create_session(
        tenant_id=user.tenant_id,
        user_id=user.user_id,
        namespace=namespace,
        title=body.title,
    )
    return SessionResponse(
        id=session.id,
        namespace=session.namespace,
        title=session.title,
        created_at=str(session.created_at),
    )


@router.get("/sessions", response_model=list[SessionResponse])
async def list_sessions(
    namespace: str,
    ns_config=Depends(ACLChecker(Permission.READ)),
    user: AuthenticatedUser = Depends(get_current_user),
    ctx=Depends(get_system_context),
):
    sm = _require_session_manager(ctx)
    sessions = await sm.list_sessions(
        tenant_id=user.tenant_id,
        user_id=user.user_id,
        namespace=namespace,
    )
    return [
        SessionResponse(
            id=s.id,
            namespace=s.namespace,
            title=s.title,
            created_at=str(s.created_at),
        )
        for s in sessions
    ]


@router.post("/sessions/{session_id}/messages", response_model=MessageResponse)
async def send_message(
    session_id: str,
    body: SendMessageRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    ctx=Depends(get_system_context),
):
    sm = _require_session_manager(ctx)
    session = await sm.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session.user_id != user.user_id:
        raise HTTPException(403, "Not your session")

    set_request_context(tenant_id=user.tenant_id, namespace=session.namespace)

    # Store user message
    await sm.add_message(session_id, "user", body.content)

    # Run QA agent if available, else plain search + LLM
    qa_agent = getattr(ctx, "qa_agent", None)
    if qa_agent:
        result = await qa_agent.answer(
            question=body.content,
            namespace=session.namespace,
            user_id=user.user_id,
            session_id=session_id,
        )
        answer = result["answer"]
        sources = result.get("sources", [])
        retrieval_ctx = result.get("retrieval_context", [])
        reasoning_trace = result.get("reasoning_trace", [])
        metadata = {"reasoning_trace": reasoning_trace}
    else:
        # Fallback: direct search only
        results = await ctx.search_service.search(
            query=body.content,
            user_id=user.user_id,
            namespace=session.namespace,
        )
        answer = "\n\n".join(r.content for r in results if r.content) or "No results found."
        sources = []
        retrieval_ctx = [
            {"id": r.id, "score": r.score, "snippet": (r.content or "")}
            for r in results
        ]
        metadata = None

    msg = await sm.add_message(
        session_id,
        "assistant",
        answer,
        retrieval_context=retrieval_ctx,
        metadata=metadata,
    )

    return MessageResponse(
        id=msg.id,
        role="assistant",
        content=answer,
        sources=[
            SearchResultItem(
                id=s.get("id", ""),
                content=s.get("snippet", ""),
                score=s.get("score", 0.0),
            )
            for s in sources
        ],
        created_at=str(msg.created_at),
    )


@router.get("/sessions/{session_id}/messages", response_model=list[MessageResponse])
async def get_messages(
    session_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
    ctx=Depends(get_system_context),
):
    sm = _require_session_manager(ctx)
    session = await sm.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session.user_id != user.user_id:
        raise HTTPException(403, "Not your session")

    messages = await sm.get_messages(session_id)
    return [
        MessageResponse(
            id=m.id, role=m.role, content=m.content, created_at=str(m.created_at)
        )
        for m in messages
    ]


@router.post("/sessions/{session_id}/documents")
async def associate_document(
    session_id: str,
    body: AssociateDocumentRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    ctx=Depends(get_system_context),
):
    sm = _require_session_manager(ctx)
    session = await sm.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session.user_id != user.user_id:
        raise HTTPException(403, "Not your session")

    await sm.associate_document(session_id, body.document_id)
    return {"status": "associated"}
