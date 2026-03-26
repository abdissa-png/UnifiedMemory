"""
Chat session manager backed by SQL.

Handles ChatSession / ChatMessage lifecycle, auto-title on first user
message, and document association.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import select, delete as sa_delete

from .models import ChatSession, ChatMessage, SessionDocument


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token for English)."""
    return max(1, len(text) // 4)


class ChatSessionManager:
    """CRUD operations for chat sessions and messages."""

    def __init__(self, session_factory) -> None:
        self._sf = session_factory

    # ---- sessions ----------------------------------------------------------

    async def create_session(
        self,
        tenant_id: str,
        user_id: str,
        namespace: str,
        title: str = "",
        agent_config: Optional[Dict[str, Any]] = None,
    ) -> ChatSession:
        async with self._sf() as db:
            session = ChatSession(
                id=uuid.uuid4().hex,
                tenant_id=tenant_id,
                user_id=user_id,
                namespace=namespace,
                title=title,
                agent_config_json=json.dumps(agent_config or {}),
            )
            db.add(session)
            await db.commit()
            await db.refresh(session)
            return session

    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        async with self._sf() as db:
            return await db.get(ChatSession, session_id)

    async def list_sessions(
        self, tenant_id: str, user_id: str, namespace: Optional[str] = None
    ) -> List[ChatSession]:
        async with self._sf() as db:
            stmt = (
                select(ChatSession)
                .where(
                    ChatSession.tenant_id == tenant_id,
                    ChatSession.user_id == user_id,
                )
                .order_by(ChatSession.updated_at.desc())
            )
            if namespace:
                stmt = stmt.where(ChatSession.namespace == namespace)
            result = await db.execute(stmt)
            return list(result.scalars().all())

    # ---- messages ----------------------------------------------------------

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        retrieval_context: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatMessage:
        async with self._sf() as db:
            msg = ChatMessage(
                id=uuid.uuid4().hex,
                session_id=session_id,
                role=role,
                content=content,
                metadata_json=json.dumps(metadata or {}),
                retrieval_context_json=(
                    json.dumps(retrieval_context) if retrieval_context else None
                ),
                token_count=_estimate_tokens(content),
            )
            db.add(msg)

            # Auto-set title from first user message
            if role == "user":
                session = await db.get(ChatSession, session_id)
                if session and not session.title:
                    session.title = content[:100].strip()

            await db.commit()
            await db.refresh(msg)
            return msg

    async def get_messages(
        self, session_id: str, limit: int = 200
    ) -> List[ChatMessage]:
        async with self._sf() as db:
            stmt = (
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.asc())
                .limit(limit)
            )
            result = await db.execute(stmt)
            return list(result.scalars().all())

    # ---- document association ----------------------------------------------

    async def associate_document(self, session_id: str, document_id: str) -> None:
        async with self._sf() as db:
            existing = await db.get(SessionDocument, (session_id, document_id))
            if existing:
                return
            db.add(SessionDocument(session_id=session_id, document_id=document_id))
            await db.commit()

    async def get_associated_documents(self, session_id: str) -> List[str]:
        async with self._sf() as db:
            stmt = select(SessionDocument.document_id).where(
                SessionDocument.session_id == session_id
            )
            result = await db.execute(stmt)
            return list(result.scalars().all())
