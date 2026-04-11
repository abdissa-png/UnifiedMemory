"""
SQLAlchemy ORM models for chat, auth, audit, and usage tracking.
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func


metadata = MetaData(
    naming_convention={
        "ix": "ix_%(table_name)s_%(column_0_name)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s",
        "pk": "pk_%(table_name)s",
    }
)


class Base(DeclarativeBase):
    metadata = metadata


# ---------------------------------------------------------------------------
# Auth / Users
# ---------------------------------------------------------------------------


class User(Base):
    __tablename__ = "users"

    id = Column(String(64), primary_key=True)
    tenant_id = Column(String(64), nullable=False, index=True)
    email = Column(String(255), nullable=False)
    display_name = Column(String(255), default="")
    password_hash = Column(String(255), nullable=False)
    roles_json = Column(Text, default="[]")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String(64), primary_key=True)
    tenant_id = Column(String(64), nullable=False)
    user_id = Column(String(64), ForeignKey("users.id"), nullable=False)
    namespace = Column(String(512), nullable=False)
    title = Column(String(255), default="")
    agent_config_json = Column(Text, default="{}")
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    messages = relationship(
        "ChatMessage",
        back_populates="session",
        order_by="ChatMessage.created_at",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_session_tenant_user", "tenant_id", "user_id"),
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String(64), primary_key=True)
    session_id = Column(
        String(64), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False
    )
    role = Column(String(16), nullable=False)  # "user", "assistant", "system"
    content = Column(Text, nullable=False)
    metadata_json = Column(Text, default="{}")
    retrieval_context_json = Column(Text, default=None, nullable=True)
    token_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    session = relationship("ChatSession", back_populates="messages")

    __table_args__ = (
        Index("ix_message_session_created", "session_id", "created_at"),
    )


class SessionDocument(Base):
    __tablename__ = "session_documents"

    session_id = Column(
        String(64),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        primary_key=True,
    )
    document_id = Column(String(128), primary_key=True)
    added_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_session_document_session_id", "session_id"),
    )

# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id = Column(String(64), primary_key=True)
    tenant_id = Column(String(64), nullable=False)
    user_id = Column(String(64), nullable=False)
    action = Column(String(64), nullable=False)
    resource_type = Column(String(64), default="")
    resource_id = Column(String(512), default="")
    details_json = Column(Text, default="{}")
    ip_address = Column(String(45), default="")
    outcome = Column(String(16), default="success")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_audit_tenant_action_created", "tenant_id", "action", "created_at"),
        Index("ix_audit_user_created", "user_id", "created_at"),
    )


# ---------------------------------------------------------------------------
# Token usage / cost tracking
# ---------------------------------------------------------------------------


class TokenUsageRecord(Base):
    __tablename__ = "token_usage"

    id = Column(String(64), primary_key=True)
    trace_id = Column(String(64), nullable=False, index=True)
    tenant_id = Column(String(64), nullable=False)
    namespace = Column(String(512), default="")
    service = Column(String(64), nullable=False)
    model = Column(String(128), nullable=False)
    operation = Column(String(64), nullable=False)

    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    reasoning_tokens = Column(Integer, default=0)
    cache_read_tokens = Column(Integer, default=0)
    cache_creation_tokens = Column(Integer, default=0)
    search_units = Column(Integer, default=0)

    duration_ms = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_usage_tenant_model_created", "tenant_id", "model", "created_at"),
    )
