"""
Pydantic request / response models for the REST API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class RegisterTenantRequest(BaseModel):
    admin_email: str
    admin_password: str
    tenant_name: str = ""
    admin_display_name: str = ""
    # Optional config overrides:
    text_embedding_provider: Optional[str] = None
    text_embedding_model: Optional[str] = None
    text_embedding_dimension: Optional[int] = None
    vision_embedding_provider: Optional[str] = None
    vision_embedding_model: Optional[str] = None
    vision_embedding_dimension: Optional[int] = None
    chunk_size: Optional[int] = None
    chunker_type: Optional[str] = None
    enable_graph_storage: Optional[bool] = None

class RegisterTenantResponse(BaseModel):
    tenant_id: str
    access_token: str
    token_type: str = "bearer"


class RegisterUserRequest(BaseModel):
    email: str
    password: str
    display_name: str = ""
    roles: List[str] = Field(default_factory=lambda: ["tenant_member"])


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ---------------------------------------------------------------------------
# Namespaces
# ---------------------------------------------------------------------------


class CreateNamespaceRequest(BaseModel):
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    scope: str = "private"


class ShareNamespaceRequest(BaseModel):
    target_user_email: str
    permissions: List[str] = Field(default_factory=lambda: ["read"])


class NamespaceResponse(BaseModel):
    namespace_id: str
    tenant_id: str
    user_id: str
    scope: str
    created_at: str


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


class IngestTextRequest(BaseModel):
    text: str
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None


class IngestResponse(BaseModel):
    document_id: Optional[str] = None
    chunk_count: int = 0
    doc_hash: str = ""
    job_id: Optional[str] = None
    status: str = "ingested"


class JobStatusResponse(BaseModel):
    job_id: str
    operation: str
    tenant_id: str
    namespace: str
    status: str
    stage: str
    document_id: str = ""
    doc_hash: str = ""
    error: Optional[str] = None
    result: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


class DocumentResponse(BaseModel):
    document_id: str
    doc_hash: str
    namespaces: List[str] = Field(default_factory=list)
    chunk_count: int = 0
    original_filename: str = ""
    content_type: str = ""
    size_bytes: int = 0


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    paths: List[str] = Field(default_factory=lambda: ["dense", "sparse", "graph"])
    rerank: bool = False
    filters: Optional[Dict[str, Any]] = None
    fusion_method: str = "rrf"  # "rrf" or "linear"
    fusion_weights: Optional[Dict[str, float]] = None
    score_threshold: Optional[float] = None
    rerank_candidates_limit: int = 50
    reranker_key: Optional[str] = "bge-local"
    target_namespaces: Optional[List[str]] = None  # cross-namespace search


class SearchResultItem(BaseModel):
    id: str
    content: str = ""
    score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source: str = ""
    document_id: Optional[str] = None
    chunk_index: Optional[int] = None
    evidence_type: Optional[str] = None  # "text", "entity", "relation", "image"


class SearchResponse(BaseModel):
    results: List[SearchResultItem]
    query: str
    namespace: str
    total_results: int = 0
    fusion_method: str = "rrf"
    paths_used: List[str] = Field(default_factory=list)


class AnswerRequest(BaseModel):
    query: str
    top_k: int = 10


class AnswerResponse(BaseModel):
    answer: str
    sources: List[SearchResultItem] = Field(default_factory=list)
    reasoning_trace: List[Dict[str, Any]] = Field(default_factory=list)
    token_usage: Optional[Dict[str, int]] = None


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    title: str = ""


class SessionResponse(BaseModel):
    id: str
    namespace: str
    title: str
    created_at: Any = None


class SendMessageRequest(BaseModel):
    content: str


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    sources: List[SearchResultItem] = Field(default_factory=list)
    created_at: Any = None


class AssociateDocumentRequest(BaseModel):
    document_id: str


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------


class EmbeddingModelResponse(BaseModel):
    provider: str = ""
    model: str = ""
    dimension: int = 0


class TenantConfigResponse(BaseModel):
    tenant_id: str
    text_embedding: Optional[EmbeddingModelResponse] = None
    vision_embedding: Optional[EmbeddingModelResponse] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    chunker_type: str = "fixed_size"
    enable_graph_storage: bool = True
    enable_visual_indexing: bool = True
    enable_entity_extraction: bool = True
    enable_relation_extraction: bool = True
    batch_size: int = 100
    deduplication_enabled: bool = True
    llm: Optional[Dict[str, Any]] = None
    created_at: str = ""
    updated_at: str = ""


class UpdateTenantConfigRequest(BaseModel):
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    chunker_type: Optional[str] = None
    enable_graph_storage: Optional[bool] = None
    enable_visual_indexing: Optional[bool] = None
    enable_entity_extraction: Optional[bool] = None
    enable_relation_extraction: Optional[bool] = None
    batch_size: Optional[int] = None
    deduplication_enabled: Optional[bool] = None
    text_embedding_provider: Optional[str] = None
    text_embedding_model: Optional[str] = None
    text_embedding_dimension: Optional[int] = None
    vision_embedding_provider: Optional[str] = None
    vision_embedding_model: Optional[str] = None
    vision_embedding_dimension: Optional[int] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
