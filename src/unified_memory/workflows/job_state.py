"""
Ingestion job state — tracks progress across durable workflow steps.

Stored in KV as ``job:{job_id}`` so that any step (or a
recovery process) can resume from the last completed stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional

from unified_memory.core.utils import utc_now
from unified_memory.workflows.error_sanitize import sanitize_workflow_error_text


class JobStatus(str, Enum):
    """High-level lifecycle state for an async job."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class JobStage(str, Enum):
    """Ordered stages for ingest and delete workflows."""

    CREATED = "created"
    DISPATCHED = "dispatched"
    RESOLVE_CONTEXT = "resolve_context"
    PARSED = "parsed"
    DEDUP_CHECKED = "dedup_checked"
    FAST_LINKED = "fast_linked"
    CHUNKED = "chunked"
    TEXT_EMBEDDED = "text_embedded"
    SPARSE_WRITTEN = "sparse_written"
    GRAPH_WRITTEN = "graph_written"
    ENTITY_REL_EMBEDDED = "entity_rel_embedded"
    VISION_EMBEDDED = "vision_embedded"
    INGEST_FINALIZED = "ingest_finalized"
    DELETE_LOADED = "delete_loaded"
    DELETE_REGISTRY_UPDATED = "delete_registry_updated"
    DELETE_STORAGE_CLEANED = "delete_storage_cleaned"
    DELETE_FINALIZED = "delete_finalized"
    FINALIZED = "finalized"
    FAILED = "failed"


@dataclass
class IngestionJobState:
    """Persistent state for a single async ingest or delete job."""

    job_id: str
    tenant_id: str
    namespace: str
    operation: str = "ingest"
    document_id: str = ""
    doc_hash: str = ""

    status: JobStatus = JobStatus.QUEUED
    stage: JobStage = JobStage.CREATED
    error: Optional[str] = None
    result: Dict[str, Any] = field(default_factory=dict)

    # Artifact URIs (populated by parse step)
    parsed_artifact_uri: str = ""
    page_image_uris: List[str] = field(default_factory=list)

    # Chunk manifest (populated by chunk step)
    chunk_manifest_uri: str = ""
    chunk_count: int = 0
    chunk_content_hashes: List[str] = field(default_factory=list)

    # IDs accumulated by write steps
    text_vector_ids: List[str] = field(default_factory=list)
    entity_vector_ids: List[str] = field(default_factory=list)
    relation_vector_ids: List[str] = field(default_factory=list)
    page_image_vector_ids: List[str] = field(default_factory=list)
    graph_node_ids: List[str] = field(default_factory=list)
    graph_edge_ids: List[str] = field(default_factory=list)

    # Timestamps
    created_at: str = field(default_factory=lambda: utc_now().isoformat())
    updated_at: str = ""

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    @staticmethod
    def kv_key(job_id: str) -> str:
        return f"job:{job_id}"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["stage"] = self.stage.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IngestionJobState":
        d = dict(d)
        d["status"] = JobStatus(d.get("status", JobStatus.QUEUED.value))
        d["stage"] = JobStage(d.get("stage", "created"))
        return cls(**{
            k: v for k, v in d.items()
            if k in {f.name for f in cls.__dataclass_fields__.values()}
        })

    def mark_stage(self, stage: JobStage) -> None:
        self.status = JobStatus.RUNNING
        self.stage = stage
        self.updated_at = utc_now().isoformat()

    def mark_succeeded(self, result: Optional[Dict[str, Any]] = None) -> None:
        self.status = JobStatus.SUCCEEDED
        self.stage = JobStage.FINALIZED
        self.result = result or {}
        self.updated_at = utc_now().isoformat()

    def mark_failed(self, error: str) -> None:
        self.status = JobStatus.FAILED
        self.stage = JobStage.FAILED
        self.error = sanitize_workflow_error_text(error)
        self.updated_at = utc_now().isoformat()


async def save_job_state(kv_store, state: IngestionJobState) -> None:
    """Persist a job state in the shared KV store."""
    await kv_store.set(state.kv_key(state.job_id), state.to_dict())


async def load_job_state(kv_store, job_id: str) -> Optional[IngestionJobState]:
    """Load a job state from the shared KV store."""
    versioned = await kv_store.get(IngestionJobState.kv_key(job_id))
    if not versioned:
        return None
    return IngestionJobState.from_dict(versioned.data)
