"""
Ingestion job state — tracks progress across durable workflow steps.

Stored in KV as ``ingest_job:{job_id}`` so that any step (or a
recovery process) can resume from the last completed stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional

from unified_memory.core.utils import utc_now


class JobStage(str, Enum):
    """Ordered stages of document ingestion."""

    CREATED = "created"
    PARSED = "parsed"
    DEDUP_CHECKED = "dedup_checked"
    FAST_LINKED = "fast_linked"
    CHUNKED = "chunked"
    TEXT_EMBEDDED = "text_embedded"
    SPARSE_WRITTEN = "sparse_written"
    GRAPH_WRITTEN = "graph_written"
    ENTITY_REL_EMBEDDED = "entity_rel_embedded"
    VISION_EMBEDDED = "vision_embedded"
    FINALIZED = "finalized"
    FAILED = "failed"


@dataclass
class IngestionJobState:
    """Persistent state for a single ingestion job."""

    job_id: str
    tenant_id: str
    namespace: str
    document_id: str
    doc_hash: str = ""

    stage: JobStage = JobStage.CREATED
    error: Optional[str] = None

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
        return f"ingest_job:{job_id}"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["stage"] = self.stage.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IngestionJobState":
        d = dict(d)
        d["stage"] = JobStage(d.get("stage", "created"))
        return cls(**{
            k: v for k, v in d.items()
            if k in {f.name for f in cls.__dataclass_fields__.values()}
        })

    def mark_stage(self, stage: JobStage) -> None:
        self.stage = stage
        self.updated_at = utc_now().isoformat()

    def mark_failed(self, error: str) -> None:
        self.stage = JobStage.FAILED
        self.error = error
        self.updated_at = utc_now().isoformat()
