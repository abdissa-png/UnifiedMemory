"""
Memory type exports extracted from ``core.types``.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
import uuid

from unified_memory.core.enums import ConsolidationAction, MemoryLayer, MemoryStatus, MemoryType
from unified_memory.core.source_types import SourceReference
from unified_memory.core.type_helpers import compute_content_hash, utc_now

# Valid combinations of MemoryType and MemoryLayer
VALID_MEMORY_LAYER_COMBINATIONS: Dict[MemoryType, Set[MemoryLayer]] = {
    MemoryType.EPISODIC: {
        MemoryLayer.L1_WORKING,
        MemoryLayer.L2_EPISODIC,
        MemoryLayer.L4_ARCHIVAL,
    },
    MemoryType.SEMANTIC: {
        MemoryLayer.L2_EPISODIC,
        MemoryLayer.L3_SEMANTIC,
        MemoryLayer.L4_ARCHIVAL,
    },
    MemoryType.PROCEDURAL: {
        MemoryLayer.L3_SEMANTIC,
        MemoryLayer.L4_ARCHIVAL,
    },
    MemoryType.CORE: {
        MemoryLayer.L3_SEMANTIC,
        MemoryLayer.L4_ARCHIVAL,
    },
    MemoryType.PREFERENCE: {
        MemoryLayer.L2_EPISODIC,
        MemoryLayer.L3_SEMANTIC,
    },
}


class MemoryValidationError(ValueError):
    """Raised when memory type/layer combination is invalid."""

    pass


@dataclass
class Memory:
    """
    Core memory record - ENHANCED with source tracking and validation.

    STORAGE MAPPING (UPDATED DESIGN):
    - Authoritative Memory record is stored in KV:
      key: "memory:{memory_id}" -> full Memory dict
    - Memory embedding is stored in the tenant-level vector store collection (memories)
    """

    content: str
    memory_type: MemoryType = MemoryType.SEMANTIC
    user_id: str = "default"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    namespace: str = "default"
    embedding: Optional[List[float]] = None
    status: MemoryStatus = MemoryStatus.VALID
    layer: MemoryLayer = MemoryLayer.L3_SEMANTIC
    importance: float = 0.5
    access_count: int = 0
    created_at: datetime = field(default_factory=utc_now)
    updated_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None
    supersedes_id: Optional[str] = None

    # Source tracking
    sources: List[SourceReference] = field(default_factory=list)
    source_chunk_ids: Set[str] = field(default_factory=set)

    # Content hash for deduplication
    content_hash: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate memory type/layer combination."""

        valid_layers = VALID_MEMORY_LAYER_COMBINATIONS.get(self.memory_type, set())
        if self.layer not in valid_layers:
            raise MemoryValidationError(
                f"Invalid combination: {self.memory_type.value} memory "
                f"cannot be in {self.layer.value} layer. "
                f"Valid layers for {self.memory_type.value}: "
                f"{[l.value for l in valid_layers]}"
            )

    def get_content_hash(self, tenant_id: str) -> str:
        """
        Get content hash using canonical function.

        Note: Renamed from compute_content_hash to clarify it delegates
        to the canonical compute_content_hash() function.
        """

        return compute_content_hash(self.content, tenant_id)

@dataclass
class ConsolidationTrigger:
    """
    Defines when memory consolidation should be triggered.

    Triggers:
    - THRESHOLD: Memory count exceeds threshold
    - SCHEDULED: Periodic consolidation (e.g., daily)
    - ON_CONFLICT: When conflicting memories detected
    - MANUAL: User-initiated consolidation
    """

    trigger_type: str  # "threshold", "scheduled", "on_conflict", "manual"
    threshold_count: Optional[int] = 100  # For threshold trigger
    schedule_cron: Optional[str] = None  # For scheduled trigger
    namespace_pattern: str = "*"  # Which namespaces to consolidate


@dataclass
class ConsolidationRule:
    """
    Rule for determining consolidation action.

    Conflict resolution strategy:
    1. Exact duplicates (same content hash): MERGE, keep older timestamp
    2. Near duplicates (>0.95 similarity): MERGE, keep richer content
    3. Conflicting facts (contradiction detected): SUPERSEDE
    4. Complementary memories: NOOP, keep both
    """

    memory_type: MemoryType
    similarity_threshold: float = 0.95
    contradiction_detection: bool = True
    prefer_recent: bool = True  # On conflict, prefer recent memory


@dataclass
class ConsolidationResult:
    """Result of a consolidation operation."""

    action: ConsolidationAction
    source_ids: List[str]  # IDs of memories that were consolidated
    result_id: Optional[str]  # ID of resulting memory (if merged/updated)
    reason: str  # Why this action was taken
    confidence: float = 1.0  # Confidence in the consolidation decision  


__all__ = [
    "ConsolidationResult",
    "ConsolidationRule",
    "ConsolidationTrigger",
    "Memory",
    "MemoryValidationError",
    "VALID_MEMORY_LAYER_COMBINATIONS",
]
