"""
Core Type Definitions - UNIFIED.

This module is the single source of truth for core dataclasses and enums
used throughout the unified memory system.

It consolidates types from:
- Existing prototypes
- MULTIMODAL_GRAPHRAG_DESIGN.md (ExtractedEntity, SourceReference, etc.)
- ARCHITECTURAL_REFACTORING_PLAN.md (enhanced consolidation)
"""

from __future__ import annotations
from .type_helpers import *
# ============================================================================
# ENUMERATIONS (Consolidated)
# ============================================================================
from .enums import *


# ============================================================================
# SOURCE PROVENANCE (from MMKG_DESIGN, extended)
# ============================================================================
from .source_types import *


# ============================================================================
# GRAPH TYPES (Refactored from God Object)
# ============================================================================
from .graph_types import *


# ============================================================================
# MEMORY TYPE (Enhanced with Validation)
# ============================================================================
from .memory_types import *


# ============================================================================
# RETRIEVAL TYPES (Consolidated)
# ============================================================================
from .retrieval_types import *


# ============================================================================
# DOCUMENT & CHUNK TYPES
# ============================================================================
from .ingestion_types import *



