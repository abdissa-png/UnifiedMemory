"""
Hierarchical Configuration System.

Location: core/config.py
Design Reference: UNIFIED_MEMORY_SYSTEM_DESIGN.md Section 8
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class SystemConfig:
    """Global system-wide defaults."""
    default_text_model: str = "text-embedding-3-small"
    default_vision_model: str = "clip-vit-base-patch32"
    default_chunk_size: int = 512
    default_chunk_overlap: int = 64
    enable_multi_tenant: bool = True


@dataclass
class IngestionConfig:
    """
    Resolved configuration for a specific ingestion request.
    
    Merges settings from:
    1. System Defaults
    2. Tenant Settings (e.g., authoritative embedding model)
    3. Namespace Settings (e.g., specific collections)
    4. Request Options (e.g., high-priority override)
    """
    
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    text_collection: str
    respect_sentence_boundaries: bool = True
    
    @classmethod
    def resolve(
        cls,
        system: SystemConfig,
        tenant_data: Optional[Dict[str, Any]] = None,
        namespace_data: Optional[Dict[str, Any]] = None,
        request_options: Optional[Dict[str, Any]] = None,
    ) -> IngestionConfig:
        """
        Hierarchical resolution logic.
        Higher levels override lower levels.
        """
        request = request_options or {}
        ns = namespace_data or {}
        tenant = tenant_data or {}
        
        # 1. Chunk Size
        chunk_size = (
            request.get("chunk_size") or 
            ns.get("chunk_size") or 
            tenant.get("chunk_size") or 
            system.default_chunk_size
        )
        
        # 2. Chunk Overlap
        chunk_overlap = (
            request.get("chunk_overlap") or 
            ns.get("chunk_overlap") or 
            tenant.get("chunk_overlap") or 
            system.default_chunk_overlap
        )
        
        # 3. Embedding Model (Tenant is usually authoritative for consistency)
        # We prioritize request if explicitly provided, else tenant.
        embedding_model = (
            request.get("embedding_model") or 
            tenant.get("text_embedding", {}).get("model") or 
            system.default_text_model
        )
        
        # 4. Collections
        text_collection = (
            ns.get("text_collection") or 
            f"tenant_{tenant.get('tenant_id', 'default')}_texts"
        )
        
        return cls(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
            text_collection=text_collection,
            respect_sentence_boundaries=request.get(
                "respect_sentence_boundaries", True
            )
        )
