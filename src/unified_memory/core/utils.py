"""
Utility helpers for the unified memory system.

This module re-exports canonical helpers from core.types so that other
modules can depend on a small, focused surface if desired.
"""

from .types import compute_content_hash, utc_now

__all__ = ["compute_content_hash", "utc_now"]

