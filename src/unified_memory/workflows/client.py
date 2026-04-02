"""
Inngest client singleton.

Import-guarded so the rest of the codebase doesn't hard-depend on the
``inngest`` package.
"""

from __future__ import annotations

from typing import Optional

_client = None


def get_inngest_client():
    """Lazy-initialise and return the shared Inngest client."""
    global _client
    if _client is not None:
        return _client
    try:
        import inngest
    except ImportError as exc:
        raise ImportError(
            "The 'inngest' package is required for workflow support. "
            "Install it with: pip install 'unified-memory-system[workflows]'"
        ) from exc
    _client = inngest.Inngest(app_id="unified-memory")
    return _client
