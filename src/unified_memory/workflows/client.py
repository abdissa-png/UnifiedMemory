"""
Inngest client singleton.

Import-guarded so the rest of the codebase doesn't hard-depend on the
``inngest`` package.
"""

from __future__ import annotations

import os
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

    event_key = os.environ.get("INNGEST_EVENT_KEY")
    event_api_base_url = os.environ.get("INNGEST_EVENT_API_BASE_URL")
    api_base_url = os.environ.get("INNGEST_API_BASE_URL")
    is_production_env = os.environ.get("INNGEST_IS_PRODUCTION", "").strip().lower()

    if is_production_env in ("1", "true", "yes", "on"):
        is_production = True
    elif is_production_env in ("0", "false", "no", "off"):
        is_production = False
    else:
        # Default to dev-server mode when no event key is configured.
        # This avoids EventKeyUnspecifiedError in local development.
        is_production = bool(event_key)

    _client = inngest.Inngest(
        app_id="unified-memory",
        event_key=event_key or None,
        event_api_base_url=event_api_base_url or None,
        api_base_url=api_base_url or None,
        is_production=is_production,
    )
    return _client
