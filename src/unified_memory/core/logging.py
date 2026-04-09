"""
Structured logging helpers with request context injection.
"""

from __future__ import annotations

import contextvars
import logging
from typing import Any

_ctx_tenant = contextvars.ContextVar("structured_log_tenant_id", default="")
_ctx_namespace = contextvars.ContextVar("structured_log_namespace", default="")


def get_logger(name: str) -> logging.Logger:
    """Return a standard logger."""
    return logging.getLogger(name)


def bind_log_context(*, tenant_id: str | None = None, namespace: str | None = None) -> None:
    """Bind request-scoped logging context for downstream structured logs."""
    if tenant_id is not None:
        _ctx_tenant.set(tenant_id)
    if namespace is not None:
        _ctx_namespace.set(namespace)


def clear_log_context() -> None:
    """Clear request-scoped logging context."""
    _ctx_tenant.set("")
    _ctx_namespace.set("")


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """Emit a structured log message with automatic context."""
    fields.setdefault("tenant_id", _ctx_tenant.get(""))
    fields.setdefault("namespace", _ctx_namespace.get(""))
    logger.log(level, event, extra=fields)
