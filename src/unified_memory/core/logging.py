"""
Structured logging helpers with request context injection.
"""

from __future__ import annotations

import contextvars
import json
import logging
from typing import Any

_ctx_tenant = contextvars.ContextVar("structured_log_tenant_id", default="")
_ctx_namespace = contextvars.ContextVar("structured_log_namespace", default="")
_ctx_user = contextvars.ContextVar("structured_log_user_id", default="")


def get_logger(name: str) -> logging.Logger:
    """Return a standard logger."""
    return logging.getLogger(name)


def _serialize_field(value: Any) -> str:
    """Render log field values as compact, stable text."""
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, default=str, separators=(",", ":"))
        except Exception:
            return str(value)
    return str(value)


def bind_log_context(
    *,
    tenant_id: str | None = None,
    namespace: str | None = None,
    user_id: str | None = None,
) -> None:
    """Bind request-scoped logging context for downstream structured logs."""
    if tenant_id is not None:
        _ctx_tenant.set(tenant_id)
    if namespace is not None:
        _ctx_namespace.set(namespace)
    if user_id is not None:
        _ctx_user.set(user_id)


def clear_log_context() -> None:
    """Clear request-scoped logging context."""
    _ctx_tenant.set("")
    _ctx_namespace.set("")
    _ctx_user.set("")


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """Emit a structured log message with automatic context."""
    fields.setdefault("tenant_id", _ctx_tenant.get(""))
    fields.setdefault("namespace", _ctx_namespace.get(""))
    fields.setdefault("user_id", _ctx_user.get(""))
    # Keep extras for structured handlers, but also render fields into message
    # text so default uvicorn/python formatters still show diagnostics.
    formatted_fields = " ".join(
        f"{key}={_serialize_field(value)}" for key, value in fields.items()
    )
    message = event if not formatted_fields else f"{event} {formatted_fields}"
    logger.log(level, message, extra=fields)
