"""
SQL-backed audit logger.

Records security-relevant events (permission denials, write operations,
GDPR actions, admin changes) to the ``audit_events`` SQL table.
"""

from __future__ import annotations

import logging
import uuid
import json
from contextvars import ContextVar
from typing import Any, Dict, Optional

from unified_memory.core.utils import utc_now
from unified_memory.core.logging import get_logger,log_event
logger = get_logger(__name__)

_request_ip_address: ContextVar[str] = ContextVar("_request_ip_address", default="")


def set_audit_ip_address(ip_address: str) -> None:
    """Bind the current request's client IP for audit events."""
    _request_ip_address.set(ip_address or "")


def get_audit_ip_address() -> str:
    """Return the current request's client IP if one is bound."""
    return _request_ip_address.get("")


class AuditLogger:
    """Writes audit events to the SQL database.

    Parameters
    ----------
    session_factory
        An ``async_sessionmaker`` (SQLAlchemy 2.0) producing async sessions.
    """

    def __init__(self, session_factory=None) -> None:
        self._session_factory = session_factory

    async def log(
        self,
        *,
        tenant_id: str,
        user_id: str,
        action: str,
        resource_type: str = "",
        resource_id: str = "",
        details: Optional[Dict[str, Any]] = None,
        ip_address: str = "",
        outcome: str = "success",
    ) -> None:
        if self._session_factory is None:
            log_event(logger, logging.INFO, "audit.event",
                tenant_id=tenant_id,
                user_id=user_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                details=details,
                ip_address=ip_address or get_audit_ip_address(),
                outcome=outcome,
            )
            return

        try:
            from unified_memory.storage.sql.models import AuditEvent

            resolved_ip_address = ip_address or get_audit_ip_address()

            async with self._session_factory() as db:
                event = AuditEvent(
                    id=uuid.uuid4().hex,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    action=action,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    details_json=json.dumps(details) if details else "{}",
                    ip_address=resolved_ip_address,
                    outcome=outcome,
                )
                db.add(event)
                await db.commit()
        except Exception as e:
            log_event(logger, logging.ERROR, "audit.event.failed",
                tenant_id=tenant_id,
                user_id=user_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                details=details,
                ip_address=ip_address or get_audit_ip_address(),
                outcome=outcome,
                error=str(e),
            )
