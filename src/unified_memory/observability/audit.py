"""
SQL-backed audit logger.

Records security-relevant events (permission denials, write operations,
GDPR actions, admin changes) to the ``audit_events`` SQL table.
"""

from __future__ import annotations

import logging
import uuid
import json
from typing import Any, Dict, Optional

from unified_memory.core.utils import utc_now

logger = logging.getLogger(__name__)


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
            logger.info(
                "audit.event",
                extra={
                    "tenant_id": tenant_id,
                    "user_id": user_id,
                    "action": action,
                    "resource_type": resource_type,
                    "resource_id": resource_id,
                    "outcome": outcome,
                },
            )
            return

        try:
            from unified_memory.storage.sql.models import AuditEvent

            async with self._session_factory() as db:
                event = AuditEvent(
                    id=uuid.uuid4().hex,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    action=action,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    details_json=json.dumps(details) if details else "{}",
                    ip_address=ip_address,
                    outcome=outcome,
                )
                db.add(event)
                await db.commit()
        except Exception:
            logger.exception("Failed to write audit event to SQL")
