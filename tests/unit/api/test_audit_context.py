import logging

import pytest

from unified_memory.observability.audit import AuditLogger, set_audit_ip_address


@pytest.mark.asyncio
async def test_audit_logger_uses_bound_request_ip(caplog):
    caplog.set_level(logging.INFO)
    set_audit_ip_address("203.0.113.7")

    try:
        logger = AuditLogger()
        await logger.log(
            tenant_id="tenant-1",
            user_id="user-1",
            action="document.create",
        )
    finally:
        set_audit_ip_address("")

    assert caplog.records
    assert caplog.records[-1].ip_address == "203.0.113.7"
