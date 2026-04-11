"""
Request-context middleware: assigns X-Request-ID and sets tracing ContextVars.
"""

from __future__ import annotations

import os
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from unified_memory.core.logging import bind_log_context, clear_log_context
from unified_memory.observability.audit import set_audit_ip_address
from unified_memory.observability.tracing import set_request_context


def _extract_client_ip(request: Request) -> str:
    """Resolve the client IP, optionally trusting proxy headers."""
    trust_proxy_headers = os.environ.get(
        "UMS_TRUST_PROXY_HEADERS", ""
    ).lower() in {"1", "true", "yes"}
    if trust_proxy_headers:
        forwarded_for = request.headers.get("X-Forwarded-For", "")
        if forwarded_for:
            first_hop = forwarded_for.split(",")[0].strip()
            if first_hop:
                return first_hop
        real_ip = request.headers.get("X-Real-IP", "").strip()
        if real_ip:
            return real_ip
    return request.client.host if request.client else ""


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        client_ip = _extract_client_ip(request)
        request.state.request_id = request_id
        request.state.client_ip = client_ip

        # Tenant / namespace are set after auth resolves
        set_request_context(tenant_id="", namespace="")
        bind_log_context(tenant_id="", namespace="")
        set_audit_ip_address(client_ip)

        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            set_audit_ip_address("")
            clear_log_context()
