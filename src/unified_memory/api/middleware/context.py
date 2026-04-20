"""
Request-context middleware: assigns X-Request-ID and sets tracing ContextVars.

Implemented as **pure ASGI** middleware (not ``BaseHTTPMiddleware``) so async
database drivers (asyncpg) stay on the same asyncio loop as the engine.
``BaseHTTPMiddleware`` runs ``call_next`` in a separate task and commonly
triggers: ``RuntimeError: ... Future ... attached to a different loop``.
"""

from __future__ import annotations

import os
import uuid

from starlette.datastructures import MutableHeaders
from starlette.requests import Request
from starlette.types import ASGIApp, Receive, Scope, Send

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


class RequestContextMiddleware:
    """ASGI middleware — safe with async SQLAlchemy / asyncpg."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        scope.setdefault("state", {})
        request = Request(scope)
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        client_ip = _extract_client_ip(request)
        request.state.request_id = request_id
        request.state.client_ip = client_ip

        # Tenant / namespace are set after auth resolves
        set_request_context(tenant_id="", namespace="", user_id="")
        set_audit_ip_address(client_ip)

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                if "x-request-id" not in headers:
                    headers["X-Request-ID"] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            set_audit_ip_address("")
            clear_log_context()
