"""
Request-context middleware: assigns X-Request-ID and sets tracing ContextVars.
"""

from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from unified_memory.observability.tracing import set_request_context


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id

        # Tenant / namespace are set after auth resolves
        set_request_context(tenant_id="", namespace="")

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
