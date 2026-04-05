"""
FastAPI dependency injection: SystemContext, auth, ACL.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Depends, HTTPException, Request

from unified_memory.auth.jwt_handler import AuthenticatedUser, decode_access_token
from unified_memory.core.types import Permission, NamespaceACL

if TYPE_CHECKING:
    from unified_memory.bootstrap import SystemContext
    from unified_memory.namespace.types import NamespaceConfig


# ---------------------------------------------------------------------------
# Core DI helpers
# ---------------------------------------------------------------------------


def get_system_context(request: Request) -> "SystemContext":
    return request.app.state.system_context


def get_jwt_secret(request: Request) -> str:
    return request.app.state.jwt_secret


def get_token_expire_minutes(request: Request) -> int:
    return getattr(request.app.state, "token_expire_minutes", 60)


async def get_current_user(
    request: Request,
    ctx: "SystemContext" = Depends(get_system_context),
    secret: str = Depends(get_jwt_secret),
) -> AuthenticatedUser:
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(401, "Missing authorization token")
    user = decode_access_token(token, secret)
    if user is None:
        raise HTTPException(401, "Invalid or expired token")
    return user


# ---------------------------------------------------------------------------
# ACL checker (used as Depends)
# ---------------------------------------------------------------------------


class ACLChecker:
    """Callable dependency that enforces namespace-level permissions."""

    def __init__(self, required: Permission) -> None:
        self.required = required

    async def __call__(
        self,
        namespace: str,
        user: AuthenticatedUser = Depends(get_current_user),
        ctx: "SystemContext" = Depends(get_system_context),
    ) -> "NamespaceConfig":
        ns_config = await ctx.namespace_manager.get_config(namespace)
        if not ns_config:
            raise HTTPException(404, f"Namespace '{namespace}' not found")

        # Cross-tenant isolation
        if user.tenant_id != ns_config.tenant_id:
            raise HTTPException(403, "Cross-tenant access denied")

        # Tenant-inherited ACL
        tenant_acl = None
        if ns_config.acl.inherit_from_tenant:
            tenant_cfg = await ctx.namespace_manager.get_tenant_config(
                ns_config.tenant_id
            )
            if tenant_cfg and tenant_cfg.default_acl:
                tenant_acl = NamespaceACL.from_dict(tenant_cfg.default_acl)

        # Unified Namespace + Tenant ACL Check
        if ns_config.acl.check_permission(
            user.user_id, self.required, user.roles, tenant_acl=tenant_acl
        ):
            return ns_config

        # Public scope grants READ
        if ns_config.scope == "public" and self.required == Permission.READ:
            return ns_config

        raise HTTPException(403, f"No {self.required.value} access to {namespace}")
