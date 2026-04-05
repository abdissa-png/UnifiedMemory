"""
Namespace CRUD and sharing endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from unified_memory.api.deps import ACLChecker, get_current_user, get_system_context
from unified_memory.api.schemas import (
    CreateNamespaceRequest,
    NamespaceResponse,
    ShareNamespaceRequest,
)
from unified_memory.auth.jwt_handler import AuthenticatedUser
from unified_memory.core.types import Permission
from unified_memory.observability.tracing import set_request_context

router = APIRouter(prefix="/v1/namespaces", tags=["namespaces"])


@router.post("", response_model=NamespaceResponse)
async def create_namespace(
    body: CreateNamespaceRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    ctx=Depends(get_system_context),
):
    ns_config = await ctx.namespace_manager.create_namespace(
        tenant_id=user.tenant_id,
        user_id=user.user_id,
        agent_id=body.agent_id,
        session_id=body.session_id,
        scope=body.scope,
    )
    if getattr(ctx, "audit_logger", None):
        await ctx.audit_logger.log(
            tenant_id=user.tenant_id,
            user_id=user.user_id,
            action="namespace.create",
            resource_type="namespace",
            resource_id=ns_config.namespace_id,
            outcome="success",
        )

    return NamespaceResponse(
        namespace_id=ns_config.namespace_id,
        tenant_id=ns_config.tenant_id,
        user_id=ns_config.user_id,
        scope=ns_config.scope,
        created_at=ns_config.created_at,
    )


@router.get("", response_model=list[NamespaceResponse])
async def list_namespaces(
    user: AuthenticatedUser = Depends(get_current_user),
    ctx=Depends(get_system_context),
):
    ns_ids = await ctx.namespace_manager.get_accessible_namespaces(
        user_id=user.user_id, tenant_id=user.tenant_id
    )
    results = []
    for ns_id in ns_ids:
        cfg = await ctx.namespace_manager.get_config(ns_id)
        if cfg:
            results.append(
                NamespaceResponse(
                    namespace_id=cfg.namespace_id,
                    tenant_id=cfg.tenant_id,
                    user_id=cfg.user_id,
                    scope=cfg.scope,
                    created_at=cfg.created_at,
                )
            )
    return results


@router.get("/{namespace:path}/config", response_model=NamespaceResponse)
async def get_namespace(
    namespace: str,
    ns_config=Depends(ACLChecker(Permission.READ)),
    user: AuthenticatedUser = Depends(get_current_user),
):
    return NamespaceResponse(
        namespace_id=ns_config.namespace_id,
        tenant_id=ns_config.tenant_id,
        user_id=ns_config.user_id,
        scope=ns_config.scope,
        created_at=ns_config.created_at,
    )


@router.delete("/{namespace:path}")
async def delete_namespace(
    namespace: str,
    ns_config=Depends(ACLChecker(Permission.ADMIN)),
    user: AuthenticatedUser = Depends(get_current_user),
    ctx=Depends(get_system_context),
):
    deleted = await ctx.namespace_manager.delete_namespace(namespace)
    if not deleted:
        raise HTTPException(404, "Namespace not found")
        
    if getattr(ctx, "audit_logger", None):
        await ctx.audit_logger.log(
            tenant_id=user.tenant_id,
            user_id=user.user_id,
            action="namespace.delete",
            resource_type="namespace",
            resource_id=namespace,
            outcome="success",
        )
        
    return {"status": "deleted"}


@router.post("/{namespace:path}/share")
async def share_namespace(
    namespace: str,
    body: ShareNamespaceRequest,
    ns_config=Depends(ACLChecker(Permission.SHARE)),
    user: AuthenticatedUser = Depends(get_current_user),
    ctx=Depends(get_system_context),
):
    from sqlalchemy import select
    from unified_memory.storage.sql.models import User

    sf = getattr(ctx, "sql_session_factory", None)
    if not sf:
        raise HTTPException(500, "SQL storage not configured")

    async with sf() as db:
        result = await db.execute(
            select(User).where(User.email == body.target_user_email)
        )
        target_user = result.scalar_one_or_none()

    if not target_user:
        raise HTTPException(404, "Target user not found")

    permissions = [Permission(p) for p in body.permissions]
    await ctx.namespace_manager.share_namespace(
        namespace_id=namespace,
        target_user_id=target_user.id,
        permissions=permissions,
    )
    
    if getattr(ctx, "audit_logger", None):
        await ctx.audit_logger.log(
            tenant_id=user.tenant_id,
            user_id=user.user_id,
            action="namespace.share",
            resource_type="namespace",
            resource_id=namespace,
            details={"target_user_id": target_user.id, "permissions": body.permissions},
            outcome="success",
        )
        
    return {"status": "shared", "target_user_id": target_user.id}


@router.delete("/{namespace:path}/share/{user_id}")
async def unshare_namespace(
    namespace: str,
    user_id: str,
    ns_config=Depends(ACLChecker(Permission.SHARE)),
    user: AuthenticatedUser = Depends(get_current_user),
    ctx=Depends(get_system_context),
):
    await ctx.namespace_manager.unshare_namespace(
        namespace_id=namespace, target_user_id=user_id
    )
    
    if getattr(ctx, "audit_logger", None):
        await ctx.audit_logger.log(
            tenant_id=user.tenant_id,
            user_id=user.user_id,
            action="namespace.unshare",
            resource_type="namespace",
            resource_id=namespace,
            details={"target_user_id": user_id},
            outcome="success",
        )
        
    return {"status": "unshared"}
