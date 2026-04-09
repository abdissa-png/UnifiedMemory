"""
Admin and GDPR endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from unified_memory.api.deps import get_current_user, get_system_context
from unified_memory.api.schemas import (
    EmbeddingModelResponse,
    TenantConfigResponse,
    UpdateTenantConfigRequest,
)
from unified_memory.auth.jwt_handler import AuthenticatedUser
from unified_memory.core.exceptions import TenantConfigNotFoundError

router = APIRouter(prefix="/v1/admin", tags=["admin"])


def _require_admin(user: AuthenticatedUser) -> None:
    if "tenant_admin" not in user.roles:
        raise HTTPException(403, "Tenant admin role required")


def _cfg_to_response(cfg) -> TenantConfigResponse:
    """Map a TenantConfig dataclass to TenantConfigResponse schema."""
    text_emb = None
    if cfg.text_embedding:
        text_emb = EmbeddingModelResponse(
            provider=cfg.text_embedding.provider,
            model=cfg.text_embedding.model,
            dimension=cfg.text_embedding.dimension,
        )
    vision_emb = None
    if cfg.vision_embedding:
        vision_emb = EmbeddingModelResponse(
            provider=cfg.vision_embedding.provider,
            model=cfg.vision_embedding.model,
            dimension=cfg.vision_embedding.dimension,
        )
    return TenantConfigResponse(
        tenant_id=cfg.tenant_id,
        text_embedding=text_emb,
        vision_embedding=vision_emb,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        chunker_type=cfg.chunker_type,
        enable_graph_storage=cfg.enable_graph_storage,
        enable_visual_indexing=cfg.enable_visual_indexing,
        enable_entity_extraction=cfg.enable_entity_extraction,
        enable_relation_extraction=cfg.enable_relation_extraction,
        batch_size=cfg.batch_size,
        deduplication_enabled=cfg.deduplication_enabled,
        llm=cfg.llm,
        created_at=cfg.created_at,
        updated_at=cfg.updated_at,
    )


@router.get("/tenants/{tenant_id}", response_model=TenantConfigResponse)
async def get_tenant_config(
    tenant_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
    ctx=Depends(get_system_context),
):
    _require_admin(user)
    if user.tenant_id != tenant_id:
        raise HTTPException(403, "Cross-tenant access denied")

    cfg = await ctx.tenant_manager.get_tenant_config(tenant_id)
    if not cfg:
        raise TenantConfigNotFoundError(f"Tenant config not found for tenant {tenant_id}")
    return _cfg_to_response(cfg)


@router.put("/tenants/{tenant_id}", response_model=TenantConfigResponse)
async def update_tenant_config(
    tenant_id: str,
    body: UpdateTenantConfigRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    ctx=Depends(get_system_context),
):
    _require_admin(user)
    if user.tenant_id != tenant_id:
        raise HTTPException(403, "Cross-tenant access denied")

    cfg = await ctx.tenant_manager.get_tenant_config(tenant_id)
    if not cfg:
        raise TenantConfigNotFoundError(f"Tenant config not found for tenant {tenant_id}")

    # Simple scalar overrides
    for field in (
        "chunk_size", "chunk_overlap", "chunker_type",
        "enable_graph_storage", "enable_visual_indexing",
        "enable_entity_extraction", "enable_relation_extraction",
        "batch_size", "deduplication_enabled",
    ):
        val = getattr(body, field, None)
        if val is not None:
            setattr(cfg, field, val)

    # Embedding overrides
    from unified_memory.namespace.types import EmbeddingModelConfig

    if body.text_embedding_provider and body.text_embedding_model:
        cfg.text_embedding = EmbeddingModelConfig(
            provider=body.text_embedding_provider,
            model=body.text_embedding_model,
            dimension=body.text_embedding_dimension or (cfg.text_embedding.dimension if cfg.text_embedding else 1536),
        )
    if body.vision_embedding_provider and body.vision_embedding_model:
        cfg.vision_embedding = EmbeddingModelConfig(
            provider=body.vision_embedding_provider,
            model=body.vision_embedding_model,
            dimension=body.vision_embedding_dimension or (cfg.vision_embedding.dimension if cfg.vision_embedding else 512),
        )

    # LLM override
    if body.llm_provider and body.llm_model:
        cfg.llm = {"provider": body.llm_provider, "model": body.llm_model}

    from dataclasses import asdict
    from unified_memory.core.types import utc_now

    cfg.updated_at = utc_now().isoformat()
    data = asdict(cfg)
    await ctx.kv_store.set(f"tenant_config:{tenant_id}", data)

    if getattr(ctx, "audit_logger", None):
        await ctx.audit_logger.log(
            tenant_id=user.tenant_id,
            user_id=user.user_id,
            action="tenant.update",
            resource_type="tenant",
            resource_id=user.tenant_id,
            outcome="success",
        )

    return _cfg_to_response(cfg)


@router.delete("/users/{user_id}/data")
async def gdpr_erase(
    user_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
    ctx=Depends(get_system_context),
):
    """GDPR right-to-erasure: delete all user data."""
    _require_admin(user)

    user_ns_key = f"idx:user_namespaces:{user.tenant_id}:{user_id}"
    versioned_own = await ctx.kv_store.get(user_ns_key)
    ns_ids = versioned_own.data.get("namespaces", []) if versioned_own else []
    
    deleted_ns = 0
    for ns_id in ns_ids:
        # Note: A complete GDPR wipe should also trigger data removal in vector
        # databases, CAS, and chat sessions. The delete_namespace method removes 
        # the config and secondary KV indexes, but currently, full systemic cleanup
        # requires an async workflow.
        await ctx.namespace_manager.delete_namespace(ns_id)
        deleted_ns += 1

    # Delete user from SQL
    sf = getattr(ctx, "sql_session_factory", None)
    if sf:
        from sqlalchemy import delete as sa_delete
        from unified_memory.storage.sql.models import User

        async with sf() as db:
            await db.execute(sa_delete(User).where(User.id == user_id))
            await db.commit()

    if getattr(ctx, "audit_logger", None):
        await ctx.audit_logger.log(
            tenant_id=user.tenant_id,
            user_id=user.user_id,
            action="admin.gdpr_erase",
            resource_type="user",
            resource_id=user_id,
            details={"namespaces_deleted": deleted_ns},
            outcome="success",
        )

    return {"status": "erased", "namespaces_deleted": deleted_ns}


@router.get("/users/{user_id}/data")
async def gdpr_export(
    user_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
    ctx=Depends(get_system_context),
):
    """GDPR data export: return summary of user's data."""
    _require_admin(user)

    user_ns_key = f"idx:user_namespaces:{user.tenant_id}:{user_id}"
    versioned_own = await ctx.kv_store.get(user_ns_key)
    ns_ids = versioned_own.data.get("namespaces", []) if versioned_own else []
    
    if getattr(ctx, "audit_logger", None):
        await ctx.audit_logger.log(
            tenant_id=user.tenant_id,
            user_id=user.user_id,
            action="admin.gdpr_export",
            resource_type="user",
            resource_id=user_id,
            details={"namespaces_count": len(ns_ids)},
            outcome="success",
        )
        
    return {
        "user_id": user_id,
        "tenant_id": user.tenant_id,
        "namespaces": ns_ids,
        "namespace_count": len(ns_ids),
        # Note: True GDPR export should generate a comprehensive file map
        # and trigger a workflow to gather and package vector and text data.
    }
