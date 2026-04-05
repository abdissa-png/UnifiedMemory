"""
Auth endpoints: tenant registration, user registration, login.
"""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Depends, HTTPException

from unified_memory.api.deps import (
    get_current_user,
    get_jwt_secret,
    get_system_context,
    get_token_expire_minutes,
)
from unified_memory.api.schemas import (
    LoginRequest,
    RegisterTenantRequest,
    RegisterUserRequest,
    TokenResponse,
    RegisterTenantResponse,
)
from unified_memory.auth.jwt_handler import AuthenticatedUser, create_access_token
from unified_memory.auth.password import async_hash_password, async_verify_password

router = APIRouter(prefix="/v1/auth", tags=["auth"])


@router.post("/register-tenant", response_model=RegisterTenantResponse)
async def register_tenant(
    body: RegisterTenantRequest,
    ctx=Depends(get_system_context),
    secret: str = Depends(get_jwt_secret),
    expire_minutes: int = Depends(get_token_expire_minutes),
):
    # Ensure tenant manager is configured
    tenant_mgr = getattr(ctx, "tenant_manager", None)
    if not tenant_mgr:
        raise HTTPException(500, "Tenant Manager is not configured")

    user_id = uuid.uuid4().hex
    tenant_id = uuid.uuid4().hex
    
    # Configure overrides safely from the request
    overrides = {}
    if body.chunk_size is not None:
        overrides["chunk_size"] = body.chunk_size
    if body.chunker_type is not None:
        overrides["chunker_type"] = body.chunker_type
    if body.enable_graph_storage is not None:
        overrides["enable_graph_storage"] = body.enable_graph_storage
        
    text_embedding = None
    if body.text_embedding_provider and body.text_embedding_model:
        from unified_memory.namespace.types import EmbeddingModelConfig
        text_embedding = EmbeddingModelConfig(
            provider=body.text_embedding_provider,
            model=body.text_embedding_model,
            dimension=body.text_embedding_dimension
        )
    vision_embedding = None
    if body.vision_embedding_provider and body.vision_embedding_model:
        from unified_memory.namespace.types import EmbeddingModelConfig
        vision_embedding = EmbeddingModelConfig(
            provider=body.vision_embedding_provider,
            model=body.vision_embedding_model,
            dimension=body.vision_embedding_dimension
        )

    await tenant_mgr.register_tenant(
        tenant_id=tenant_id,
        admin_user_id=user_id,
        text_embedding=text_embedding,
        vision_embedding=vision_embedding,
        **overrides
    )

    # Create admin user in SQL
    sf = getattr(ctx, "sql_session_factory", None)
    if sf:
        from unified_memory.storage.sql.models import User

        async with sf() as db:
            db.add(
                User(
                    id=user_id,
                    tenant_id=tenant_id,
                    email=body.admin_email,
                    display_name=body.admin_display_name or body.admin_email,
                    password_hash=await async_hash_password(body.admin_password),
                    roles_json=json.dumps(["tenant_admin", "tenant_member"]),
                )
            )
            await db.commit()
             
    if getattr(ctx, "audit_logger", None):
        await ctx.audit_logger.log(
            tenant_id=tenant_id,
            user_id=user_id,
            action="tenant.register",
            resource_type="tenant",
            resource_id=tenant_id,
            outcome="success"
        )

    token = create_access_token(
        user_id=user_id,
        tenant_id=tenant_id,
        email=body.admin_email,
        roles=["tenant_admin", "tenant_member"],
        secret=secret,
        expire_minutes=expire_minutes,
    )
    return RegisterTenantResponse(tenant_id=tenant_id, access_token=token)


@router.post("/register-user", response_model=TokenResponse)
async def register_user(
    body: RegisterUserRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    ctx=Depends(get_system_context),
    secret: str = Depends(get_jwt_secret),
    expire_minutes: int = Depends(get_token_expire_minutes),
):
    if "tenant_admin" not in user.roles:
        raise HTTPException(403, "Only tenant admins can register users")

    sf = getattr(ctx, "sql_session_factory", None)
    new_id = uuid.uuid4().hex
    if sf:
        from unified_memory.storage.sql.models import User

        async with sf() as db:
            db.add(
                User(
                    id=new_id,
                    tenant_id=user.tenant_id,
                    email=body.email,
                    display_name=body.display_name or body.email,
                    password_hash=await async_hash_password(body.password),
                    roles_json=json.dumps(body.roles),
                )
            )
            await db.commit()

    if getattr(ctx, "audit_logger", None):
        await ctx.audit_logger.log(
            tenant_id=user.tenant_id,
            user_id=user.user_id,
            action="user.register",
            resource_type="user",
            resource_id=new_id,
            outcome="success"
        )

    token = create_access_token(
        user_id=new_id,
        tenant_id=user.tenant_id,
        email=body.email,
        roles=body.roles,
        secret=secret,
        expire_minutes=expire_minutes,
    )
    return TokenResponse(access_token=token)


@router.post("/login", response_model=TokenResponse)
async def login(
    body: LoginRequest,
    ctx=Depends(get_system_context),
    secret: str = Depends(get_jwt_secret),
    expire_minutes: int = Depends(get_token_expire_minutes),
):
    sf = getattr(ctx, "sql_session_factory", None)
    if not sf:
        raise HTTPException(500, "SQL storage not configured")

    from sqlalchemy import select
    from unified_memory.storage.sql.models import User

    async with sf() as db:
        result = await db.execute(select(User).where(User.email == body.email))
        db_user = result.scalar_one_or_none()

    if not db_user or not await async_verify_password(body.password, db_user.password_hash):
        if getattr(ctx, "audit_logger", None) and db_user:
            await ctx.audit_logger.log(
                tenant_id=db_user.tenant_id,
                user_id=db_user.id,
                action="user.login",
                outcome="failure"
            )
        raise HTTPException(401, "Invalid credentials")

    import json as _json

    roles = _json.loads(db_user.roles_json) if db_user.roles_json else []
    token = create_access_token(
        user_id=db_user.id,
        tenant_id=db_user.tenant_id,
        email=db_user.email,
        roles=roles,
        secret=secret,
        expire_minutes=expire_minutes,
    )
    if getattr(ctx, "audit_logger", None):
        await ctx.audit_logger.log(
            tenant_id=db_user.tenant_id,
            user_id=db_user.id,
            action="user.login",
            outcome="success"
        )

    return TokenResponse(access_token=token)
