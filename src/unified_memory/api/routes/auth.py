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


def _ensure_pair(value_a: str | None, value_b: str | None, label_a: str, label_b: str) -> None:
    if bool(value_a) != bool(value_b):
        raise HTTPException(400, f"Provide both {label_a} and {label_b} together")


def _ensure_embedding_registered(ctx, provider: str, model: str, *, vision: bool = False) -> None:
    registry = getattr(ctx, "provider_registry", None)
    if not registry:
        return

    if vision:
        embedder = registry.resolve_vision_embedding_provider(provider, model)
    else:
        embedder = registry.resolve_embedding_provider(provider, model)

    if embedder is None:
        kind = "vision embedding" if vision else "text embedding"
        raise HTTPException(
            400,
            (
                f"{kind} provider '{provider}:{model}' is not registered. "
                "Specify a configured provider/model pair for the tenant."
            ),
        )


def _ensure_llm_registered(ctx, provider: str, model: str) -> None:
    registry = getattr(ctx, "provider_registry", None)
    if not registry:
        return
    if registry.get_llm_provider(f"{provider}:{model}") is None:
        raise HTTPException(
            400,
            f"LLM provider '{provider}:{model}' is not registered. Specify a configured provider/model pair.",
        )


def _ensure_extractor_registered(ctx, extractor_key: str) -> None:
    registry = getattr(ctx, "provider_registry", None)
    if not registry:
        return
    if registry.get_extractor(extractor_key) is None:
        raise HTTPException(
            400,
            (
                f"Extractor '{extractor_key}' is not registered. "
                "Specify a configured extractor key/type."
            ),
        )


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

    _ensure_pair(
        body.text_embedding_provider,
        body.text_embedding_model,
        "text_embedding_provider",
        "text_embedding_model",
    )
    _ensure_pair(
        body.vision_embedding_provider,
        body.vision_embedding_model,
        "vision_embedding_provider",
        "vision_embedding_model",
    )
    _ensure_pair(body.llm_provider, body.llm_model, "llm_provider", "llm_model")

    effective_text_provider = body.text_embedding_provider or "openai"
    effective_text_model = body.text_embedding_model or "text-embedding-3-small"
    _ensure_embedding_registered(ctx, effective_text_provider, effective_text_model)

    enable_visual_indexing = (
        body.enable_visual_indexing if body.enable_visual_indexing is not None else True
    )
    if enable_visual_indexing:
        effective_vision_provider = body.vision_embedding_provider or "openai"
        effective_vision_model = body.vision_embedding_model or "clip-vit-base-patch32"
        _ensure_embedding_registered(
            ctx,
            effective_vision_provider,
            effective_vision_model,
            vision=True,
        )

    if body.llm_provider and body.llm_model:
        _ensure_llm_registered(ctx, body.llm_provider, body.llm_model)

    effective_graph_storage = (
        body.enable_graph_storage if body.enable_graph_storage is not None else True
    )
    effective_entity_extraction = (
        body.enable_entity_extraction
        if body.enable_entity_extraction is not None
        else True
    )
    effective_relation_extraction = (
        body.enable_relation_extraction
        if body.enable_relation_extraction is not None
        else True
    )
    effective_extractor_type = body.extraction_extractor_type or "llm"
    if (
        effective_graph_storage
        and (effective_entity_extraction or effective_relation_extraction)
    ):
        _ensure_extractor_registered(ctx, effective_extractor_type)

    # Configure overrides safely from the request
    overrides = {}
    for field in (
        "chunk_size",
        "chunk_overlap",
        "chunker_type",
        "enable_graph_storage",
        "enable_visual_indexing",
        "enable_entity_extraction",
        "enable_relation_extraction",
        "batch_size",
        "deduplication_enabled",
    ):
        value = getattr(body, field, None)
        if value is not None:
            overrides[field] = value

    extraction_overrides = {}
    if body.extraction_extractor_type:
        extraction_overrides["extractor_type"] = body.extraction_extractor_type
    if body.extraction_llm_model:
        extraction_overrides["llm_model"] = body.extraction_llm_model
    if body.extraction_entity_types is not None:
        extraction_overrides["entity_types"] = body.extraction_entity_types
    if body.extraction_relation_types is not None:
        extraction_overrides["relation_types"] = body.extraction_relation_types
    if body.extraction_confidence_threshold is not None:
        extraction_overrides["confidence_threshold"] = body.extraction_confidence_threshold
    if body.extraction_batch_size is not None:
        extraction_overrides["batch_size"] = body.extraction_batch_size
    if body.extraction_strict_type_filtering is not None:
        extraction_overrides["strict_type_filtering"] = body.extraction_strict_type_filtering
    if extraction_overrides:
        from unified_memory.namespace.types import ExtractionConfig

        overrides["extraction"] = ExtractionConfig(**extraction_overrides)

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
    if body.llm_provider and body.llm_model:
        overrides["llm"] = {"provider": body.llm_provider, "model": body.llm_model}

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
