"""
FastAPI application factory.

Usage::

    uvicorn unified_memory.api.app:app --reload
"""

from __future__ import annotations

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise SystemContext and wire everything at startup."""
    from unified_memory.bootstrap import SystemContext
    from unified_memory.storage.sql.engine import create_sql_engine, create_session_factory, init_db
    from unified_memory.storage.sql.session_manager import ChatSessionManager
    from unified_memory.observability.audit import AuditLogger
    from unified_memory.observability.tracing import set_flush_callback
    from unified_memory.namespace.tenant_manager import TenantManager

    config_path = os.environ.get("UMS_CONFIG", "config/app.example.yaml")
    jwt_secret = os.environ.get("UMS_JWT_SECRET", "change-me-in-production")
    token_expire_minutes = int(os.environ.get("UMS_TOKEN_EXPIRE_MINUTES", "60"))
    db_url = os.environ.get("UMS_DATABASE_URL", "sqlite+aiosqlite:///./memory_system.db")
    enable_inngest = os.environ.get("UMS_ENABLE_INNGEST", "").lower() in ("1", "true")

    # Build system context
    if os.path.exists(config_path):
        ctx = SystemContext.from_config_file(config_path)
    else:
        ctx = SystemContext()

    ctx.build_services(enable_inngest=enable_inngest)

    # SQL engine
    sql_engine = create_sql_engine(db_url)
    sf = create_session_factory(sql_engine)
    await init_db(sql_engine)

    ctx.sql_session_factory = sf
    ctx.chat_session_manager = ChatSessionManager(sf)
    ctx.audit_logger = AuditLogger(sf)
    ctx.tenant_manager = TenantManager(ctx.kv_store)

    # Wire usage flush callback
    async def _flush_usage(trace_ctx):
        import uuid as _uuid
        from unified_memory.storage.sql.models import TokenUsageRecord

        async with sf() as db:
            for r in trace_ctx.records:
                db.add(
                    TokenUsageRecord(
                        id=_uuid.uuid4().hex,
                        trace_id=trace_ctx.trace_id,
                        tenant_id=trace_ctx.tenant_id,
                        namespace=trace_ctx.namespace,
                        service=r.service,
                        model=r.model,
                        operation=r.operation,
                        input_tokens=r.input_tokens,
                        output_tokens=r.output_tokens,
                        reasoning_tokens=r.reasoning_tokens,
                        cache_read_tokens=r.cache_read_tokens,
                        cache_creation_tokens=r.cache_creation_tokens,
                        search_units=r.search_units,
                        duration_ms=r.duration_ms,
                    )
                )
            await db.commit()

    set_flush_callback(_flush_usage)

    # Store on app state
    app.state.system_context = ctx
    app.state.jwt_secret = jwt_secret
    app.state.token_expire_minutes = token_expire_minutes

    logger.info("Unified Memory System started")

    yield

    # Shutdown
    await sql_engine.dispose()
    logger.info("Unified Memory System stopped")


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    app = FastAPI(
        title="Unified Memory System",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    from unified_memory.api.middleware.context import RequestContextMiddleware

    app.add_middleware(RequestContextMiddleware)

    # Routes
    from unified_memory.api.routes import auth, namespaces, ingestion, search, chat, admin

    app.include_router(auth.router)
    app.include_router(namespaces.router)
    app.include_router(ingestion.router)
    app.include_router(search.router)
    app.include_router(chat.router)
    app.include_router(admin.router)

    # Inngest serve handler (if available)
    @app.on_event("startup")
    async def _register_inngest():
        ctx = app.state.system_context
        if ctx.inngest_client and ctx.inngest_functions:
            try:
                import inngest.fast_api

                inngest.fast_api.serve(
                    app, ctx.inngest_client, ctx.inngest_functions
                )
                logger.info("Inngest functions registered")
            except ImportError:
                logger.warning("inngest.fast_api not available; skipping")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


app = create_app()
