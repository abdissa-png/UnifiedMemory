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
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def _configure_application_logging() -> None:
    """Ensure app/module loggers emit through uvicorn or stdout."""
    level_name = os.environ.get("UMS_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
        return

    # uvicorn config usually installs handlers on uvicorn.error only.
    if root_logger.level > level:
        root_logger.setLevel(level)

    uvicorn_error_logger = logging.getLogger("uvicorn.error")
    if uvicorn_error_logger.handlers:
        app_logger = logging.getLogger("unified_memory")
        if not app_logger.handlers:
            app_logger.handlers = list(uvicorn_error_logger.handlers)
        app_logger.setLevel(level)
        app_logger.propagate = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise SystemContext and wire everything at startup."""
    from unified_memory.bootstrap import SystemContext
    from unified_memory.storage.sql.engine import create_sql_engine, create_session_factory, init_db
    from unified_memory.storage.sql.session_manager import ChatSessionManager
    from unified_memory.observability.audit import AuditLogger
    from unified_memory.observability.tracing import set_flush_callback, user_id_for_token_usage
    from unified_memory.namespace.tenant_manager import TenantManager

    config_path = os.environ.get("UMS_CONFIG", "config/app.example.yaml")
    jwt_secret = os.environ.get("UMS_JWT_SECRET", "change-me-in-production")
    token_expire_minutes = int(os.environ.get("UMS_TOKEN_EXPIRE_MINUTES", "60"))
    db_url = os.environ.get("UMS_DATABASE_URL","postgresql+asyncpg://postgres:postgres@localhost:5432/unified_memory_test")
    enable_inngest = os.environ.get("UMS_ENABLE_INNGEST", "").lower() in ("1", "true")
    skip_db_init = os.environ.get("UMS_SKIP_DB_INIT", "").lower() in ("1", "true")
    allow_create_all_fallback = os.environ.get(
        "UMS_SQL_CREATE_ALL_FALLBACK", ""
    ).lower() in ("1", "true")

    logger.info("startup.step begin config_path=%s", config_path)

    # Build system context
    if os.path.exists(config_path):
        ctx = SystemContext.from_config_file(config_path)
    else:
        ctx = SystemContext()
    logger.info("startup.step system_context_built")

    ctx.build_services(enable_inngest=enable_inngest)
    logger.info("startup.step services_built enable_inngest=%s", enable_inngest)

    # SQL engine
    sql_engine = create_sql_engine(db_url)
    sf = create_session_factory(sql_engine)
    logger.info("startup.step sql_engine_ready")
    if skip_db_init:
        logger.warning("startup.step init_db_skipped UMS_SKIP_DB_INIT=true")
    else:
        await init_db(
            sql_engine,
            db_url,
            allow_create_all_fallback=allow_create_all_fallback,
        )
        logger.info("startup.step init_db_complete")
        # Alembic's fileConfig can alter logging; re-apply app handler wiring.
        _configure_application_logging()

    ctx.sql_session_factory = sf
    ctx.chat_session_manager = ChatSessionManager(sf)
    if ctx.ingestion_pipeline is not None:
        ctx.ingestion_pipeline.chat_session_manager = ctx.chat_session_manager
    if ctx.qa_agent is not None:
        ctx.qa_agent.session_manager = ctx.chat_session_manager
    ctx.audit_logger = AuditLogger(sf)
    ctx.tenant_manager = TenantManager(ctx.kv_store,ctx.vector_store)

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
                        user_id=user_id_for_token_usage(trace_ctx),
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
    logger.info("startup.step flush_callback_registered")

    # Store on app state
    app.state.system_context = ctx
    app.state.jwt_secret = jwt_secret
    app.state.token_expire_minutes = token_expire_minutes

    # Register Inngest serve handler (if available/enabled).
    if ctx.inngest_client and ctx.inngest_functions:
        try:
            import inngest.fast_api

            inngest.fast_api.serve(
                app, ctx.inngest_client, ctx.inngest_functions
            )
            logger.info("Inngest functions registered")
        except ImportError:
            logger.warning("inngest.fast_api not available; skipping")

    logger.info("Unified Memory System started")

    yield

    # Shutdown
    await ctx.close()
    await sql_engine.dispose()
    logger.info("Unified Memory System stopped")


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    _configure_application_logging()
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

    try:
        from prometheus_client import make_asgi_app

        app.mount("/metrics", make_asgi_app())
    except ImportError:
        pass

    # Routes
    from unified_memory.api.routes import auth, namespaces, ingestion, search, chat, admin

    app.include_router(auth.router)
    app.include_router(namespaces.router)
    app.include_router(ingestion.router)
    app.include_router(search.router)
    app.include_router(chat.router)
    app.include_router(admin.router)

    @app.get("/health/live")
    async def health_live():
        return {"status": "ok"}

    @app.get("/health")
    @app.get("/health/ready")
    async def health_ready():
        ctx = app.state.system_context
        checks: dict[str, str] = {}

        try:
            await ctx.kv_store.get("__health_probe__")
            checks["kv"] = "ok"
        except Exception as exc:
            checks["kv"] = f"error:{type(exc).__name__}"

        try:
            await ctx.vector_store.list_collections(prefix="__health__")
            checks["vector"] = "ok"
        except Exception as exc:
            checks["vector"] = f"error:{type(exc).__name__}"

        try:
            await ctx.graph_store.query_nodes({}, namespace=None, limit=1)
            checks["graph"] = "ok"
        except Exception as exc:
            checks["graph"] = f"error:{type(exc).__name__}"

        if ctx.elasticsearch_store is not None:
            try:
                ping_result = await ctx.elasticsearch_store._es.ping()
                checks["elasticsearch"] = "ok" if ping_result else "error:ping_failed"
            except Exception as exc:
                checks["elasticsearch"] = f"error:{type(exc).__name__}"

        overall = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
        status_code = 200 if overall == "ok" else 503
        return JSONResponse(
            {"status": overall, "checks": checks},
            status_code=status_code,
        )

    return app


app = create_app()
