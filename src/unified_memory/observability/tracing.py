"""
Tracing, usage recording, and the @traced decorator.

Providers call ``record_usage()`` after each external API call.  The
``@traced`` decorator manages a ContextVar-based ``TraceUsageContext``
that accumulates records for the current operation tree, logs structured
spans, emits Prometheus metrics (when available), and flushes token
usage to SQL at the root span boundary.
"""

from __future__ import annotations

import functools
import logging
import time
import uuid
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from unified_memory.core.logging import bind_log_context
from unified_memory.core.utils import utc_now

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus guards
# ---------------------------------------------------------------------------

PROMETHEUS_AVAILABLE = False
try:
    from prometheus_client import Counter, Histogram

    operation_duration = Histogram(
        "memory_operation_duration_seconds",
        "Duration of traced operations",
        ["operation"],
    )
    token_counter = Counter(
        "memory_tokens_total",
        "Tokens consumed by external API calls",
        ["model", "token_type"],
    )
    api_call_counter = Counter(
        "memory_api_calls_total",
        "Count of external API calls",
        ["service", "model", "operation"],
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class UsageRecord:
    """Raw token / unit counts from a single external API call."""

    service: str  # "openai", "cohere", "local"
    model: str  # "text-embedding-3-small", "gpt-4o-mini"
    operation: str  # "embedding", "completion", "rerank", "extraction"

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    search_units: int = 0  # Cohere reranking

    duration_ms: float = 0.0
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = utc_now().isoformat()


@dataclass
class TraceUsageContext:
    """Accumulates UsageRecords for the duration of a root span."""

    trace_id: str
    tenant_id: str = ""
    namespace: str = ""
    records: List[UsageRecord] = field(default_factory=list)


# Request-scoped context vars
_usage_context: ContextVar[Optional[TraceUsageContext]] = ContextVar(
    "_usage_context", default=None
)
_request_tenant_id: ContextVar[str] = ContextVar("_request_tenant_id", default="")
_request_namespace: ContextVar[str] = ContextVar("_request_namespace", default="")

# SQL flush callback — set by bootstrap when SQL is available
_flush_callback: Optional[Callable] = None


def set_flush_callback(cb: Callable) -> None:
    """Register the async callback used to persist usage records to SQL."""
    global _flush_callback
    _flush_callback = cb


def set_request_context(*, tenant_id: str = "", namespace: str = "") -> None:
    """Set request-scoped tenant/namespace (called by middleware)."""
    if tenant_id:
        _request_tenant_id.set(tenant_id)
    if namespace:
        _request_namespace.set(namespace)
    bind_log_context(tenant_id=tenant_id, namespace=namespace)


def record_usage(record: UsageRecord) -> None:
    """Deposit a usage record into the current trace context (no-op if none)."""
    ctx = _usage_context.get()
    if ctx is not None:
        ctx.records.append(record)


# ---------------------------------------------------------------------------
# @traced decorator
# ---------------------------------------------------------------------------


def traced(operation_name: str):
    """Decorator that creates a traced span around an async function.

    - Root spans create a new ``TraceUsageContext``.
    - Child spans share the parent's context.
    - On completion the span is logged with duration + usage.
    - Prometheus metrics are emitted when available.
    - At root span exit, accumulated records are flushed to SQL.
    """

    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            parent_ctx = _usage_context.get()
            is_root = parent_ctx is None

            if is_root:
                ctx = TraceUsageContext(
                    trace_id=uuid.uuid4().hex,
                    tenant_id=_request_tenant_id.get(""),
                    namespace=_request_namespace.get(""),
                )
                _usage_context.set(ctx)
            else:
                ctx = parent_ctx

            records_before = len(ctx.records)
            start = time.perf_counter()

            try:
                result = await fn(*args, **kwargs)
                duration_ms = (time.perf_counter() - start) * 1000
                new_records = ctx.records[records_before:]

                _emit_span_log(
                    operation_name, ctx, duration_ms, new_records, error=None
                )
                _emit_prometheus(operation_name, duration_ms, new_records)

                if is_root:
                    await _flush_to_sql(ctx)
                    _usage_context.set(None)

                return result

            except Exception as exc:
                duration_ms = (time.perf_counter() - start) * 1000
                new_records = ctx.records[records_before:]

                _emit_span_log(
                    operation_name, ctx, duration_ms, new_records, error=str(exc)
                )

                if is_root:
                    await _flush_to_sql(ctx)
                    _usage_context.set(None)

                raise

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _emit_span_log(
    operation: str,
    ctx: TraceUsageContext,
    duration_ms: float,
    records: List[UsageRecord],
    error: Optional[str],
) -> None:
    extra: Dict[str, Any] = {
        "trace_id": ctx.trace_id,
        "operation": operation,
        "duration_ms": round(duration_ms, 2),
        "tenant_id": ctx.tenant_id,
        "namespace": ctx.namespace,
    }
    if records:
        extra["usage"] = [asdict(r) for r in records]
    if error:
        extra["error"] = error
        logger.error("trace.span.failed", extra=extra)
    else:
        logger.info("trace.span.complete", extra=extra)


def _emit_prometheus(
    operation: str,
    duration_ms: float,
    records: List[UsageRecord],
) -> None:
    if not PROMETHEUS_AVAILABLE:
        return
    operation_duration.labels(operation=operation).observe(duration_ms / 1000)
    for r in records:
        api_call_counter.labels(
            service=r.service, model=r.model, operation=r.operation
        ).inc()
        if r.input_tokens:
            token_counter.labels(model=r.model, token_type="input").inc(
                r.input_tokens
            )
        if r.output_tokens:
            token_counter.labels(model=r.model, token_type="output").inc(
                r.output_tokens
            )
        if r.reasoning_tokens:
            token_counter.labels(model=r.model, token_type="reasoning").inc(
                r.reasoning_tokens
            )


async def _flush_to_sql(ctx: TraceUsageContext) -> None:
    if not ctx.records or _flush_callback is None:
        return
    try:
        await _flush_callback(ctx)
    except Exception:
        logger.exception("Failed to flush usage records to SQL")
