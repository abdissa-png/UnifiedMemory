"""Tests for token_usage user_id resolution from namespace."""

from __future__ import annotations

from unified_memory.observability.tracing import TraceUsageContext, user_id_for_token_usage


def test_user_id_from_canonical_namespace():
    ctx = TraceUsageContext(
        trace_id="t1",
        tenant_id="acme",
        namespace="tenant:acme/user:alice/agent:bot",
        user_id="ignored-for-sql",
    )
    assert user_id_for_token_usage(ctx) == "alice"


def test_user_id_fallback_when_namespace_empty():
    ctx = TraceUsageContext(
        trace_id="t2",
        tenant_id="acme",
        namespace="",
        user_id="from-jwt",
    )
    assert user_id_for_token_usage(ctx) == "from-jwt"


def test_user_id_anonymous_when_no_user_segment():
    ctx = TraceUsageContext(
        trace_id="t3",
        tenant_id="acme",
        namespace="tenant:acme/other:foo",
        user_id="",
    )
    assert user_id_for_token_usage(ctx) == "anonymous"
