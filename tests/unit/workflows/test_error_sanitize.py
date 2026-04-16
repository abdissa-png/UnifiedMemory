"""Tests for workflow error string sanitization."""

from __future__ import annotations

from unified_memory.workflows.error_sanitize import sanitize_workflow_error_text


def test_strips_data_url_base64():
    raw = "422 detail input data:image/png;base64," + "A" * 2000 + " tail"
    out = sanitize_workflow_error_text(raw)
    assert "data:image" not in out
    assert "[redacted base64 image]" in out
    assert "tail" in out


def test_strips_long_raw_base64_run():
    raw = "error: " + "B" * 600
    out = sanitize_workflow_error_text(raw)
    assert "[redacted base64]" in out


def test_truncates_total_length():
    # Avoid a single long run of [A-Za-z0-9+/] so the base64 heuristic does not fire.
    raw = "e:" + ("α" * 20000)
    out = sanitize_workflow_error_text(raw, max_len=100)
    assert len(out) <= 100
    assert "truncated" in out
