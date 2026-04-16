"""Bound size and redact sensitive payloads in workflow-visible error text."""

from __future__ import annotations

import re

_DEFAULT_MAX_LEN = 8000


def sanitize_workflow_error_text(
    message: str,
    *,
    max_len: int = _DEFAULT_MAX_LEN,
) -> str:
    """Strip embedded base64 / image payloads and cap length for durable step I/O.

    Covers OpenAI-style 422 bodies where ``detail`` embeds a non-string ``input``
    list with ``image_url`` / ``data:`` URLs.
    """
    if not message:
        return ""
    s = str(message)
    # Standard data URLs
    s = re.sub(
        r"data:image/[^;]+;base64,[A-Za-z0-9+/=\s]+",
        "[redacted base64 image]",
        s,
        flags=re.IGNORECASE,
    )
    # JSON-ish image_url / url fields (quoted or not)
    s = re.sub(
        r"(image_url|url)\s*[:=]\s*['\"]?data:image/[^'\"\s]+",
        r"\1=[redacted base64 image]",
        s,
        flags=re.IGNORECASE,
    )
    # Long base64 runs without a data: prefix (defensive)
    s = re.sub(
        r"[A-Za-z0-9+/]{512,}={0,2}",
        "[redacted base64]",
        s,
    )
    if len(s) > max_len:
        s = s[: max_len - 20] + "…[truncated]"
    return s
