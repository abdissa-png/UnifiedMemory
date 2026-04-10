"""
Shared retry helpers for external service calls.
"""

from __future__ import annotations

from typing import Type

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


def external_call(*exceptions: Type[BaseException]):
    """Retry transient external failures with exponential backoff."""
    retryable = exceptions or (ConnectionError, TimeoutError, OSError)
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=10),
        retry=retry_if_exception_type(retryable),
        reraise=True,
    )
