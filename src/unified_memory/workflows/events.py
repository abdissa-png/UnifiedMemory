"""
Inngest event name constants and typed payload descriptors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Event names
DOCUMENT_UPLOADED = "document/uploaded"
DOCUMENT_DELETE_REQUESTED = "document/delete-requested"


@dataclass
class IngestEventData:
    """Payload shape for ``document/uploaded`` events."""

    tenant_id: str
    namespace: str
    document_id: str
    job_id: str = ""
    source_path: Optional[str] = None
    source_text: Optional[str] = None
    title: Optional[str] = None
    original_filename: Optional[str] = None
    content_type: Optional[str] = None
    session_id: Optional[str] = None
    options: Optional[dict] = None


@dataclass
class DeleteEventData:
    """Payload shape for ``document/delete-requested`` events."""

    tenant_id: str
    namespace: str
    doc_hash: str
    job_id: str
