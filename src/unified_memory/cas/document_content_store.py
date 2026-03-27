"""
DocumentContentStore — persistent binary content store for original documents.

Mirrors ``ImageContentStore`` but is file-format-aware.  Original uploaded
documents (PDFs, DOCX, etc.) are stored on a durable blob backend
(local FS for dev, pluggable for S3/GCS in prod) and referenced by their
tenant-scoped document hash.

Key design decisions:
- Documents are keyed by ``{tenant_id}/{doc_hash}`` to ensure tenant isolation
  and content-addressable deduplication.
- Original filename and content type are stored as sidecar metadata so the
  file can be served with correct headers on download.
- Idempotent writes: re-uploading the same content for the same tenant is a no-op.

Usage::

    store = LocalFSDocumentContentStore("/data/documents")
    meta = await store.store_document(tenant_id, doc_hash, file_bytes, "report.pdf", "application/pdf")
    data = await store.get_document(tenant_id, doc_hash)
    info = await store.get_document_metadata(tenant_id, doc_hash)
    await store.delete_document(tenant_id, doc_hash)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class DocumentStorageMetadata:
    """Sidecar metadata stored alongside the document bytes."""

    tenant_id: str
    doc_hash: str
    original_filename: str = ""
    content_type: str = "application/octet-stream"
    size_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "doc_hash": self.doc_hash,
            "original_filename": self.original_filename,
            "content_type": self.content_type,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentStorageMetadata":
        return cls(
            tenant_id=data["tenant_id"],
            doc_hash=data["doc_hash"],
            original_filename=data.get("original_filename", ""),
            content_type=data.get("content_type", "application/octet-stream"),
            size_bytes=data.get("size_bytes", 0),
        )


class DocumentContentStore(ABC):
    """Abstract interface for persistent original-document storage."""

    @abstractmethod
    async def store_document(
        self,
        tenant_id: str,
        doc_hash: str,
        data: bytes,
        original_filename: str = "",
        content_type: str = "application/octet-stream",
    ) -> DocumentStorageMetadata:
        """Store original document bytes.  Idempotent — no-op if already stored."""

    @abstractmethod
    async def get_document(self, tenant_id: str, doc_hash: str) -> Optional[bytes]:
        """Retrieve original document bytes by tenant and hash."""

    @abstractmethod
    async def get_document_metadata(
        self, tenant_id: str, doc_hash: str
    ) -> Optional[DocumentStorageMetadata]:
        """Retrieve sidecar metadata (filename, content_type, size)."""

    @abstractmethod
    async def delete_document(self, tenant_id: str, doc_hash: str) -> bool:
        """Delete a stored document.  Returns True if removed."""

    @abstractmethod
    async def document_exists(self, tenant_id: str, doc_hash: str) -> bool:
        """Check if a document exists without reading its bytes."""

    @staticmethod
    def compute_hash(data: bytes) -> str:
        """Compute a raw SHA-256 hash over arbitrary bytes."""
        return hashlib.sha256(data).hexdigest()


class LocalFSDocumentContentStore(DocumentContentStore):
    """File-system backed document store for development.

    Directory layout::

        {base_dir}/{tenant_id}/{doc_hash[:2]}/{doc_hash}.bin
        {base_dir}/{tenant_id}/{doc_hash[:2]}/{doc_hash}.meta.json
    """

    def __init__(self, base_dir: str = "/tmp/memory_documents") -> None:
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def _dir(self, tenant_id: str, doc_hash: str) -> str:
        subdir = doc_hash[:2]
        full_dir = os.path.join(self.base_dir, tenant_id, subdir)
        os.makedirs(full_dir, exist_ok=True)
        return full_dir

    def _data_path(self, tenant_id: str, doc_hash: str) -> str:
        return os.path.join(self._dir(tenant_id, doc_hash), f"{doc_hash}.bin")

    def _meta_path(self, tenant_id: str, doc_hash: str) -> str:
        return os.path.join(self._dir(tenant_id, doc_hash), f"{doc_hash}.meta.json")

    async def store_document(
        self,
        tenant_id: str,
        doc_hash: str,
        data: bytes,
        original_filename: str = "",
        content_type: str = "application/octet-stream",
    ) -> DocumentStorageMetadata:
        data_path = self._data_path(tenant_id, doc_hash)
        meta_path = self._meta_path(tenant_id, doc_hash)

        meta = DocumentStorageMetadata(
            tenant_id=tenant_id,
            doc_hash=doc_hash,
            original_filename=original_filename,
            content_type=content_type,
            size_bytes=len(data),
        )

        # Idempotent: skip if already stored
        if not os.path.exists(data_path):
            with open(data_path, "wb") as fh:
                fh.write(data)
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump(meta.to_dict(), fh)
            logger.debug(
                "Stored document %s for tenant %s (%d bytes)",
                doc_hash,
                tenant_id,
                len(data),
            )
        else:
            logger.debug(
                "Document %s already stored for tenant %s; skipping",
                doc_hash,
                tenant_id,
            )

        return meta

    async def get_document(self, tenant_id: str, doc_hash: str) -> Optional[bytes]:
        path = self._data_path(tenant_id, doc_hash)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as fh:
            return fh.read()

    async def get_document_metadata(
        self, tenant_id: str, doc_hash: str
    ) -> Optional[DocumentStorageMetadata]:
        path = self._meta_path(tenant_id, doc_hash)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            return DocumentStorageMetadata.from_dict(json.load(fh))

    async def delete_document(self, tenant_id: str, doc_hash: str) -> bool:
        data_path = self._data_path(tenant_id, doc_hash)
        meta_path = self._meta_path(tenant_id, doc_hash)
        deleted = False
        if os.path.exists(data_path):
            os.remove(data_path)
            deleted = True
        if os.path.exists(meta_path):
            os.remove(meta_path)
        return deleted

    async def document_exists(self, tenant_id: str, doc_hash: str) -> bool:
        return os.path.exists(self._data_path(tenant_id, doc_hash))


class InMemoryDocumentContentStore(DocumentContentStore):
    """In-memory document store for unit tests."""

    def __init__(self) -> None:
        self._documents: Dict[str, bytes] = {}  # key: "{tenant}:{hash}"
        self._metadata: Dict[str, DocumentStorageMetadata] = {}

    def _key(self, tenant_id: str, doc_hash: str) -> str:
        return f"{tenant_id}:{doc_hash}"

    async def store_document(
        self,
        tenant_id: str,
        doc_hash: str,
        data: bytes,
        original_filename: str = "",
        content_type: str = "application/octet-stream",
    ) -> DocumentStorageMetadata:
        key = self._key(tenant_id, doc_hash)
        meta = DocumentStorageMetadata(
            tenant_id=tenant_id,
            doc_hash=doc_hash,
            original_filename=original_filename,
            content_type=content_type,
            size_bytes=len(data),
        )
        if key not in self._documents:
            self._documents[key] = data
            self._metadata[key] = meta
        return meta

    async def get_document(self, tenant_id: str, doc_hash: str) -> Optional[bytes]:
        return self._documents.get(self._key(tenant_id, doc_hash))

    async def get_document_metadata(
        self, tenant_id: str, doc_hash: str
    ) -> Optional[DocumentStorageMetadata]:
        return self._metadata.get(self._key(tenant_id, doc_hash))

    async def delete_document(self, tenant_id: str, doc_hash: str) -> bool:
        key = self._key(tenant_id, doc_hash)
        if key in self._documents:
            del self._documents[key]
            self._metadata.pop(key, None)
            return True
        return False

    async def document_exists(self, tenant_id: str, doc_hash: str) -> bool:
        return self._key(tenant_id, doc_hash) in self._documents
