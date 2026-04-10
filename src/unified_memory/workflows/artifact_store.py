"""
Artifact Store — externalises large binary and JSON payloads so that
workflow step outputs stay well under orchestrator payload limits.

Every step that produces large data (parsed documents, page images,
chunk manifests) writes to this store and passes only the returned URI
to the next step.

Backends:
- ``LocalFSArtifactStore``  — file-system based; suitable for dev/test.
- (future) S3 / GCS / MinIO — production deployments.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from unified_memory.core.config import DEFAULT_ARTIFACT_DIR
from unified_memory.core.logging import get_logger,log_event
logger = get_logger(__name__)


class ArtifactStore(ABC):
    """Abstract interface for large-payload storage."""

    @abstractmethod
    async def put_bytes(self, data: bytes, key: Optional[str] = None) -> str:
        """Store a binary blob.  Returns an opaque URI."""

    @abstractmethod
    async def put_json(self, data: dict, key: Optional[str] = None) -> str:
        """Store a JSON-serialisable dict.  Returns an opaque URI."""

    @abstractmethod
    async def get_bytes(self, uri: str) -> Optional[bytes]:
        """Retrieve a binary blob by URI."""

    @abstractmethod
    async def get_json(self, uri: str) -> Optional[dict]:
        """Retrieve a JSON dict by URI."""

    @abstractmethod
    async def delete(self, uri: str) -> bool:
        """Delete an artifact.  Returns True if something was removed."""

    async def cleanup_job(self, job_id: str) -> int:
        """Remove all artifacts for a given job.  Returns count deleted."""
        raise NotImplementedError


class LocalFSArtifactStore(ArtifactStore):
    """File-system backed artifact store for development and testing."""

    SCHEME = "local"

    def __init__(self, base_dir: str = DEFAULT_ARTIFACT_DIR) -> None:
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def _resolve_path(self, key: str) -> str:
        return os.path.join(self.base_dir, key)

    def _uri(self, key: str) -> str:
        return f"{self.SCHEME}://{key}"

    def _key_from_uri(self, uri: str) -> str:
        return uri.removeprefix(f"{self.SCHEME}://")

    @staticmethod
    def _content_key(data: bytes, prefix: str = "blob") -> str:
        digest = hashlib.sha256(data).hexdigest()
        return f"{prefix}/{digest}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def put_bytes(self, data: bytes, key: Optional[str] = None) -> str:
        key = key or self._content_key(data)
        path = self._resolve_path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(data)
        return self._uri(key)

    async def put_json(self, data: dict, key: Optional[str] = None) -> str:
        raw = json.dumps(data, default=str, ensure_ascii=False).encode("utf-8")
        key = key or self._content_key(raw, prefix="json")
        path = self._resolve_path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(raw)
        return self._uri(key)

    async def get_bytes(self, uri: str) -> Optional[bytes]:
        key = self._key_from_uri(uri)
        path = self._resolve_path(key)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as fh:
            return fh.read()

    async def get_json(self, uri: str) -> Optional[dict]:
        raw = await self.get_bytes(uri)
        if raw is None:
            return None
        return json.loads(raw)

    async def delete(self, uri: str) -> bool:
        key = self._key_from_uri(uri)
        path = self._resolve_path(key)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    async def cleanup_job(self, job_id: str) -> int:
        """Remove the ``jobs/{job_id}/`` directory tree."""
        job_dir = self._resolve_path(f"jobs/{job_id}")
        if not os.path.isdir(job_dir):
            return 0
        count = 0
        for root, _dirs, files in os.walk(job_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
                count += 1
            os.rmdir(root)
        return count


class InMemoryArtifactStore(ArtifactStore):
    """Pure in-memory artifact store for unit tests."""

    def __init__(self) -> None:
        self._blobs: dict[str, bytes] = {}

    def _uri(self, key: str) -> str:
        return f"mem://{key}"

    def _key_from_uri(self, uri: str) -> str:
        return uri.removeprefix("mem://")

    async def put_bytes(self, data: bytes, key: Optional[str] = None) -> str:
        key = key or f"blob/{hashlib.sha256(data).hexdigest()}"
        self._blobs[key] = data
        return self._uri(key)

    async def put_json(self, data: dict, key: Optional[str] = None) -> str:
        raw = json.dumps(data, default=str, ensure_ascii=False).encode("utf-8")
        key = key or f"json/{hashlib.sha256(raw).hexdigest()}"
        self._blobs[key] = raw
        return self._uri(key)

    async def get_bytes(self, uri: str) -> Optional[bytes]:
        return self._blobs.get(self._key_from_uri(uri))

    async def get_json(self, uri: str) -> Optional[dict]:
        raw = await self.get_bytes(uri)
        return json.loads(raw) if raw else None

    async def delete(self, uri: str) -> bool:
        key = self._key_from_uri(uri)
        if key in self._blobs:
            del self._blobs[key]
            return True
        return False

    async def cleanup_job(self, job_id: str) -> int:
        prefix = f"jobs/{job_id}/"
        keys_to_delete = [k for k in self._blobs if k.startswith(prefix)]
        for k in keys_to_delete:
            del self._blobs[k]
        return len(keys_to_delete)
