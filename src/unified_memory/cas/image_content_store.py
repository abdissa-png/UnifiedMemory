"""
ImageContentStore — persistent binary content store for images.

Mirrors ``ContentStore`` but is binary-native.  Images are stored on a
durable blob backend (local FS for dev, pluggable for S3/GCS in prod)
and referenced by their SHA-256 hash.  CAS tracks ownership and
ref-counting so images are cleaned up when no documents reference them.

Usage::

    store = LocalFSImageContentStore("/data/images")
    await store.store_image(image_hash, image_bytes)
    img = await store.get_image(image_hash)
    await store.delete_image(image_hash)
"""

from __future__ import annotations

import hashlib
import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class ImageContentStore(ABC):
    """Abstract interface for persistent image storage."""

    @abstractmethod
    async def store_image(self, image_hash: str, data: bytes) -> str:
        """Store image bytes.  Returns a content_id / path."""

    @abstractmethod
    async def get_image(self, image_hash: str) -> Optional[bytes]:
        """Retrieve image bytes by hash."""

    @abstractmethod
    async def delete_image(self, image_hash: str) -> bool:
        """Delete an image.  Returns True if removed."""

    @staticmethod
    def compute_hash(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()


class LocalFSImageContentStore(ImageContentStore):
    """File-system backed image store for development."""

    def __init__(self, base_dir: str = "/tmp/memory_images") -> None:
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def _path(self, image_hash: str) -> str:
        subdir = image_hash[:2]
        full_dir = os.path.join(self.base_dir, subdir)
        os.makedirs(full_dir, exist_ok=True)
        return os.path.join(full_dir, f"{image_hash}.bin")

    async def store_image(self, image_hash: str, data: bytes) -> str:
        path = self._path(image_hash)
        if not os.path.exists(path):
            with open(path, "wb") as fh:
                fh.write(data)
        return f"image:{image_hash}"

    async def get_image(self, image_hash: str) -> Optional[bytes]:
        path = self._path(image_hash)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as fh:
            return fh.read()

    async def delete_image(self, image_hash: str) -> bool:
        path = self._path(image_hash)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False


class InMemoryImageContentStore(ImageContentStore):
    """In-memory image store for unit tests."""

    def __init__(self) -> None:
        self._images: dict[str, bytes] = {}

    async def store_image(self, image_hash: str, data: bytes) -> str:
        self._images[image_hash] = data
        return f"image:{image_hash}"

    async def get_image(self, image_hash: str) -> Optional[bytes]:
        return self._images.get(image_hash)

    async def delete_image(self, image_hash: str) -> bool:
        if image_hash in self._images:
            del self._images[image_hash]
            return True
        return False
