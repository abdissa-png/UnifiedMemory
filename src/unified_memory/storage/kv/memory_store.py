"""
In-memory implementation of KVStoreBackend.
Useful for testing and local development.
"""

from __future__ import annotations

import asyncio
import fnmatch
from typing import Any, Dict, List, Optional
from datetime import datetime

from unified_memory.storage.base import KVStoreBackend, VersionedValue


class MemoryKVStore(KVStoreBackend):
    """
    In-memory Key-Value store implementation using a dict.
    
    Thread-safe for async operations using asyncio.Lock.
    Note: TTL (Time To Live) is NOT implemented in this basic version.
    """

    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[VersionedValue]:
        """Get value by key."""
        async with self._lock:
            entry = self._store.get(key)
            if entry:
                return VersionedValue(
                    data=entry["data"].copy(),
                    version=entry["version"]
                )
            return None

    async def set(
        self,
        key: str,
        value: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """Set value and increment version."""
        async with self._lock:
            current = self._store.get(key, {"version": 0})
            new_version = current["version"] + 1
            self._store[key] = {
                "data": value.copy(),
                "version": new_version
            }
            return True

    async def delete(self, key: str) -> bool:
        """Delete key."""
        async with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    async def set_if_not_exists(
        self,
        key: str,
        value: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """Atomic set if not exists. Initializes version to 1."""
        async with self._lock:
            if key in self._store:
                return False
            self._store[key] = {
                "data": value.copy(),
                "version": 1
            }
            return True

    async def compare_and_swap(
        self,
        key: str,
        expected_version: int,
        new_value: Dict[str, Any],
    ) -> bool:
        """Atomic compare and swap. Increments version on success."""
        async with self._lock:
            current = self._store.get(key)
            if not current:
                return False
            
            if current["version"] != expected_version:
                return False
            
            self._store[key] = {
                "data": new_value.copy(),
                "version": current["version"] + 1
            }
            return True

    async def delete_if_version(
        self,
        key: str,
        expected_version: int,
    ) -> bool:
        """Atomic delete if version matches."""
        async with self._lock:
            current = self._store.get(key)
            if not current:
                return False
            
            if current["version"] != expected_version:
                return False
            
            del self._store[key]
            return True

    async def scan(
        self,
        pattern: str,
        limit: int = 1000,
    ) -> List[str]:
        """Scan keys matching pattern."""
        async with self._lock:
            matches = fnmatch.filter(self._store.keys(), pattern)
            return matches[:limit]

    async def delete_pattern(self, pattern: str) -> int:
        """Delete keys matching pattern."""
        async with self._lock:
            matches = fnmatch.filter(self._store.keys(), pattern)
            count = 0
            for key in matches:
                data = self._store.get(key)
                if data:
                    # We store data/version dict internally
                    pass
                del self._store[key]
                count += 1
            return count
