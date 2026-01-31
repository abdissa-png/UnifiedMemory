"""
In-memory implementation of KVStoreBackend.
Useful for testing and local development.
"""

from __future__ import annotations

import asyncio
import fnmatch
from typing import Any, Dict, List, Optional
from datetime import datetime

from unified_memory.storage.base import KVStoreBackend


class MemoryKVStore(KVStoreBackend):
    """
    In-memory Key-Value store implementation using a dict.
    
    Thread-safe for async operations using asyncio.Lock.
    Note: TTL (Time To Live) is NOT implemented in this basic version.
    """

    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value by key."""
        async with self._lock:
            # Return a copy to prevent mutation of stored objects
            value = self._store.get(key)
            if value:
                return value.copy()
            return None

    async def set(
        self,
        key: str,
        value: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """Set value."""
        async with self._lock:
            # Store a copy
            self._store[key] = value.copy()
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
        """Atomic set if not exists."""
        async with self._lock:
            if key in self._store:
                return False
            self._store[key] = value.copy()
            return True

    async def compare_and_swap(
        self,
        key: str,
        expected_version: int,
        new_value: Dict[str, Any],
    ) -> bool:
        """Atomic compare and swap."""
        async with self._lock:
            current = self._store.get(key)
            if not current:
                return False
            
            # Check version
            if current.get("version") != expected_version:
                return False
            
            self._store[key] = new_value.copy()
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
            
            if current.get("version") != expected_version:
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
                del self._store[key]
                count += 1
            return count
