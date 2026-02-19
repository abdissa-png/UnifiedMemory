"""
Redis implementation of KVStoreBackend.

Uses ``redis.asyncio`` for async access. Each value is stored as a
Redis hash with fields ``data`` (JSON-encoded) and ``version`` (int).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from unified_memory.storage.base import KVStoreBackend, VersionedValue

logger = logging.getLogger(__name__)


class RedisKVStore(KVStoreBackend):
    """
    Redis-backed key-value store with optimistic-locking support.

    Atomic compare-and-swap is implemented via Lua scripts executed
    server-side so that read-modify-write is atomic.
    """

    _CAS_SCRIPT = """
    local key = KEYS[1]
    local expected = tonumber(ARGV[1])
    local new_data = ARGV[2]
    local cur = redis.call('HGET', key, 'version')
    if cur == false then return 0 end
    if tonumber(cur) ~= expected then return 0 end
    local new_ver = expected + 1
    redis.call('HSET', key, 'data', new_data, 'version', new_ver)
    return 1
    """

    _DELETE_IF_VERSION_SCRIPT = """
    local key = KEYS[1]
    local expected = tonumber(ARGV[1])
    local cur = redis.call('HGET', key, 'version')
    if cur == false then return 0 end
    if tonumber(cur) ~= expected then return 0 end
    redis.call('DEL', key)
    return 1
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
    ) -> None:
        try:
            import redis.asyncio as aioredis
        except ImportError as exc:
            raise ImportError(
                "The 'redis' package is required for RedisKVStore. "
                "Install it with: pip install redis"
            ) from exc

        self._redis = aioredis.from_url(url, decode_responses=True)

    async def get(self, key: str) -> Optional[VersionedValue]:
        raw = await self._redis.hgetall(key)
        if not raw:
            return None
        data = json.loads(raw["data"])
        version = int(raw["version"])
        return VersionedValue(data=data, version=version)

    async def set(
        self,
        key: str,
        value: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        cur = await self._redis.hget(key, "version")
        new_version = (int(cur) + 1) if cur else 1
        await self._redis.hset(key, mapping={
            "data": json.dumps(value),
            "version": str(new_version),
        })
        if ttl_seconds:
            await self._redis.expire(key, ttl_seconds)
        return True

    async def delete(self, key: str) -> bool:
        return bool(await self._redis.delete(key))

    async def set_if_not_exists(
        self,
        key: str,
        value: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        created = await self._redis.hsetnx(key, "version", "1")
        if not created:
            return False
        await self._redis.hset(key, "data", json.dumps(value))
        if ttl_seconds:
            await self._redis.expire(key, ttl_seconds)
        return True

    async def compare_and_swap(
        self,
        key: str,
        expected_version: int,
        new_value: Dict[str, Any],
    ) -> bool:
        result = await self._redis.eval(
            self._CAS_SCRIPT,
            1,
            key,
            str(expected_version),
            json.dumps(new_value),
        )
        return bool(result)

    async def delete_if_version(
        self,
        key: str,
        expected_version: int,
    ) -> bool:
        result = await self._redis.eval(
            self._DELETE_IF_VERSION_SCRIPT,
            1,
            key,
            str(expected_version),
        )
        return bool(result)

    async def scan(
        self,
        pattern: str,
        limit: int = 1000,
    ) -> List[str]:
        keys: List[str] = []
        async for key in self._redis.scan_iter(match=pattern, count=limit):
            keys.append(key)
            if len(keys) >= limit:
                break
        return keys

    async def delete_pattern(self, pattern: str) -> int:
        keys = await self.scan(pattern, limit=100_000)
        if not keys:
            return 0
        return await self._redis.delete(*keys)
