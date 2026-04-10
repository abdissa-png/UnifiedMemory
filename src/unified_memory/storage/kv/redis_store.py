"""
Redis implementation of KVStoreBackend.

Uses ``redis.asyncio`` for async access. Each value is stored as a
Redis hash with fields ``data`` (JSON-encoded) and ``version`` (int).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from unified_memory.core.config import DEFAULT_REDIS_URL
from unified_memory.core.logging import get_logger, log_event
from unified_memory.core.resilience import external_call
from unified_memory.storage.base import KVStoreBackend, VersionedValue

logger = get_logger(__name__)


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

    _SET_SCRIPT = """
    local key = KEYS[1]
    local new_data = ARGV[1]
    local ttl = tonumber(ARGV[2])
    local cur = redis.call('HGET', key, 'version')
    local new_ver = cur and (tonumber(cur) + 1) or 1
    redis.call('HSET', key, 'data', new_data, 'version', new_ver)
    if ttl > 0 then redis.call('EXPIRE', key, ttl) end
    return new_ver
    """

    _SET_IF_NOT_EXISTS_SCRIPT = """
    local key = KEYS[1]
    local new_data = ARGV[1]
    local ttl = tonumber(ARGV[2])
    if redis.call('EXISTS', key) == 1 then return 0 end
    redis.call('HSET', key, 'data', new_data, 'version', 1)
    if ttl > 0 then redis.call('EXPIRE', key, ttl) end
    return 1
    """

    def __init__(
        self,
        url: str = DEFAULT_REDIS_URL,
    ) -> None:
        try:
            import redis.asyncio as aioredis
        except ImportError as exc:
            raise ImportError(
                "The 'redis' package is required for RedisKVStore. "
                "Install it with: pip install redis"
            ) from exc

        self._redis = aioredis.from_url(url or DEFAULT_REDIS_URL, decode_responses=True)

    @external_call()
    async def get(self, key: str) -> Optional[VersionedValue]:
        raw = await self._redis.hgetall(key)
        if not raw:
            return None
        data = json.loads(raw["data"])
        version = int(raw["version"])
        return VersionedValue(data=data, version=version)

    @external_call()
    async def set(
        self,
        key: str,
        value: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        ttl = ttl_seconds or 0
        await self._redis.eval(
            self._SET_SCRIPT,
            1,
            key,
            json.dumps(value),
            str(ttl),
        )
        return True

    @external_call()
    async def delete(self, key: str) -> bool:
        return bool(await self._redis.delete(key))

    @external_call()
    async def set_if_not_exists(
        self,
        key: str,
        value: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        ttl = ttl_seconds or 0
        created = await self._redis.eval(
            self._SET_IF_NOT_EXISTS_SCRIPT,
            1,
            key,
            json.dumps(value),
            str(ttl),
        )
        return bool(created)

    @external_call()
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

    @external_call()
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

    @external_call()
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

    @external_call()
    async def delete_pattern(self, pattern: str) -> int:
        keys = await self.scan(pattern, limit=100_000)
        if not keys:
            return 0
        return await self._redis.delete(*keys)

    async def close(self) -> None:
        try:
            await self._redis.aclose()
        except AttributeError:
            log_event(logger, 10, "redis_close_skipped")
