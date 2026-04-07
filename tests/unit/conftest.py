"""
Unit-test helpers.

- CI sets REDIS_URL after starting Redis (see .github/workflows/ci.yml).
- Locally, run with Docker available and::

    UMS_START_REDIS_DOCKER=1 pytest tests/unit/storage/test_kv_store.py -k redis

  to spawn an ephemeral ``redis:7-alpine`` on a random host port and set REDIS_URL.
"""

from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import time
import uuid

import pytest
import pytest_asyncio

_UMS_REDIS_CONTAINER_ID: str | None = None


def _docker_rm(cid: str) -> None:
    docker = shutil.which("docker")
    if not docker or not cid:
        return
    subprocess.run([docker, "rm", "-f", cid], capture_output=True, check=False)


def _start_ephemeral_redis() -> str | None:
    """Return redis:// URL or None if Docker is unavailable."""
    global _UMS_REDIS_CONTAINER_ID
    docker = shutil.which("docker")
    if not docker:
        return None
    name = f"ums-redis-pytest-{uuid.uuid4().hex[:12]}"
    try:
        cid = subprocess.check_output(
            [docker, "run", "-d", "--name", name, "-p", "0:6379", "redis:7-alpine"],
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return None
    _UMS_REDIS_CONTAINER_ID = cid
    atexit.register(_docker_rm, cid)
    host_port: str | None = None
    for _ in range(60):
        try:
            out = subprocess.check_output(
                [docker, "port", cid, "6379"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            line = out.strip().split("\n")[0]
            host_port = line.rsplit(":", 1)[-1]
            break
        except subprocess.CalledProcessError:
            time.sleep(0.5)
    if not host_port:
        _docker_rm(cid)
        _UMS_REDIS_CONTAINER_ID = None
        return None
    url = f"redis://127.0.0.1:{host_port}/0"
    for _ in range(60):
        r = subprocess.run(
            [docker, "exec", cid, "redis-cli", "ping"],
            capture_output=True,
        )
        if r.returncode == 0 and b"PONG" in r.stdout:
            return url
        time.sleep(0.5)
    _docker_rm(cid)
    _UMS_REDIS_CONTAINER_ID = None
    return None


def pytest_configure(config) -> None:
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return
    flag = os.environ.get("UMS_START_REDIS_DOCKER", "").lower()
    if flag not in ("1", "true", "yes"):
        return
    if os.environ.get("REDIS_URL", "").strip():
        return
    url = _start_ephemeral_redis()
    if url:
        os.environ["REDIS_URL"] = url


@pytest.fixture
def redis_url() -> str:
    return os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0").strip() or "redis://127.0.0.1:6379/0"


@pytest_asyncio.fixture
async def redis_kv_store(redis_url: str):
    """RedisKVStore; skips if nothing is listening (no Redis in CI/local)."""
    if not redis_url:
        pytest.skip("REDIS_URL is empty.")
    from unified_memory.storage.kv.redis_store import RedisKVStore

    store = RedisKVStore(url=redis_url)
    try:
        await store._redis.ping()
    except Exception:
        await store._redis.aclose()
        pytest.skip(
            "Redis not reachable. For local runs: start Redis, set REDIS_URL, or use "
            "UMS_START_REDIS_DOCKER=1 with Docker."
        )
    try:
        yield store
    finally:
        await store._redis.aclose()
