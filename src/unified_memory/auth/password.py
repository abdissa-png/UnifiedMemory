"""
Password hashing via passlib + bcrypt.
"""

from __future__ import annotations


import asyncio
from concurrent.futures import ThreadPoolExecutor

_executor = ThreadPoolExecutor(max_workers=4)

def _get_ctx():
    from passlib.context import CryptContext
    return CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(plain: str) -> str:
    """Sync password hashing (blocking)."""
    return _get_ctx().hash(plain)

def verify_password(plain: str, hashed: str) -> bool:
    """Sync password verification (blocking)."""
    return _get_ctx().verify(plain, hashed)

async def async_hash_password(plain: str) -> str:
    """Async password hashing to prevent blocking the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, hash_password, plain)

async def async_verify_password(plain: str, hashed: str) -> bool:
    """Async password verification to prevent blocking the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, verify_password, plain, hashed)
