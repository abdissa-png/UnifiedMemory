"""
Content Store Implementation.

Location: cas/content_store.py
Design Reference: UNIFIED_MEMORY_SYSTEM_DESIGN.md Section 6

Stores the actual content payload (text/bytes), keyed by content ID.
Separating content from the index (vector DB) and registry allows:
- Sharing content across multiple embeddings/chunks
- Optional lazy loading
- Compression/deduplication
"""

from __future__ import annotations

from typing import Optional

from unified_memory.storage.base import KVStoreBackend


class ContentStore:
    """
    Storage for raw content payloads.
    
    Implementation is a thin wrapper around a KVStoreBackend,
    but provides semantics for content addressing.
    """
    
    def __init__(self, kv_store: KVStoreBackend):
        self._store = kv_store
    
    def _key(self, content_id: str) -> str:
        return f"content:{content_id}"
    
    async def get_content(self, content_id: str) -> Optional[str]:
        """Retrieve content payload."""
        data = await self._store.get(self._key(content_id))
        if not data:
            return None
        return data.get("payload")
    
    async def store_content(
        self,
        content_id: str,
        payload: str,
    ) -> bool:
        """
        Store content payload.
        Returns True if stored (or already existed).
        """
        key = self._key(content_id)
        
        # Check existence first? Or just overwrite (idempotent)?
        # For simplicity and CAS correctness, simple set is fine.
        # But set_if_not_exists is safer to avoid unnecessary writes.
        
        return await self._store.set_if_not_exists(key, {"payload": payload})
    
    async def delete_content(self, content_id: str) -> bool:
        """Delete content (e.g. during garbage collection)."""
        return await self._store.delete(self._key(content_id))
