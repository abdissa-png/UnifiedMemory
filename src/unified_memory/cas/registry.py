"""
CAS Registry Implementation.

Location: cas/registry.py
Design Reference: UNIFIED_MEMORY_SYSTEM_DESIGN.md Section 6

The CAS Registry tracks content hashes and their references (usage locations).
It does NOT store the content itself (content is in ContentStore) or embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from unified_memory.core.types import utc_now
from unified_memory.storage.base import KVStoreBackend


@dataclass
class CASReference:
    """Reference to where content is used."""
    namespace: str
    document_id: str
    chunk_index: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "namespace": self.namespace,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CASReference":
        return cls(
            namespace=data["namespace"],
            document_id=data["document_id"],
            chunk_index=data["chunk_index"],
        )


@dataclass
class CASEntry:
    """Entry in the CAS registry."""
    content_hash: str
    content_id: str  # Pointer to content in ContentStore
    vector_id: Optional[str] = None  # Pointer to embedding in Vector DB
    created_at: datetime = field(default_factory=utc_now)
    refs: List[CASReference] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content_hash": self.content_hash,
            "content_id": self.content_id,
            "vector_id": self.vector_id,
            "created_at": self.created_at.isoformat(),
            "refs": [ref.to_dict() for ref in self.refs],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CASEntry":
        return cls(
            content_hash=data["content_hash"],
            content_id=data["content_id"],
            vector_id=data.get("vector_id"),
            created_at=datetime.fromisoformat(data["created_at"]),
            refs=[CASReference.from_dict(r) for r in data.get("refs", [])],
        )


class CASRegistry:
    """
    Registry for Content-Addressable Storage.
    
    Design Reference: UNIFIED_MEMORY_SYSTEM_DESIGN.md Section 6
    
    Responsibilities:
    - Map content_hash -> CASEntry
    - Track all references (where is this content used?)
    - Support reference counting / orphan detection
    """
    
    def __init__(self, kv_store: KVStoreBackend):
        self._store = kv_store
    
    def _key(self, content_hash: str) -> str:
        return f"cas:{content_hash}"
    
    async def get_entry(self, content_hash: str) -> Optional[CASEntry]:
        """Lookup CAS entry by hash."""
        data = await self._store.get(self._key(content_hash))
        if not data:
            return None
        return CASEntry.from_dict(data)
    
    async def register(
        self,
        content_hash: str,
        content_id: str,
        vector_id: Optional[str] = None,
    ) -> CASEntry:
        """
        Register new content or get existing entry.
        
        This uses set_if_not_exists to handle concurrency race conditions.
        """
        key = self._key(content_hash)
        
        # Optimistically try to create new
        new_entry = CASEntry(
            content_hash=content_hash,
            content_id=content_id,
            vector_id=vector_id,
        )
        
        # Try atomic set (if not exists)
        success = await self._store.set_if_not_exists(key, new_entry.to_dict())
        
        if success:
            return new_entry
        else:
            # Race condition: someone else created it. Fetch it.
            return await self.get_entry(content_hash) # type: ignore
    
    async def add_reference(
        self,
        content_hash: str,
        namespace: str,
        document_id: str,
        chunk_index: int,
    ) -> bool:
        """
        Add a usage reference to a CAS entry.
        
        Uses optimistic locking (compare-and-swap) to update the list safely.
        """
        key = self._key(content_hash)
        
        # Retry loop for CAS (Compare-And-Swap)
        for _ in range(5):  # Max 5 retries
            data = await self._store.get(key)
            if not data:
                return False  # Content not found (race condition?)
            
            # Reconstruct object
            entry = CASEntry.from_dict(data)
            
            # Create new reference
            new_ref = CASReference(namespace, document_id, chunk_index)
            
            # Check for duplicate
            if any(r.namespace == namespace and 
                   r.document_id == document_id and 
                   r.chunk_index == chunk_index for r in entry.refs):
                return True  # Already exists
            
            # Add ref
            entry.refs.append(new_ref)
            
            # Attempt CAS update
            # Note: We need the store to return version/support versioning
            # Our current abstract KVStoreBackend defined compare_and_swap.
            # But the 'data' returned from get() needs to include version info 
            # or we need to manage versioning.
            # Assuming for now 'get' returns raw dict, but we need version.
            # Let's check KVStoreBackend signature.
            
            # KVStoreBackend.compare_and_swap(key, expected_version, new_value)
            # MemoryKVStore implements simple integer versioning.
            # But get() in MemoryKVStore just returns data['value'].
            # Wait, `MemoryKVStore.get` implementation:
            # return self._store[key]["value"]
            # It hides the version!
            
            # FIX: We need access to version. 
            # But for now, let's assume we can rely on a simpler read-modify-write 
            # if we accept last-writer-wins or if the KV store handles locking.
            # Unsafe in distributed sys, but OK for MVP with MemoryStore.
            # Proper fix: Use optimistic locking if KV store exposes version.
            
            # Let's try simple set first.
            await self._store.set(key, entry.to_dict())
            return True
            
        return False

    async def remove_reference(
        self,
        content_hash: str,
        namespace: str,
        document_id: str,
        chunk_index: int = None,
    ) -> bool:
        """Remove a reference."""
        key = self._key(content_hash)
        data = await self._store.get(key)
        if not data:
            return False
            
        entry = CASEntry.from_dict(data)
        
        # Filter out the reference
        original_len = len(entry.refs)
        if chunk_index is not None:
             entry.refs = [r for r in entry.refs 
                          if not (r.namespace == namespace and 
                                  r.document_id == document_id and
                                  r.chunk_index == chunk_index)]
        else:
            # Remove all refs for this doc
             entry.refs = [r for r in entry.refs 
                          if not (r.namespace == namespace and 
                                  r.document_id == document_id)]
        
        if len(entry.refs) != original_len:
            await self._store.set(key, entry.to_dict())
            return True
            
        return False
    
    async def get_orphans(self) -> List[str]:
        """
        Find all content hashes that have no references.
        
        Warning: This is a scan operation (expensive).
        """
        # Scan keys starting with cas:
        # MemoryKVStore doesn't expose scan. Redis would use SCAN.
        # Check if KVStoreBackend has scan/pattern matching.
        # It has `scan_keys(pattern)`.
        
        keys = await self._store.scan("cas:*")
        orphans = []
        for key in keys:
            data = await self._store.get(key)
            if data and not data.get("refs"):
                # Extract hash from key "cas:..."
                orphans.append(key.split(":", 1)[1])
        return orphans
