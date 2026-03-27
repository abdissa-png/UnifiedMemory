"""
Document Registry for Tenant-Level Deduplication.

Design Reference: COMPREHENSIVE_CODEBASE_ANALYSIS.md Section 11.2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import asyncio

from unified_memory.storage.base import KVStoreBackend
from unified_memory.core.types import utc_now


@dataclass
class DocumentEntry:
    """Entry in the Document Registry."""
    
    tenant_id: str
    doc_hash: str
    document_id: str  # The FIRST document ID created for this content
    created_at: str = field(default_factory=lambda: utc_now().isoformat())
    namespaces: List[str] = field(default_factory=list)
    # Typed IDs for all stored objects that belong to this document.
    # These are used for fast-path namespace linking and ref-count-aware delete.
    text_vector_ids: List[str] = field(default_factory=list)
    entity_vector_ids: List[str] = field(default_factory=list)
    relation_vector_ids: List[str] = field(default_factory=list)
    page_image_vector_ids: List[str] = field(default_factory=list)
    graph_node_ids: List[str] = field(default_factory=list)
    graph_edge_ids: List[str] = field(default_factory=list)
    # All chunk-level content hashes that make up this document.
    chunk_content_hashes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "doc_hash": self.doc_hash,
            "document_id": self.document_id,
            "created_at": self.created_at,
            "namespaces": self.namespaces,
            "text_vector_ids": self.text_vector_ids,
            "entity_vector_ids": self.entity_vector_ids,
            "relation_vector_ids": self.relation_vector_ids,
            "page_image_vector_ids": self.page_image_vector_ids,
            "graph_node_ids": self.graph_node_ids,
            "graph_edge_ids": self.graph_edge_ids,
            "chunk_content_hashes": self.chunk_content_hashes,
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentEntry":
        return cls(
            tenant_id=data["tenant_id"],
            doc_hash=data["doc_hash"],
            document_id=data["document_id"],
            created_at=data["created_at"],
            namespaces=data.get("namespaces", []),
            text_vector_ids=data.get("text_vector_ids", data.get("vector_ids", [])),
            entity_vector_ids=data.get("entity_vector_ids", []),
            relation_vector_ids=data.get("relation_vector_ids", []),
            page_image_vector_ids=data.get("page_image_vector_ids", []),
            graph_node_ids=data.get("graph_node_ids", data.get("node_ids", [])),
            graph_edge_ids=data.get("graph_edge_ids", []),
            chunk_content_hashes=data.get("chunk_content_hashes", []),
        )


class DocumentRegistry:
    """
    Registry for tracking document uniqueness at the tenant level.
    
    This prevents re-ingestion of full documents that have already been processed
    for the same tenant, even if requested by a different user/namespace.
    Instead of re-importing, we just grant the new namespace access (add to ACL/namespace list).
    """

    def __init__(self, kv_store: KVStoreBackend):
        self._store = kv_store

    def _key(self, tenant_id: str, doc_hash: str) -> str:
        """Key schema: doc_reg:{tenant_id}:{doc_hash}"""
        return f"doc_reg:{tenant_id}:{doc_hash}"

    @staticmethod
    def _idx_key(document_id: str) -> str:
        """Reverse-index key: doc_id_idx:{document_id} → {tenant_id, doc_hash}."""
        return f"doc_id_idx:{document_id}"

    async def get_document(self, tenant_id: str, doc_hash: str) -> Optional[DocumentEntry]:
        """Look up document by content hash within a tenant."""
        key = self._key(tenant_id, doc_hash)
        versioned = await self._store.get(key)
        if not versioned:
            return None
        return DocumentEntry.from_dict(versioned.data)

    async def get_document_by_document_id(self, document_id: str) -> Optional[DocumentEntry]:
        """Reverse lookup: find a document entry by its ``document_id``.

        This relies on the secondary index written during
        ``register_document``.
        """
        idx = await self._store.get(self._idx_key(document_id))
        if not idx:
            return None
        return await self.get_document(idx.data["tenant_id"], idx.data["doc_hash"])

    async def register_document(
        self, 
        tenant_id: str, 
        doc_hash: str, 
        namespace: str,
        document_id: str
    ) -> bool:
        """
        Register a new document.
        
        Returns:
            True if NEW document registered.
            False if document ALREADY existed (in which case, namespace is added).
        """
        key = self._key(tenant_id, doc_hash)
        
        # Optimistic check
        existing = await self.get_document(tenant_id, doc_hash)
        if existing:
            # Document exists, just add namespace if missing
            await self.add_namespace(tenant_id, doc_hash, namespace)
            return False
        
        # Try to create new
        new_entry = DocumentEntry(
            tenant_id=tenant_id,
            doc_hash=doc_hash,
            document_id=document_id,
            namespaces=[namespace]
        )
        
        success = await self._store.set_if_not_exists(key, new_entry.to_dict())
        if success:
            await self._store.set(
                self._idx_key(document_id),
                {"tenant_id": tenant_id, "doc_hash": doc_hash},
            )
            return True
        else:
            # Race condition: someone else created it. Add namespace to theirs.
            await self.add_namespace(tenant_id, doc_hash, namespace)
            return False

    async def add_namespace(self, tenant_id: str, doc_hash: str, namespace: str) -> bool:
        """Add a namespace to an existing document entry (CAS loop)."""
        key = self._key(tenant_id, doc_hash)
        
        for attempt in range(5):
            versioned = await self._store.get(key)
            if not versioned:
                return False
            
            entry = DocumentEntry.from_dict(versioned.data)
            
            if namespace in entry.namespaces:
                return True
                
            entry.namespaces.append(namespace)
            
            success = await self._store.compare_and_swap(
                key, versioned.version, entry.to_dict()
            )
            
            if success:
                return True
            
            await asyncio.sleep(0.01 * (attempt + 1))
            
        return False

    async def remove_namespace(self, tenant_id: str, doc_hash: str, namespace: str) -> bool:
        """Remove a namespace from a document entry."""
        key = self._key(tenant_id, doc_hash)
        
        for attempt in range(5):
            versioned = await self._store.get(key)
            if not versioned:
                return False
                
            entry = DocumentEntry.from_dict(versioned.data)
            
            if namespace not in entry.namespaces:
                return True  # Already removed
                
            entry.namespaces.remove(namespace)
            
            # If no namespaces left, we might delete the registry entry?
            # Design choice: keep it for history or delete? 
            # For now, we update it empty or with remaining namespaces.
            # Ideally we keep it to know "we saw this hash before" but 
            # if we truly delete the doc from system, we should remove registry too.
            # Let's clean up if empty to match vector store behavior.
            
            if not entry.namespaces:
                success = await self._store.delete_if_version(key, versioned.version)
            else:
                success = await self._store.compare_and_swap(
                    key, versioned.version, entry.to_dict()
                )
                
            if success:
                return True
                
            await asyncio.sleep(0.01 * (attempt + 1))
            
        return False
    async def add_ids(
        self,
        tenant_id: str,
        doc_hash: str,
        text_vector_ids: List[str],
        entity_vector_ids: List[str],
        relation_vector_ids: List[str],
        page_image_vector_ids: List[str],
        graph_node_ids: List[str],
        graph_edge_ids: List[str],
        chunk_content_hashes: List[str],
    ) -> bool:
        """
        Add typed IDs and chunk hashes to an existing document entry.

        This is called after a *full* ingestion finishes so that subsequent
        duplicate ingestions can use the fast path (namespace linking) and
        delete flows can perform ref-count-aware cleanup.
        """
        key = self._key(tenant_id, doc_hash)
        
        for attempt in range(5):
            versioned = await self._store.get(key)
            if not versioned:
                return False
            
            entry = DocumentEntry.from_dict(versioned.data)

            # Use sets to avoid duplicates for each ID type
            def _merge(existing: List[str], incoming: List[str]) -> List[str]:
                if not incoming:
                    return existing
                merged = set(existing)
                merged.update(incoming)
                return list(merged)

            new_text = _merge(entry.text_vector_ids, text_vector_ids)
            new_entity = _merge(entry.entity_vector_ids, entity_vector_ids)
            new_relation = _merge(entry.relation_vector_ids, relation_vector_ids)
            new_page = _merge(entry.page_image_vector_ids, page_image_vector_ids)
            new_nodes = _merge(entry.graph_node_ids, graph_node_ids)
            new_edges = _merge(entry.graph_edge_ids, graph_edge_ids)
            new_hashes = _merge(entry.chunk_content_hashes, chunk_content_hashes)

            # If nothing changed, short‑circuit
            if (
                new_text == entry.text_vector_ids
                and new_entity == entry.entity_vector_ids
                and new_relation == entry.relation_vector_ids
                and new_page == entry.page_image_vector_ids
                and new_nodes == entry.graph_node_ids
                and new_edges == entry.graph_edge_ids
                and new_hashes == entry.chunk_content_hashes
            ):
                return True

            entry.text_vector_ids = new_text
            entry.entity_vector_ids = new_entity
            entry.relation_vector_ids = new_relation
            entry.page_image_vector_ids = new_page
            entry.graph_node_ids = new_nodes
            entry.graph_edge_ids = new_edges
            entry.chunk_content_hashes = new_hashes
            
            success = await self._store.compare_and_swap(
                key, versioned.version, entry.to_dict()
            )
            
            if success:
                return True
                
            await asyncio.sleep(0.01 * (attempt + 1))
            
        return False

    # ------------------------------------------------------------------
    # Namespace → Documents secondary index
    # ------------------------------------------------------------------

    @staticmethod
    def _ns_docs_key(namespace: str) -> str:
        """Index key: ``idx:ns_docs:{namespace}`` → list of doc entries."""
        return f"idx:ns_docs:{namespace}"

    async def add_doc_to_namespace_index(
        self,
        namespace: str,
        doc_hash: str,
        document_id: str,
    ) -> bool:
        """Add a document to the namespace-level document index (CAS loop).

        This enables ``list_documents`` to enumerate all documents in a
        namespace without scanning the entire registry.
        """
        key = self._ns_docs_key(namespace)

        for attempt in range(5):
            versioned = await self._store.get(key)

            if versioned:
                docs: List[Dict[str, str]] = versioned.data.get("docs", [])
            else:
                docs = []

            # Check if already present
            for entry in docs:
                if entry.get("doc_hash") == doc_hash:
                    return True  # Already indexed

            docs.append({"doc_hash": doc_hash, "document_id": document_id})

            if versioned:
                success = await self._store.compare_and_swap(
                    key, versioned.version, {"docs": docs}
                )
            else:
                success = await self._store.set_if_not_exists(key, {"docs": docs})

            if success:
                return True

            await asyncio.sleep(0.01 * (attempt + 1))

        return False

    async def remove_doc_from_namespace_index(
        self,
        namespace: str,
        doc_hash: str,
    ) -> bool:
        """Remove a document from the namespace-level document index."""
        key = self._ns_docs_key(namespace)

        for attempt in range(5):
            versioned = await self._store.get(key)
            if not versioned:
                return True  # Nothing to remove

            docs: List[Dict[str, str]] = versioned.data.get("docs", [])
            new_docs = [d for d in docs if d.get("doc_hash") != doc_hash]

            if len(new_docs) == len(docs):
                return True  # Not present

            if not new_docs:
                success = await self._store.delete_if_version(key, versioned.version)
            else:
                success = await self._store.compare_and_swap(
                    key, versioned.version, {"docs": new_docs}
                )

            if success:
                return True

            await asyncio.sleep(0.01 * (attempt + 1))

        return False

    async def get_namespace_documents(
        self, namespace: str
    ) -> List[Dict[str, str]]:
        """Return all ``{doc_hash, document_id}`` entries for a namespace."""
        key = self._ns_docs_key(namespace)
        versioned = await self._store.get(key)
        if not versioned:
            return []
        return versioned.data.get("docs", [])
