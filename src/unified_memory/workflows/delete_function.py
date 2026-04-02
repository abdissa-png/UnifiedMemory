"""
Inngest durable function for document deletion.

Mirrors ``IngestionPipeline.delete_document`` but runs each clean-up
phase as an individually retryable Inngest step.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from unified_memory.ingestion.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)


def create_delete_function(pipeline: "IngestionPipeline"):
    """Factory: build an Inngest delete function bound to a pipeline.

    Called once at bootstrap time; the returned function object is
    registered with the Inngest serve handler.
    """
    from unified_memory.workflows.client import get_inngest_client
    from unified_memory.workflows.events import DOCUMENT_DELETE_REQUESTED

    inngest_client = get_inngest_client()

    try:
        import inngest
    except ImportError as exc:
        raise ImportError(
            "inngest is required for workflow support"
        ) from exc

    @inngest_client.create_function(
        fn_id="delete-document",
        trigger=inngest.TriggerEvent(event=DOCUMENT_DELETE_REQUESTED),
        retries=3,
        concurrency=[
            inngest.Concurrency(limit=3, key="event.data.tenant_id"),
        ],
    )
    async def delete_document(
        ctx: inngest.Context,
        step: inngest.Step,
    ) -> dict:
        data = ctx.event.data
        tenant_id = data["tenant_id"]
        namespace = data["namespace"]
        doc_hash = data["doc_hash"]

        # Step 1 — Load document entry
        async def _load_entry():
            entry = await pipeline.document_registry.get_document(
                tenant_id, doc_hash
            )
            if not entry:
                return None
            return {
                "document_id": entry.document_id,
                "text_vector_ids": entry.text_vector_ids,
                "entity_vector_ids": entry.entity_vector_ids,
                "relation_vector_ids": entry.relation_vector_ids,
                "page_image_vector_ids": entry.page_image_vector_ids,
                "graph_node_ids": entry.graph_node_ids,
                "graph_edge_ids": entry.graph_edge_ids,
                "chunk_content_hashes": entry.chunk_content_hashes,
            }

        entry_dict = await step.run("load-entry", _load_entry)

        if entry_dict is None:
            return {"status": "not_found"}

        # Step 2 — Remove namespace from registry
        await step.run(
            "remove-registry-namespace",
            lambda: pipeline.document_registry.remove_namespace(
                tenant_id, doc_hash, namespace
            ),
        )

        # Step 3 — Clean up text vectors (shared-doc-aware)
        async def _cleanup_text_vectors():
            from unified_memory.core.types import CollectionType
            from unified_memory.ingestion.pipeline import DeleteResult

            res = DeleteResult(found=True)
            text_col = await pipeline.namespace_manager.get_collection_name(
                namespace, CollectionType.TEXTS
            )
            for vid in entry_dict.get("text_vector_ids", []):
                await pipeline._remove_vector_smart(
                    vid, namespace, entry_dict["document_id"], text_col, res
                )
            return {"deleted": res.vectors_deleted, "unlinked": res.vectors_unlinked}

        await step.run("cleanup-text-vectors", _cleanup_text_vectors)

        # Step 4 — Clean up entity vectors
        async def _cleanup_entity_vectors():
            from unified_memory.core.types import CollectionType
            from unified_memory.ingestion.pipeline import DeleteResult

            res = DeleteResult(found=True)
            ent_col = await pipeline.namespace_manager.get_collection_name(
                namespace, CollectionType.ENTITIES
            )
            for vid in entry_dict.get("entity_vector_ids", []):
                await pipeline._remove_vector_smart(
                    vid, namespace, entry_dict["document_id"], ent_col, res
                )
            return {"deleted": res.vectors_deleted, "unlinked": res.vectors_unlinked}

        await step.run("cleanup-entity-vectors", _cleanup_entity_vectors)

        # Step 5 — Clean up relation vectors
        async def _cleanup_relation_vectors():
            from unified_memory.core.types import CollectionType
            from unified_memory.ingestion.pipeline import DeleteResult

            res = DeleteResult(found=True)
            rel_col = await pipeline.namespace_manager.get_collection_name(
                namespace, CollectionType.RELATIONS
            )
            for vid in entry_dict.get("relation_vector_ids", []):
                await pipeline._remove_vector_smart(
                    vid, namespace, entry_dict["document_id"], rel_col, res
                )
            return {"deleted": res.vectors_deleted, "unlinked": res.vectors_unlinked}

        await step.run("cleanup-relation-vectors", _cleanup_relation_vectors)

        # Step 6 — Image CAS clean-up (must run BEFORE page image vectors
        #          are removed so we can still read content_hash from metadata)
        async def _cleanup_images():
            if not getattr(pipeline, "image_content_store", None):
                return {"cleaned": 0}
            from unified_memory.core.types import CollectionType

            cleaned = 0
            page_col = await pipeline.namespace_manager.get_collection_name(
                namespace, CollectionType.PAGE_IMAGES
            )
            for vid in entry_dict.get("page_image_vector_ids", []):
                try:
                    vec = await pipeline.vector_store.get_by_id(
                        vid, collection=page_col, namespace=namespace
                    )
                    if not vec:
                        continue
                    img_content_hash = vec.metadata.get("content_hash")
                    if not img_content_hash:
                        continue

                    await pipeline.cas_registry.remove_reference(
                        content_hash=img_content_hash,
                        namespace=namespace,
                        document_id=entry_dict["document_id"],
                        chunk_index=None,
                    )
                    remaining = await pipeline.cas_registry.get_entry(
                        img_content_hash
                    )
                    if not remaining or remaining.refs:
                        continue
                    await pipeline.image_content_store.delete_image(
                        img_content_hash
                    )
                    await pipeline.cas_registry.delete_if_orphan(
                        img_content_hash
                    )
                    cleaned += 1
                except Exception:
                    logger.exception(
                        "Error during image cleanup for vector %s", vid
                    )
            return {"cleaned": cleaned}

        await step.run("cleanup-images", _cleanup_images)

        # Step 7 — Clean up page image vectors
        async def _cleanup_page_image_vectors():
            from unified_memory.core.types import CollectionType
            from unified_memory.ingestion.pipeline import DeleteResult

            res = DeleteResult(found=True)
            page_col = await pipeline.namespace_manager.get_collection_name(
                namespace, CollectionType.PAGE_IMAGES
            )
            for vid in entry_dict.get("page_image_vector_ids", []):
                await pipeline._remove_vector_smart(
                    vid, namespace, entry_dict["document_id"], page_col, res
                )
            return {"deleted": res.vectors_deleted, "unlinked": res.vectors_unlinked}

        await step.run("cleanup-page-image-vectors", _cleanup_page_image_vectors)

        # Step 8 — Clean up graph nodes + edges (shared-doc-aware)
        async def _cleanup_graph():
            if not pipeline.graph_store:
                return {"deleted": 0}
            from unified_memory.ingestion.pipeline import DeleteResult

            res = DeleteResult(found=True)
            for nid in entry_dict.get("graph_node_ids", []):
                await pipeline._remove_graph_smart(
                    nid, namespace, entry_dict["document_id"], res, is_node=True
                )
            for eid in entry_dict.get("graph_edge_ids", []):
                await pipeline._remove_graph_smart(
                    eid, namespace, entry_dict["document_id"], res, is_node=False
                )
            return {"deleted": res.nodes_deleted, "unlinked": res.nodes_unlinked}

        await step.run("cleanup-graph", _cleanup_graph)

        # Step 9 — CAS / ContentStore clean-up
        async def _cleanup_cas():
            cleaned = 0
            for content_hash in entry_dict.get("chunk_content_hashes", []):
                try:
                    await pipeline.cas_registry.remove_reference(
                        content_hash=content_hash,
                        namespace=namespace,
                        document_id=entry_dict["document_id"],
                        chunk_index=None,
                    )
                    cas_entry = await pipeline.cas_registry.get_entry(content_hash)
                    if not cas_entry or cas_entry.refs:
                        continue
                    try:
                        await pipeline.content_store.delete_content(
                            cas_entry.content_id
                        )
                    except Exception:
                        logger.exception(
                            "Failed to delete content for orphaned hash %s",
                            content_hash,
                        )
                    await pipeline.cas_registry.delete_if_orphan(content_hash)
                    cleaned += 1
                except Exception:
                    logger.exception(
                        "Error during CAS cleanup for hash %s", content_hash
                    )
            return {"cleaned": cleaned}

        await step.run("cleanup-cas", _cleanup_cas)

        # Step 10 — Sparse store clean-up (shared-doc-aware)
        async def _cleanup_sparse():
            if not pipeline.sparse_store:
                return {"deleted": 0}
            chunk_hashes = entry_dict.get("chunk_content_hashes", [])
            if not chunk_hashes:
                return {"deleted": 0}
            try:
                if hasattr(pipeline.sparse_store, "remove_document_reference"):
                    remaining_map = await pipeline.sparse_store.remove_document_reference(
                        doc_ids=chunk_hashes,
                        namespace=namespace,
                        document_id=entry_dict["document_id"],
                    )
                    hashes_to_delete = [
                        h for h, docs in remaining_map.items()
                        if not docs
                        or not await pipeline._namespace_still_needed(
                            docs, namespace
                        )
                    ]
                    if hashes_to_delete:
                        await pipeline.sparse_store.delete(
                            doc_ids=hashes_to_delete,
                            namespace=namespace,
                        )
                    return {"deleted": len(hashes_to_delete)}
                else:
                    await pipeline.sparse_store.delete(
                        doc_ids=chunk_hashes,
                        namespace=namespace,
                        document_id=entry_dict["document_id"],
                    )
                    return {"deleted": len(chunk_hashes)}
            except Exception as e:
                logger.error(f"Sparse index delete failed: {e}")
                return {"deleted": 0}

        await step.run("cleanup-sparse", _cleanup_sparse)

        return {"status": "deleted", "doc_hash": doc_hash}

    return delete_document
