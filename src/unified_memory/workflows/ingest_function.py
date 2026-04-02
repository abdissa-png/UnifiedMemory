"""
Inngest durable function for document ingestion.

Each logical stage of the ingestion pipeline runs as an individually
retryable, memoised Inngest step.  Large payloads (images, embeddings)
are externalised to an ``ArtifactStore`` so step outputs stay small.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from unified_memory.ingestion.pipeline import IngestionPipeline
    from unified_memory.workflows.artifact_store import ArtifactStore

logger = logging.getLogger(__name__)


def create_ingest_function(
    pipeline: "IngestionPipeline",
    artifact_store: "ArtifactStore",
):
    """Factory: build an Inngest function bound to a pipeline + artifact store.

    Called once at bootstrap time; the returned function object is
    registered with the Inngest serve handler.
    """
    from unified_memory.workflows.client import get_inngest_client
    from unified_memory.workflows.events import DOCUMENT_UPLOADED, DOCUMENT_DELETE_REQUESTED

    inngest_client = get_inngest_client()

    try:
        import inngest
    except ImportError as exc:
        raise ImportError(
            "inngest is required for workflow support"
        ) from exc

    @inngest_client.create_function(
        fn_id="ingest-document",
        trigger=inngest.TriggerEvent(event=DOCUMENT_UPLOADED),
        retries=3,
        concurrency=[
            inngest.Concurrency(limit=5, key="event.data.tenant_id"),
        ],
        cancel=[
            inngest.Cancel(
                event=DOCUMENT_DELETE_REQUESTED,
                if_exp=(
                    "event.data.tenant_id == async.data.tenant_id && "
                    "event.data.doc_hash == async.data.doc_hash"
                ),
            ),
        ],
    )
    async def ingest_document(
        ctx: inngest.Context,
        step: inngest.Step,
    ) -> dict:
        data = ctx.event.data
        tenant_id = data["tenant_id"]
        namespace = data["namespace"]
        document_id = data.get("document_id") or str(uuid.uuid4())
        source_path = data.get("source_path")
        source_text = data.get("source_text")
        title = data.get("title")
        options = data.get("options") or {}
        job_id = data.get("job_id") or str(uuid.uuid4())

        is_file = bool(source_path)

        # ----------------------------------------------------------
        # Step 1 — Resolve tenant/namespace context
        # ----------------------------------------------------------
        resolve_ctx = await step.run(
            "resolve-context",
            lambda: pipeline.step_resolve_context(namespace),
        )

        # ----------------------------------------------------------
        # Step 2 — Parse and externalise heavy payloads
        # ----------------------------------------------------------
        parse_result = await step.run(
            "parse-and-externalize",
            lambda: pipeline.step_parse_and_externalize(
                source_path_or_text=source_path if is_file else source_text,
                document_id=document_id,
                job_id=job_id,
                artifact_store=artifact_store,
                is_file=is_file,
                title=title,
                metadata=options.get("metadata"),
                **{k: v for k, v in options.items() if k != "metadata"},
            ),
        )

        if parse_result.get("error"):
            return {"status": "parse_error", "error": parse_result["error"]}

        full_text = parse_result["full_text"]

        # Compute doc_hash from full text (same logic as ingest_file/ingest_text)
        from unified_memory.core.types import compute_document_hash as _cdh
        doc_hash = _cdh(full_text, tenant_id)

        # ----------------------------------------------------------
        # Step 3 — Dedup check
        # ----------------------------------------------------------
        dedup = await step.run(
            "dedup-check",
            lambda: pipeline.step_dedup_check(tenant_id, doc_hash, namespace),
        )

        if dedup["decision"] == "skip":
            return {
                "status": "deduped",
                "document_id": dedup.get("existing_document_id"),
                "doc_hash": doc_hash,
            }

        # ----------------------------------------------------------
        # Step 4 — Fast link (existing doc, new namespace)
        # ----------------------------------------------------------
        if dedup["decision"] == "fast_link":
            link_result = await step.run(
                "fast-link",
                lambda: pipeline.step_fast_link(
                    tenant_id, doc_hash, namespace, dedup,
                    ctx=resolve_ctx,
                ),
            )
            return {
                "status": "linked",
                "document_id": link_result["document_id"],
                "doc_hash": doc_hash,
            }

        # ----------------------------------------------------------
        # Step 5 — Register document + chunk
        # ----------------------------------------------------------
        chunk_result = await step.run(
            "register-and-chunk",
            lambda: pipeline.step_register_and_chunk(
                tenant_id=tenant_id,
                doc_hash=doc_hash,
                namespace=namespace,
                document_id=document_id,
                parsed_artifact_uri=parse_result["parsed_artifact_uri"],
                job_id=job_id,
                artifact_store=artifact_store,
                ctx=resolve_ctx,
            ),
        )

        chunk_content_hashes = chunk_result["chunk_content_hashes"]
        parsed_artifact_uri = parse_result["parsed_artifact_uri"]
        page_image_uris = parse_result.get("page_image_uris", [])

        # ----------------------------------------------------------
        # Step 6 — Embed + upsert text vectors
        # ----------------------------------------------------------
        text_result = await step.run(
            "embed-and-upsert-text",
            lambda: pipeline.step_embed_and_upsert_text(
                namespace=namespace,
                tenant_id=tenant_id,
                chunk_content_hashes=chunk_content_hashes,
                document_id=document_id,
                ctx=resolve_ctx,
            ),
        )

        # ----------------------------------------------------------
        # Step 7 — Sparse index
        # ----------------------------------------------------------
        await step.run(
            "sparse-upsert",
            lambda: pipeline.step_sparse_upsert(
                namespace=namespace,
                chunk_content_hashes=chunk_content_hashes,
                document_id=document_id,
            ),
        )

        # ----------------------------------------------------------
        # Step 8 — Graph extraction + upsert
        # ----------------------------------------------------------
        graph_result = await step.run(
            "extract-and-upsert-graph",
            lambda: pipeline.step_extract_and_upsert_graph(
                namespace=namespace,
                tenant_id=tenant_id,
                chunk_content_hashes=chunk_content_hashes,
                document_id=document_id,
                parsed_artifact_uri=parsed_artifact_uri,
                artifact_store=artifact_store,
                ctx=resolve_ctx,
                job_id=job_id,
            ),
        )

        # ----------------------------------------------------------
        # Step 9 — Embed entities + relations
        # ----------------------------------------------------------
        ent_rel_result = await step.run(
            "embed-entities-relations",
            lambda: pipeline.step_embed_and_upsert_entities_relations(
                namespace=namespace,
                tenant_id=tenant_id,
                document_id=document_id,
                ctx=resolve_ctx,
                artifact_store=artifact_store,
                entity_descriptors_uri=graph_result.get("entity_descriptors_uri", ""),
                relation_descriptors_uri=graph_result.get("relation_descriptors_uri", ""),
            ),
        )

        # ----------------------------------------------------------
        # Step 10 — Vision embeddings
        # ----------------------------------------------------------
        vision_result = await step.run(
            "embed-and-upsert-vision",
            lambda: pipeline.step_embed_and_upsert_vision(
                namespace=namespace,
                tenant_id=tenant_id,
                document_id=document_id,
                page_image_uris=page_image_uris,
                artifact_store=artifact_store,
                ctx=resolve_ctx,
            ),
        )

        # ----------------------------------------------------------
        # Step 11 — Finalise document registry
        # ----------------------------------------------------------
        await step.run(
            "finalize-registry",
            lambda: pipeline.step_finalize_registry(
                tenant_id=tenant_id,
                doc_hash=doc_hash,
                text_result=text_result,
                graph_result=graph_result,
                ent_rel_result=ent_rel_result,
                vision_result=vision_result,
                chunk_content_hashes=chunk_content_hashes,
            ),
        )

        # ----------------------------------------------------------
        # (Optional) Clean up artifacts
        # ----------------------------------------------------------
        await step.run(
            "cleanup-artifacts",
            lambda: artifact_store.cleanup_job(job_id),
        )

        return {
            "status": "ingested",
            "document_id": document_id,
            "doc_hash": doc_hash,
            "chunk_count": chunk_result["chunk_count"],
        }

    return ingest_document
