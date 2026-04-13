"""
Inngest durable function for document ingestion.

Each logical stage of the ingestion pipeline runs as an individually
retryable, memoised Inngest step.  Large payloads (images, embeddings)
are externalised to an ``ArtifactStore`` so step outputs stay small.
"""

from __future__ import annotations

import logging
import os
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
    from unified_memory.workflows.job_state import (
        IngestionJobState,
        JobStage,
        save_job_state,
    )

    inngest_client = get_inngest_client()
    kv_store = pipeline.namespace_manager.kv_store

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
            inngest.Concurrency(limit=5, key="event.data.namespace"),
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
        original_filename = data.get("original_filename") or (os.path.basename(source_path) if source_path else "")
        content_type = data.get("content_type") or "application/octet-stream"
        session_id = data.get("session_id")
        options = data.get("options") or {}
        job_id = data.get("job_id") or str(uuid.uuid4())

        is_file = bool(source_path)
        job_state = IngestionJobState(
            job_id=job_id,
            tenant_id=tenant_id,
            namespace=namespace,
            operation="ingest",
            document_id=document_id,
            stage=JobStage.DISPATCHED,
        )
        await save_job_state(kv_store, job_state)

        try:
            # ----------------------------------------------------------
            # Step 1 — Resolve tenant/namespace context
            # ----------------------------------------------------------
            resolve_ctx = await step.run(
                "resolve-context",
                lambda: pipeline.step_resolve_context(namespace),
            )
            job_state.mark_stage(JobStage.RESOLVE_CONTEXT)
            await save_job_state(kv_store, job_state)

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
                job_state.mark_failed(parse_result["error"])
                await save_job_state(kv_store, job_state)
                if is_file and source_path and os.path.exists(source_path):
                    await step.run(
                        "cleanup-source-file",
                        lambda: os.unlink(source_path),
                    )
                return {"status": "parse_error", "error": parse_result["error"]}

            job_state.mark_stage(JobStage.PARSED)
            job_state.parsed_artifact_uri = parse_result.get("parsed_artifact_uri", "")
            job_state.page_image_uris = parse_result.get("page_image_uris", [])
            await save_job_state(kv_store, job_state)

            full_text = parse_result["full_text"]

            # Compute doc_hash from full text (same logic as ingest_file/ingest_text)
            from unified_memory.core.types import compute_document_hash as _cdh
            doc_hash = _cdh(full_text, tenant_id)
            job_state.doc_hash = doc_hash

            # ----------------------------------------------------------
            # Step 3 — Dedup check
            # ----------------------------------------------------------
            dedup = await step.run(
                "dedup-check",
                lambda: pipeline.step_dedup_check(tenant_id, doc_hash, namespace),
            )
            job_state.mark_stage(JobStage.DEDUP_CHECKED)
            await save_job_state(kv_store, job_state)

            if dedup["decision"] == "skip":
                job_state.document_id = dedup.get("existing_document_id", document_id)
                await pipeline.finalize_ingest_lifecycle(
                    tenant_id=tenant_id,
                    namespace=namespace,
                    document_id=job_state.document_id,
                    doc_hash=doc_hash,
                    session_id=session_id,
                    source_path=source_path,
                    original_filename=original_filename,
                    content_type=content_type,
                )
                job_state.mark_succeeded(
                    {
                        "status": "deduped",
                        "document_id": job_state.document_id,
                        "doc_hash": doc_hash,
                    }
                )
                await save_job_state(kv_store, job_state)
                if is_file and source_path and os.path.exists(source_path):
                    await step.run(
                        "cleanup-source-file",
                        lambda: os.unlink(source_path),
                    )
                return job_state.result

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
                job_state.document_id = link_result["document_id"]
                job_state.mark_stage(JobStage.FAST_LINKED)
                await pipeline.finalize_ingest_lifecycle(
                    tenant_id=tenant_id,
                    namespace=namespace,
                    document_id=job_state.document_id,
                    doc_hash=doc_hash,
                    session_id=session_id,
                    source_path=source_path,
                    original_filename=original_filename,
                    content_type=content_type,
                )
                job_state.mark_succeeded(
                    {
                        "status": "linked",
                        "document_id": link_result["document_id"],
                        "doc_hash": doc_hash,
                    }
                )
                await save_job_state(kv_store, job_state)
                if is_file and source_path and os.path.exists(source_path):
                    await step.run(
                        "cleanup-source-file",
                        lambda: os.unlink(source_path),
                    )
                return job_state.result

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
            job_state.mark_stage(JobStage.CHUNKED)
            job_state.chunk_count = chunk_result["chunk_count"]
            job_state.chunk_content_hashes = chunk_result["chunk_content_hashes"]
            await save_job_state(kv_store, job_state)

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
            job_state.mark_stage(JobStage.TEXT_EMBEDDED)
            job_state.text_vector_ids = text_result.get("text_vector_ids", [])
            await save_job_state(kv_store, job_state)

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
            job_state.mark_stage(JobStage.SPARSE_WRITTEN)
            await save_job_state(kv_store, job_state)

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
            job_state.mark_stage(JobStage.GRAPH_WRITTEN)
            job_state.graph_node_ids = graph_result.get("graph_node_ids", [])
            job_state.graph_edge_ids = graph_result.get("graph_edge_ids", [])
            await save_job_state(kv_store, job_state)

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
            job_state.mark_stage(JobStage.ENTITY_REL_EMBEDDED)
            job_state.entity_vector_ids = ent_rel_result.get("entity_vector_ids", [])
            job_state.relation_vector_ids = ent_rel_result.get("relation_vector_ids", [])
            await save_job_state(kv_store, job_state)

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
            job_state.mark_stage(JobStage.VISION_EMBEDDED)
            job_state.page_image_vector_ids = vision_result.get(
                "page_image_vector_ids", []
            )
            await save_job_state(kv_store, job_state)

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
            job_state.mark_stage(JobStage.INGEST_FINALIZED)
            await step.run(
                "finalize-ingest-lifecycle",
                lambda: pipeline.finalize_ingest_lifecycle(
                    tenant_id=tenant_id,
                    namespace=namespace,
                    document_id=document_id,
                    doc_hash=doc_hash,
                    session_id=session_id,
                    source_path=source_path,
                    original_filename=original_filename,
                    content_type=content_type,
                ),
            )
            await save_job_state(kv_store, job_state)

            # ----------------------------------------------------------
            # (Optional) Clean up artifacts
            # ----------------------------------------------------------
            await step.run(
                "cleanup-artifacts",
                lambda: artifact_store.cleanup_job(job_id),
            )
            if is_file and source_path and os.path.exists(source_path):
                await step.run(
                    "cleanup-source-file",
                    lambda: os.unlink(source_path),
                )

            result = {
                "status": "ingested",
                "document_id": document_id,
                "doc_hash": doc_hash,
                "chunk_count": chunk_result["chunk_count"],
            }
            job_state.mark_succeeded(result)
            await save_job_state(kv_store, job_state)
            return result
        except Exception as exc:
            job_state.mark_failed(str(exc))
            await save_job_state(kv_store, job_state)
            raise

    return ingest_document
