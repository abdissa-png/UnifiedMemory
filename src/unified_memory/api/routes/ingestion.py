"""
Ingestion and document management endpoints.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import Response

import logging
import uuid

from unified_memory.api.deps import ACLChecker, get_current_user, get_system_context
from unified_memory.api.schemas import (
    DocumentResponse,
    IngestResponse,
    IngestTextRequest,
    JobStatusResponse,
)
from unified_memory.auth.jwt_handler import AuthenticatedUser
from unified_memory.core.types import Permission
from unified_memory.observability.tracing import set_request_context
from unified_memory.workflows.job_state import (
    IngestionJobState,
    JobStage,
    load_job_state,
    save_job_state,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["ingestion"])


# ---------------------------------------------------------------------------
# Text ingestion
# ---------------------------------------------------------------------------


@router.post("/ingest/text/{namespace}", response_model=IngestResponse)
async def ingest_text(
    namespace: str,
    body: IngestTextRequest,
    background: bool = True,
    ns_config=Depends(ACLChecker(Permission.WRITE)),
    user: AuthenticatedUser = Depends(get_current_user),
    ctx=Depends(get_system_context),
):
    set_request_context(tenant_id=user.tenant_id, namespace=namespace)

    # ---- async Inngest path ----
    inngest_client = getattr(ctx, "inngest_client", None)
    if background and inngest_client:
        job_id = uuid.uuid4().hex
        document_id = uuid.uuid4().hex
        job_state = IngestionJobState(
            job_id=job_id,
            tenant_id=user.tenant_id,
            namespace=namespace,
            operation="ingest",
            document_id=document_id,
            stage=JobStage.DISPATCHED,
        )
        await save_job_state(ctx.kv_store, job_state)
        try:
            import inngest
            await inngest_client.send(
                inngest.Event(
                    name="document/uploaded",
                    data={
                        "tenant_id": user.tenant_id,
                        "namespace": namespace,
                        "document_id": document_id,
                        "source_text": body.text,
                        "title": body.title,
                        "job_id": job_id,
                        "session_id": body.session_id,
                        "options": {"metadata": body.metadata} if body.metadata else {},
                    },
                )
            )
            logger.info("Dispatched async ingest job %s", job_id)
            return IngestResponse(
                job_id=job_id,
                status="processing",
            )
        except Exception:
            await ctx.kv_store.delete(IngestionJobState.kv_key(job_id))
            logger.exception("Inngest dispatch failed, falling back to sync")

    # ---- sync path ----

    result = await ctx.ingestion_pipeline.ingest_text(
        text=body.text,
        namespace=namespace,
        title=body.title,
        metadata=body.metadata,
        session_id=body.session_id,
    )
    if not result.success:
        raise HTTPException(422, "; ".join(result.errors))
        
    if getattr(ctx, "audit_logger", None):
        await ctx.audit_logger.log(
            tenant_id=user.tenant_id,
            user_id=user.user_id,
            action="document.create",
            resource_type="document",
            resource_id=result.doc_hash or "unknown",
            details={"type": "text"},
            outcome="success",
        )

    return IngestResponse(
        document_id=result.document_id,
        chunk_count=result.chunk_count,
        doc_hash=result.doc_hash,
        status="deduped" if result.deduped else "ingested",
    )


# ---------------------------------------------------------------------------
# File ingestion — uses permanent DocumentContentStore
# ---------------------------------------------------------------------------


@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    namespace: str = Form(...),
    file: UploadFile = File(...),
    title: str = Form(None),
    session_id: str = Form(None),
    background: bool = Form(True),
    user: AuthenticatedUser = Depends(get_current_user),
    ns_config=Depends(ACLChecker(Permission.WRITE)),
    ctx=Depends(get_system_context),
):
    import tempfile

    set_request_context(tenant_id=user.tenant_id, namespace=namespace)
    contents = await file.read()
    filename = file.filename or "unknown"
    content_type = file.content_type or "application/octet-stream"
    suffix = os.path.splitext(filename)[1]

    # ---- async Inngest path ----
    inngest_client = getattr(ctx, "inngest_client", None)
    if background and inngest_client:
        job_id = uuid.uuid4().hex
        document_id = uuid.uuid4().hex
        job_state = IngestionJobState(
            job_id=job_id,
            tenant_id=user.tenant_id,
            namespace=namespace,
            operation="ingest",
            document_id=document_id,
            stage=JobStage.DISPATCHED,
        )
        await save_job_state(ctx.kv_store, job_state)
        tmp_permanent = None
        # Write temp file for the workflow to read
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_permanent = tmp.name

        try:
            import inngest
            await inngest_client.send(
                inngest.Event(
                    name="document/uploaded",
                    data={
                        "tenant_id": user.tenant_id,
                        "namespace": namespace,
                        "document_id": document_id,
                        "source_path": tmp_permanent,
                        "title": title or filename,
                        "job_id": job_id,
                        "original_filename": filename,
                        "content_type": content_type,
                        "session_id": session_id,
                    },
                )
            )
            logger.info("Dispatched async file ingest job %s", job_id)
            return IngestResponse(
                job_id=job_id,
                status="processing",
            )
        except Exception:
            await ctx.kv_store.delete(IngestionJobState.kv_key(job_id))
            logger.exception("Inngest dispatch failed, falling back to sync")
            if tmp_permanent:
                os.unlink(tmp_permanent)

    # The pipeline still expects a file path, so write a temp file for
    # parsing.  The permanent copy is created after ingestion succeeds
    # using the canonical doc_hash returned by the pipeline.
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = await ctx.ingestion_pipeline.ingest_file(
            path=Path(tmp_path),
            namespace=namespace,
            title=title,
            session_id=session_id,
            original_filename=filename,
            content_type=content_type,
        )
    finally:
        os.unlink(tmp_path)

    if not result.success:
        raise HTTPException(422, "; ".join(result.errors))

    if getattr(ctx, "audit_logger", None):
        await ctx.audit_logger.log(
            tenant_id=user.tenant_id,
            user_id=user.user_id,
            action="document.create",
            resource_type="document",
            resource_id=result.doc_hash or "unknown",
            details={"type": "file", "filename": filename},
            outcome="success",
        )

    return IngestResponse(
        document_id=result.document_id,
        chunk_count=result.chunk_count,
        doc_hash=result.doc_hash,
        status="deduped" if result.deduped else "ingested",
    )


# ---------------------------------------------------------------------------
# Document listing — uses the namespace-docs index
# ---------------------------------------------------------------------------


@router.get("/documents", response_model=list[DocumentResponse])
async def list_documents(
    namespace: str,
    ns_config=Depends(ACLChecker(Permission.READ)),
    user: AuthenticatedUser = Depends(get_current_user),
    ctx=Depends(get_system_context),
):
    """List all documents ingested into *namespace*."""
    registry = ctx.document_registry
    doc_entries = await registry.get_namespace_documents(namespace)
    if not doc_entries:
        return []

    doc_store = getattr(ctx, "document_content_store", None)
    docs = []
    for entry in doc_entries:
        dh = entry.get("doc_hash", "")
        did = entry.get("document_id", "")
        reg_entry = await registry.get_document(user.tenant_id, dh)

        # Pull metadata from document content store if available
        original_filename = ""
        content_type_val = ""
        size_bytes = 0
        if doc_store:
            meta = await doc_store.get_document_metadata(user.tenant_id, dh)
            if meta:
                original_filename = meta.original_filename
                content_type_val = meta.content_type
                size_bytes = meta.size_bytes

        docs.append(
            DocumentResponse(
                document_id=did,
                doc_hash=dh,
                namespaces=list(reg_entry.namespaces) if reg_entry else [],
                chunk_count=len(reg_entry.chunk_content_hashes) if reg_entry else 0,
                original_filename=original_filename,
                content_type=content_type_val,
                size_bytes=size_bytes,
            )
        )
    return docs


# ---------------------------------------------------------------------------
# Document download — retrieves the original uploaded file
# ---------------------------------------------------------------------------


@router.get("/documents/{doc_hash}/download")
async def download_document(
    doc_hash: str,
    namespace: str,
    ns_config=Depends(ACLChecker(Permission.READ)),
    user: AuthenticatedUser = Depends(get_current_user),
    ctx=Depends(get_system_context),
):
    """Download the original uploaded document by its content hash."""
    doc_store = getattr(ctx, "document_content_store", None)
    if not doc_store:
        raise HTTPException(501, "Document content store not configured")

    data = await doc_store.get_document(user.tenant_id, doc_hash)
    if data is None:
        raise HTTPException(404, "Document not found")

    meta = await doc_store.get_document_metadata(user.tenant_id, doc_hash)
    filename = meta.original_filename if meta else f"{doc_hash}.bin"
    media_type = meta.content_type if meta else "application/octet-stream"

    return Response(
        content=data,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


# ---------------------------------------------------------------------------
# Document deletion
# ---------------------------------------------------------------------------


@router.delete("/documents/{doc_hash}")
async def delete_document(
    doc_hash: str,
    namespace: str,
    background: bool = True,
    ns_config=Depends(ACLChecker(Permission.DELETE)),
    user: AuthenticatedUser = Depends(get_current_user),
    ctx=Depends(get_system_context),
):
    # ---- async Inngest path ----
    inngest_client = getattr(ctx, "inngest_client", None)
    if background and inngest_client:
        job_id = uuid.uuid4().hex
        await save_job_state(
            ctx.kv_store,
            IngestionJobState(
                job_id=job_id,
                tenant_id=user.tenant_id,
                namespace=namespace,
                operation="delete",
                doc_hash=doc_hash,
                stage=JobStage.DISPATCHED,
            ),
        )
        try:
            import inngest
            await inngest_client.send(
                inngest.Event(
                    name="document/delete-requested",
                    data={
                        "tenant_id": user.tenant_id,
                        "namespace": namespace,
                        "doc_hash": doc_hash,
                        "job_id": job_id,
                    },
                )
            )
            logger.info("Dispatched async delete for doc %s", doc_hash)

            if getattr(ctx, "audit_logger", None):
                await ctx.audit_logger.log(
                    tenant_id=user.tenant_id,
                    user_id=user.user_id,
                    action="document.delete",
                    resource_type="document",
                    resource_id=doc_hash,
                    details={"mode": "background"},
                    outcome="success",
                )

            return {
                "status": "delete_processing",
                "doc_hash": doc_hash,
                "job_id": job_id,
            }
        except Exception:
            await ctx.kv_store.delete(IngestionJobState.kv_key(job_id))
            logger.exception("Inngest delete dispatch failed, falling back to sync")

    # ---- sync path ----
    result = await ctx.ingestion_pipeline.delete_document(
        tenant_id=user.tenant_id,
        document_hash=doc_hash,
        namespace=namespace,
    )
    if not result.found:
        raise HTTPException(404, "Document not found")
            
    if getattr(ctx, "audit_logger", None):
        await ctx.audit_logger.log(
            tenant_id=user.tenant_id,
            user_id=user.user_id,
            action="document.delete",
            resource_type="document",
            resource_id=doc_hash,
            outcome="success",
        )

    return {
        "status": "deleted",
        "doc_hash": doc_hash,
        "vectors_deleted": result.vectors_deleted,
        "nodes_deleted": result.nodes_deleted,
    }


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
    ctx=Depends(get_system_context),
):
    job = await load_job_state(ctx.kv_store, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.tenant_id != user.tenant_id:
        raise HTTPException(403, "Cross-tenant access denied")

    return JobStatusResponse(
        job_id=job.job_id,
        operation=job.operation,
        tenant_id=job.tenant_id,
        namespace=job.namespace,
        status=job.status.value,
        stage=job.stage.value,
        document_id=job.document_id,
        doc_hash=job.doc_hash,
        error=job.error,
        result=job.result,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )
