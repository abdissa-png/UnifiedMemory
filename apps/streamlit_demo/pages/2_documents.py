"""Document upload and management page."""

import streamlit as st
import sys, os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from api_client import MemoryAPIClient

st.title("Documents")


def get_client() -> MemoryAPIClient:
    return MemoryAPIClient(
        base_url=st.session_state.get("api_base_url", "http://localhost:8000"),
        token=st.session_state.get("api_token", ""),
    )


client = get_client()

if "ingest_jobs" not in st.session_state:
    st.session_state.ingest_jobs = {}


def _job_key(namespace_id: str, job_id: str) -> str:
    return f"{namespace_id}:{job_id}"


def _remember_job(namespace_id: str, result: dict, label: str) -> None:
    job_id = result.get("job_id")
    if not job_id:
        return
    st.session_state.ingest_jobs[_job_key(namespace_id, job_id)] = {
        "job_id": job_id,
        "namespace": namespace_id,
        "label": label,
    }


def _show_ingest_result(namespace_id: str, result: dict, label: str) -> None:
    if result.get("job_id"):
        _remember_job(namespace_id, result, label)
        st.success(
            f"{label} queued. Job ID: `{result['job_id']}`. "
            "Use the job tracker below to monitor progress."
        )
    else:
        st.success(result.get("status", "submitted"))
    st.json(result)

# Namespace selector
try:
    namespaces = client.list_namespaces()
    ns_ids = [ns["namespace_id"] for ns in namespaces]
except Exception:
    ns_ids = []

namespace = st.selectbox("Namespace", ns_ids) if ns_ids else st.text_input("Namespace ID")

if not namespace:
    st.info("Select or enter a namespace first.")
    st.stop()

# Upload
tab1, tab2 = st.tabs(["Upload File", "Paste Text"])

with tab1:
    uploaded = st.file_uploader("Upload Document", type=["pdf", "txt", "md"])
    file_title = st.text_input("Title override (optional)")
    file_session_id = st.text_input("Session ID (optional)", key="file_session_id")
    file_background = st.checkbox("Process in background", value=True, key="file_background")
    if uploaded and st.button("Ingest File"):
        with st.spinner("Submitting upload..."):
            try:
                result = client.ingest_file(
                    namespace,
                    uploaded.read(),
                    uploaded.name,
                    title=file_title or None,
                    session_id=file_session_id or None,
                    background=file_background,
                )
                _show_ingest_result(namespace, result, uploaded.name)
            except Exception as e:
                st.error(str(e))

with tab2:
    text = st.text_area("Text content", height=200)
    title = st.text_input("Title (optional)")
    text_session_id = st.text_input("Session ID (optional)", key="text_session_id")
    text_background = st.checkbox("Process in background", value=True, key="text_background")
    metadata_text = st.text_area(
        "Metadata JSON (optional)",
        value="",
        help='Example: {"source": "demo", "category": "notes"}',
    )
    if st.button("Ingest Text") and text:
        with st.spinner("Submitting text..."):
            try:
                metadata = json.loads(metadata_text) if metadata_text.strip() else None
                result = client.ingest_text(
                    namespace,
                    text,
                    title=title or None,
                    metadata=metadata,
                    session_id=text_session_id or None,
                    background=text_background,
                )
                _show_ingest_result(namespace, result, title or "text ingestion")
            except json.JSONDecodeError as e:
                st.error(f"Invalid metadata JSON: {e}")
            except Exception as e:
                st.error(str(e))

# Document list
st.subheader("Documents in namespace")
try:
    docs = client.list_documents(namespace)
    for doc in docs:
        col1, col2, col3 = st.columns([3, 1, 1])
        col1.write(
            f"**{doc['document_id']}** — {doc.get('chunk_count', 0)} chunks — "
            f"{doc.get('original_filename') or doc['doc_hash']}"
        )
        try:
            download_resp = client.download_document(namespace, doc["doc_hash"])
            col2.download_button(
                "Download",
                data=download_resp.content,
                file_name=doc.get("original_filename") or f"{doc['doc_hash']}.bin",
                mime=doc.get("content_type") or "application/octet-stream",
                key=f"download-{doc['doc_hash']}",
            )
        except Exception:
            col2.caption("Download unavailable")
        if col3.button("Delete", key=doc["doc_hash"]):
            try:
                client.delete_document(namespace, doc["doc_hash"])
                st.rerun()
            except Exception as e:
                st.error(str(e))
except Exception as e:
    st.info(f"No documents or error: {e}")

with st.expander("Job Status"):
    current_jobs = [
        job for job in st.session_state.ingest_jobs.values() if job["namespace"] == namespace
    ]
    if current_jobs:
        st.caption("Background jobs submitted from this page")
        if st.button("Refresh Job Statuses"):
            st.rerun()
        for job in current_jobs:
            try:
                status = client.get_job_status(job["job_id"])
                with st.container(border=True):
                    st.write(f"**{job['label']}**")
                    st.json(status)
                    if status.get("status") in {"succeeded", "failed"} or status.get("stage") in {
                        "finalized",
                        "failed",
                    }:
                        st.session_state.ingest_jobs.pop(
                            _job_key(namespace, job["job_id"]),
                            None,
                        )
            except Exception as e:
                st.error(f"{job['job_id']}: {e}")
    else:
        st.caption("No tracked background jobs for this namespace yet.")

    job_id = st.text_input("Manual Job ID Lookup")
    if st.button("Check Job", disabled=not job_id):
        try:
            st.json(client.get_job_status(job_id))
        except Exception as e:
            st.error(str(e))
