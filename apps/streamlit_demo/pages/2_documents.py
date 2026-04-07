"""Document upload and management page."""

import streamlit as st
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from api_client import MemoryAPIClient

st.title("Documents")


def get_client() -> MemoryAPIClient:
    return MemoryAPIClient(
        base_url=st.session_state.get("api_base_url", "http://localhost:8000"),
        token=st.session_state.get("api_token", ""),
    )


client = get_client()

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
    if uploaded and st.button("Ingest File"):
        with st.spinner("Processing..."):
            try:
                result = client.ingest_file(namespace, uploaded.read(), uploaded.name)
                st.success(f"Ingested: {result.get('chunk_count', 0)} chunks")
            except Exception as e:
                st.error(str(e))

with tab2:
    text = st.text_area("Text content", height=200)
    title = st.text_input("Title (optional)")
    if st.button("Ingest Text") and text:
        with st.spinner("Processing..."):
            try:
                result = client.ingest_text(namespace, text, title=title)
                st.success(f"Ingested: {result.get('chunk_count', 0)} chunks")
            except Exception as e:
                st.error(str(e))

# Document list
st.subheader("Documents in namespace")
try:
    docs = client.list_documents(namespace)
    for doc in docs:
        col1, col2 = st.columns([3, 1])
        col1.write(f"**{doc['document_id']}** — {doc.get('chunk_count', 0)} chunks")
        if col2.button("Delete", key=doc["doc_hash"]):
            try:
                client.delete_document(namespace, doc["doc_hash"])
                st.rerun()
            except Exception as e:
                st.error(str(e))
except Exception as e:
    st.info(f"No documents or error: {e}")
