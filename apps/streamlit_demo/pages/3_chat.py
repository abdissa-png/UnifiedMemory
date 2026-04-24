"""Chat interface page."""

import streamlit as st
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from api_client import MemoryAPIClient

st.title("Chat with Documents")


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

# Session management
col_new, col_list = st.columns([1, 2])
with col_new:
    new_session_title = st.text_input("New session title", key="new_session_title")
    if st.button("New Session"):
        try:
            session = client.create_session(namespace, title=new_session_title)
            st.session_state["chat_session_id"] = session["id"]
            st.rerun()
        except Exception as e:
            st.error(str(e))

with col_list:
    try:
        sessions = client.list_sessions(namespace)
        session_options = {s["id"]: s.get("title") or s["id"][:12] for s in sessions}
        if session_options:
            selected = st.selectbox(
                "Session",
                list(session_options.keys()),
                format_func=lambda x: session_options[x],
            )
            st.session_state["chat_session_id"] = selected
    except Exception:
        pass

session_id = st.session_state.get("chat_session_id")
if not session_id:
    st.info("Create or select a chat session.")
    st.stop()

with st.expander("Associate Document"):
    try:
        docs = client.list_documents(namespace)
    except Exception:
        docs = []
    doc_options = {
        doc["document_id"]: doc.get("original_filename") or doc["document_id"]
        for doc in docs
    }
    selected_doc_id = st.selectbox(
        "Document",
        [""] + list(doc_options.keys()),
        format_func=lambda doc_id: doc_options.get(doc_id, "Select a document") if doc_id else "Select a document",
    )
    if st.button("Associate Document", disabled=not selected_doc_id):
        try:
            result = client.associate_document(session_id, selected_doc_id)
            st.success(result.get("status", "associated"))
        except Exception as e:
            st.error(str(e))

# Display messages
try:
    messages = client.get_messages(session_id)
except Exception:
    messages = []

for msg in messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question..."):
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = client.send_message(session_id, prompt)
                st.write(response["content"])
                sources = response.get("sources", [])
                if sources:
                    with st.expander("Sources"):
                        for src in sources:
                            st.caption(f"Score: {src.get('score', 0):.2f}")
                            st.text(src.get("content", "")[:300])
            except Exception as e:
                st.error(str(e))
