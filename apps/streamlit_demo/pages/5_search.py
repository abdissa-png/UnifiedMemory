"""Search and QA page."""

import json
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from api_client import MemoryAPIClient

st.title("Search")


def get_client() -> MemoryAPIClient:
    return MemoryAPIClient(
        base_url=st.session_state.get("api_base_url", "http://localhost:8000"),
        token=st.session_state.get("api_token", ""),
    )


client = get_client()

try:
    namespaces = client.list_namespaces()
    ns_ids = [ns["namespace_id"] for ns in namespaces]
except Exception:
    ns_ids = []

namespace = st.selectbox("Namespace", ns_ids) if ns_ids else st.text_input("Namespace ID")

if not namespace:
    st.info("Select or enter a namespace first.")
    st.stop()

tab1, tab2 = st.tabs(["Search", "Answer"])

with tab1:
    query = st.text_input("Query", key="search_query")
    top_k = st.number_input("Top K", min_value=1, max_value=100, value=10)
    paths = st.multiselect("Paths", ["dense", "sparse", "graph"], default=["dense", "sparse", "graph"])
    rerank = st.checkbox("Rerank", value=False)
    fusion_method = st.selectbox("Fusion method", ["rrf", "linear"])
    score_threshold_enabled = st.checkbox("Use score threshold", value=False)
    score_threshold = st.number_input(
        "Score threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        disabled=not score_threshold_enabled,
    )
    target_namespaces = st.multiselect(
        "Target namespaces (optional)",
        [ns_id for ns_id in ns_ids if ns_id != namespace],
    )
    filters_text = st.text_area(
        "Filters JSON (optional)",
        value="",
        help='Example: {"document_id": "abc123"}',
    )

    if st.button("Run Search") and query:
        try:
            filters = json.loads(filters_text) if filters_text.strip() else None
            result = client.search(
                namespace=namespace,
                query=query,
                top_k=int(top_k),
                paths=paths,
                rerank=rerank,
                filters=filters,
                fusion_method=fusion_method,
                score_threshold=score_threshold if score_threshold_enabled else None,
                target_namespaces=target_namespaces or None,
            )
            st.caption(
                f"{result.get('total_results', 0)} results using "
                f"{result.get('fusion_method', fusion_method)}"
            )
            for item in result.get("results", []):
                with st.container(border=True):
                    st.write(item.get("content", ""))
                    st.caption(
                        f"id={item.get('id', '')} | score={item.get('score', 0):.3f} | "
                        f"source={item.get('source', '')}"
                    )
                    if item.get("metadata"):
                        st.json(item["metadata"])
        except json.JSONDecodeError as e:
            st.error(f"Invalid filters JSON: {e}")
        except Exception as e:
            st.error(str(e))

with tab2:
    question = st.text_area("Question", height=140)
    answer_top_k = st.number_input("Top K for answer", min_value=1, max_value=100, value=10)
    if st.button("Generate Answer") and question:
        try:
            result = client.search_answer(namespace, question, top_k=int(answer_top_k))
            st.subheader("Answer")
            st.write(result.get("answer", ""))
            sources = result.get("sources", [])
            if sources:
                with st.expander("Sources", expanded=True):
                    for src in sources:
                        st.caption(f"Score: {src.get('score', 0):.3f}")
                        st.write(src.get("content", ""))
            trace = result.get("reasoning_trace")
            if trace:
                with st.expander("Reasoning Trace"):
                    st.json(trace)
        except Exception as e:
            st.error(str(e))
