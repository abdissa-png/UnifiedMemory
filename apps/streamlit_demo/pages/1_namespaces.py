"""Namespace management page."""

import streamlit as st
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from api_client import MemoryAPIClient

st.title("Namespaces")


def get_client() -> MemoryAPIClient:
    return MemoryAPIClient(
        base_url=st.session_state.get("api_base_url", "http://localhost:8000"),
        token=st.session_state.get("api_token", ""),
    )


client = get_client()

# Create namespace
with st.expander("Create Namespace", expanded=True):
    agent_id = st.text_input("Agent ID (optional)")
    session_id = st.text_input("Session ID (optional)")
    scope = st.selectbox("Scope", ["private", "shared", "public"])
    if st.button("Create"):
        try:
            result = client.create_namespace(
                agent_id=agent_id or None,
                session_id=session_id or None,
                scope=scope,
            )
            st.success(f"Created: {result['namespace_id']}")
        except Exception as e:
            st.error(str(e))

# List namespaces
st.subheader("Your Namespaces")
try:
    namespaces = client.list_namespaces()
    ns_ids = [ns["namespace_id"] for ns in namespaces]
    if ns_ids:
        selected_namespace = st.selectbox("Select namespace", ns_ids)
        selected = next(ns for ns in namespaces if ns["namespace_id"] == selected_namespace)
        st.json(selected)
    else:
        selected_namespace = ""
        st.info("No namespaces found yet.")

    for ns in namespaces:
        st.write(f"**{ns['namespace_id']}** — scope: {ns['scope']}")
except Exception as e:
    selected_namespace = ""
    st.warning(f"Could not load namespaces: {e}")

with st.expander("Namespace Config / Delete"):
    config_namespace = st.text_input(
        "Namespace ID",
        value=selected_namespace,
        key="namespace_config_id",
    )
    col1, col2 = st.columns(2)
    if col1.button("Load Config", disabled=not config_namespace):
        try:
            st.json(client.get_namespace_config(config_namespace))
        except Exception as e:
            st.error(str(e))
    if col2.button("Delete Namespace", disabled=not config_namespace):
        try:
            result = client.delete_namespace(config_namespace)
            st.success(result.get("status", "deleted"))
            st.rerun()
        except Exception as e:
            st.error(str(e))

# Share namespace
with st.expander("Share Namespace"):
    ns_to_share = st.text_input("Namespace ID to share", value=selected_namespace)
    target_email = st.text_input("Target user email")
    perms = st.multiselect("Permissions", ["read", "write", "delete", "share"], default=["read"])
    if st.button("Share") and ns_to_share and target_email:
        try:
            result = client.share_namespace(ns_to_share, target_email, perms)
            st.success(f"Shared with {result.get('target_user_id', 'user')}")
        except Exception as e:
            st.error(str(e))

with st.expander("Unshare Namespace"):
    ns_to_unshare = st.text_input("Namespace ID", value=selected_namespace, key="unshare_ns")
    target_user_id = st.text_input("Target user ID")
    if st.button("Unshare") and ns_to_unshare and target_user_id:
        try:
            result = client.unshare_namespace(ns_to_unshare, target_user_id)
            st.success(result.get("status", "unshared"))
        except Exception as e:
            st.error(str(e))
