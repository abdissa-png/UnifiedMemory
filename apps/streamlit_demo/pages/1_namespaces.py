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
    scope = st.selectbox("Scope", ["private", "shared", "public"])
    if st.button("Create"):
        try:
            result = client.create_namespace(agent_id=agent_id or None, scope=scope)
            st.success(f"Created: {result['namespace_id']}")
        except Exception as e:
            st.error(str(e))

# List namespaces
st.subheader("Your Namespaces")
try:
    namespaces = client.list_namespaces()
    for ns in namespaces:
        st.write(f"**{ns['namespace_id']}** — scope: {ns['scope']}")
except Exception as e:
    st.warning(f"Could not load namespaces: {e}")

# Share namespace
with st.expander("Share Namespace"):
    ns_to_share = st.text_input("Namespace ID to share")
    target_email = st.text_input("Target user email")
    perms = st.multiselect("Permissions", ["read", "write", "delete", "share"])
    if st.button("Share") and ns_to_share and target_email:
        try:
            result = client.share_namespace(ns_to_share, target_email, perms)
            st.success(f"Shared with {result.get('target_user_id', 'user')}")
        except Exception as e:
            st.error(str(e))
