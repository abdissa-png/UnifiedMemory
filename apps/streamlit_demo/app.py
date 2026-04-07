"""
Streamlit demo — main entry point.

Run with:
    streamlit run apps/streamlit_demo/app.py
"""

import streamlit as st

st.set_page_config(page_title="Unified Memory System", layout="wide")

st.title("Unified Memory System")

if "api_token" not in st.session_state:
    st.session_state.api_token = ""
if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = "http://localhost:8000"
if "tenant_id" not in st.session_state:
    st.session_state.tenant_id = ""

from api_client import MemoryAPIClient


def get_client() -> MemoryAPIClient:
    return MemoryAPIClient(
        base_url=st.session_state.api_base_url,
        token=st.session_state.api_token,
    )


# Login / register form
if not st.session_state.api_token:
    st.subheader("Login")

    with st.form("login_form"):
        base_url = st.text_input("API Base URL", value=st.session_state.api_base_url)
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        col1, col2 = st.columns(2)
        login_btn = col1.form_submit_button("Login")
        register_btn = col2.form_submit_button("Register Tenant")

    if login_btn and email and password:
        st.session_state.api_base_url = base_url
        client = get_client()
        try:
            client.login(email, password)
            st.session_state.api_token = client.token
            st.rerun()
        except Exception as e:
            st.error(f"Login failed: {e}")

    if register_btn and email and password:
        st.session_state.api_base_url = base_url
        tenant_id = st.text_input("Tenant ID", value="default", key="reg_tenant")
        client = get_client()
        try:
            client.register_tenant(tenant_id or "default", email, password)
            st.session_state.api_token = client.token
            st.session_state.tenant_id = tenant_id or "default"
            st.rerun()
        except Exception as e:
            st.error(f"Registration failed: {e}")
else:
    st.success("Authenticated")
    if st.button("Logout"):
        st.session_state.api_token = ""
        st.rerun()

    st.info("Use the sidebar pages to manage namespaces, documents, and chat.")
