"""
Streamlit demo — main entry point.

Run with:
    streamlit run apps/streamlit_demo/app.py
"""

import os

import streamlit as st

st.set_page_config(page_title="Unified Memory System", layout="wide")

st.title("Unified Memory System")

if "api_token" not in st.session_state:
    st.session_state.api_token = ""
if "api_base_url" not in st.session_state:
    # In Docker Compose, set UMS_API_BASE_URL=http://ums-backend:8000 (server-side HTTP from this container).
    st.session_state.api_base_url = os.environ.get(
        "UMS_API_BASE_URL", "http://localhost:8000"
    )
if "tenant_id" not in st.session_state:
    st.session_state.tenant_id = ""

from api_client import MemoryAPIClient
from tenant_registration import build_tenant_config_overrides, render_tenant_config_inputs


def get_client() -> MemoryAPIClient:
    return MemoryAPIClient(
        base_url=st.session_state.api_base_url,
        token=st.session_state.api_token,
    )


# Login / register form
if not st.session_state.api_token:
    login_tab, register_tab = st.tabs(["Login", "Register Tenant"])

    with login_tab:
        with st.form("login_form"):
            base_url = st.text_input("API Base URL", value=st.session_state.api_base_url)
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            login_btn = st.form_submit_button("Login")

    if login_btn and email and password:
        st.session_state.api_base_url = base_url
        client = get_client()
        try:
            client.login(email, password)
            st.session_state.api_token = client.token
            st.rerun()
        except Exception as e:
            st.error(f"Login failed: {e}")

    with register_tab:
        with st.form("register_tenant_form"):
            reg_base_url = st.text_input(
                "API Base URL",
                value=st.session_state.api_base_url,
                key="reg_base_url",
            )
            admin_email = st.text_input("Admin Email", key="reg_admin_email")
            admin_password = st.text_input(
                "Admin Password",
                type="password",
                key="reg_admin_password",
            )
            tenant_name = st.text_input("Tenant Name", key="reg_tenant_name")
            display_name = st.text_input("Admin Display Name", key="reg_display_name")
            raw_config_values = render_tenant_config_inputs("main_register")
            register_btn = st.form_submit_button("Register Tenant")

    if register_btn and admin_email and admin_password:
        st.session_state.api_base_url = reg_base_url
        client = get_client()
        try:
            config_overrides = build_tenant_config_overrides(raw_config_values)
            result = client.register_tenant(
                admin_email=admin_email,
                admin_password=admin_password,
                tenant_name=tenant_name,
                admin_display_name=display_name,
                **config_overrides,
            )
            st.session_state.api_token = client.token
            st.session_state.tenant_id = result.get("tenant_id", "")
            st.rerun()
        except ValueError as e:
            st.error(f"Registration failed: {e}")
        except Exception as e:
            st.error(f"Registration failed: {e}")
else:
    st.success("Authenticated")
    if st.button("Logout"):
        st.session_state.api_token = ""
        st.rerun()

    st.info("Use the sidebar pages to manage namespaces, documents, search, chat, and admin tools.")
