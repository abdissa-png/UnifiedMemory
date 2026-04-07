"""Tenant admin page."""

import streamlit as st
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from api_client import MemoryAPIClient

st.title("Tenant Administration")


def get_client() -> MemoryAPIClient:
    return MemoryAPIClient(
        base_url=st.session_state.get("api_base_url", "http://localhost:8000"),
        token=st.session_state.get("api_token", ""),
    )


client = get_client()

tab1, tab2 = st.tabs(["Register User", "Register New Tenant"])

with tab1:
    st.subheader("Add User to Tenant")
    user_email = st.text_input("Email")
    user_password = st.text_input("Password", type="password")
    user_name = st.text_input("Display Name")
    if st.button("Add User") and user_email and user_password:
        try:
            result = client.register_user(user_email, user_password, user_name)
            st.success("User created and token generated.")
        except Exception as e:
            st.error(str(e))

with tab2:
    st.subheader("Create New Tenant")
    new_tenant_id = st.text_input("Tenant ID")
    admin_email = st.text_input("Admin Email", key="new_admin_email")
    admin_password = st.text_input("Admin Password", type="password", key="new_admin_pw")
    if st.button("Create Tenant") and new_tenant_id:
        try:
            result = client.register_tenant(new_tenant_id, admin_email, admin_password)
            st.success(f"Tenant '{new_tenant_id}' created. Token set.")
            st.session_state.api_token = result["access_token"]
        except Exception as e:
            st.error(str(e))
