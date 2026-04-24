"""Tenant admin page."""

import streamlit as st
import sys, os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from api_client import MemoryAPIClient
from tenant_registration import build_tenant_config_overrides, render_tenant_config_inputs

st.title("Tenant Administration")


def get_client() -> MemoryAPIClient:
    return MemoryAPIClient(
        base_url=st.session_state.get("api_base_url", "http://localhost:8000"),
        token=st.session_state.get("api_token", ""),
    )


client = get_client()

tab1, tab2, tab3, tab4 = st.tabs(
    ["Register User", "Register New Tenant", "Tenant Config", "GDPR Tools"]
)

with tab1:
    st.subheader("Add User to Tenant")
    user_email = st.text_input("Email")
    user_password = st.text_input("Password", type="password")
    user_name = st.text_input("Display Name")
    user_roles = st.multiselect(
        "Roles",
        ["tenant_member", "tenant_admin"],
        default=["tenant_member"],
    )
    if st.button("Add User") and user_email and user_password:
        try:
            client.register_user(user_email, user_password, user_name, roles=user_roles)
            st.success("User created and token generated.")
        except Exception as e:
            st.error(str(e))

with tab2:
    st.subheader("Create New Tenant")
    tenant_name = st.text_input("Tenant Name")
    admin_email = st.text_input("Admin Email", key="new_admin_email")
    admin_password = st.text_input("Admin Password", type="password", key="new_admin_pw")
    admin_display_name = st.text_input("Admin Display Name", key="new_admin_display_name")
    raw_config_values = render_tenant_config_inputs("admin_register")
    if st.button("Create Tenant") and admin_email and admin_password:
        try:
            config_overrides = build_tenant_config_overrides(raw_config_values)
            result = client.register_tenant(
                admin_email=admin_email,
                admin_password=admin_password,
                tenant_name=tenant_name,
                admin_display_name=admin_display_name,
                **config_overrides,
            )
            st.success(f"Tenant created: {result.get('tenant_id', '')}")
            st.session_state.api_token = result["access_token"]
            st.session_state.tenant_id = result.get("tenant_id", "")
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(str(e))

with tab3:
    st.subheader("Tenant Configuration")
    tenant_id = st.text_input(
        "Tenant ID",
        value=st.session_state.get("tenant_id", ""),
        key="tenant_config_tenant_id",
    )
    current_config = None
    if st.button("Load Tenant Config", disabled=not tenant_id):
        try:
            current_config = client.get_tenant_config(tenant_id)
            st.session_state["tenant_config_cache"] = current_config
        except Exception as e:
            st.error(str(e))
    else:
        current_config = st.session_state.get("tenant_config_cache")

    if current_config:
        st.json(current_config)
        default_payload = json.dumps(
            {
                "chunk_size": current_config.get("chunk_size"),
                "chunk_overlap": current_config.get("chunk_overlap"),
                "chunker_type": current_config.get("chunker_type"),
                "enable_graph_storage": current_config.get("enable_graph_storage"),
                "enable_visual_indexing": current_config.get("enable_visual_indexing"),
                "enable_entity_extraction": current_config.get("enable_entity_extraction"),
                "enable_relation_extraction": current_config.get("enable_relation_extraction"),
                "batch_size": current_config.get("batch_size"),
                "deduplication_enabled": current_config.get("deduplication_enabled"),
            },
            indent=2,
        )
    else:
        default_payload = "{}"

    update_payload_text = st.text_area(
        "Update payload JSON",
        value=default_payload,
        height=220,
        help="Only include fields accepted by UpdateTenantConfigRequest.",
    )
    if st.button("Update Tenant Config", disabled=not tenant_id):
        try:
            payload = json.loads(update_payload_text)
            updated = client.update_tenant_config(tenant_id, payload)
            st.session_state["tenant_config_cache"] = updated
            st.success("Tenant config updated.")
            st.json(updated)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
        except Exception as e:
            st.error(str(e))

with tab4:
    st.subheader("GDPR Export / Erase")
    gdpr_user_id = st.text_input("User ID")
    col1, col2 = st.columns(2)
    if col1.button("Export User Data", disabled=not gdpr_user_id):
        try:
            st.json(client.gdpr_export(gdpr_user_id))
        except Exception as e:
            st.error(str(e))
    if col2.button("Erase User Data", disabled=not gdpr_user_id):
        try:
            st.json(client.gdpr_erase(gdpr_user_id))
        except Exception as e:
            st.error(str(e))
