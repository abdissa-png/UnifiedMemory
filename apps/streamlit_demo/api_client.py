"""
HTTP client for the Unified Memory System REST API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx


class MemoryAPIClient:
    """Thin wrapper around the UMS API for use by the Streamlit frontend."""

    def __init__(self, base_url: str = "http://localhost:8000", token: str = "") -> None:
        self.base_url = base_url
        self.token = token

    @property
    def _headers(self) -> Dict[str, str]:
        h: Dict[str, str] = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    # ---- auth --------------------------------------------------------------

    def register_tenant(
        self, email: str, password: str, tenant_name: str = "", display_name: str = ""
    ) -> Dict[str, Any]:
        resp = httpx.post(
            f"{self.base_url}/v1/auth/register-tenant",
            json={
                "admin_email": email,
                "admin_password": password,
                "tenant_name": tenant_name,
                "admin_display_name": display_name,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        self.token = data["access_token"]
        return data

    def register_user(self, email: str, password: str, display_name: str = "") -> Dict[str, Any]:
        resp = httpx.post(
            f"{self.base_url}/v1/auth/register-user",
            json={"email": email, "password": password, "display_name": display_name},
            headers=self._headers,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def login(self, email: str, password: str) -> str:
        resp = httpx.post(
            f"{self.base_url}/v1/auth/login",
            json={"email": email, "password": password},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        self.token = data["access_token"]
        return self.token

    # ---- namespaces --------------------------------------------------------

    def create_namespace(
        self, agent_id: Optional[str] = None, scope: str = "private"
    ) -> Dict[str, Any]:
        resp = httpx.post(
            f"{self.base_url}/v1/namespaces",
            json={"agent_id": agent_id, "scope": scope},
            headers=self._headers,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def list_namespaces(self) -> List[Dict[str, Any]]:
        resp = httpx.get(
            f"{self.base_url}/v1/namespaces",
            headers=self._headers,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def share_namespace(
        self, namespace: str, target_email: str, permissions: List[str]
    ) -> Dict[str, Any]:
        resp = httpx.post(
            f"{self.base_url}/v1/namespaces/{namespace}/share",
            json={"target_user_email": target_email, "permissions": permissions},
            headers=self._headers,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    # ---- ingestion ---------------------------------------------------------

    def ingest_text(
        self, namespace: str, text: str, title: Optional[str] = None
    ) -> Dict[str, Any]:
        resp = httpx.post(
            f"{self.base_url}/v1/ingest/text/{namespace}",
            json={"text": text, "title": title},
            headers=self._headers,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()

    def ingest_file(self, namespace: str, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        resp = httpx.post(
            f"{self.base_url}/v1/ingest/file",
            data={"namespace": namespace},
            files={"file": (filename, file_bytes)},
            headers={"Authorization": f"Bearer {self.token}"} if self.token else {},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()

    def list_documents(self, namespace: str) -> List[Dict[str, Any]]:
        resp = httpx.get(
            f"{self.base_url}/v1/documents",
            params={"namespace": namespace},
            headers=self._headers,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def delete_document(self, namespace: str, doc_hash: str) -> Dict[str, Any]:
        resp = httpx.delete(
            f"{self.base_url}/v1/documents/{doc_hash}",
            params={"namespace": namespace},
            headers=self._headers,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    # ---- search ------------------------------------------------------------

    def search(
        self,
        namespace: str,
        query: str,
        top_k: int = 10,
        paths: Optional[List[str]] = None,
        fusion_method: str = "rrf",
        score_threshold: Optional[float] = None,
        target_namespaces: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"query": query, "top_k": top_k, "fusion_method": fusion_method}
        if paths:
            payload["paths"] = paths
        if score_threshold is not None:
            payload["score_threshold"] = score_threshold
        if target_namespaces:
            payload["target_namespaces"] = target_namespaces
        resp = httpx.post(
            f"{self.base_url}/v1/search/{namespace}",
            json=payload,
            headers=self._headers,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()

    def search_answer(
        self, namespace: str, query: str, top_k: int = 10
    ) -> Dict[str, Any]:
        """Call the QA agent endpoint for an LLM-generated answer."""
        resp = httpx.post(
            f"{self.base_url}/v1/search/answer/{namespace}",
            json={"query": query, "top_k": top_k},
            headers=self._headers,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()

    # ---- chat --------------------------------------------------------------

    def create_session(self, namespace: str, title: str = "") -> Dict[str, Any]:
        resp = httpx.post(
            f"{self.base_url}/v1/chat/sessions/{namespace}",
            json={"title": title},
            headers=self._headers,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def list_sessions(self, namespace: str) -> List[Dict[str, Any]]:
        resp = httpx.get(
            f"{self.base_url}/v1/chat/sessions",
            params={"namespace": namespace},
            headers=self._headers,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def send_message(self, session_id: str, content: str) -> Dict[str, Any]:
        resp = httpx.post(
            f"{self.base_url}/v1/chat/sessions/{session_id}/messages",
            json={"content": content},
            headers=self._headers,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        resp = httpx.get(
            f"{self.base_url}/v1/chat/sessions/{session_id}/messages",
            headers=self._headers,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
