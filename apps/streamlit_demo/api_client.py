"""
HTTP client for the Unified Memory System REST API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx


class MemoryAPIClient:
    """Thin wrapper around the UMS API for use by the Streamlit frontend."""

    def __init__(self, base_url: str = "http://localhost:8000", token: str = "") -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token

    @property
    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    @property
    def _auth_headers(self) -> Dict[str, str]:
        if not self.token:
            return {}
        return {"Authorization": f"Bearer {self.token}"}

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
    ) -> Any:
        resp = httpx.request(
            method,
            f"{self.base_url}{path}",
            json=json,
            params=params,
            headers=self._headers,
            timeout=timeout,
        )
        self._raise_for_status(resp)
        return resp.json()

    @staticmethod
    def _raise_for_status(resp: httpx.Response) -> None:
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            details = ""
            try:
                payload = resp.json()
                if isinstance(payload, dict):
                    detail = payload.get("detail")
                    if detail:
                        details = str(detail)
                    else:
                        details = str(payload)
                else:
                    details = str(payload)
            except Exception:
                details = resp.text.strip()

            if details:
                raise RuntimeError(
                    f"{resp.status_code} {resp.reason_phrase}: {details}"
                ) from exc
            raise

    # ---- auth --------------------------------------------------------------

    def register_tenant(
        self,
        admin_email: str,
        admin_password: str,
        tenant_name: str = "",
        admin_display_name: str = "",
        *,
        text_embedding_provider: Optional[str] = None,
        text_embedding_model: Optional[str] = None,
        text_embedding_dimension: Optional[int] = None,
        vision_embedding_provider: Optional[str] = None,
        vision_embedding_model: Optional[str] = None,
        vision_embedding_dimension: Optional[int] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        chunker_type: Optional[str] = None,
        enable_graph_storage: Optional[bool] = None,
        enable_visual_indexing: Optional[bool] = None,
        enable_entity_extraction: Optional[bool] = None,
        enable_relation_extraction: Optional[bool] = None,
        batch_size: Optional[int] = None,
        deduplication_enabled: Optional[bool] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        extraction_extractor_type: Optional[str] = None,
        extraction_llm_model: Optional[str] = None,
        extraction_entity_types: Optional[List[str]] = None,
        extraction_relation_types: Optional[List[str]] = None,
        extraction_confidence_threshold: Optional[float] = None,
        extraction_batch_size: Optional[int] = None,
        extraction_strict_type_filtering: Optional[bool] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "admin_email": admin_email,
            "admin_password": admin_password,
            "tenant_name": tenant_name,
            "admin_display_name": admin_display_name,
        }
        optional_fields = {
            "text_embedding_provider": text_embedding_provider,
            "text_embedding_model": text_embedding_model,
            "text_embedding_dimension": text_embedding_dimension,
            "vision_embedding_provider": vision_embedding_provider,
            "vision_embedding_model": vision_embedding_model,
            "vision_embedding_dimension": vision_embedding_dimension,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunker_type": chunker_type,
            "enable_graph_storage": enable_graph_storage,
            "enable_visual_indexing": enable_visual_indexing,
            "enable_entity_extraction": enable_entity_extraction,
            "enable_relation_extraction": enable_relation_extraction,
            "batch_size": batch_size,
            "deduplication_enabled": deduplication_enabled,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "extraction_extractor_type": extraction_extractor_type,
            "extraction_llm_model": extraction_llm_model,
            "extraction_entity_types": extraction_entity_types,
            "extraction_relation_types": extraction_relation_types,
            "extraction_confidence_threshold": extraction_confidence_threshold,
            "extraction_batch_size": extraction_batch_size,
            "extraction_strict_type_filtering": extraction_strict_type_filtering,
        }
        payload.update({key: value for key, value in optional_fields.items() if value is not None})
        data = self._request_json(
            "POST",
            "/v1/auth/register-tenant",
            json=payload,
        )
        self.token = data["access_token"]
        return data

    def register_user(
        self,
        email: str,
        password: str,
        display_name: str = "",
        roles: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "email": email,
            "password": password,
            "display_name": display_name,
        }
        if roles is not None:
            payload["roles"] = roles
        return self._request_json(
            "POST",
            "/v1/auth/register-user",
            json=payload,
        )

    def login(self, email: str, password: str) -> str:
        data = self._request_json(
            "POST",
            "/v1/auth/login",
            json={"email": email, "password": password},
        )
        self.token = data["access_token"]
        return self.token

    # ---- namespaces --------------------------------------------------------

    def create_namespace(
        self,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        scope: str = "private",
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"scope": scope}
        if agent_id is not None:
            payload["agent_id"] = agent_id
        if session_id is not None:
            payload["session_id"] = session_id
        return self._request_json(
            "POST",
            "/v1/namespaces",
            json=payload,
        )

    def list_namespaces(self) -> List[Dict[str, Any]]:
        return self._request_json("GET", "/v1/namespaces")

    def get_namespace_config(self, namespace: str) -> Dict[str, Any]:
        return self._request_json("GET", f"/v1/namespaces/{namespace}/config")

    def delete_namespace(self, namespace: str) -> Dict[str, Any]:
        return self._request_json("DELETE", f"/v1/namespaces/{namespace}")

    def share_namespace(
        self, namespace: str, target_email: str, permissions: List[str]
    ) -> Dict[str, Any]:
        return self._request_json(
            "POST",
            f"/v1/namespaces/{namespace}/share",
            json={"target_user_email": target_email, "permissions": permissions},
        )

    def unshare_namespace(self, namespace: str, user_id: str) -> Dict[str, Any]:
        return self._request_json(
            "DELETE",
            f"/v1/namespaces/{namespace}/share/{user_id}",
        )

    # ---- ingestion ---------------------------------------------------------

    def ingest_text(
        self,
        namespace: str,
        text: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        background: bool = True,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"text": text}
        if title is not None:
            payload["title"] = title
        if metadata is not None:
            payload["metadata"] = metadata
        if session_id is not None:
            payload["session_id"] = session_id
        return self._request_json(
            "POST",
            f"/v1/ingest/text/{namespace}",
            json=payload,
            params={"background": str(background).lower()},
            timeout=30 if background else 120,
        )

    def ingest_file(
        self,
        namespace: str,
        file_bytes: bytes,
        filename: str,
        *,
        title: Optional[str] = None,
        session_id: Optional[str] = None,
        background: bool = True,
    ) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "namespace": namespace,
            "background": str(background).lower(),
        }
        if title is not None:
            data["title"] = title
        if session_id is not None:
            data["session_id"] = session_id
        resp = httpx.post(
            f"{self.base_url}/v1/ingest/file",
            data=data,
            files={"file": (filename, file_bytes)},
            headers=self._auth_headers,
            timeout=30 if background else 120,
        )
        self._raise_for_status(resp)
        return resp.json()

    def list_documents(self, namespace: str) -> List[Dict[str, Any]]:
        return self._request_json(
            "GET",
            "/v1/documents",
            params={"namespace": namespace},
        )

    def download_document(self, namespace: str, doc_hash: str) -> httpx.Response:
        resp = httpx.get(
            f"{self.base_url}/v1/documents/{doc_hash}/download",
            params={"namespace": namespace},
            headers=self._auth_headers,
            timeout=60,
        )
        self._raise_for_status(resp)
        return resp

    def delete_document(
        self,
        namespace: str,
        doc_hash: str,
        *,
        background: bool = True,
    ) -> Dict[str, Any]:
        return self._request_json(
            "DELETE",
            f"/v1/documents/{doc_hash}",
            params={"namespace": namespace, "background": str(background).lower()},
        )

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        return self._request_json("GET", f"/v1/jobs/{job_id}")

    # ---- search ------------------------------------------------------------

    def search(
        self,
        namespace: str,
        query: str,
        top_k: int = 10,
        paths: Optional[List[str]] = None,
        rerank: bool = False,
        filters: Optional[Dict[str, Any]] = None,
        fusion_method: str = "rrf",
        fusion_weights: Optional[Dict[str, float]] = None,
        score_threshold: Optional[float] = None,
        rerank_candidates_limit: int = 50,
        reranker_key: Optional[str] = "bge-local",
        target_namespaces: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "rerank": rerank,
            "fusion_method": fusion_method,
            "rerank_candidates_limit": rerank_candidates_limit,
            "reranker_key": reranker_key,
        }
        if paths is not None:
            payload["paths"] = paths
        if filters is not None:
            payload["filters"] = filters
        if fusion_weights is not None:
            payload["fusion_weights"] = fusion_weights
        if score_threshold is not None:
            payload["score_threshold"] = score_threshold
        if target_namespaces is not None:
            payload["target_namespaces"] = target_namespaces
        return self._request_json(
            "POST",
            f"/v1/search/{namespace}",
            json=payload,
            timeout=60,
        )

    def search_answer(
        self,
        namespace: str,
        query: str,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        return self._request_json(
            "POST",
            f"/v1/search/answer/{namespace}",
            json={"query": query, "top_k": top_k},
            timeout=120,
        )

    # ---- chat --------------------------------------------------------------

    def create_session(self, namespace: str, title: str = "") -> Dict[str, Any]:
        return self._request_json(
            "POST",
            f"/v1/chat/sessions/{namespace}",
            json={"title": title},
        )

    def list_sessions(self, namespace: str) -> List[Dict[str, Any]]:
        return self._request_json(
            "GET",
            "/v1/chat/sessions",
            params={"namespace": namespace},
        )

    def send_message(self, session_id: str, content: str) -> Dict[str, Any]:
        return self._request_json(
            "POST",
            f"/v1/chat/sessions/{session_id}/messages",
            json={"content": content},
            timeout=120,
        )

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        return self._request_json(
            "GET",
            f"/v1/chat/sessions/{session_id}/messages",
        )

    def associate_document(self, session_id: str, document_id: str) -> Dict[str, Any]:
        return self._request_json(
            "POST",
            f"/v1/chat/sessions/{session_id}/documents",
            json={"document_id": document_id},
        )

    # ---- admin -------------------------------------------------------------

    def get_tenant_config(self, tenant_id: str) -> Dict[str, Any]:
        return self._request_json("GET", f"/v1/admin/tenants/{tenant_id}")

    def update_tenant_config(
        self,
        tenant_id: str,
        updates: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self._request_json(
            "PUT",
            f"/v1/admin/tenants/{tenant_id}",
            json=updates,
        )

    def gdpr_export(self, user_id: str) -> Dict[str, Any]:
        return self._request_json("GET", f"/v1/admin/users/{user_id}/data")

    def gdpr_erase(self, user_id: str) -> Dict[str, Any]:
        return self._request_json("DELETE", f"/v1/admin/users/{user_id}/data")
