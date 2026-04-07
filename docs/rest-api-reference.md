# REST API reference

Base URL is the host where **FastAPI** is mounted (e.g. `http://localhost:8000`). All versioned routes use the **`/v1`** prefix unless noted.

**Authentication:** Most endpoints require `Authorization: Bearer <JWT>`. Obtain a token via **`POST /v1/auth/login`** (or registration flows). Exceptions are noted below.

**ACL:** Namespace-scoped routes use **`ACLChecker`** with a required [`Permission`](./domain-model-and-types.md#permissions-and-acl) (`READ`, `WRITE`, `DELETE`, `ADMIN`, `SHARE`). Cross-tenant access returns **403**.

---

## Health

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| `GET` | `/health` | No | Liveness (`{"status": "ok"}`) |

---

## Auth (`/v1/auth`)

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| `POST` | `/v1/auth/register-tenant` | No | Create tenant + initial admin user; returns JWT |
| `POST` | `/v1/auth/register-user` | Yes | Register a user under the caller’s tenant |
| `POST` | `/v1/auth/login` | No | Email/password → JWT |

Implementation: [`api/routes/auth.py`](../src/unified_memory/api/routes/auth.py).

---

## Namespaces (`/v1/namespaces`)

| Method | Path | ACL / notes |
| --- | --- | --- |
| `POST` | `/v1/namespaces` | Authenticated; creates namespace for tenant |
| `GET` | `/v1/namespaces` | List for current user/tenant |
| `GET` | `/v1/namespaces/{namespace}/config` | **READ** on namespace |
| `DELETE` | `/v1/namespaces/{namespace}` | **ADMIN** |
| `POST` | `/v1/namespaces/{namespace}/share` | **SHARE** |
| `DELETE` | `/v1/namespaces/{namespace}/share/{user_id}` | **SHARE** |

`{namespace}` is a path segment (may contain slashes per FastAPI `path` type).

---

## Ingestion & documents (`/v1`)

| Method | Path | ACL |
| --- | --- | --- |
| `POST` | `/v1/ingest/text/{namespace}` | **WRITE** |
| `POST` | `/v1/ingest/file` | **WRITE** (multipart upload) |
| `GET` | `/v1/documents` | **READ** (query params for listing) |
| `GET` | `/v1/documents/{doc_hash}/download` | **READ** |
| `DELETE` | `/v1/documents/{doc_hash}` | **DELETE** |

---

## Search (`/v1`)

| Method | Path | ACL |
| --- | --- | --- |
| `POST` | `/v1/search/{namespace}` | **READ** — unified hybrid search |
| `POST` | `/v1/search/answer/{namespace}` | **READ** — retrieval + grounded answer |

---

## Chat (`/v1/chat`)

| Method | Path | ACL / notes |
| --- | --- | --- |
| `POST` | `/v1/chat/sessions/{namespace}` | **READ** — create session |
| `GET` | `/v1/chat/sessions?namespace=...` | **READ** on namespace — list sessions for that namespace |
| `POST` | `/v1/chat/sessions/{session_id}/messages` | Authenticated + session ownership |
| `GET` | `/v1/chat/sessions/{session_id}/messages` | Authenticated |
| `POST` | `/v1/chat/sessions/{session_id}/documents` | Authenticated — attach docs to session |

---

## Admin (`/v1/admin`)

| Method | Path | Notes |
| --- | --- | --- |
| `GET` | `/v1/admin/tenants/{tenant_id}` | Authenticated; tenant isolation enforced in handler |
| `PUT` | `/v1/admin/tenants/{tenant_id}` | Update tenant config |
| `DELETE` | `/v1/admin/users/{user_id}/data` | User data purge |
| `GET` | `/v1/admin/users/{user_id}/data` | Export / inspect user data |

Review handler logic before relying on admin guarantees in production.

---

## Inngest

When enabled, the Inngest **serve** handler is mounted by the FastAPI app (see [workflows.md](./workflows.md)). Event shapes and external URLs depend on your Inngest deployment.

---

## Request/response schemas

Shared Pydantic models live in [`api/schemas.py`](../src/unified_memory/api/schemas.py). Prefer importing types from there when building clients.

## Related

- [api-http-and-observability.md](./api-http-and-observability.md) — lifespan, middleware, tracing
- [namespaces-tenants-auth.md](./namespaces-tenants-auth.md) — tenant and namespace semantics
