# Domain model and core types

The **`unified_memory.core.types`** module is the **single source of truth** for shared dataclasses, enums, and hash helpers used across ingestion, retrieval, and storage. This page summarizes the **most important concepts**; for exhaustive fields, read the source.

## Content identity and hashing

| Function | Purpose |
| --- | --- |
| `compute_content_hash(content, tenant_id, modality?)` | Chunk-level hash; **tenant** and **modality** scoped (SHA256). Used for deduplication. |
| `compute_document_hash(content, tenant_id)` | Document-level hash; **does not** include embedding model (same normalized doc → same hash per tenant). |
| `compute_vector_id(content_hash, embedding_model, prefix?)` | Deterministic vector id so different embedding models never collide for the same content. |

**Design intent:** explicit **tenant_id** in hashes supports compliance and isolation (see docstring in `types.py`).

## Key dataclasses (conceptual)

- **`Chunk`** — Text (or multimodal) segment with **`SourceReference`**, metadata, links to embeddings.
- **`SourceReference` / `SourceLocation`** — Provenance (file, URL, page, offsets).
- **`RetrievalResult` / `QueryResult`** — Normalized search hits returned by **`UnifiedSearchService`**.
- **`Relation` / graph node-edge types** — Used with **`GraphStoreBackend`** (see `GraphNode`, `GraphEdge` in storage/base and types).

## Enumerations

Examples (non-exhaustive):

| Enum | Role |
| --- | --- |
| `Modality` | `text`, `image`, etc. — routing embeddings and hashes |
| `SourceType` | How content was sourced (file, URL, conversation, …) |
| `MemoryStatus` | Validity of a memory record (`valid`, superseded, …) |
| `CollectionType` | Vector collection routing |
| `Permission` | ACL permissions (see below) |

Many enums exist for forward-compatible **memory taxonomies** (`MemoryType`, `MemoryLayer`); retrieval and ingestion paths use the subset relevant to each feature.

## Permissions and ACL

**`Permission`:** `READ`, `WRITE`, `DELETE`, `ADMIN`, `SHARE`.

**`NamespaceACL`** + **`ACLEntry`** implement **deny-over-allow**, optional **inheritance from tenant** defaults, and **fail-closed** default (no permission → denied).

HTTP enforcement is centralized in **`ACLChecker`** ([`api/deps.py`](../src/unified_memory/api/deps.py)):

- Loads **`NamespaceConfig`** for the path’s namespace.
- Rejects **`user.tenant_id != ns_config.tenant_id`** (cross-tenant).
- Evaluates **`NamespaceACL.check_permission`** with optional tenant default ACL.
- **`public`** namespace **scope** can grant **READ** without a full ACL match (see implementation).

## SQL persistence (API mode)

When using the FastAPI app, **SQLAlchemy** models in **`storage/sql/models.py`** persist:

| Table | Role |
| --- | --- |
| `users` | `tenant_id`, email, password hash, roles |
| `chat_sessions` / `chat_messages` | Conversation history and optional retrieval context JSON |
| `session_documents` | Documents pinned to a chat session |
| `audit_events` | Security-relevant actions |
| `token_usage` | Per-trace token and timing metrics (flushed from tracing) |

These are **orthogonal** to the KV/vector/graph stores: they track **who** did **what** in the API, not the full content plane.

## Related

- [storage-and-cas.md](./storage-and-cas.md) — CAS vs SQL boundaries
- [retrieval-and-search.md](./retrieval-and-search.md) — what `RetrievalResult` carries
- [glossary.md](./glossary.md) — short definitions
