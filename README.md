# Unified Memory System

A **multi-tenant**, **multimodal-aware** memory layer for LLM applications: ingestion (parse → chunk → embed → index), **hybrid retrieval** (dense, sparse, graph) with score fusion and optional reranking, **content-addressable** deduplication, and an optional **FastAPI** service with JWT auth, chat, and observability hooks.

Python package name: `**unified-memory-system`** (`import unified_memory`). See `**pyproject.toml**` for dependencies and extras.

## What this project is (and why it exists)

This is **retrieval and memory infrastructure**, not a thin wrapper around a single vector database or a demo RAG script. The goal is to give you **one coherent system** you can run as a library (`SystemContext`) or as an **HTTP service**: ingest content, deduplicate it with **content-addressable** semantics, index it into **vector**, **lexical** (BM25 or Elasticsearch), and **graph** stores, then **search** through a single `**UnifiedSearchService`** that **fuses** those signals and optionally **reranks**—with **tenants, namespaces, and ACL-aware** APIs when you use FastAPI.

**What tends to be distinctive:** the **combination**—hybrid retrieval as a first-class design, **multi-tenant** auth and permissions in the API, **pluggable** backends behind one YAML-shaped `infra` block, and **operational** hooks (SQL for users/chat/audit/usage, optional **Inngest** workflows). Individual building blocks exist elsewhere; here they are **wired into one deployable package** with documentation in `[docs/](./docs/README.md)`.

**Who it is for:** teams building **assistant or agent platforms**, **internal knowledge APIs**, or **B2B products** that need isolated customer memory and **better-than-vectors-only** search without owning three separate pipelines and score-merge logic by hand.

**If you only need** a quick embeddings + cosine demo, this repo is heavier than you need. **If you need** a serious baseline for **tenant-scoped, hybrid, auditable memory**, start with `[docs/architecture-overview.md](./docs/architecture-overview.md)` and the [REST reference](docs/rest-api-reference.md).

## Technical documentation

Canonical documentation for `**src/unified_memory/`**, `**apps/**`, and `**tests/**` is in `**[docs/README.md](./docs/README.md)**` (full index, diagrams, and reading order).


| Topic                         | Document                                                                                     |
| ----------------------------- | -------------------------------------------------------------------------------------------- |
| **Index / map**               | `[docs/README.md](./docs/README.md)`                                                         |
| Install, env, YAML            | `[docs/setup-and-configuration.md](./docs/setup-and-configuration.md)`                       |
| **Architecture & deployment** | `[docs/architecture-overview.md](./docs/architecture-overview.md)`                           |
| **Layers & dependency rules** | `[docs/system-design-layers.md](./docs/system-design-layers.md)`                             |
| `SystemContext` and bootstrap | `[docs/system-context-and-bootstrap.md](./docs/system-context-and-bootstrap.md)`             |
| **Domain types, hashes, ACL** | `[docs/domain-model-and-types.md](./docs/domain-model-and-types.md)`                         |
| Providers and registry        | `[docs/providers-and-registry.md](./docs/providers-and-registry.md)`                         |
| Ingestion                     | `[docs/ingestion-pipeline.md](./docs/ingestion-pipeline.md)`                                 |
| Search and fusion             | `[docs/retrieval-and-search.md](./docs/retrieval-and-search.md)`                             |
| Storage and CAS               | `[docs/storage-and-cas.md](./docs/storage-and-cas.md)`                                       |
| Namespaces, tenants, auth     | `[docs/namespaces-tenants-auth.md](./docs/namespaces-tenants-auth.md)`                       |
| **REST API reference**        | `[docs/rest-api-reference.md](./docs/rest-api-reference.md)`                                 |
| API lifespan, observability   | `[docs/api-http-and-observability.md](./docs/api-http-and-observability.md)`                 |
| **Security & operations**     | `[docs/security-deployment-and-operations.md](./docs/security-deployment-and-operations.md)` |
| Agents and chat               | `[docs/agents-and-chat.md](./docs/agents-and-chat.md)`                                       |
| Workflows (Inngest)           | `[docs/workflows.md](./docs/workflows.md)`                                                   |
| Streamlit demo                | `[docs/apps-streamlit-demo.md](./docs/apps-streamlit-demo.md)`                               |
| Testing                       | `[docs/testing-strategy.md](./docs/testing-strategy.md)`                                     |
| Class and protocol maps       | `[docs/inheritance-class-diagrams.md](./docs/inheritance-class-diagrams.md)`                 |
| **Glossary**                  | `[docs/glossary.md](./docs/glossary.md)`                                                     |
| **MDX deep dive** (extended diagrams) | `[docs/mdx/README.md](./docs/mdx/README.md)`                                         |

## Features (as implemented in this repository)

- **Ingestion pipeline** — Document parsing (e.g. text; optional PDF via MinerU), **fixed-size / recursive / semantic** chunking, embeddings (text and vision providers), optional LLM **extractors**, writes to **vector**, **graph**, and **sparse** (BM25 in-process or **Elasticsearch**) indexes, plus **CAS** metadata and deduplication.
- **Unified search** — `**UnifiedSearchService`** orchestrates **dense**, **sparse**, and **graph** retrieval (graph path supports **Personalized PageRank**-style operations where configured), then **RRF or linear fusion** and optional **reranking** (e.g. BGE, Cohere, depending on extras).
- **Multi-tenancy** — **Namespaces** and **tenant** configuration via `**NamespaceManager`** / `**TenantManager**`; HTTP API with **JWT** registration and login (`/v1/auth/...`).
- **Agents** — `**QAAgent`** performs iterative planning and search-backed answer generation over the unified retriever.
- **HTTP API** — **FastAPI** app (`unified_memory.api.app`), SQL-backed chat sessions, token usage flush, audit logging; optional **Inngest**-based durable workflows when enabled.
- **Demo UI** — `**apps/streamlit_demo`** Streamlit client against the REST API.
- **Observability** — Tracing hooks (`@traced`), structured logging dependencies; metrics client included with the `**server`** extra.

## Source layout

```
memory_system/
├── src/unified_memory/   # Library: api, agents, auth, cas, core, embeddings,
│                         # ingestion, namespace, observability, retrieval,
│                         # storage, workflows, llm, ...
├── apps/streamlit_demo/  # Streamlit UI against the HTTP API
├── tests/                # pytest: unit + integration
├── config/               # Example YAML (e.g. app.example.yaml)
└── docs/                 # Technical documentation (Markdown + Mermaid)
```

## Quick start

### Install

```bash
pip install -e ".[dev]"
```

For the API server, databases, OpenAI, rerankers, and workflows (see `**pyproject.toml**` extras):

```bash
pip install -e ".[server,streamlit]"
```

### Run the HTTP API

```bash
export UMS_CONFIG=config/app.example.yaml
uvicorn unified_memory.api.app:app --reload
```

Health check: `GET http://localhost:8000/health`.

### Use the library (`SystemContext`)

The composition root is `**SystemContext**` (`unified_memory.bootstrap`). It wires KV, vector, and graph stores, **CAS**, namespaces, **ProviderRegistry**, and—after `**build_services()`**—the **ingestion pipeline** and **unified search** service.

```python
from unified_memory.bootstrap import SystemContext

ctx = SystemContext(
    config={
        "embedding_providers": {
            "mock:test-model": {
                "provider": "mock",
                "model": "test-model",
                "dimension": 32,
            }
        }
    }
)
ctx.build_services(default_text_embedding_key="mock:test-model")

assert ctx.ingestion_pipeline is not None
assert ctx.search_service is not None
# Ingestion and search require a valid tenant/namespace setup; see docs/ and tests/.
```

Loading from YAML (recommended for real runs):

```python
from unified_memory.bootstrap import SystemContext

ctx = SystemContext.from_config_file("config/app.example.yaml")
ctx.build_services()
```

### Run the Streamlit demo

```bash
cd apps/streamlit_demo
streamlit run app.py
```

Point the app at your API base URL (default `http://localhost:8000`).

## Configuration

- **Application and infrastructure** — Edit `**config/app.example.yaml`** (copy to your own path and set `**UMS_CONFIG**`). Defines `infra` (KV, vector, graph, sparse backend), **embedding**, **LLM**, **extractor**, and **reranker** sections.
- **Environment** — Copy `**.env.example`** to `.env` for URLs, API keys, `**UMS_JWT_SECRET**`, `**UMS_DATABASE_URL**`, `**UMS_ENABLE_INNGEST**`, etc.

Details: `[docs/setup-and-configuration.md](./docs/setup-and-configuration.md)`.

## Continuous integration

GitHub Actions runs **unit tests** (Python 3.10 and 3.12) and **integration tests** (Docker services via `docker-compose.test.yml`). See `[.github/workflows/ci.yml](./.github/workflows/ci.yml)` and `[docs/testing-strategy.md](./docs/testing-strategy.md)`.

## License

MIT License