# Unified Memory System — Documentation

This folder is the **canonical** technical documentation for the `unified-memory-system` Python package (`src/unified_memory/`), the **`apps/`** demos, and the **`tests/`** suite. It uses **Markdown** and **Mermaid** diagrams (GitHub and most editors render both).

### Extended visual deep dive (MDX)

For a **longer, diagram-rich** walkthrough (multi-page **`.mdx`** chapters with many Mermaid figures), see **[`mdx/README.md`](./mdx/README.md)**. Those files are ideal for **Docusaurus / Nextra / Fumadocs** or VS Code preview; they complement—not replace—the Markdown guides here.

## Documentation map (coverage)

```mermaid
flowchart TB
  subgraph start["Start here"]
    SET[setup-and-configuration]
    ARC[architecture-overview]
    SDL[system-design-layers]
  end
  subgraph core["Core system"]
    BOOT[system-context-and-bootstrap]
    PROV[providers-and-registry]
    DOM[domain-model-and-types]
    INH[inheritance-class-diagrams]
  end
  subgraph paths["Data paths"]
    ING[ingestion-pipeline]
    RET[retrieval-and-search]
    STO[storage-and-cas]
  end
  subgraph surface["API & clients"]
    API[api-http-and-observability]
    REST[rest-api-reference]
    NS[namespaces-tenants-auth]
    SEC[security-deployment-and-operations]
  end
  subgraph extras["Agents & automation"]
    AG[agents-and-chat]
    WF[workflows]
    APP[apps-streamlit-demo]
  end
  subgraph quality["Quality"]
    TST[testing-strategy]
    GLO[glossary]
  end
  SET --> ARC
  ARC --> SDL
  ARC --> BOOT
  BOOT --> PROV
  BOOT --> ING
  BOOT --> RET
  ING --> STO
  RET --> STO
  ARC --> REST
  REST --> API
  NS --> REST
  SEC --> SET
  DOM --> INH
  AG --> RET
  WF --> ING
  APP --> REST
```

## Reading order (recommended)

| Order | Document | What you get |
| :---: | --- | --- |
| 1 | [Setup and configuration](./setup-and-configuration.md) | Python version, extras, `.env`, YAML, run API and tests |
| 2 | [Architecture overview](./architecture-overview.md) | **System context, containers, layers, deployment topology, data-store matrix**, ingest/search flows |
| 3 | [System design — layers](./system-design-layers.md) | **Layered architecture**, dependency direction, library vs HTTP |
| 4 | [System context and bootstrap](./system-context-and-bootstrap.md) | `SystemContext`, stores, `build_services`, hot reload |
| 5 | [Domain model and types](./domain-model-and-types.md) | **Hashes, ACL, core types, SQL tables overview** |
| 6 | [Providers and registry](./providers-and-registry.md) | `ProviderRegistry`, embedding/LLM/extractor/reranker keys |
| 7 | [Ingestion pipeline](./ingestion-pipeline.md) | Parse → chunk → embed → CAS/graph/vector/sparse |
| 8 | [Retrieval and search](./retrieval-and-search.md) | `UnifiedSearchService`, fusion, reranking |
| 9 | [Storage and CAS](./storage-and-cas.md) | KV, vector, graph, Elasticsearch, CAS |
| 10 | [Namespaces, tenants, and auth](./namespaces-tenants-auth.md) | Multi-tenancy, JWT flows |
| 11 | [REST API reference](./rest-api-reference.md) | **Full route table**, methods, ACL requirements |
| 12 | [API, HTTP, and observability](./api-http-and-observability.md) | Lifespan, middleware, deps, tracing, audit |
| 13 | [Security, deployment, and operations](./security-deployment-and-operations.md) | **Secrets, TLS, CORS, scaling, backups, checklist** |
| 14 | [Agents and chat](./agents-and-chat.md) | `QAAgent`, chat |
| 15 | [Workflows (Inngest)](./workflows.md) | Durable ingest/delete |
| 16 | [Streamlit demo app](./apps-streamlit-demo.md) | `apps/streamlit_demo` |
| 17 | [Testing strategy](./testing-strategy.md) | Unit vs integration, layout |
| 18 | [Inheritance and class diagrams](./inheritance-class-diagrams.md) | ABCs, protocols, subclass maps |
| — | [Glossary](./glossary.md) | Short definitions of terms |
| — | **[MDX deep dive](./mdx/README.md)** (optional) | Extended multi-page **`.mdx`** chapters with **many diagrams**; for doc sites or VS Code |

## Package map (quick reference)

```mermaid
flowchart TB
  subgraph api["unified_memory.api"]
    APP[app.py FastAPI]
    R[routes: auth namespaces ingestion search chat admin]
    MW[middleware context]
  end
  subgraph core["unified_memory.core"]
    CFG[config]
    TYPES[types]
    REG[registry ProviderRegistry]
  end
  subgraph boot["bootstrap"]
    CTX[SystemContext]
  end
  subgraph ingest["ingestion"]
    PIPE[pipeline IngestionPipeline]
    PAR[parsers]
    CHK[chunkers]
    EXT[extractors]
  end
  subgraph retr["retrieval"]
    UNI[unified UnifiedSearchService]
    DNS[dense]
    GPH[graph]
    FUS[fusion]
  end
  subgraph stor["storage"]
    KV[kv]
    VEC[vector]
    GRA[graph]
    SRCH[search elasticsearch]
    SQL[sql]
  end
  APP --> CTX
  CTX --> PIPE
  CTX --> UNI
  CTX --> REG
  PIPE --> stor
  UNI --> stor
```

## Legacy notes

Older narrative documents may exist under `old_docs/`. Prefer this `docs/` tree for accuracy; migrate any unique content from `old_docs/` when in doubt.
