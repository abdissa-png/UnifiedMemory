# Architecture overview

This document describes the **Unified Memory System** as implemented in this repository: a multi-tenant, multimodal memory layer with ingestion, hybrid retrieval, and an optional FastAPI surface.

## 1. System context (C4 Level 1)

External actors and systems interact with the **Unified Memory API** (and optional **Streamlit demo**). The service depends on pluggable **infrastructure**: KV, vector DB, graph DB, optional Elasticsearch, and optional **LLM/embedding APIs**.

```mermaid
C4Context
  title System context — Unified Memory System
  Person(user, "Client / App", "Sends documents and queries")
  Person(admin, "Operator", "Configures tenants and namespaces")
  System(ums, "Unified Memory API", "Ingest, search, chat, admin")
  System_Ext(llm, "LLM / Embedding APIs", "OpenAI-compatible providers")
  System_Ext(stores, "Data plane", "Redis, Qdrant, Neo4j, Elasticsearch, SQLite")
  Rel(user, ums, "HTTPS + JWT")
  Rel(admin, ums, "HTTPS + JWT")
  Rel(ums, llm, "Embeddings, chat, extraction")
  Rel(ums, stores, "Vectors, graph, sparse text, metadata, audit")
```

If your renderer does not support **C4** syntax, use this equivalent:

```mermaid
flowchart LR
  user((Client / App))
  admin((Operator))
  ums[Unified Memory API]
  llm[LLM / Embedding APIs]
  stores[Data plane\nRedis Qdrant Neo4j ES SQLite]
  user -->|HTTPS JWT| ums
  admin -->|HTTPS JWT| ums
  ums -->|embeddings chat extraction| llm
  ums -->|vectors graph sparse metadata audit| stores
```

## 2. Container view (logical)

```mermaid
flowchart TB
  subgraph clients["Clients"]
    APP[Application]
    ST[Streamlit demo]
  end
  subgraph runtime["Python process"]
    API[FastAPI unified_memory.api.app]
    CTX[SystemContext]
    PIPE[IngestionPipeline]
    SRCH[UnifiedSearchService]
    AG[QAAgent]
    SQL[(Async SQLAlchemy\nchat + usage + audit)]
  end
  subgraph infra["Infrastructure"]
    KV[(KV store)]
    VEC[(Vector store)]
    G[(Graph store)]
    ES[(Elasticsearch\noptional)]
  end
  APP --> API
  ST --> API
  API --> CTX
  CTX --> PIPE
  CTX --> SRCH
  CTX --> AG
  CTX --> KV
  CTX --> VEC
  CTX --> G
  CTX --> ES
  API --> SQL
```

## 3. Request-centric architecture

### 3.1 API startup sequence

```mermaid
sequenceDiagram
  participant Uvicorn
  participant App as FastAPI lifespan
  participant Ctx as SystemContext
  participant SQL as SQL engine
  Uvicorn->>App: startup
  App->>Ctx: from_config_file(UMS_CONFIG) or SystemContext()
  App->>Ctx: build_services(enable_inngest=...)
  App->>SQL: create_engine, init_db
  App->>App: ChatSessionManager, AuditLogger, TenantManager
  App->>App: set_flush_callback(tracing → TokenUsageRecord)
  App-->>Uvicorn: ready
```

### 3.2 Ingestion path (conceptual)

```mermaid
flowchart LR
  D[Document bytes / text] --> P[Parse DocumentParser]
  P --> C[Chunk Chunker]
  C --> E[Embed EmbeddingProvider]
  E --> CAS[CAS + DocumentRegistry]
  E --> V[Vector upsert]
  E --> GR[Graph upsert]
  E --> SP[Sparse index\nBM25 or ES]
```

### 3.3 Search path (conceptual)

```mermaid
flowchart TB
  Q[Query string] --> NS[Resolve namespace + RetrievalConfig]
  NS --> PAR[Parallel retrieval]
  PAR --> DENSE[DenseRetriever\nvectors]
  PAR --> SPRS[SparseRetriever\nBM25 or ES]
  PAR --> GRP[GraphRetriever]
  DENSE --> FUS[Fusion RRF / linear]
  SPRS --> FUS
  GRP --> FUS
  FUS --> RR[Reranker optional]
  RR --> OUT[List of RetrievalResult]
```

## 4. Multi-tenant model (high level)

```mermaid
flowchart TB
  T[Tenant] --> TN[TenantConfig\nembeddings LLM limits]
  T --> NS1[Namespace 1]
  T --> NS2[Namespace 2]
  NS1 --> RC[RetrievalConfig]
  NS2 --> RC
  RC --> RET[UnifiedSearchService.search]
```

Tenants and namespaces are persisted and validated through `NamespaceManager` and `TenantManager` (see [namespaces-tenants-auth.md](./namespaces-tenants-auth.md)).

## 5. Optional asynchronous workflows

When `UMS_ENABLE_INNGEST` is enabled and `build_services(enable_inngest=True)` runs, **Inngest** functions are created for durable **ingest** and **delete** pipelines using an artifact store for large payloads. The FastAPI app registers `inngest.fast_api.serve` when the client is present.

```mermaid
flowchart LR
  ING[Inngest Cloud / Dev Server] --> FN[ingest / delete functions]
  FN --> PIPE[IngestionPipeline]
  FN --> ART[LocalFSArtifactStore]
```

## 6. Logical layers (design view)

The same system can be viewed as **layers**: HTTP adapters → orchestration (`SystemContext`, pipelines, search) → domain types → infrastructure implementations. This complements the container diagram above.

```mermaid
flowchart TB
  subgraph L1["Adapters"]
    HTTP[FastAPI]
    CLI[Streamlit / scripts]
  end
  subgraph L2["Application services"]
    IP[IngestionPipeline]
    US[UnifiedSearchService]
    QA[QAAgent]
  end
  subgraph L3["Domain"]
    T[core.types + namespace types]
  end
  subgraph L4["Infrastructure"]
    S[KV Vector Graph SQL ES]
  end
  HTTP --> IP
  HTTP --> US
  HTTP --> QA
  CLI --> HTTP
  IP --> T
  US --> T
  IP --> S
  US --> S
```

Full narrative: [system-design-layers.md](./system-design-layers.md).

## 7. Deployment topology (typical production)

One **stateless** API tier talks to **shared** data services. In-memory backends are for **single-process** dev/test only.

```mermaid
flowchart LR
  subgraph edge["Edge"]
    TLS[TLS termination]
  end
  subgraph compute["Compute"]
    API1[API replica]
    API2[API replica]
  end
  subgraph data["Shared data plane"]
    PG[(Postgres\nSQL users chat audit usage)]
    RD[(Redis\nKV metadata)]
    QD[(Qdrant)]
    N4J[(Neo4j)]
    ES[(Elasticsearch\noptional)]
  end
  TLS --> API1
  TLS --> API2
  API1 --> PG
  API2 --> PG
  API1 --> RD
  API2 --> RD
  API1 --> QD
  API2 --> QD
  API1 --> N4J
  API2 --> N4J
  API1 --> ES
  API2 --> ES
```

Operational checklist: [security-deployment-and-operations.md](./security-deployment-and-operations.md).

## 8. Data-store responsibility matrix

| Store | Primary responsibility | Used for |
| --- | --- | --- |
| **KV** (`KVStoreBackend`) | Versioned metadata, CAS registry, namespace/tenant docs | Fast optimistic concurrency, registry state |
| **Vector** (`VectorStoreBackend`) | Embeddings and similarity search | Dense retrieval |
| **Graph** (`GraphStoreBackend`) | Entities, edges, PPR-style walks | Graph retrieval, provenance |
| **Elasticsearch** (optional) | Full-text inverted index | Sparse retrieval when `sparse_retriever: elasticsearch` |
| **SQL** (API mode) | Relational consistency | Users, passwords, chat, audit, **token_usage** |
| **CAS / artifact FS** | Blobs and large payloads | Content dedup, workflow externalization |

## 9. End-to-end request journeys (summary)

| Journey | Starts at | Core components |
| --- | --- | --- |
| **Register / login** | `/v1/auth/*` | `TenantManager`, JWT, SQL `users` |
| **Ingest document** | `/v1/ingest/*` | `IngestionPipeline`, stores, CAS |
| **Search** | `POST /v1/search/{namespace}` | `UnifiedSearchService`, fusion, reranker |
| **Chat** | `/v1/chat/*` | `ChatSessionManager`, `QAAgent`, SQL messages |

Detailed route table: [rest-api-reference.md](./rest-api-reference.md).

## 10. Where to read next

| Topic | Document |
| --- | --- |
| Layering and dependency rules | [system-design-layers.md](./system-design-layers.md) |
| Class wiring and stores | [system-context-and-bootstrap.md](./system-context-and-bootstrap.md) |
| Domain types and ACL | [domain-model-and-types.md](./domain-model-and-types.md) |
| Ingestion internals | [ingestion-pipeline.md](./ingestion-pipeline.md) |
| Search and fusion | [retrieval-and-search.md](./retrieval-and-search.md) |
| REST API | [rest-api-reference.md](./rest-api-reference.md) |
| Security / ops | [security-deployment-and-operations.md](./security-deployment-and-operations.md) |
| HTTP internals | [api-http-and-observability.md](./api-http-and-observability.md) |
