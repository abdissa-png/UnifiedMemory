# MDX deep-dive documentation

This folder contains **extended, diagram-heavy** documentation in **MDX** (Markdown + optional JSX). It is **more narrative and visual** than the canonical Markdown guides in [`docs/README.md`](../README.md) and targets **maintainers**, **architects**, and **integrators** who want end-to-end figures.

## Structure (reading order)

Files are prefixed with **`NN-`** so they sort in the repo; **Docusaurus doc IDs drop the numeric prefix** (e.g. `00-system-overview.mdx` → route **`/docs/system-overview`**).

| Order | File | Doc ID / slug | Topic |
| :---: | --- | --- | --- |
| 0 | `00-system-overview.mdx` | `system-overview` | Context, deployment, ingest/search journeys |
| 1 | `01-bootstrap-and-providers.mdx` | `bootstrap-and-providers` | `SystemContext`, registry, infra matrix, hot reload |
| 2 | `02-ingestion-pipeline.mdx` | `ingestion-pipeline` | Parse → chunk → embed → indexes; CAS; saga |
| 3 | `03-retrieval-fusion-rerank.mdx` | `retrieval-fusion-rerank` | Unified search, fusion, rerank |
| 4 | `04-storage-data-plane.mdx` | `storage-data-plane` | KV, vector, graph, ES, SQL, CAS boundaries |
| 5 | `05-api-tenants-auth-observability.mdx` | `api-tenants-auth-observability` | FastAPI, ACL, JWT, tracing, SQL sidecar |
| 6 | `06-agents-workflows-and-apps.mdx` | `agents-workflows-and-apps` | QAAgent, Inngest, artifacts, Streamlit demo |
| 7 | `07-domain-validation-and-quality.mdx` | `domain-validation-and-quality` | Namespace validation, JSON repair, cache, doc map |
| 8 | `08-configuration-matrix.mdx` | `configuration-matrix` | YAML sections, env interpolation, validation vs API env |

**Sidebar:** `docs/website/sidebars.js` lists the same IDs under **`deepDive`**.

## How this maps to `docs/*.md` (canonical)

MDX chapters **summarize and diagram** behavior that is specified in more detail (tables, checklists, full route lists) in **`docs/`**:

| MDX chapter | Primary canonical companions |
| --- | --- |
| System overview | `architecture-overview.md`, `system-design-layers.md` |
| Bootstrap | `system-context-and-bootstrap.md`, `providers-and-registry.md` |
| Ingestion | `ingestion-pipeline.md`, `storage-and-cas.md` |
| Retrieval | `retrieval-and-search.md` |
| Storage | `storage-and-cas.md` |
| API / auth / observability | `api-http-and-observability.md`, `rest-api-reference.md`, `namespaces-tenants-auth.md`, `security-deployment-and-operations.md` |
| Agents / workflows / apps | `agents-and-chat.md`, `workflows.md`, `apps-streamlit-demo.md` |
| Domain / quality | `domain-model-and-types.md`, `testing-strategy.md`, `glossary.md` |
| Configuration | `setup-and-configuration.md` (with **`config/app.example.yaml`** and **`.env.example`**) |

If **code**, **canonical Markdown**, and **MDX** disagree, treat **code** as truth and update the docs.

## What is intentionally not duplicated here

- **Exhaustive REST tables** — `docs/rest-api-reference.md`
- **Production security and runbooks** — `docs/security-deployment-and-operations.md`
- **Full class diagrams** — `docs/inheritance-class-diagrams.md`

## Viewing the diagrams

- **GitHub** often renders **Mermaid** in ` ```mermaid ` fences; raw `.mdx` may show as source in some views.
- **VS Code**: Mermaid preview extensions.
- **Local site (recommended):** **`docs/website/`** (Docusaurus + Mermaid).

### Run the documentation site locally

From the repository root:

```bash
cd docs/website
npm install
npm start
```

Open **http://localhost:3000** — the sidebar **Deep dive** lists all chapters.

Build a static bundle:

```bash
cd docs/website
npm run build
npm run serve
```

Output: **`docs/website/build/`** (typically gitignored).

## Relationship to `docs/*.md`

The Markdown tree under **`docs/`** remains the **stable, repo-native** documentation. **`docs/mdx/`** is an **optional** deep dive with richer figures; keep behavioral truth aligned when you change either layer.
