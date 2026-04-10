# Glossary

| Term | Meaning in this codebase |
| --- | --- |
| **CAS** | Content-addressable storage: storing and referencing blobs by cryptographic hash (`CASRegistry`, `ContentStore`). |
| **Chunk** | A segment of a parsed document used for embedding and retrieval (`core.types.Chunk`). |
| **Dense retrieval** | Vector similarity search over embeddings (`DenseRetriever`). |
| **Document hash** | Tenant-scoped hash of normalized document content (`compute_document_hash`). |
| **Extractor** | Component that runs over chunks to produce entities/relations (`ingestion/extractors`). |
| **Fusion** | Combining ranked lists from dense/sparse/graph (`reciprocal_rank_fusion`, `linear_fusion`). |
| **Graph store** | Backend for entities and edges, plus PPR-style operations. |
| **Ingestion pipeline** | Orchestrator from raw input to indexed chunks (`IngestionPipeline`). |
| **KV store** | Versioned key-value metadata store (`KVStoreBackend`). |
| **Namespace** | Isolation boundary for memory; string id + `NamespaceConfig` (ACL, retrieval defaults). |
| **Provider registry** | Central registry for embedders, LLMs, extractors, rerankers, parsers (`ProviderRegistry`). |
| **Reranker** | Second-stage scorer over candidate passages (`Reranker` protocol). |
| **Sparse retrieval** | Lexical search — BM25 in-process or Elasticsearch. |
| **SystemContext** | Bootstrap container for all major services (`bootstrap.py`). |
| **Tenant** | Top-level organization; owns users and default policies (`TenantConfig`). |
| **Unified search** | Single entry point combining dense + sparse + graph (`UnifiedSearchService`). |
| **Vision embedding** | Image embedding provider registered separately from text keys (`modality: vision`). |

## Related

- [domain-model-and-types.md](./domain-model-and-types.md)
- [README.md](./README.md) (documentation index)
