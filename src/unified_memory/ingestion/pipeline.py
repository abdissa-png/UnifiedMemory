"""
Ingestion Pipeline Orchestrator.

Coordinates the flow: Document -> Parse -> Chunk -> Embed -> Store (CAS, Vector, Graph, Sparse).
Implements the Saga pattern for atomic-like processing of documents.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Tuple

from unified_memory.core.types import (
    Chunk,
    SourceReference,
    SourceLocation,
    SourceType,
    Modality,
    CollectionType,
    Relation,
    compute_content_hash,
    compute_document_hash,
    normalize_relation_type,
    make_entity_id,
    source_locations_to_parallel_arrays,
)
from unified_memory.ingestion.parsers.base import DocumentParser, ParsedDocument
from unified_memory.ingestion.chunkers.base import Chunker, ChunkingConfig
from unified_memory.embeddings.base import EmbeddingProvider
from unified_memory.storage.base import VectorStoreBackend, GraphStoreBackend
from unified_memory.cas.registry import CASRegistry
from unified_memory.cas.content_store import ContentStore
from unified_memory.ingestion.extractors.base import Extractor
from unified_memory.namespace.manager import NamespaceManager
from unified_memory.namespace.types import ExtractionConfig
from unified_memory.cas.document_registry import DocumentRegistry
from unified_memory.ingestion.parsers.registry import get_parser_registry, ParserRegistry
from unified_memory.ingestion.chunkers.fixed_size import FixedSizeChunker, ChunkingConfig
from unified_memory.ingestion.chunkers.recursive import RecursiveChunker
from unified_memory.ingestion.chunkers.semantic import SemanticChunker
from unified_memory.namespace.types import TenantConfig
from unified_memory.observability.tracing import traced
logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Result of ingesting a document."""
    document_id: str
    source: SourceReference
    chunk_count: int = 0
    page_count: int = 0
    chunks: List[Chunk] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    deduped: bool = False  # True if document was already present
    doc_hash: str = ""  # Content-addressable hash for the document

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


@dataclass
class DeleteResult:
    """
    Result of deleting a document for a specific namespace.

    - If found == False, no document entry existed for the given (tenant, hash).
    - vectors_deleted: number of vectors hard-deleted because this was the last namespace.
    - vectors_unlinked: number of vectors that simply had this namespace removed.
    - nodes_deleted / nodes_unlinked: same semantics for graph nodes/edges.
    """

    found: bool
    vectors_deleted: int = 0
    vectors_unlinked: int = 0
    nodes_deleted: int = 0
    nodes_unlinked: int = 0




class IngestionPipeline:
    """
    Enhanced ingestion pipeline with CAS, Embedding, and Saga support.
    
    Design Reference: UNIFIED_MEMORY_SYSTEM_DESIGN.md and INITIAL_PLAN.md
    """
    
    def __init__(
        self,
        vector_store: VectorStoreBackend,
        cas_registry: CASRegistry,
        content_store: ContentStore,
        namespace_manager: NamespaceManager,
        document_registry: DocumentRegistry,
        graph_store: Optional[GraphStoreBackend] = None,
        sparse_store: Optional['ElasticSearchStore'] = None,
        vision_embedding_provider: Optional[EmbeddingProvider] = None,
        chunker: Optional[Chunker] = None,
        parser_registry: Optional[ParserRegistry] = None,
        provider_registry: Optional['ProviderRegistry'] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        default_extractor_key: str = "default",
        artifact_store: Optional[Any] = None,
        image_content_store: Optional[Any] = None,
        document_content_store: Optional[Any] = None,
    ) -> None:
        from unified_memory.ingestion.chunkers.fixed_size import FixedSizeChunker

        self.vector_store = vector_store
        self.cas_registry = cas_registry
        self.content_store = content_store
        self.namespace_manager = namespace_manager
        self.document_registry = document_registry
        from unified_memory.core.registry import ProviderRegistry as _PR
        self.provider_registry = provider_registry or _PR()

        self.graph_store = graph_store
        self.sparse_store = sparse_store
        self.vision_embedding_provider = vision_embedding_provider
        self.extractors: List[Extractor] = []

        # Base chunker: when a custom chunker is provided, it is used for the
        # "fixed_size" path.  Built-in recursive / semantic chunkers are kept
        # as singleton instances and selected based on tenant chunker_type.
        self.chunker = chunker or FixedSizeChunker()
        self._fixed_chunker = self.chunker
        self._recursive_chunker = RecursiveChunker()
        self._semantic_chunker = SemanticChunker(
            provider_registry=self.provider_registry,
            namespace_manager=self.namespace_manager,
        )

        # Prefer the parser registry attached to ProviderRegistry when no
        # explicit registry is supplied, so that parsers are centralised.
        if parser_registry is not None:
            self.parser_registry = parser_registry
        else:
            self.parser_registry = self.provider_registry.get_parser_registry()
            from unified_memory.ingestion.parsers.text import TextParser

            self.parser_registry.register(TextParser())

            from unified_memory.ingestion.parsers.mineru_pdf import is_mineru_available
            if is_mineru_available():
                from unified_memory.ingestion.parsers.mineru_pdf import MinerUPDFParser
                self.parser_registry.register(MinerUPDFParser())
                logger.info("MinerU PDF parser registered")

        # Legacy: if an explicit embedding_provider is given, register it as
        # a process-wide fallback.  New code should resolve embedders via
        # ProviderRegistry and tenant configuration instead of touching this.
        self._fallback_embedder = embedding_provider
        self._default_extractor_key = default_extractor_key
        self._artifact_store = artifact_store
        self.image_content_store = image_content_store
        self.document_content_store = document_content_store

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        """
        Backward-compatible accessor for the legacy embedding provider.

        Integration tests and older code paths still expect
        ``ingestion_pipeline.embedding_provider`` to exist.  In the new
        design, embedders are resolved per-tenant via ProviderRegistry and
        TenantConfig, but when a fallback embedder was supplied at
        construction time we expose it here so existing call sites (e.g.
        tests that build a simple DenseRetriever) continue to work.
        """
        if self._fallback_embedder is None:
            raise AttributeError(
                "IngestionPipeline was constructed without an embedding_provider. "
                "Resolve an embedder via ProviderRegistry instead."
            )
        return self._fallback_embedder

    def _build_chunker_for_tenant(
        self, tenant_config: TenantConfig
    ) -> Tuple[Chunker, ChunkingConfig]:
        """
        Build a chunker instance based on tenant ingestion config.

        Supports:
        - "fixed_size"  (default)
        - "recursive"
        """
        chunk_size = (
            tenant_config.chunk_size if tenant_config.chunk_size is not None else 512
        )
        chunk_overlap = (
            tenant_config.chunk_overlap if tenant_config.chunk_overlap is not None else 64
        )
        respect_sentence_boundaries = tenant_config.respect_sentence_boundaries if tenant_config.respect_sentence_boundaries is not None else True
        similarity_threshold = tenant_config.similarity_threshold if tenant_config.similarity_threshold is not None else 0.5

        config = ChunkingConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            respect_sentence_boundaries=respect_sentence_boundaries,
            similarity_threshold=similarity_threshold,
        )

        chunker_type = (tenant_config.chunker_type or "fixed_size").lower()

        if chunker_type == "recursive":
            return self._recursive_chunker, config
        if chunker_type == "semantic":
            return self._semantic_chunker, config

        # Fallback / default: fixed-size, using either the user-supplied
        # chunker or the built-in FixedSizeChunker singleton.
        return self._fixed_chunker, config

    @staticmethod
    def _cfg_attr(model_cfg, attr: str):
        """Access an attribute on an EmbeddingModelConfig-like object,
        whether it is a dataclass instance or a plain dict (serialised)."""
        if isinstance(model_cfg, dict):
            return model_cfg.get(attr)
        return getattr(model_cfg, attr, None)

    def _resolve_embedder_from_tenant(self, tenant_config: Optional[TenantConfig]) -> EmbeddingProvider:
        """Resolve the embedding provider for a tenant via registry or fallback."""
        if tenant_config:
            model_cfg = tenant_config.text_embedding
            provider = self._cfg_attr(model_cfg, "provider")
            model = self._cfg_attr(model_cfg, "model")
            if provider and model:
                embedder = self.provider_registry.resolve_embedding_provider(
                    provider, model, fallback_key=model,
                )
                if embedder:
                    return embedder
        if self._fallback_embedder:
            return self._fallback_embedder
        raise ValueError(
            "No embedding provider available for tenant. "
            "Register one in the ProviderRegistry."
        )

    def _resolve_extractor_from_config(
        self,
        extraction_config: Optional[ExtractionConfig],
    ) -> Optional[Extractor]:
        """Resolve a single extractor from ExtractionConfig, falling back to self.extractors."""
        if extraction_config and extraction_config.extractor_type:
            extractor = self.provider_registry.get_extractor(extraction_config.extractor_type)
            if extractor:
                return extractor
        # Tenant didn't specify, or registry doesn't have that extractor: use registry default if present.
        if self._default_extractor_key:
            extractor = self.provider_registry.get_extractor(self._default_extractor_key)
            if extractor:
                return extractor
        if self.extractors:
            return self.extractors[0]
        return None

    def _resolve_vision_embedder_from_tenant(
        self,
        tenant_config: Optional[TenantConfig],
    ) -> Optional[EmbeddingProvider]:
        """Resolve the vision embedding provider for a tenant.

        Resolution order:
        1. ``vision_embedding_providers`` slot in the ProviderRegistry
           (keyed by ``provider:model`` from ``tenant_config.vision_embedding``).
        2. Legacy fallback: the vision embedder registered in the text
           embedding slot under the same key (backward-compat).
        3. ``self.vision_embedding_provider`` (direct constructor injection).
        """
        if tenant_config and tenant_config.vision_embedding:
            model_cfg = tenant_config.vision_embedding
            provider = self._cfg_attr(model_cfg, "provider")
            model = self._cfg_attr(model_cfg, "model")
            if provider and model:
                embedder = self.provider_registry.resolve_vision_embedding_provider(
                    provider, model,
                )
                if embedder:
                    return embedder
        return self.vision_embedding_provider

    # ===================================================================
    # Step methods — granular, resumable ingestion steps
    #
    # Each method is self-contained: all data inputs arrive as
    # parameters and outputs are JSON-serialisable dicts.  The methods
    # mirror the logic in _process_chunks / ingest_file / ingest_text
    # so they can later replace the monolithic code path.
    # ===================================================================

    @traced("ingestion.resolve_context")
    async def step_resolve_context(self, namespace: str) -> dict:
        """Resolve tenant/namespace config and provider availability.

        Returns a JSON-serialisable dict with the **full** tenant config
        plus convenience flags derived from it, so downstream steps never
        need to re-fetch the config from the namespace manager.
        """
        from dataclasses import asdict as _asdict

        ns_config = await self.namespace_manager.get_config(namespace)
        tenant_id = ns_config.tenant_id if ns_config else "default"
        tenant_config = await self.namespace_manager.get_tenant_config(tenant_id)

        text_embedding_model = (
            tenant_config.text_embedding.model if tenant_config else "default"
        )
        vision_embedding_model = (
            tenant_config.vision_embedding.model
            if tenant_config and tenant_config.vision_embedding
            else ""
        )

        raw_ext = tenant_config.extraction if tenant_config else None
        if isinstance(raw_ext, dict):
            extraction_config = (
                ExtractionConfig(
                    **{
                        k: v
                        for k, v in raw_ext.items()
                        if k in ExtractionConfig.__dataclass_fields__
                    }
                )
                if raw_ext
                else None
            )
        else:
            extraction_config = raw_ext

        enable_graph = bool(
            getattr(tenant_config, "enable_graph_storage", False)
            if tenant_config
            else False
        )
        enable_visual = bool(
            getattr(tenant_config, "enable_visual_indexing", False)
            if tenant_config
            else False
        )
        enable_entity_extraction = bool(
            getattr(tenant_config, "enable_entity_extraction", False)
            if tenant_config
            else False
        )
        enable_relation_extraction = bool(
            getattr(tenant_config, "enable_relation_extraction", False)
            if tenant_config
            else False
        )

        page_snippet_length = (
            tenant_config.page_snippet_length
            if tenant_config and hasattr(tenant_config, "page_snippet_length")
            else 200
        )

        extraction_config_dict = (
            _asdict(extraction_config) if extraction_config else None
        )

        tenant_config_dict = _asdict(tenant_config) if tenant_config else {}

        return {
            "tenant_id": tenant_id,
            "namespace": namespace,
            "text_embedding_model": text_embedding_model,
            "vision_embedding_model": vision_embedding_model,
            "tenant_config": tenant_config_dict,
            # Convenience flags (derived from tenant_config for readability)
            "enable_graph": enable_graph,
            "enable_visual": enable_visual,
            "enable_entity_extraction": enable_entity_extraction,
            "enable_relation_extraction": enable_relation_extraction,
            "page_snippet_length": page_snippet_length,
            "extraction_config": extraction_config_dict,
        }

    @traced("ingestion.parse")
    async def step_parse_and_externalize(
        self,
        source_path_or_text,
        document_id: str,
        job_id: str,
        artifact_store: "ArtifactStore",
        is_file: bool = True,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **options: Any,
    ) -> dict:
        """Parse a document (file or raw text) and externalise heavy payloads.

        For *file* input the parser registry selects the right parser.
        For *text* input a minimal ``ParsedDocument`` is built inline.

        Page images, figure image bytes, and table image bytes are stored
        in *artifact_store* and stripped from the in-memory document.  The
        text-only ``ParsedDocument`` is then serialised to the artifact
        store as JSON.
        """
        from unified_memory.workflows.serialization import (
            parsed_doc_to_dict,
            source_ref_to_dict,
        )
        from unified_memory.core.types import PageContent

        if is_file:
            path = (
                Path(source_path_or_text)
                if not isinstance(source_path_or_text, Path)
                else source_path_or_text
            )
            parser = self.parser_registry.get_parser_for_file(path)
            if not parser:
                return {"error": f"No parser for {path.suffix}"}
            parsed = await parser.parse_file(path, document_id, **options)
        else:
            parsed = ParsedDocument(
                document_id=document_id,
                source=SourceReference(
                    source_id=document_id, source_type=SourceType.TEXT_BLOCK
                ),
                title=title,
                pages=[
                    PageContent(
                        page_number=1,
                        document_id=document_id,
                        text_blocks=[{"text": source_path_or_text}],
                        full_text=source_path_or_text,
                    )
                ],
                full_text=source_path_or_text,
                metadata=metadata or {},
            )

        page_image_uris: List[dict] = []

        for page in parsed.pages:
            if page.full_page_image:
                uri = await artifact_store.put_bytes(
                    page.full_page_image,
                    key=f"jobs/{job_id}/pages/{document_id}/{page.page_number}/full_page.bin",
                )
                page_image_uris.append(
                    {"page_number": page.page_number, "uri": uri}
                )
                page.full_page_image = None

            for fig_idx, fig in enumerate(page.figures):
                if fig.get("image_bytes"):
                    fig_uri = await artifact_store.put_bytes(
                        fig["image_bytes"],
                        key=(
                            f"jobs/{job_id}/pages/{document_id}/"
                            f"{page.page_number}/fig_{fig_idx}.bin"
                        ),
                    )
                    fig["image_uri"] = fig_uri
                    del fig["image_bytes"]

            for tbl_idx, tbl in enumerate(page.tables):
                if tbl.get("image_bytes"):
                    tbl_uri = await artifact_store.put_bytes(
                        tbl["image_bytes"],
                        key=(
                            f"jobs/{job_id}/pages/{document_id}/"
                            f"{page.page_number}/tbl_{tbl_idx}.bin"
                        ),
                    )
                    tbl["image_uri"] = tbl_uri
                    del tbl["image_bytes"]

        doc_dict = parsed_doc_to_dict(parsed)
        parsed_uri = await artifact_store.put_json(
            doc_dict,
            key=f"jobs/{job_id}/parsed/{document_id}.json",
        )

        return {
            "document_id": document_id,
            "parsed_artifact_uri": parsed_uri,
            "page_image_uris": page_image_uris,
            "page_count": parsed.page_count,
            "full_text": parsed.full_text,
            "source": source_ref_to_dict(parsed.source),
            "parse_errors": parsed.parse_errors,
        }

    @traced("ingestion.dedup_check")
    async def step_dedup_check(
        self, tenant_id: str, doc_hash: str, namespace: str
    ) -> dict:
        """Check whether a document already exists in the registry.

        Returns a dict whose ``decision`` field is one of:
        - ``"skip"``         — document already present in this namespace.
        - ``"fast_link"``    — document exists in another namespace.
        - ``"full_ingest"``  — document is new.
        """
        existing_doc = await self.document_registry.get_document(
            tenant_id, doc_hash
        )

        if not existing_doc:
            return {"decision": "full_ingest"}

        entry_data = {
            "existing_document_id": existing_doc.document_id,
            "chunk_content_hashes": existing_doc.chunk_content_hashes,
            "text_vector_ids": existing_doc.text_vector_ids,
            "entity_vector_ids": existing_doc.entity_vector_ids,
            "relation_vector_ids": existing_doc.relation_vector_ids,
            "page_image_vector_ids": existing_doc.page_image_vector_ids,
            "graph_node_ids": existing_doc.graph_node_ids,
            "graph_edge_ids": existing_doc.graph_edge_ids,
        }

        if namespace in existing_doc.namespaces:
            return {"decision": "skip", **entry_data}

        return {"decision": "fast_link", **entry_data}

    @traced("ingestion.fast_link")
    async def step_fast_link(
        self,
        tenant_id: str,
        doc_hash: str,
        namespace: str,
        existing_entry_dict: dict,
        ctx: Optional[dict] = None,
    ) -> dict:
        """Fast-link: add a new namespace to an already-ingested document.

        Replicates Path B from ``ingest_file`` / ``ingest_text``:
        registers the namespace, adds CAS references, grants vector and
        graph access, and re-indexes sparse storage.

        When *ctx* (from ``step_resolve_context``) is provided the method
        avoids re-fetching the tenant config.
        """
        existing_document_id = existing_entry_dict["existing_document_id"]
        chunk_content_hashes = existing_entry_dict["chunk_content_hashes"]
        text_vector_ids = existing_entry_dict.get("text_vector_ids", [])
        entity_vector_ids = existing_entry_dict.get("entity_vector_ids", [])
        relation_vector_ids = existing_entry_dict.get("relation_vector_ids", [])
        page_image_vector_ids = existing_entry_dict.get(
            "page_image_vector_ids", []
        )
        graph_node_ids = existing_entry_dict.get("graph_node_ids", [])
        graph_edge_ids = existing_entry_dict.get("graph_edge_ids", [])

        await self.document_registry.add_namespace(
            tenant_id, doc_hash, namespace
        )

        for i, content_hash in enumerate(chunk_content_hashes):
            await self.cas_registry.add_reference(
                content_hash=content_hash,
                namespace=namespace,
                document_id=existing_document_id,
                chunk_index=i,
            )

        all_vec_ids = (
            text_vector_ids
            + entity_vector_ids
            + relation_vector_ids
            + page_image_vector_ids
        )
        for vec_id in all_vec_ids:
            await self.vector_store.add_namespace(vec_id, namespace)

        enable_graph = (
            ctx.get("enable_graph", False) if ctx else False
        )
        if not ctx:
            tc = await self.namespace_manager.get_tenant_config(tenant_id)
            enable_graph = bool(
                getattr(tc, "enable_graph_storage", False) if tc else False
            )
        if self.graph_store and enable_graph:
            all_graph_ids = graph_node_ids + graph_edge_ids
            for gid in all_graph_ids:
                await self.graph_store.add_namespace(gid, namespace)

        if self.sparse_store and chunk_content_hashes:
            sparse_docs: List[Dict[str, Any]] = []
            for h in chunk_content_hashes:
                content = await self.content_store.get_content(h)
                if content:
                    sparse_docs.append(
                        {
                            "id": h,
                            "content": content,
                            "metadata": {
                                "document_id": existing_document_id
                            },
                        }
                    )
            if sparse_docs:
                await self.sparse_store.index(sparse_docs, namespace)

        return {
            "document_id": existing_document_id,
            "chunk_content_hashes": chunk_content_hashes,
            "chunk_count": len(chunk_content_hashes),
            "deduped": True,
        }

    @traced("ingestion.register_and_chunk")
    async def step_register_and_chunk(
        self,
        tenant_id: str,
        doc_hash: str,
        namespace: str,
        document_id: str,
        parsed_artifact_uri: str,
        job_id: str,
        artifact_store: "ArtifactStore",
        ctx: dict,
    ) -> dict:
        """Register a new document and chunk it.

        1. Registers the document in ``DocumentRegistry``.
        2. Loads the parsed document from the artifact store.
        3. Chunks using the tenant-configured chunker.
        4. Stores chunk text in ``ContentStore``.
        5. Returns a chunk manifest dict.
        """
        from unified_memory.workflows.serialization import (
            parsed_doc_from_dict,
            chunk_ref_to_dict,
        )

        await self.document_registry.register_document(
            tenant_id, doc_hash, namespace, document_id
        )

        doc_dict = await artifact_store.get_json(parsed_artifact_uri)
        parsed = parsed_doc_from_dict(doc_dict)

        tc_dict = ctx.get("tenant_config", {})
        tenant_config = TenantConfig(**{
            k: v for k, v in tc_dict.items()
            if k in TenantConfig.__dataclass_fields__
        }) if tc_dict else await self.namespace_manager.get_tenant_config(tenant_id)
        req_chunker, chunk_cfg = self._build_chunker_for_tenant(tenant_config)
        chunks = await req_chunker.chunk(
            parsed, namespace, tenant_id=tenant_id, config=chunk_cfg
        )

        chunk_refs: List[dict] = []
        chunk_content_hashes: List[str] = []

        for chunk in chunks:
            if not chunk.content_hash:
                chunk.content_hash = compute_content_hash(
                    chunk.content, tenant_id, Modality.TEXT
                )
            await self.content_store.store_content(
                chunk.content_hash, chunk.content
            )
            chunk_refs.append(chunk_ref_to_dict(chunk))
            chunk_content_hashes.append(chunk.content_hash)

        manifest_uri = await artifact_store.put_json(
            {"chunks": chunk_refs},
            key=f"jobs/{job_id}/chunks/{document_id}.json",
        )

        return {
            "document_id": document_id,
            "chunk_manifest_uri": manifest_uri,
            "chunk_count": len(chunks),
            "chunk_content_hashes": chunk_content_hashes,
        }

    @traced("ingestion.embed_text")
    async def step_embed_and_upsert_text(
        self,
        namespace: str,
        tenant_id: str,
        chunk_content_hashes: List[str],
        document_id: str,
        ctx: dict,
        batch_size: int = 64,
    ) -> dict:
        """CAS-aware text embedding and vector upsert.

        For each chunk content hash:
        1. Check CAS for an existing vector with the current model.
        2. Embed missing chunks in batches.
        3. Register/update CAS entries.
        4. Upsert new vectors; add namespace to existing vectors.
        """
        from unified_memory.core.types import compute_vector_id

        embedding_model = ctx["text_embedding_model"]
        tc_dict = ctx.get("tenant_config", {})
        tenant_config = TenantConfig(**{
            k: v for k, v in tc_dict.items()
            if k in TenantConfig.__dataclass_fields__
        }) if tc_dict else await self.namespace_manager.get_tenant_config(tenant_id)
        embedder = self._resolve_embedder_from_tenant(tenant_config)

        text_collection = await self.namespace_manager.get_collection_name(
            namespace, CollectionType.TEXTS
        )

        text_vector_ids: List[str] = []
        errors: List[str] = []

        hashes_needing_embed: List[str] = []
        seen: set = set()

        for content_hash in chunk_content_hashes:
            if content_hash in seen:
                continue
            seen.add(content_hash)

            expected_vid = compute_vector_id(content_hash, embedding_model)
            entry = await self.cas_registry.get_entry(content_hash)

            if entry and entry.vector_id == expected_vid:
                pass
            else:
                hashes_needing_embed.append(content_hash)

        hash_to_embedding: Dict[str, Any] = {}
        if hashes_needing_embed:
            contents: List[str] = []
            valid_hashes: List[str] = []
            for h in hashes_needing_embed:
                content = await self.content_store.get_content(h)
                if content:
                    contents.append(content)
                    valid_hashes.append(h)

            try:
                for start in range(0, len(valid_hashes), batch_size):
                    batch_h = valid_hashes[start : start + batch_size]
                    batch_c = contents[start : start + batch_size]
                    embeddings = await embedder.embed_batch(
                        batch_c, modality=Modality.TEXT
                    )
                    for h, emb in zip(batch_h, embeddings):
                        hash_to_embedding[h] = emb
            except Exception as e:
                logger.exception("Text embedding failed")
                return {
                    "text_vector_ids": [],
                    "errors": [f"Embedding error: {str(e)}"],
                }

        vectors_to_upsert: List[Dict[str, Any]] = []
        upserted_hashes: set = set()

        for idx, content_hash in enumerate(chunk_content_hashes):
            vector_id = compute_vector_id(
                content_hash, embedding_model, "text"
            )
            content_id = f"content:{content_hash}"
            chunk_loc = SourceLocation(
                document_id=document_id, chunk_index=idx
            )

            try:
                await self.cas_registry.register(
                    content_hash=content_hash,
                    content_id=content_id,
                    vector_id=vector_id,
                )
                await self.cas_registry.add_reference(
                    content_hash=content_hash,
                    namespace=namespace,
                    document_id=document_id,
                    chunk_index=idx,
                )

                if (
                    content_hash in hash_to_embedding
                    and content_hash not in upserted_hashes
                ):
                    vectors_to_upsert.append(
                        {
                            "id": vector_id,
                            "embedding": hash_to_embedding[content_hash],
                            "metadata": {
                                "content_hash": content_hash,
                                "document_id": document_id,
                                "source_locations": [chunk_loc.to_dict()],
                            },
                        }
                    )
                    upserted_hashes.add(content_hash)
                else:
                    await self.vector_store.add_namespace(
                        vector_id,
                        namespace,
                        collection=text_collection,
                        document_id=document_id,
                    )

                text_vector_ids.append(vector_id)
            except Exception as e:
                logger.exception(
                    "Failed to process chunk hash %s", content_hash
                )
                errors.append(f"Chunk processing error: {str(e)}")

        if vectors_to_upsert:
            try:
                await self.vector_store.upsert(
                    vectors_to_upsert, namespace, collection=text_collection
                )
            except Exception as e:
                logger.error(f"Batch text vector upsert failed: {e}")
                errors.append(f"Vector batch error: {str(e)}")

        return {"text_vector_ids": text_vector_ids, "errors": errors}

    @traced("ingestion.sparse_upsert")
    async def step_sparse_upsert(
        self,
        namespace: str,
        chunk_content_hashes: List[str],
        document_id: str,
    ) -> dict:
        """Index chunks in the sparse (BM25) store."""
        if not self.sparse_store:
            return {"indexed": 0, "errors": []}

        sparse_docs: List[Dict[str, Any]] = []
        for content_hash in chunk_content_hashes:
            content = await self.content_store.get_content(content_hash)
            if content:
                sparse_docs.append(
                    {
                        "id": content_hash,
                        "content": content,
                        "metadata": {"document_id": document_id},
                    }
                )

        errors: List[str] = []
        if sparse_docs:
            try:
                await self.sparse_store.index(sparse_docs, namespace)
            except Exception as e:
                logger.error(f"Sparse index upsert failed: {e}")
                errors.append(f"Sparse index error: {str(e)}")

        return {"indexed": len(sparse_docs), "errors": errors}

    @traced("ingestion.extract_graph")
    async def step_extract_and_upsert_graph(
        self,
        namespace: str,
        tenant_id: str,
        chunk_content_hashes: List[str],
        document_id: str,
        parsed_artifact_uri: str,
        artifact_store: "ArtifactStore",
        ctx: dict,
        job_id: str = "",
    ) -> dict:
        """Build graph nodes/edges, extract entities/relations, deduplicate,
        and batch-write.

        Entity/relation descriptors are externalised to the artifact store
        so the step output stays small.  The returned dict contains URIs
        pointing to the descriptor payloads.
        """
        from unified_memory.core.types import (
            PageNode,
            PassageNode,
            EntityNode,
            GraphEdge,
            NodeType,
        )
        from unified_memory.workflows.serialization import parsed_doc_from_dict

        empty_result: dict = {
            "graph_node_ids": [],
            "graph_edge_ids": [],
            "entity_descriptors": [],
            "relation_descriptors": [],
            "errors": [],
        }

        if not self.graph_store:
            return empty_result

        errors: List[str] = []
        nodes_to_create: List[Any] = []
        edges_to_create: List[Any] = []
        graph_node_ids: List[str] = []

        doc_dict = await artifact_store.get_json(parsed_artifact_uri)
        parsed = parsed_doc_from_dict(doc_dict) if doc_dict else None
        snippet_len = ctx.get("page_snippet_length", 200)

        if parsed:
            for page in parsed.pages:
                page_node = PageNode(
                    id=PageNode.make_id(document_id, page.page_number),
                    content=page.full_text[:snippet_len],
                    namespace=namespace,
                    page_number=page.page_number,
                    document_id=document_id,
                    node_type=NodeType.PAGE,
                )
                nodes_to_create.append(page_node)
                graph_node_ids.append(page_node.id)

        extraction_config_dict = ctx.get("extraction_config")
        extraction_config: Optional[ExtractionConfig] = None
        if extraction_config_dict:
            extraction_config = ExtractionConfig(
                **{
                    k: v
                    for k, v in extraction_config_dict.items()
                    if k in ExtractionConfig.__dataclass_fields__
                }
            )
        extractor = self._resolve_extractor_from_config(extraction_config)

        for idx, content_hash in enumerate(chunk_content_hashes):
            content = await self.content_store.get_content(content_hash)
            if not content:
                continue

            chunk_loc = SourceLocation(
                document_id=document_id, chunk_index=idx
            )

            passage_node = PassageNode(
                id=PassageNode.make_id(tenant_id, content_hash),
                content="",
                namespace=namespace,
                node_type=NodeType.PASSAGE,
                page_number=None,
                source_locations=[chunk_loc],
                properties={"content_hash": content_hash},
            )
            nodes_to_create.append(passage_node)
            graph_node_ids.append(passage_node.id)

            if extractor:
                chunk_obj = Chunk(
                    document_id=document_id,
                    content=content,
                    chunk_index=idx,
                    content_hash=content_hash,
                )
                extraction_result = await extractor.extract(chunk_obj)

                entities = extraction_result.entities
                relations = extraction_result.relations

                if (
                    extraction_config
                    and extraction_config.strict_type_filtering
                ):
                    if extraction_config.entity_types:
                        entities = [
                            e
                            for e in entities
                            if e.type in extraction_config.entity_types
                        ]
                    if extraction_config.relation_types:
                        relations = [
                            r
                            for r in relations
                            if r.relation_type
                            in extraction_config.relation_types
                        ]

                for ent in entities:
                    entity_id = make_entity_id(ent.name, tenant_id)
                    node = EntityNode(
                        id=entity_id,
                        content=ent.description or "",
                        namespace=namespace,
                        entity_name=ent.name,
                        entity_type=ent.type,
                        source_locations=[chunk_loc],
                        properties=ent.properties,
                        node_type=NodeType.ENTITY,
                    )
                    nodes_to_create.append(node)
                    graph_node_ids.append(node.id)
                    edges_to_create.append(
                        GraphEdge(
                            source_id=passage_node.id,
                            target_id=node.id,
                            relation="MENTIONS",
                            source_locations=[chunk_loc],
                            namespace=namespace,
                        )
                    )

                for rel in relations:
                    src_id = make_entity_id(rel.source_entity, tenant_id)
                    tgt_id = make_entity_id(rel.target_entity, tenant_id)
                    extra_props: Dict[str, Any] = {**rel.properties}
                    if rel.description:
                        extra_props["description"] = rel.description
                    if rel.confidence != 1.0:
                        extra_props["confidence"] = rel.confidence
                    if rel.keywords:
                        extra_props["keywords"] = rel.keywords
                    if rel.source_type:
                        extra_props["source_type"] = rel.source_type
                    if rel.target_type:
                        extra_props["target_type"] = rel.target_type
                    edge = GraphEdge(
                        source_id=src_id,
                        target_id=tgt_id,
                        relation=normalize_relation_type(rel.relation_type),
                        weight=rel.weight,
                        is_bidirectional=rel.is_bidirectional,
                        inverse_relation=rel.inverse_relation,
                        source_locations=[chunk_loc],
                        namespace=namespace,
                        properties=extra_props,
                        source_entity_name=rel.source_entity,
                        target_entity_name=rel.target_entity,
                    )
                    edges_to_create.append(edge)

        # Deduplicate entities (merge provenance)
        unique_entity_nodes: Dict[str, Any] = {}
        other_nodes: List[Any] = []
        for node in nodes_to_create:
            if (
                hasattr(node, "node_type")
                and node.node_type == NodeType.ENTITY
            ):
                if node.id not in unique_entity_nodes:
                    unique_entity_nodes[node.id] = node
                else:
                    existing = unique_entity_nodes[node.id]
                    existing_locs = getattr(existing, "source_locations", [])
                    new_locs = getattr(node, "source_locations", [])
                    for loc in new_locs:
                        if loc not in existing_locs:
                            existing_locs.append(loc)
            else:
                other_nodes.append(node)

        final_nodes = list(unique_entity_nodes.values()) + other_nodes

        # Deduplicate relations (merge provenance and keywords)
        unique_relations: Dict[tuple, Any] = {}
        for edge in edges_to_create:
            key = (edge.source_id, edge.relation, edge.target_id)
            if key not in unique_relations:
                unique_relations[key] = edge
            else:
                existing = unique_relations[key]
                existing_locs = getattr(existing, "source_locations", [])
                new_locs = getattr(edge, "source_locations", [])
                for loc in new_locs:
                    if loc not in existing_locs:
                        existing_locs.append(loc)
                existing_kw = set(
                    existing.properties.get("keywords") or []
                )
                new_kw = edge.properties.get("keywords") or []
                if new_kw:
                    existing_kw.update(new_kw)
                    existing.properties["keywords"] = list(existing_kw)

        final_edges = list(unique_relations.values())

        try:
            if final_nodes:
                await self.graph_store.create_nodes_batch(
                    final_nodes, namespace
                )
            if final_edges:
                await self.graph_store.create_edges_batch(
                    final_edges, namespace
                )
        except Exception as e:
            logger.error(f"Batch graph creation failed: {e}")
            errors.append(f"Graph batch error: {str(e)}")

        graph_edge_ids: List[str] = [edge.id for edge in final_edges]

        entity_descriptors: List[dict] = []
        for node in unique_entity_nodes.values():
            locs = getattr(node, "source_locations", [])
            entity_descriptors.append(
                {
                    "id": node.id,
                    "entity_name": node.entity_name,
                    "entity_type": node.entity_type,
                    "description": node.content or "",
                    "source_locations": [
                        {
                            "document_id": loc.document_id,
                            "chunk_index": loc.chunk_index,
                        }
                        for loc in locs
                    ],
                }
            )

        relation_descriptors: List[dict] = []
        for edge in final_edges:
            if edge.relation in ("HAS_CHUNK", "MENTIONS"):
                continue
            locs = getattr(edge, "source_locations", [])
            relation_descriptors.append(
                {
                    "id": edge.id,
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "relation": edge.relation,
                    "source_entity_name": (
                        getattr(edge, "source_entity_name", "")
                        or edge.source_id
                    ),
                    "target_entity_name": (
                        getattr(edge, "target_entity_name", "")
                        or edge.target_id
                    ),
                    "description": edge.properties.get("description", ""),
                    "keywords": edge.properties.get("keywords", []),
                    "inverse_relation": edge.inverse_relation,
                    "weight": edge.weight,
                    "is_bidirectional": edge.is_bidirectional,
                    "source_locations": [
                        {
                            "document_id": loc.document_id,
                            "chunk_index": loc.chunk_index,
                        }
                        for loc in locs
                    ],
                }
            )

        ent_desc_uri = ""
        rel_desc_uri = ""
        if artifact_store and job_id:
            if entity_descriptors:
                ent_desc_uri = await artifact_store.put_json(
                    {"descriptors": entity_descriptors},
                    key=f"jobs/{job_id}/graph/{document_id}_entities.json",
                )
            if relation_descriptors:
                rel_desc_uri = await artifact_store.put_json(
                    {"descriptors": relation_descriptors},
                    key=f"jobs/{job_id}/graph/{document_id}_relations.json",
                )

        result: dict = {
            "graph_node_ids": graph_node_ids,
            "graph_edge_ids": graph_edge_ids,
            "entity_count": len(entity_descriptors),
            "relation_count": len(relation_descriptors),
            "errors": errors,
        }

        if ent_desc_uri:
            result["entity_descriptors_uri"] = ent_desc_uri
        else:
            result["entity_descriptors"] = entity_descriptors

        if rel_desc_uri:
            result["relation_descriptors_uri"] = rel_desc_uri
        else:
            result["relation_descriptors"] = relation_descriptors

        return result

    @traced("ingestion.embed_entities_relations")
    async def step_embed_and_upsert_entities_relations(
        self,
        namespace: str,
        tenant_id: str,
        document_id: str,
        entity_descriptors: Optional[List[dict]] = None,
        relation_descriptors: Optional[List[dict]] = None,
        ctx: Optional[dict] = None,
        artifact_store: "ArtifactStore" = None,
        entity_descriptors_uri: str = "",
        relation_descriptors_uri: str = "",
    ) -> dict:
        """Embed entity and relation descriptions and upsert to vector store.

        Descriptors can be passed inline *or* as artifact store URIs.
        When URIs are provided, the descriptors are loaded from the store.
        """
        from unified_memory.core.types import Entity as _EntityType

        if entity_descriptors_uri and artifact_store:
            data = await artifact_store.get_json(entity_descriptors_uri)
            entity_descriptors = data.get("descriptors", []) if data else []
        entity_descriptors = entity_descriptors or []

        if relation_descriptors_uri and artifact_store:
            data = await artifact_store.get_json(relation_descriptors_uri)
            relation_descriptors = data.get("descriptors", []) if data else []
        relation_descriptors = relation_descriptors or []

        errors: List[str] = []
        entity_vector_ids: List[str] = []
        relation_vector_ids: List[str] = []

        ctx = ctx or {}
        tc_dict = ctx.get("tenant_config", {})
        tenant_config = TenantConfig(**{
            k: v for k, v in tc_dict.items()
            if k in TenantConfig.__dataclass_fields__
        }) if tc_dict else await self.namespace_manager.get_tenant_config(tenant_id)
        embedder = self._resolve_embedder_from_tenant(tenant_config)

        if entity_descriptors:
            entity_collection = (
                await self.namespace_manager.get_collection_name(
                    namespace, CollectionType.ENTITIES
                )
            )

            ent_contents: List[str] = []
            for ed in entity_descriptors:
                tmp_entity = _EntityType(
                    name=ed["entity_name"],
                    entity_type=ed["entity_type"],
                    description=ed.get("description", ""),
                )
                ent_contents.append(tmp_entity.get_embedding_text())

            try:
                ent_embeddings = await embedder.embed_batch(
                    ent_contents, modality=Modality.TEXT
                )

                ent_vector_data: List[Dict[str, Any]] = []
                for ed, emb in zip(entity_descriptors, ent_embeddings):
                    provenance_doc_ids = list(
                        {
                            loc["document_id"]
                            for loc in ed.get("source_locations", [])
                        }
                    )
                    ent_vector_data.append(
                        {
                            "id": ed["id"],
                            "embedding": emb,
                            "metadata": {
                                "entity_name": ed["entity_name"],
                                "entity_type": ed["entity_type"],
                                "content_hash": compute_content_hash(
                                    ed.get("description")
                                    or ed["entity_name"],
                                    tenant_id,
                                    Modality.TEXT,
                                ),
                                "document_id": document_id,
                                "source_doc_ids": provenance_doc_ids,
                                "source_locations": ed.get(
                                    "source_locations", []
                                ),
                            },
                        }
                    )

                await self.vector_store.upsert(
                    ent_vector_data, namespace, collection=entity_collection
                )
                entity_vector_ids = [v["id"] for v in ent_vector_data]
            except Exception as e:
                logger.error(f"Entity embedding/storage failed: {e}")
                errors.append(f"Entity semantic index error: {str(e)}")

        if relation_descriptors:
            rel_collection = (
                await self.namespace_manager.get_collection_name(
                    namespace, CollectionType.RELATIONS
                )
            )

            rel_texts: List[str] = []
            for rd in relation_descriptors:
                tmp_rel = Relation(
                    subject_id=rd["source_id"],
                    predicate=rd["relation"],
                    object_id=rd["target_id"],
                    subject=(
                        rd.get("source_entity_name", "") or rd["source_id"]
                    ),
                    object=(
                        rd.get("target_entity_name", "") or rd["target_id"]
                    ),
                    description=rd.get("description", ""),
                    keywords=rd.get("keywords", []),
                    inverse_relation=rd.get("inverse_relation"),
                )
                rel_texts.append(tmp_rel.get_embedding_text())

            try:
                rel_embeddings = await embedder.embed_batch(
                    rel_texts, modality=Modality.TEXT
                )

                rel_vector_data: List[Dict[str, Any]] = []
                for rd, emb, text in zip(
                    relation_descriptors, rel_embeddings, rel_texts
                ):
                    provenance_doc_ids = list(
                        {
                            loc["document_id"]
                            for loc in rd.get("source_locations", [])
                        }
                    )
                    rel_vector_data.append(
                        {
                            "id": rd["id"],
                            "embedding": emb,
                            "metadata": {
                                "source_id": rd["source_id"],
                                "target_id": rd["target_id"],
                                "relation": rd["relation"],
                                "content_hash": compute_content_hash(
                                    text, tenant_id, Modality.TEXT
                                ),
                                "text": text,
                                "document_id": document_id,
                                "source_doc_ids": provenance_doc_ids,
                                "source_locations": rd.get(
                                    "source_locations", []
                                ),
                            },
                        }
                    )

                await self.vector_store.upsert(
                    rel_vector_data, namespace, collection=rel_collection
                )
                relation_vector_ids = [v["id"] for v in rel_vector_data]
            except Exception as e:
                logger.error(f"Relation embedding failed: {e}")
                errors.append(f"Relation embedding error: {str(e)}")

        return {
            "entity_vector_ids": entity_vector_ids,
            "relation_vector_ids": relation_vector_ids,
            "errors": errors,
        }

    @traced("ingestion.embed_vision")
    async def step_embed_and_upsert_vision(
        self,
        namespace: str,
        tenant_id: str,
        document_id: str,
        page_image_uris: List[dict],
        artifact_store: "ArtifactStore",
        ctx: dict,
    ) -> dict:
        """Embed page images and upsert to the vision vector collection.

        ``page_image_uris`` is a list of dicts with ``page_number`` and
        ``uri`` keys produced by ``step_parse_and_externalize``.
        Images are loaded from the artifact store by URI.

        Vector IDs are content-addressable and model-aware via
        ``compute_vector_id(image_hash, vision_model, "image")``.

        When an ``image_content_store`` is available on ``self``, images
        are persisted there for later retrieval; otherwise they remain
        only in the ephemeral artifact store.
        """
        import hashlib
        from unified_memory.core.types import compute_vector_id

        errors: List[str] = []
        page_image_vector_ids: List[str] = []

        if not ctx.get("enable_visual") or not page_image_uris:
            return {"page_image_vector_ids": [], "errors": []}

        vision_model = ctx.get("vision_embedding_model", "")

        tc_dict = ctx.get("tenant_config", {})
        tenant_config = TenantConfig(**{
            k: v for k, v in tc_dict.items()
            if k in TenantConfig.__dataclass_fields__
        }) if tc_dict else await self.namespace_manager.get_tenant_config(tenant_id)
        vision_embedder = self._resolve_vision_embedder_from_tenant(
            tenant_config
        )

        if not vision_embedder:
            return {"page_image_vector_ids": [], "errors": []}

        try:
            vision_collection = (
                await self.namespace_manager.get_collection_name(
                    namespace, CollectionType.PAGE_IMAGES
                )
            )

            images: List[bytes] = []
            page_numbers: List[int] = []
            for entry in page_image_uris:
                img_bytes = await artifact_store.get_bytes(entry["uri"])
                if img_bytes:
                    images.append(img_bytes)
                    page_numbers.append(entry["page_number"])

            if images:
                vision_embeddings = await vision_embedder.embed_batch(
                    images, modality=Modality.IMAGE
                )

                vision_vectors: List[Dict[str, Any]] = []
                for page_num, img_bytes, emb in zip(
                    page_numbers, images, vision_embeddings
                ):
                    raw_hash = hashlib.sha256(img_bytes).hexdigest()
                    img_content_hash = compute_content_hash(
                        raw_hash, tenant_id, Modality.IMAGE
                    )
                    vector_id = compute_vector_id(
                        img_content_hash, vision_model or "default", "image"
                    )

                    if getattr(self, "image_content_store", None):
                        await self.image_content_store.store_image(
                            img_content_hash, img_bytes
                        )
                        await self.cas_registry.register(
                            content_hash=img_content_hash,
                            content_id=img_content_hash,
                            vector_id=vector_id,
                        )
                        await self.cas_registry.add_reference(
                            content_hash=img_content_hash,
                            namespace=namespace,
                            document_id=document_id,
                            chunk_index=page_num,
                        )

                    vision_vectors.append(
                        {
                            "id": vector_id,
                            "embedding": emb,
                            "metadata": {
                                "document_id": document_id,
                                "page_number": page_num,
                                "content_hash": img_content_hash,
                                "source_locations": [
                                    {
                                        "document_id": document_id,
                                        "chunk_index": page_num,
                                    }
                                ],
                            },
                        }
                    )

                await self.vector_store.upsert(
                    vision_vectors, namespace, collection=vision_collection
                )
                page_image_vector_ids = [
                    v["id"] for v in vision_vectors
                ]
        except Exception as e:
            logger.error(f"Vision embedding failed: {e}")
            errors.append(f"Vision embedding error: {str(e)}")

        return {
            "page_image_vector_ids": page_image_vector_ids,
            "errors": errors,
        }

    @traced("ingestion.finalize")
    async def step_finalize_registry(
        self,
        tenant_id: str,
        doc_hash: str,
        text_result: dict,
        graph_result: dict,
        ent_rel_result: dict,
        vision_result: dict,
        chunk_content_hashes: List[str],
    ) -> None:
        """Persist all accumulated IDs to the document registry."""
        await self.document_registry.add_ids(
            tenant_id,
            doc_hash,
            text_vector_ids=text_result.get("text_vector_ids", []),
            entity_vector_ids=ent_rel_result.get("entity_vector_ids", []),
            relation_vector_ids=ent_rel_result.get(
                "relation_vector_ids", []
            ),
            page_image_vector_ids=vision_result.get(
                "page_image_vector_ids", []
            ),
            graph_node_ids=graph_result.get("graph_node_ids", []),
            graph_edge_ids=graph_result.get("graph_edge_ids", []),
            chunk_content_hashes=chunk_content_hashes,
        )

    async def _process_chunks(
        self,
        chunks: List[Chunk],
        namespace: str,
        parsed_document: Optional[ParsedDocument] = None,
        skip_embedding: bool = False,
    ) -> Tuple[
        List[str],  # errors
        List[str],  # all_vector_ids (aggregate)
        List[str],  # all_node_ids (aggregate)
        List[str],  # text_vector_ids
        List[str],  # entity_vector_ids
        List[str],  # relation_vector_ids
        List[str],  # page_image_vector_ids
        List[str],  # graph_node_ids
        List[str],  # graph_edge_ids
        List[str],  # chunk_content_hashes
    ]:
        """
        Process a batch of chunks through Embed -> CAS -> Storage.

        Derives tenant_id, embedding_model, extraction_config, and providers
        internally from the namespace.
        """
        errors: List[str] = []
        all_vector_ids: List[str] = []
        all_node_ids: List[str] = []
        # Typed ID buckets for DocumentRegistry / delete flows
        text_vector_ids: List[str] = []
        entity_vector_ids: List[str] = []
        relation_vector_ids: List[str] = []
        page_image_vector_ids: List[str] = []
        graph_node_ids: List[str] = []
        graph_edge_ids: List[str] = []
        chunk_hashes: List[str] = []

        _empty = (
            errors, all_vector_ids, all_node_ids,
            text_vector_ids, entity_vector_ids, relation_vector_ids,
            page_image_vector_ids, graph_node_ids, graph_edge_ids, chunk_hashes,
        )

        if not chunks:
            return _empty

        # Derive tenant context from namespace
        ns_config = await self.namespace_manager.get_config(namespace)
        tenant_id = ns_config.tenant_id if ns_config else "default"
        tenant_config = await self.namespace_manager.get_tenant_config(tenant_id)
        embedding_model = tenant_config.text_embedding.model if tenant_config else "default"
        raw_ext = tenant_config.extraction if tenant_config else None
        if isinstance(raw_ext, dict):
            extraction_config = ExtractionConfig(**{
                k: v for k, v in raw_ext.items()
                if k in ExtractionConfig.__dataclass_fields__
            }) if raw_ext else None
        else:
            extraction_config = raw_ext

        # Resolve providers (avoid re-fetching ns/tenant configs)
        embedder = self._resolve_embedder_from_tenant(tenant_config)
        extractor = self._resolve_extractor_from_config(extraction_config)
 
        from unified_memory.core.types import compute_vector_id

        # 1. Pre-calculate hashes and check CAS for existing embeddings
        chunks_to_embed = []
        seen_hashes = set()
        
        # Use the tenant-configured model ID for all hashing/ID decisions so
        # that vector IDs remain stable and testable.
        actual_model_id = embedding_model

        for chunk in chunks:
            if not chunk.content_hash:
                chunk.content_hash = compute_content_hash(chunk.content, tenant_id, Modality.TEXT)
            
            # Deterministic vector ID for CURRENT model
            expected_vector_id = compute_vector_id(chunk.content_hash, actual_model_id)
            
            if chunk.content_hash in seen_hashes:
                continue

            # Optimization: Check if this content already has the expected vector for the current model.
            entry = await self.cas_registry.get_entry(chunk.content_hash)
            
            if entry and entry.vector_id == expected_vector_id:
                # Already embedded with current model. 
                # Fetch existing embedding if we really need it? 
                # For now, we only embed if missing.
                pass
            else:
                chunks_to_embed.append(chunk)
                seen_hashes.add(chunk.content_hash)

        # 2. Batch Embedding (only for missing)
        if not skip_embedding and chunks_to_embed:
            try:
                contents_to_embed = [c.content for c in chunks_to_embed]
                embeddings = await embedder.embed_batch(
                    contents_to_embed, modality=Modality.TEXT
                )
                for i, emb in enumerate(embeddings):
                    current_hash = chunks_to_embed[i].content_hash
                    # Apply this embedding to ALL chunks with the same hash
                    for chunk in chunks:
                        if chunk.content_hash == current_hash:
                            chunk.embedding = emb
            except Exception as e:
                logger.exception("Embedding failed")
                return (
                    [f"Embedding error: {str(e)}"],
                    [], [], [], [], [], [], [], [], [],
                )

        # 1.5 Structural Graph Storage (Page Nodes) - Batching Preparation
        nodes_to_create: List[Any] = []
        edges_to_create: List[Any] = []
        
        if self.graph_store and parsed_document:
            from unified_memory.core.types import PageNode, NodeType
            snippet_len = (
                tenant_config.page_snippet_length
                if tenant_config and hasattr(tenant_config, "page_snippet_length")
                else 200
            )
            for page in parsed_document.pages:
                page_node = PageNode(
                    id=PageNode.make_id(parsed_document.document_id, page.page_number),
                    # Store a short snippet only — full text lives in ContentStore.
                    # Length is controlled by TenantConfig.page_snippet_length.
                    content=page.full_text[:snippet_len],
                    namespace=namespace,
                    page_number=page.page_number,
                    document_id=parsed_document.document_id,
                    node_type=NodeType.PAGE
                )
                nodes_to_create.append(page_node)
                all_node_ids.append(page_node.id)
                graph_node_ids.append(page_node.id)

        # 2. Sequential processing for CAS/Content (requires ordering/locking)
        errors = []
        processed_chunks: List[Chunk] = []
        
        # Resolve collection name (P0 fix #15)
        text_collection = await self.namespace_manager.get_collection_name(
            namespace, CollectionType.TEXTS
        )

        # Accumulator for batched text vector upsert
        text_vectors_to_upsert: List[Dict[str, Any]] = []
        sparse_docs_to_upsert: List[Dict[str, Any]] = []
        
        for i, chunk in enumerate(chunks):
            chunk_hashes.append(chunk.content_hash)
            try:
                # Use current model for vector ID
                vector_id = compute_vector_id(chunk.content_hash, actual_model_id,"text")
                content_id = f"content:{chunk.content_hash}"
                
                # CAS Registration (idempotent, updates vector_id to latest model used)
                await self.cas_registry.register(
                    content_hash=chunk.content_hash,
                    content_id=content_id,
                    vector_id=vector_id,
                )
                
                # Add usage reference
                await self.cas_registry.add_reference(
                    content_hash=chunk.content_hash,
                    namespace=namespace,
                    document_id=chunk.document_id,
                    chunk_index=chunk.chunk_index
                )
                
                # Content Storage (deduplicated)
                await self.content_store.store_content(chunk.content_hash, chunk.content)
                
                # Vector Storage Preparation (Shared Entries strategy)
                chunk_loc = SourceLocation(
                    document_id=chunk.document_id,
                    chunk_index=chunk.chunk_index,
                )
 
                if chunk.embedding is not None:
                    vector_data = {
                        "id": vector_id,
                        "embedding": chunk.embedding,
                        "metadata": {
                            **chunk.metadata,
                            "content_hash": chunk.content_hash,
                            "document_id": chunk.document_id,
                            "source_locations": [chunk_loc.to_dict()],
                        },
                    }
                    text_vectors_to_upsert.append(vector_data)
                    all_vector_ids.append(vector_id)
                    text_vector_ids.append(vector_id)
                else:
                    # Content was already in CAS (deduplicated).
                    # We must still link it to the current namespace in the vector store.
                    await self.vector_store.add_namespace(
                        vector_id, namespace, collection=text_collection, document_id=chunk.document_id
                    )
                    all_vector_ids.append(vector_id)
                    text_vector_ids.append(vector_id)
                
                # Sparse Storage (BM25 Index)
                sparse_docs_to_upsert.append({
                    "id": chunk.content_hash,
                    "content": chunk.content,
                    "metadata": {
                        "document_id": chunk.document_id,
                    },
                })
                
                # 3. Extraction and Graph Storage Preparation (P0 fix #3 - batching)
                if self.graph_store:
                    from unified_memory.core.types import EntityNode, GraphEdge, NodeType, PassageNode

                    chunk_loc = SourceLocation(
                        document_id=chunk.document_id,
                        chunk_index=chunk.chunk_index,
                    )
                    passage_node = PassageNode(
                        id=PassageNode.make_id(tenant_id, chunk.content_hash),
                        content="",
                        namespace=namespace,
                        node_type=NodeType.PASSAGE,
                        page_number=chunk.page_number,
                        source_locations=[chunk_loc],
                        properties={**chunk.metadata, "content_hash": chunk.content_hash},
                    )
                    nodes_to_create.append(passage_node)
                    all_node_ids.append(passage_node.id)
                    graph_node_ids.append(passage_node.id)

                    if chunk.page_number is not None:
                        edges_to_create.append(GraphEdge(
                            source_id=PageNode.make_id(chunk.document_id, chunk.page_number),
                            target_id=passage_node.id,
                            relation="HAS_CHUNK",
                            source_locations=[chunk_loc],
                            namespace=namespace,
                        ))

                    if extractor:
                        rels_buffer = []
                        extraction = await extractor.extract(chunk)

                        entities = extraction.entities
                        relations = extraction.relations
                        if extraction_config and extraction_config.strict_type_filtering:
                            # Drop entities/relations whose type is not in the
                            # allow-list.  Only active when strict_type_filtering
                            # is True; the default is False because LLM-generated
                            # type labels are open-ended and a closed list would
                            # silently discard valid extractions.
                            if extraction_config.entity_types:
                                entities = [
                                    e for e in entities
                                    if e.type in extraction_config.entity_types
                                ]
                            if extraction_config.relation_types:
                                relations = [
                                    r for r in relations
                                    if r.relation_type in extraction_config.relation_types
                                ]

                        for ent in entities:
                            entity_id = make_entity_id(ent.name, tenant_id)
                            node = EntityNode(
                                id=entity_id,
                                content=ent.description or "",
                                namespace=namespace,
                                entity_name=ent.name,
                                entity_type=ent.type,
                                source_locations=[chunk_loc],
                                properties=ent.properties,
                                node_type=NodeType.ENTITY,
                            )
                            nodes_to_create.append(node)
                            all_node_ids.append(node.id)
                            graph_node_ids.append(node.id)
                            edges_to_create.append(
                                GraphEdge(
                                    source_id=passage_node.id,
                                    target_id=node.id,
                                    relation="MENTIONS",
                                    source_locations=[chunk_loc],
                                    namespace=namespace,
                                )
                            )

                        for rel in relations:
                            src_id = make_entity_id(rel.source_entity, tenant_id)
                            tgt_id = make_entity_id(rel.target_entity, tenant_id)
                            # source_type / target_type are optional hints from the
                            # extractor.  They do NOT change the stable entity ID
                            # (which is name-only) but are stored as edge properties
                            # so downstream consumers can use them for disambiguation.
                            extra_props: Dict[str, Any] = {**rel.properties}
                            if rel.description:
                                extra_props["description"] = rel.description
                            if rel.confidence != 1.0:
                                extra_props["confidence"] = rel.confidence
                            if rel.keywords:
                                extra_props["keywords"] = rel.keywords
                            if rel.source_type:
                                extra_props["source_type"] = rel.source_type
                            if rel.target_type:
                                extra_props["target_type"] = rel.target_type
                            edge = GraphEdge(
                                source_id=src_id,
                                target_id=tgt_id,
                                relation=normalize_relation_type(rel.relation_type),
                                weight=rel.weight,
                                is_bidirectional=rel.is_bidirectional,
                                inverse_relation=rel.inverse_relation,
                                source_locations=[chunk_loc],
                                namespace=namespace,
                                properties=extra_props,
                                source_entity_name=rel.source_entity,
                                target_entity_name=rel.target_entity,
                            )
                            edges_to_create.append(edge)
                            rels_buffer.append(edge)

                
                processed_chunks.append(chunk)

            except Exception as e:
                logger.exception(f"Failed to process chunk {i}")
                errors.append(f"Chunk {i} processing error: {str(e)}")


        # 4. Batch Operations for Vector and Graph stores (P0 fix #3)
        
        # Deduplicate Entities for both Vector and Graph (merge provenance)
        unique_entity_nodes: Dict[str, EntityNode] = {}
        other_nodes = []
        for node in nodes_to_create:
            if hasattr(node, "node_type") and node.node_type == NodeType.ENTITY:
                if node.id not in unique_entity_nodes:
                    unique_entity_nodes[node.id] = node
                else:
                    existing = unique_entity_nodes[node.id]
                    existing_locs = getattr(existing, "source_locations", [])
                    new_locs = getattr(node, "source_locations", [])
                    for loc in new_locs:
                        if loc not in existing_locs:
                            existing_locs.append(loc)
            else:
                other_nodes.append(node)
        
        final_nodes_to_create = list(unique_entity_nodes.values()) + other_nodes

        # Deduplicate Relations for both Vector and Graph (merge provenance and keywords)
        unique_relations: Dict[tuple, GraphEdge] = {}
        for edge in edges_to_create:
            key = (edge.source_id, edge.relation, edge.target_id)
            if key not in unique_relations:
                unique_relations[key] = edge
            else:
                existing = unique_relations[key]
                # Merge source_locations
                existing_locs = getattr(existing, "source_locations", [])
                new_locs = getattr(edge, "source_locations", [])
                for loc in new_locs:
                    if loc not in existing_locs:
                        existing_locs.append(loc)
                
                # Merge keywords
                existing_kw = set(existing.properties.get("keywords") or [])
                new_kw = edge.properties.get("keywords") or []
                if new_kw:
                    existing_kw.update(new_kw)
                    existing.properties["keywords"] = list(existing_kw)

        final_edges_to_create = list(unique_relations.values())

        # 4.1 Batch vector upsert for text chunks
        if text_vectors_to_upsert:
            try:
                await self.vector_store.upsert(
                    text_vectors_to_upsert, namespace, collection=text_collection
                )
            except Exception as e:
                logger.error(f"Batch text vector upsert failed: {e}")
                errors.append(f"Vector batch error: {str(e)}")
 
        # 4.1.5 Batch sparse upsert (BM25)
        if self.sparse_store and sparse_docs_to_upsert:
            try:
                await self.sparse_store.index(sparse_docs_to_upsert, namespace)
            except Exception as e:
                logger.error(f"Sparse index upsert failed: {e}")
                # We don't fail the whole ingest for sparse index errors, but log it
                errors.append(f"Sparse index error: {str(e)}")

        # 4.2 Graph nodes/edges
        if self.graph_store and (final_nodes_to_create or final_edges_to_create):
            try:
                if final_nodes_to_create:
                    await self.graph_store.create_nodes_batch(final_nodes_to_create, namespace)
                if final_edges_to_create:
                    await self.graph_store.create_edges_batch(final_edges_to_create, namespace)
            except Exception as e:
                logger.error(f"Batch graph creation failed: {e}")
                errors.append(f"Graph batch error: {str(e)}")

        # 4.5 Vision Embeddings (Phase 2.3)
        vision_embedder = self._resolve_vision_embedder_from_tenant(tenant_config)
        enable_visual = bool(
            getattr(tenant_config, "enable_visual_indexing", False) if tenant_config else False
        )
        if vision_embedder and parsed_document and not skip_embedding and enable_visual:
            try:
                pages_with_images = [p for p in parsed_document.pages if p.full_page_image]
                if pages_with_images:
                    vision_collection = await self.namespace_manager.get_collection_name(
                        namespace, CollectionType.PAGE_IMAGES
                    )
                    
                    # Embed batch
                    images = [p.full_page_image for p in pages_with_images]
                    vision_embeddings = await vision_embedder.embed_batch(
                        images, modality=Modality.IMAGE
                    )
                    
                    vision_vectors: List[Dict[str, Any]] = []
                    import hashlib
                    from unified_memory.core.types import compute_vector_id

                    vision_model_name = (
                        tenant_config.vision_embedding.model
                        if tenant_config and tenant_config.vision_embedding
                        else "default"
                    )
                    for page, emb in zip(pages_with_images, vision_embeddings):
                        raw_hash = hashlib.sha256(page.full_page_image).hexdigest()
                        img_content_hash = compute_content_hash(
                            raw_hash, tenant_id, Modality.IMAGE
                        )
                        vector_id = compute_vector_id(
                            img_content_hash, vision_model_name, "image"
                        )

                        if getattr(self, "image_content_store", None):
                            await self.image_content_store.store_image(
                                img_content_hash, page.full_page_image
                            )
                            await self.cas_registry.register(
                                content_hash=img_content_hash,
                                content_id=img_content_hash,
                                vector_id=vector_id,
                            )
                            await self.cas_registry.add_reference(
                                content_hash=img_content_hash,
                                namespace=namespace,
                                document_id=parsed_document.document_id,
                                chunk_index=page.page_number,
                            )

                        vision_vectors.append({
                            "id": vector_id,
                            "embedding": emb,
                            "metadata": {
                                "document_id": parsed_document.document_id,
                                "page_number": page.page_number,
                                "content_hash": img_content_hash,
                                "source_locations": [
                                    {
                                        "document_id": parsed_document.document_id,
                                        "chunk_index": page.page_number,
                                    }
                                ],
                            }
                        })
                    
                    await self.vector_store.upsert(
                        vision_vectors, namespace, collection=vision_collection
                    )
                    ids = [v["id"] for v in vision_vectors]
                    all_vector_ids.extend(ids)
                    page_image_vector_ids.extend(ids)
            except Exception as e:
                logger.error(f"Vision embedding failed: {e}")
                errors.append(f"Vision embedding error: {str(e)}")

        # 5. Entity/Relation Embeddings (P0 fix #2)
        # Skip if skip_embedding is True (assuming entities are also covered by existing doc check? 
        # Actually entities might be new if extractor changed, but for now we follow the doc-level dedup logic)
        if self.vector_store and final_nodes_to_create and not skip_embedding:
            entity_nodes = [
                node
                for node in final_nodes_to_create
                if hasattr(node, "node_type") and node.node_type == NodeType.ENTITY
            ]
            if entity_nodes:
                entity_collection = await self.namespace_manager.get_collection_name(
                    namespace, CollectionType.ENTITIES
                )

                if entity_nodes:
                    # Use Entity.get_embedding_text() for richer, centralised logic
                    from unified_memory.core.types import Entity as _EntityType

                    ent_contents = []
                    for e in entity_nodes:
                        # EntityNode stores denormalised fields; build a lightweight Entity
                        tmp_entity = _EntityType(
                            name=getattr(e, "entity_name", ""),
                            entity_type=getattr(e, "entity_type", "entity"),
                            description=getattr(e, "content", "") or "",
                        )
                        ent_contents.append(tmp_entity.get_embedding_text())
                    try:
                        ent_embeddings = await embedder.embed_batch(
                            ent_contents, modality=Modality.TEXT
                        )
                        
                        ent_vector_data = []
                        for ent, emb in zip(entity_nodes, ent_embeddings):
                            # Get provenance for metadata
                            ent_locs = getattr(ent, "source_locations", [])
                            provenance = source_locations_to_parallel_arrays(ent_locs)
                            
                            ent_vector_data.append(
                                {
                                    "id": ent.id,
                                    "embedding": emb,
                                    "metadata": {
                                        "entity_name": ent.entity_name,
                                        "entity_type": ent.entity_type,
                                        "content_hash": compute_content_hash(
                                            ent.content or ent.entity_name,
                                            tenant_id,
                                            Modality.TEXT,
                                        ),
                                        "document_id": parsed_document.document_id,
                                        "source_doc_ids": provenance["source_doc_ids"],
                                        "source_locations": [
                                            loc.to_dict() for loc in ent_locs
                                        ],
                                    },
                                }
                            )
                        
                        await self.vector_store.upsert(
                            ent_vector_data, namespace, collection=entity_collection
                        )
                        ids = [v["id"] for v in ent_vector_data]
                        all_vector_ids.extend(ids)
                        entity_vector_ids.extend(ids)
                    except Exception as e:
                        logger.error(f"Entity embedding/storage failed: {e}")
                        errors.append(f"Entity semantic index error: {str(e)}")

        # 5.5 Relation Embeddings (Phase 2.4)
        if self.vector_store and final_edges_to_create and not skip_embedding:
            try:
                rel_collection = await self.namespace_manager.get_collection_name(
                    namespace, CollectionType.RELATIONS
                )

                # final_edges_to_create contains unique relations with merged provenance and keywords
                rel_list = final_edges_to_create

                rel_texts: List[str] = []
                for e in rel_list:
                    tmp_rel = Relation(
                        subject_id=e.source_id,
                        predicate=e.relation,
                        object_id=e.target_id,
                        subject=getattr(e, "source_entity_name", "") or e.source_id,
                        object=getattr(e, "target_entity_name", "") or e.target_id,
                        description=e.properties.get("description", ""),
                        keywords=e.properties.get("keywords", []),
                        inverse_relation=e.inverse_relation,
                    )
                    rel_texts.append(tmp_rel.get_embedding_text())

                if rel_texts:
                    rel_embeddings = await embedder.embed_batch(
                        rel_texts, modality=Modality.TEXT
                    )

                    rel_vector_data = []
                    for edge, emb, text in zip(rel_list, rel_embeddings, rel_texts):
                        # Get provenance
                        rel_locs = getattr(edge, "source_locations", [])
                        provenance = source_locations_to_parallel_arrays(rel_locs)
                        
                        rel_vector_data.append(
                            {
                                "id": edge.id,
                                "embedding": emb,
                                "metadata": {
                                    "source_id": edge.source_id,
                                    "target_id": edge.target_id,
                                    "relation": edge.relation,
                                    "content_hash": compute_content_hash(
                                        text, tenant_id, Modality.TEXT
                                    ),
                                    "text": text,  # helpful for debugging/retrieval context
                                    "document_id": parsed_document.document_id,
                                    "source_doc_ids": provenance["source_doc_ids"],
                                    "source_locations": [loc.to_dict() for loc in rel_locs],
                                },
                            }
                        )

                    await self.vector_store.upsert(
                        rel_vector_data, namespace, collection=rel_collection
                    )
                    ids = [v["id"] for v in rel_vector_data]
                    all_vector_ids.extend(ids)
                    relation_vector_ids.extend(ids)
            except Exception as e:
                logger.error(f"Relation embedding failed: {e}")
                errors.append(f"Relation embedding error: {str(e)}")

        # Collect all edge IDs as well for registry/delete flows
        for edge in edges_to_create:
            all_node_ids.append(edge.id)
            graph_edge_ids.append(edge.id)

        return (
            errors,
            all_vector_ids,
            all_node_ids,
            text_vector_ids,
            entity_vector_ids,
            relation_vector_ids,
            page_image_vector_ids,
            graph_node_ids,
            graph_edge_ids,
            chunk_hashes,
        )

    @traced("ingestion.file")
    async def ingest_file(
        self,
        path: Path,
        namespace: str = "default",
        document_id: Optional[str] = None,
        skip_embedding: bool = False,
        **options: Any,
    ) -> IngestionResult:
        doc_id = document_id or str(uuid.uuid4())
        # Initial source_ref for error paths; will be replaced with parser-provided
        # source once parsing succeeds.
        source_ref = SourceReference(source_id=doc_id, source_type=SourceType.TEXT_BLOCK)

        parser = self.parser_registry.get_parser_for_file(path)
        if not parser:
            return IngestionResult(doc_id, source_ref, errors=[f"No parser for {path.suffix}"])
        
        try:
            # P1 fix #5 - Validate embedding model against tenant config
            ns_config = await self.namespace_manager.get_config(namespace)
            if not ns_config:
                return IngestionResult(doc_id, source_ref, errors=[f"Namespace {namespace} not found"])
            
            tenant_config = await self.namespace_manager.get_tenant_config(ns_config.tenant_id)

            # options["embedding_model"] is accepted for backward-compatibility
            # logging but NEVER used for hashing or storage decisions.  All
            # stored artefacts (chunks, embeddings, CAS entries) are keyed by
            # the tenant-canonical model so that cross-namespace retrieval
            # always operates in a single, compatible embedding space.
            requested_model = options.get("embedding_model")
            if requested_model and requested_model != tenant_config.text_embedding.model:
                logger.warning(
                    "options['embedding_model']='%s' is ignored; tenant '%s' uses '%s'. "
                    "Remove this override to suppress the warning.",
                    requested_model,
                    ns_config.tenant_id,
                    tenant_config.text_embedding.model,
                )

            parsed = await parser.parse_file(path, doc_id, **options)

            # Use the parser's source reference (with its default_source_type)
            # instead of hard-coding TEXT_BLOCK here.
            source_ref = parsed.source
            
            # Phase 1.3: Document Registry Check
            # Use tenant-scoped document hash (independent of embedding model)
            doc_hash = compute_document_hash(parsed.full_text, ns_config.tenant_id)
            existing_doc = await self.document_registry.get_document(ns_config.tenant_id, doc_hash)
            
            if existing_doc:
                # If this namespace already has this document, no-op fast path
                if namespace in existing_doc.namespaces:
                    logger.info(
                        "Document %s already exists in namespace %s; skipping ingestion.",
                        doc_hash,
                        namespace,
                    )
                    return IngestionResult(
                        document_id=existing_doc.document_id,
                        source=source_ref,
                        chunk_count=len(existing_doc.chunk_content_hashes),
                        chunks=[
                            Chunk(
                                document_id=existing_doc.document_id,
                                content="",
                                chunk_index=i,
                                content_hash=content_hash,
                            )
                            for i, content_hash in enumerate(existing_doc.chunk_content_hashes)
                        ],
                        page_count=parsed.page_count,
                        deduped=True,
                        doc_hash=doc_hash,
                    )

                # PATH B: Fast link
                logger.info(f"Document {doc_hash} already exists. Path B (Fast Link) for {namespace}")
                
                # Add namespace to registry
                await self.document_registry.add_namespace(
                    ns_config.tenant_id, doc_hash, namespace
                )

                # Update CAS references so CAS remains the source of truth for
                # (namespace, document_id, chunk_index) ownership.
                for i, content_hash in enumerate(existing_doc.chunk_content_hashes):
                    await self.cas_registry.add_reference(
                        content_hash=content_hash,
                        namespace=namespace,
                        document_id=existing_doc.document_id,
                        chunk_index=i,
                    )

                # Grant access to all vectors (text + entities + relations + page images)
                all_vec_ids: List[str] = (
                    existing_doc.text_vector_ids
                    + existing_doc.entity_vector_ids
                    + existing_doc.relation_vector_ids
                    + existing_doc.page_image_vector_ids
                )
                for vec_id in all_vec_ids:
                    await self.vector_store.add_namespace(vec_id, namespace)

                # Grant access to all nodes/edges
                if self.graph_store and tenant_config.enable_graph_storage:
                    all_node_ids: List[str] = (
                        existing_doc.graph_node_ids + existing_doc.graph_edge_ids
                    )
                    for node_id in all_node_ids:
                        await self.graph_store.add_namespace(node_id, namespace)
 
                # Sparse Storage (BM25 Index) - re-index with new namespace
                if self.sparse_store and existing_doc.chunk_content_hashes:
                    sparse_docs = []
                    for h in existing_doc.chunk_content_hashes:
                        content = await self.content_store.get_content(h)
                        if content:
                            sparse_docs.append({
                                "id": h,
                                "content": content,
                                "metadata": {"document_id": existing_doc.document_id}
                            })
                    if sparse_docs:
                        await self.sparse_store.index(sparse_docs, namespace)

                # Prepare chunk results for the response
                chunks_for_result = [
                    Chunk(
                        document_id=existing_doc.document_id,
                        content="",
                        chunk_index=i,
                        content_hash=content_hash,
                    )
                    for i, content_hash in enumerate(existing_doc.chunk_content_hashes)
                ]
                
                # Update namespace-docs index
                await self.document_registry.add_doc_to_namespace_index(
                    namespace, doc_hash, existing_doc.document_id
                )

                return IngestionResult(
                    document_id=existing_doc.document_id,
                    source=source_ref,
                    chunk_count=len(chunks_for_result),
                    chunks=chunks_for_result,
                    page_count=parsed.page_count,
                    deduped=True,
                    doc_hash=doc_hash,
                )
            
            # PATH A: Full Ingestion
            # Register new doc
            await self.document_registry.register_document(
                ns_config.tenant_id, doc_hash, namespace, doc_id
            )
            
            # Use tenant-configured chunker; always use the tenant-canonical
            # embedding model for hashing so stored artefacts are consistent.
            req_chunker, chunk_cfg = self._build_chunker_for_tenant(tenant_config)
            chunks = await req_chunker.chunk(
                parsed,
                namespace,
                tenant_id=ns_config.tenant_id,
                config=chunk_cfg,
            )

            (
                processing_errors,
                _all_vec_ids,
                _all_node_ids,
                text_vec_ids,
                entity_vec_ids,
                relation_vec_ids,
                page_image_vec_ids,
                graph_node_ids,
                graph_edge_ids,
                chunk_hashes,
            ) = await self._process_chunks(
                chunks,
                namespace,
                parsed_document=parsed,
                skip_embedding=skip_embedding,
            )

            await self.document_registry.add_ids(
                ns_config.tenant_id,
                doc_hash,
                text_vector_ids=text_vec_ids,
                entity_vector_ids=entity_vec_ids,
                relation_vector_ids=relation_vec_ids,
                page_image_vector_ids=page_image_vec_ids,
                graph_node_ids=graph_node_ids,
                graph_edge_ids=graph_edge_ids,
                chunk_content_hashes=chunk_hashes,
            )

            # Update namespace-docs index
            await self.document_registry.add_doc_to_namespace_index(
                namespace, doc_hash, doc_id
            )

            return IngestionResult(
                document_id=doc_id,
                source=source_ref,
                chunk_count=len(chunks),
                page_count=parsed.page_count,
                chunks=chunks,
                errors=parsed.parse_errors + processing_errors,
                deduped=False,
                doc_hash=doc_hash,
            )
        except Exception as e:
            logger.exception(f"Error ingesting file: {path}")
            return IngestionResult(doc_id, source_ref, errors=[str(e)])

    @traced("ingestion.text")
    async def ingest_text(
        self,
        text: str,
        namespace: str = "default",
        document_id: Optional[str] = None,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        skip_embedding: bool = False,
    ) -> IngestionResult:
        import uuid
        from unified_memory.core.types import PageContent
        
        doc_id = document_id or str(uuid.uuid4())
        source_ref = SourceReference(source_id=doc_id, source_type=SourceType.TEXT_BLOCK)
        
        parsed = ParsedDocument(
            document_id=doc_id,
            source=source_ref,
            title=title,
            pages=[PageContent(page_number=1, document_id=doc_id, text_blocks=[{"text": text}], full_text=text)],
            full_text=text,
            metadata=metadata or {},
        )
        
        try:
            # P1 fix #5 - Validate embedding model
            ns_config = await self.namespace_manager.get_config(namespace)
            if not ns_config:
                 return IngestionResult(doc_id, source_ref, errors=[f"Namespace {namespace} not found"])
            
            tenant_config = await self.namespace_manager.get_tenant_config(ns_config.tenant_id)
            embedding_model = tenant_config.text_embedding.model
            
            # Phase 1.3: Document Registry Check
            # Use tenant-scoped document hash (independent of embedding model)
            doc_hash = compute_document_hash(parsed.full_text, ns_config.tenant_id)
            existing_doc = await self.document_registry.get_document(ns_config.tenant_id, doc_hash)
            
            if existing_doc:
                # If this namespace already has this document, no-op fast path
                if namespace in existing_doc.namespaces:
                    logger.info(
                        "Text doc %s already exists in namespace %s; skipping ingestion.",
                        doc_hash,
                        namespace,
                    )
                    # For text ingestion, we only have text vectors recorded
                    text_vec_ids = existing_doc.text_vector_ids
                    chunks_for_result = [
                        Chunk(
                            document_id=existing_doc.document_id,
                            content="",
                            chunk_index=i,
                            content_hash=vid.split(":")[-1],
                        )
                        for i, vid in enumerate(text_vec_ids)
                    ]
                    return IngestionResult(
                        document_id=existing_doc.document_id,
                        source=source_ref,
                        chunk_count=len(chunks_for_result),
                        chunks=chunks_for_result,
                        page_count=1,
                        deduped=True,
                        doc_hash=doc_hash,
                    )

            if existing_doc:
                # PATH B: Fast link
                logger.info(
                    f"Text doc {doc_hash} already exists. Path B (Fast Link) for {namespace}"
                )
                await self.document_registry.add_namespace(
                    ns_config.tenant_id, doc_hash, namespace
                )

                for i, content_hash in enumerate(existing_doc.chunk_content_hashes):
                    await self.cas_registry.add_reference(
                        content_hash=content_hash,
                        namespace=namespace,
                        document_id=existing_doc.document_id,
                        chunk_index=i,
                    )

                # Grant access to all vectors (text + entities + relations + page images)
                all_vec_ids: List[str] = (
                    existing_doc.text_vector_ids
                    + existing_doc.entity_vector_ids
                    + existing_doc.relation_vector_ids
                    + existing_doc.page_image_vector_ids
                )
                for vec_id in all_vec_ids:
                    await self.vector_store.add_namespace(vec_id, namespace)

                # Graph nodes/edges
                if self.graph_store and tenant_config.enable_graph_storage:
                    all_node_ids: List[str] = (
                        existing_doc.graph_node_ids + existing_doc.graph_edge_ids
                    )
                    for node_id in all_node_ids:
                        await self.graph_store.add_namespace(node_id, namespace)
 
                # Sparse Storage (BM25 Index) - re-index with new namespace
                if self.sparse_store and existing_doc.chunk_content_hashes:
                    sparse_docs = []
                    for h in existing_doc.chunk_content_hashes:
                        content = await self.content_store.get_content(h)
                        if content:
                            sparse_docs.append({
                                "id": h,
                                "content": content,
                                "metadata": {"document_id": existing_doc.document_id}
                            })
                    if sparse_docs:
                        await self.sparse_store.index(sparse_docs, namespace)

                # Prepare chunk results for the response (text vectors only)
                text_vec_ids = existing_doc.text_vector_ids
                chunks_for_result = [
                    Chunk(
                        document_id=existing_doc.document_id,
                        content="",
                        chunk_index=i,
                        content_hash=vid.split(":")[-1],
                    )
                    for i, vid in enumerate(text_vec_ids)
                ]

                # Update namespace-docs index
                await self.document_registry.add_doc_to_namespace_index(
                    namespace, doc_hash, existing_doc.document_id
                )

                return IngestionResult(
                    document_id=existing_doc.document_id,
                    source=source_ref,
                    chunk_count=len(chunks_for_result),
                    chunks=chunks_for_result,
                    page_count=1,
                    deduped=True,
                    doc_hash=doc_hash,
                )
            
            # PATH A: Full Ingestion
            await self.document_registry.register_document(
                ns_config.tenant_id, doc_hash, namespace, doc_id
            )
            
            # Phase 1.5: Tenant Ingestion Config (chunker type + params)
            req_chunker, chunk_cfg = self._build_chunker_for_tenant(tenant_config)
            chunks = await req_chunker.chunk(
                parsed,
                namespace,
                tenant_id=ns_config.tenant_id,
                config=chunk_cfg,
            )
            
            (
                processing_errors,
                _all_vec_ids,
                _all_node_ids,
                text_vec_ids,
                entity_vec_ids,
                relation_vec_ids,
                page_image_vec_ids,
                graph_node_ids,
                graph_edge_ids,
                chunk_hashes,
            ) = await self._process_chunks(
                chunks,
                namespace,
                parsed_document=parsed,
                skip_embedding=skip_embedding,
            )

            await self.document_registry.add_ids(
                ns_config.tenant_id,
                doc_hash,
                text_vector_ids=text_vec_ids,
                entity_vector_ids=entity_vec_ids,
                relation_vector_ids=relation_vec_ids,
                page_image_vector_ids=page_image_vec_ids,
                graph_node_ids=graph_node_ids,
                graph_edge_ids=graph_edge_ids,
                chunk_content_hashes=chunk_hashes,
            )
            
            # Update namespace-docs index
            await self.document_registry.add_doc_to_namespace_index(
                namespace, doc_hash, doc_id
            )

            return IngestionResult(
                document_id=doc_id,
                source=source_ref,
                chunk_count=len(chunks),
                page_count=1,
                chunks=chunks,
                errors=processing_errors,
                deduped=False,
                doc_hash=doc_hash,
            )
        except Exception as e:
            logger.exception(f"Error ingesting text: {doc_id}")
            return IngestionResult(doc_id, source_ref, errors=[str(e)])

    async def _namespace_still_needed(
        self,
        remaining_doc_ids: List[str],
        namespace: str,
    ) -> bool:
        """Return ``True`` if any of *remaining_doc_ids* still belongs to
        *namespace* according to the ``DocumentRegistry``.

        This is the check that prevents shared vectors from being
        prematurely removed when one of several contributing documents
        is deleted.
        """
        for doc_id in remaining_doc_ids:
            doc_entry = await self.document_registry.get_document_by_document_id(
                doc_id
            )
            if doc_entry and namespace in doc_entry.namespaces:
                return True
        return False

    async def _remove_vector_smart(
        self,
        vid: str,
        namespace: str,
        document_id: str,
        collection: str,
        result: DeleteResult,
    ) -> None:
        """Two-step vector removal: dissociate the document, then
        conditionally remove the namespace."""
        remaining = await self.vector_store.remove_document_reference(
            vid, document_id, collection=collection
        )
        if remaining and await self._namespace_still_needed(remaining, namespace):
            result.vectors_unlinked += 1
            return

        success, was_last = await self.vector_store.remove_namespace(
            vid, namespace, collection=collection
        )
        if not success:
            return
        if was_last:
            result.vectors_deleted += 1
        else:
            result.vectors_unlinked += 1

    async def _remove_graph_smart(
        self,
        gid: str,
        namespace: str,
        document_id: str,
        result: DeleteResult,
        is_node: bool = True,
    ) -> None:
        """Two-step graph element removal."""
        remaining = await self.graph_store.remove_document_reference(
            gid, document_id
        )
        if remaining and await self._namespace_still_needed(remaining, namespace):
            if is_node:
                result.nodes_unlinked += 1
            return

        success, was_last = await self.graph_store.remove_namespace(gid, namespace)
        if not success:
            return
        if was_last:
            result.nodes_deleted += 1
        else:
            result.nodes_unlinked += 1

    @traced("ingestion.delete")
    async def delete_document(
        self,
        tenant_id: str,
        document_hash: str,
        namespace: str,
    ) -> DeleteResult:
        """Ref-count-aware delete for a document in a given namespace.

        For each vector / graph element the method first removes the
        document's metadata (``source_doc_ids``, ``source_locations``).
        It then checks, via ``DocumentRegistry``, whether any of the
        *remaining* documents still use this namespace.  Only when the
        namespace is no longer needed does it actually remove the
        namespace from the element (and hard-delete if it was the last).

        This prevents shared vectors from being prematurely deleted
        when two documents contribute the same chunk to the same
        namespace.
        """

        # Lookup document entry
        entry = await self.document_registry.get_document(tenant_id, document_hash)
        if not entry:
            return DeleteResult(found=False)

        # Update registry namespaces first (typed IDs remain on entry)
        await self.document_registry.remove_namespace(tenant_id, document_hash, namespace)

        result = DeleteResult(found=True)

        # --- Vector store clean-up per collection ---

        if entry.text_vector_ids:
            text_col = await self.namespace_manager.get_collection_name(
                namespace, CollectionType.TEXTS
            )
            for vid in entry.text_vector_ids:
                await self._remove_vector_smart(
                    vid, namespace, entry.document_id, text_col, result
                )

        # ENTITIES
        if entry.entity_vector_ids:
            ent_col = await self.namespace_manager.get_collection_name(
                namespace, CollectionType.ENTITIES
            )
            for vid in entry.entity_vector_ids:
                await self._remove_vector_smart(
                    vid, namespace, entry.document_id, ent_col, result
                )

        # RELATIONS
        if entry.relation_vector_ids:
            rel_col = await self.namespace_manager.get_collection_name(
                namespace, CollectionType.RELATIONS
            )
            for vid in entry.relation_vector_ids:
                await self._remove_vector_smart(
                    vid, namespace, entry.document_id, rel_col, result
                )

        # PAGE IMAGES — image CAS cleanup first (needs vector metadata), then vector removal
        if entry.page_image_vector_ids:
            page_col = await self.namespace_manager.get_collection_name(
                namespace, CollectionType.PAGE_IMAGES
            )
            if getattr(self, "image_content_store", None):
                for vid in entry.page_image_vector_ids:
                    try:
                        vec = await self.vector_store.get_by_id(
                            vid, collection=page_col, namespace=namespace
                        )
                        if not vec:
                            continue
                        img_content_hash = vec.metadata.get("content_hash")
                        if not img_content_hash:
                            continue
                        await self.cas_registry.remove_reference(
                            content_hash=img_content_hash,
                            namespace=namespace,
                            document_id=entry.document_id,
                            chunk_index=None,
                        )
                        remaining = await self.cas_registry.get_entry(
                            img_content_hash
                        )
                        if not remaining or remaining.refs:
                            continue
                        await self.image_content_store.delete_image(
                            img_content_hash
                        )
                        await self.cas_registry.delete_if_orphan(
                            img_content_hash
                        )
                    except Exception:
                        logger.exception(
                            "Error during image CAS cleanup for vector %s",
                            vid,
                        )
            for vid in entry.page_image_vector_ids:
                await self._remove_vector_smart(
                    vid, namespace, entry.document_id, page_col, result
                )

        # --- Graph clean-up ---

        if self.graph_store:
            # Nodes
            for nid in entry.graph_node_ids:
                await self._remove_graph_smart(
                    nid, namespace, entry.document_id, result, is_node=True
                )
            # Edges
            for eid in entry.graph_edge_ids:
                await self._remove_graph_smart(
                    eid, namespace, entry.document_id, result, is_node=False
                )

        # --- CAS / ContentStore clean-up ---

        # CAS / ContentStore clean-up using chunk content hashes.
        # We conservatively remove this namespace's references for the document_id
        # tracked in the registry, and only hard-delete content + CAS entry when
        # *no* references remain.
        for content_hash in entry.chunk_content_hashes:
            try:
                # Remove all references for this (namespace, document_id) pair.
                await self.cas_registry.remove_reference(
                    content_hash=content_hash,
                    namespace=namespace,
                    document_id=entry.document_id,
                    chunk_index=None,
                )

                cas_entry = await self.cas_registry.get_entry(content_hash)
                if not cas_entry or cas_entry.refs:
                    # Either already deleted or still referenced elsewhere.
                    continue

                # Last reference: delete content payload and CAS entry.
                try:
                    await self.content_store.delete_content(cas_entry.content_id)
                except Exception:
                    logger.exception(
                        "Failed to delete content for orphaned hash %s",
                        content_hash,
                    )

                await self.cas_registry.delete_if_orphan(content_hash)
            except Exception:
                logger.exception(
                    "Error during CAS/content cleanup for hash %s",
                    content_hash,
                )

        # --- Sparse store clean-up ---

        if self.sparse_store and entry.chunk_content_hashes:
            try:
                if hasattr(self.sparse_store, "remove_document_reference"):
                    remaining_map = await self.sparse_store.remove_document_reference(
                        doc_ids=entry.chunk_content_hashes,
                        namespace=namespace,
                        document_id=entry.document_id,
                    )
                    hashes_to_delete = [
                        h for h, docs in remaining_map.items()
                        if not docs or not await self._namespace_still_needed(docs, namespace)
                    ]
                    if hashes_to_delete:
                        await self.sparse_store.delete(
                            doc_ids=hashes_to_delete,
                            namespace=namespace,
                        )
                else:
                    await self.sparse_store.delete(
                        doc_ids=entry.chunk_content_hashes,
                        namespace=namespace,
                        document_id=entry.document_id,
                    )
            except Exception as e:
                logger.error(f"Sparse index delete failed: {e}")

        return result
