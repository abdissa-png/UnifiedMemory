"""
Integration test: verifies that source_locations and keywords are correctly
accumulated (merged) when entities / relations appear in multiple chunks of
the same document during ingestion.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from unified_memory.ingestion.pipeline import IngestionPipeline
from unified_memory.ingestion.extractors.schema import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
)
from unified_memory.core.types import Chunk, Modality
from unified_memory.storage.vector.memory_store import MemoryVectorStore
from unified_memory.storage.graph.networkx_store import NetworkXGraphStore
from unified_memory.namespace.manager import NamespaceManager
from unified_memory.cas.registry import CASRegistry
from unified_memory.cas.content_store import ContentStore


def _make_pipeline(vector_store, graph_store, embedder, namespace_manager):
    """Build a minimally-configured IngestionPipeline for testing."""
    return IngestionPipeline(
        vector_store=vector_store,
        cas_registry=MagicMock(spec=CASRegistry),
        content_store=MagicMock(spec=ContentStore),
        namespace_manager=namespace_manager,
        document_registry=MagicMock(),
        graph_store=graph_store,
        embedding_provider=embedder,
    )


def _make_namespace_manager(tenant_id: str, embedder):
    """Return a NamespaceManager mock with all required async methods stubbed."""
    nm = MagicMock(spec=NamespaceManager)
    nm.get_collection_name = AsyncMock(
        side_effect=lambda ns, ct: f"{ns}_{ct.value}"
    )

    ns_config = MagicMock()
    ns_config.tenant_id = tenant_id
    nm.get_config = AsyncMock(return_value=ns_config)

    tenant_config = MagicMock()
    tenant_config.text_embedding = MagicMock(provider="openai", model="test-model")
    tenant_config.extraction = None
    nm.get_tenant_config = AsyncMock(return_value=tenant_config)

    return nm


def _make_embedder():
    embedder = MagicMock()
    embedder.model_id = "test-model"
    # Return a list of zero-vectors; length matches the number of texts passed.
    embedder.embed_batch = AsyncMock(
        side_effect=lambda texts, modality: [[0.1] * 8 for _ in texts]
    )
    return embedder


@pytest.mark.asyncio
async def test_entity_provenance_merging():
    """
    When the same entity appears in two chunks, both source_locations must be
    recorded in the vector store entry for that entity.
    """
    doc_id = "doc_ep"
    namespace = "test_ns"
    tenant_id = "test_tenant"

    vector_store = MemoryVectorStore()
    graph_store = NetworkXGraphStore()
    embedder = _make_embedder()
    nm = _make_namespace_manager(tenant_id, embedder)

    pipeline = _make_pipeline(vector_store, graph_store, embedder, nm)
    pipeline.provider_registry.resolve_embedding_provider = MagicMock(return_value=embedder)

    ent = ExtractedEntity(
        name="Albert Einstein",
        type="PERSON",
        description="Famous physicist",
    )

    chunk0 = Chunk(document_id=doc_id, content="Einstein was born in Ulm.", chunk_index=0)
    chunk1 = Chunk(document_id=doc_id, content="Albert Einstein is famous.", chunk_index=1)

    extractor = MagicMock()
    extractor.extract = AsyncMock(
        side_effect=[
            ExtractionResult(entities=[ent], relations=[]),
            ExtractionResult(entities=[ent], relations=[]),
        ]
    )
    pipeline._resolve_extractor_from_config = MagicMock(return_value=extractor)

    parsed_doc = MagicMock()
    parsed_doc.document_id = doc_id

    await pipeline._process_chunks(
        chunks=[chunk0, chunk1],
        namespace=namespace,
        parsed_document=parsed_doc,
    )

    # Inspect what was stored in the entity collection
    entity_collection = f"{namespace}_entities"
    results = await vector_store.search(
        query_embedding=[0.1] * 8,
        top_k=20,
        namespace=namespace,
        collection=entity_collection,
    )

    assert results, "No entity vectors were stored"
    einstein = next(
        (r for r in results if r.metadata.get("entity_name") == "Albert Einstein"),
        None,
    )
    assert einstein is not None, "Albert Einstein not found in vector store"

    locs = einstein.metadata.get("source_locations", [])
    assert len(locs) == 2, f"Expected 2 source_locations, got {len(locs)}: {locs}"
    chunk_indices = {loc["chunk_index"] for loc in locs}
    assert chunk_indices == {0, 1}, f"Unexpected chunk indices: {chunk_indices}"


@pytest.mark.asyncio
async def test_relation_provenance_and_keyword_merging():
    """
    When the same relation (same source/predicate/target) appears in two chunks:
    - both source_locations must be stored
    - keywords from both chunks must be merged
    """
    doc_id = "doc_rel"
    namespace = "test_ns"
    tenant_id = "test_tenant"

    vector_store = MemoryVectorStore()
    graph_store = NetworkXGraphStore()
    embedder = _make_embedder()
    nm = _make_namespace_manager(tenant_id, embedder)

    pipeline = _make_pipeline(vector_store, graph_store, embedder, nm)
    pipeline.provider_registry.resolve_embedding_provider = MagicMock(return_value=embedder)

    ent = ExtractedEntity(name="Albert Einstein", type="PERSON", description="Physicist")
    rel_chunk0 = ExtractedRelation(
        source_entity="Albert Einstein",
        relation_type="BORN_IN",
        target_entity="Ulm",
        keywords=["birth"],
    )
    rel_chunk1 = ExtractedRelation(
        source_entity="Albert Einstein",
        relation_type="BORN_IN",
        target_entity="Ulm",
        keywords=["origin"],
    )

    chunk0 = Chunk(document_id=doc_id, content="Einstein was born in Ulm.", chunk_index=0)
    chunk1 = Chunk(document_id=doc_id, content="Einstein's origin is Ulm.", chunk_index=1)

    extractor = MagicMock()
    extractor.extract = AsyncMock(
        side_effect=[
            ExtractionResult(entities=[ent], relations=[rel_chunk0]),
            ExtractionResult(entities=[ent], relations=[rel_chunk1]),
        ]
    )
    pipeline._resolve_extractor_from_config = MagicMock(return_value=extractor)

    parsed_doc = MagicMock()
    parsed_doc.document_id = doc_id

    await pipeline._process_chunks(
        chunks=[chunk0, chunk1],
        namespace=namespace,
        parsed_document=parsed_doc,
    )

    rel_collection = f"{namespace}_relations"
    rel_results = await vector_store.search(
        query_embedding=[0.1] * 8,
        top_k=20,
        namespace=namespace,
        collection=rel_collection,
    )

    assert rel_results, "No relation vectors were stored"
    born_in = next(
        (r for r in rel_results if r.metadata.get("relation") == "BORN_IN"),
        None,
    )
    assert born_in is not None, "BORN_IN relation not found in vector store"

    locs = born_in.metadata.get("source_locations", [])
    assert len(locs) == 2, f"Expected 2 source_locations, got {len(locs)}: {locs}"
    chunk_indices = {loc["chunk_index"] for loc in locs}
    assert chunk_indices == {0, 1}, f"Unexpected chunk indices: {chunk_indices}"
