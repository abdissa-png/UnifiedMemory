import pytest

from unified_memory.retrieval.unified import UnifiedSearchService
from unified_memory.retrieval.fusion import (
    canonical_key,
    reciprocal_rank_fusion,
    linear_fusion,
)
from unified_memory.core.types import RetrievalResult, Modality


def test_validate_filters_drops_unsupported_and_namespace_keys():
    raw = {
        "document_id": "doc-1",
        "status": "valid",
        "namespace": "ns1",
        "namespaces": ["ns1", "ns2"],
        "unexpected": "foo",
    }

    cleaned = UnifiedSearchService._validate_filters(
        raw,
        allowed=["document_id", "status"],
        store_name="dense",
    )

    # namespace fields removed
    assert "namespace" not in cleaned
    assert "namespaces" not in cleaned
    # unsupported key removed
    assert "unexpected" not in cleaned
    # allowed keys preserved
    assert cleaned == {"document_id": "doc-1", "status": "valid"}


def test_validate_filters_allows_all_when_allowed_empty():
    raw = {
        "namespace": "ns1",
        "foo": 1,
        "bar": 2,
    }

    cleaned = UnifiedSearchService._validate_filters(
        raw,
        allowed=[],
        store_name="sparse",
    )

    # namespace stripped, other keys passed through
    assert cleaned == {"foo": 1, "bar": 2}


def test_canonical_key_prefers_content_hash():
    r_with_hash = RetrievalResult(
        id="vec-1",
        content="x",
        score=1.0,
        metadata={"content_hash": "hash-123"},
        source="dense",
        modality=Modality.TEXT,
    )
    r_without_hash = RetrievalResult(
        id="vec-2",
        content="y",
        score=0.5,
        metadata={},
        source="dense",
        modality=Modality.TEXT,
    )

    assert canonical_key(r_with_hash) == "hash-123"
    assert canonical_key(r_without_hash) == "vec-2"


def test_rrf_merges_results_with_same_content_hash():
    # Two hits from different sources but same underlying chunk hash
    r_dense = RetrievalResult(
        id="dense-1",
        content="A",
        score=1.0,
        metadata={"content_hash": "h1"},
        source="dense",
        modality=Modality.TEXT,
    )
    r_sparse = RetrievalResult(
        id="sparse-1",
        content="A sparse",
        score=0.8,
        metadata={"content_hash": "h1"},
        source="sparse",
        modality=Modality.TEXT,
    )

    fused = reciprocal_rank_fusion([r_dense, r_sparse], k=10)

    # Both should collapse into a single fused result keyed by "h1"
    assert len(fused) == 1
    fused_item = fused[0]
    assert fused_item.metadata.get("content_hash") == "h1"
    assert fused_item.source == "hybrid:rrf"
    assert fused_item.score > 0.0


def test_linear_fusion_merges_results_with_same_content_hash():
    r_dense = RetrievalResult(
        id="dense-1",
        content="A",
        score=0.9,
        metadata={"content_hash": "h1"},
        source="dense",
        modality=Modality.TEXT,
    )
    r_sparse = RetrievalResult(
        id="sparse-1",
        content="A sparse",
        score=0.7,
        metadata={"content_hash": "h1"},
        source="sparse",
        modality=Modality.TEXT,
    )

    fused = linear_fusion(
        [r_dense, r_sparse],
        weights={"dense": 1.0, "sparse": 1.0},
        normalize_first=False,
    )

    assert len(fused) == 1
    fused_item = fused[0]
    assert fused_item.metadata.get("content_hash") == "h1"
    assert fused_item.source == "hybrid:linear"
    # Score should be the sum of weighted scores
    assert pytest.approx(fused_item.score, rel=1e-6) == 0.9 + 0.7


def test_linear_fusion_keeps_separate_items_for_different_hashes():
    r1 = RetrievalResult(
        id="a",
        content="A",
        score=0.5,
        metadata={"content_hash": "h1"},
        source="dense",
        modality=Modality.TEXT,
    )
    r2 = RetrievalResult(
        id="b",
        content="B",
        score=0.4,
        metadata={"content_hash": "h2"},
        source="dense",
        modality=Modality.TEXT,
    )

    fused = linear_fusion(
        [r1, r2],
        weights={"dense": 1.0},
        normalize_first=False,
    )

    assert len(fused) == 2
    keys = {item.metadata.get("content_hash") for item in fused}
    assert keys == {"h1", "h2"}

