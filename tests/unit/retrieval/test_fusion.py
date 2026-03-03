import pytest
from unified_memory.core.types import RetrievalResult
from unified_memory.retrieval.fusion import reciprocal_rank_fusion, linear_fusion, normalize_scores

def create_result(id: str, score: float, source: str) -> RetrievalResult:
    return RetrievalResult(
        id=id,
        content=f"Content {id}",
        score=score,
        source=source
    )

def test_normalize_scores():
    results = [
        create_result("a", 100.0, "dense"),
        create_result("b", 50.0, "dense"),
        create_result("c", 0.0, "dense"),
        create_result("d", 1.0, "sparse"),
        create_result("e", 0.5, "sparse"),
    ]
    
    normalized = normalize_scores(results)
    
    # Check dense
    dense_results = [r for r in normalized if r.source == "dense"]
    scores = {r.id: r.score for r in dense_results}
    assert scores["a"] == 1.0
    assert scores["b"] == 0.5
    assert scores["c"] == 0.0
    
    # Check sparse
    sparse_results = [r for r in normalized if r.source == "sparse"]
    scores = {r.id: r.score for r in sparse_results}
    assert scores["d"] == 1.0
    assert scores["e"] == 0.0

def test_rrf_configurable_k():
    results = [
        create_result("doc1", 0.9, "dense"),
        create_result("doc2", 0.8, "dense"),
        create_result("doc1", 0.7, "sparse"),
        create_result("doc3", 0.6, "sparse"),
    ]
    
    # Test with default k=60
    fused_60 = reciprocal_rank_fusion(results, k=60)
    # Test with k=0
    fused_0 = reciprocal_rank_fusion(results, k=0)
    
    # Scores should differ
    assert fused_60[0].score != fused_0[0].score
    
    # With k=0, doc1 = 1/(0+1) + 1/(0+1) = 2.0
    # doc2 = 1/(0+2) = 0.5
    # doc3 = 1/(0+2) = 0.5
    scores_0 = {r.id: r.score for r in fused_0}
    assert scores_0["doc1"] == 2.0
    assert scores_0["doc2"] == 0.5
    assert scores_0["doc3"] == 0.5

def test_linear_fusion_normalization():
    # Large scores vs small scores
    results = [
        create_result("doc1", 1000.0, "dense"),
        create_result("doc2", 500.0, "dense"),
        create_result("doc1", 0.1, "sparse"),
        create_result("doc2", 0.9, "sparse"),
    ]
    
    weights = {"dense": 0.5, "sparse": 0.5}
    
    # Without normalization, dense dominates completely
    fused_no_norm = linear_fusion(results, weights, normalize_first=False)
    assert fused_no_norm[0].id == "doc1"
    
    # With normalization:
    # dense: doc1=1.0, doc2=0.0
    # sparse: doc1=0.0, doc2=1.0
    # doc1 score = 1.0*0.5 + 0.0*0.5 = 0.5
    # doc2 score = 0.0*0.5 + 1.0*0.5 = 0.5
    fused_norm = linear_fusion(results, weights, normalize_first=True)
    scores = {r.id: r.score for r in fused_norm}
    assert scores["doc1"] == scores["doc2"] == 0.5
