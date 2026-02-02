"""
Result Fusion Algorithms.

Implements RRF and Linear fusion methods.
"""

from typing import Dict, List, Tuple
from unified_memory.core.types import RetrievalResult

def normalize_scores(results: List[RetrievalResult]) -> List[RetrievalResult]:
    """
    Normalize scores to [0, 1] range per source.
    """
    if not results:
        return []
    
    # Group by source
    by_source: Dict[str, List[RetrievalResult]] = {}
    for r in results:
        # Handle source subtypes like "dense:texts" -> groups by "dense" if simplest, 
        # or treat each unique source string as its own group.
        # Here we treat unique source string as group for safety.
        source = r.source
        if source not in by_source:
            by_source[source] = []
        by_source[source].append(r)
    
    normalized_results: List[RetrievalResult] = []
    
    for _, rs in by_source.items():
        if not rs:
            continue
            
        scores = [r.score for r in rs]
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score
        
        for r in rs:
            if score_range > 0:
                new_score = (r.score - min_score) / score_range
            else:
                new_score = 1.0 # All scores equal
                
            # Create a copy with new score
            # We can't modify frozen dataclass if frozen=True, assuming it's standard dataclass
            # But let's construct a new one to be safe/clean
            new_r = RetrievalResult(
                id=r.id,
                content=r.content,
                score=new_score,
                metadata=r.metadata,
                source=r.source,
                evidence_type=r.evidence_type,
                entity_ids=r.entity_ids,
                relation_ids=r.relation_ids,
                page_number=r.page_number,
                modality=r.modality,
            )
            normalized_results.append(new_r)
            
    return normalized_results

def reciprocal_rank_fusion(
    results: List[RetrievalResult],
    k: int = 60,
) -> List[RetrievalResult]:
    """
    Reciprocal Rank Fusion (RRF).
    
    Score = sum(1 / (k + rank_i)) for each source.
    """
    # Group by source to determine rank within source
    by_source: Dict[str, List[RetrievalResult]] = {}
    for r in results:
        if r.source not in by_source:
            by_source[r.source] = []
        by_source[r.source].append(r)
        
    # Sort each source group by score descending (just in case input mixed)
    for s in by_source:
        by_source[s].sort(key=lambda x: x.score, reverse=True)
        
    # Calculate RRF scores
    rrf_scores: Dict[str, float] = {}
    id_to_result: Dict[str, RetrievalResult] = {}
    
    for source, source_results in by_source.items():
        for rank, r in enumerate(source_results):
            if r.id not in rrf_scores:
                rrf_scores[r.id] = 0.0
                id_to_result[r.id] = r
            
            rrf_scores[r.id] += 1.0 / (k + rank + 1)
            
    # Create fused results
    fused: List[RetrievalResult] = []
    for rid, score in rrf_scores.items():
        orig = id_to_result[rid]
        fused.append(RetrievalResult(
            id=orig.id,
            content=orig.content,
            score=score,
            metadata=orig.metadata,
            source="hybrid:rrf", # Consolidated source
            evidence_type=orig.evidence_type,
            # Merge other fields if needed, for now pick one
        ))
        
    fused.sort(key=lambda x: x.score, reverse=True)
    return fused

def linear_fusion(
    results: List[RetrievalResult],
    weights: Dict[str, float],
) -> List[RetrievalResult]:
    """
    Weighted linear fusion. Assumes scores are normalized.
    """
    fused_scores: Dict[str, float] = {}
    id_to_result: Dict[str, RetrievalResult] = {}
    
    for r in results:
        source_weight = weights.get(r.source, 1.0)
        weighted_score = r.score * source_weight
        
        if r.id not in fused_scores:
            fused_scores[r.id] = 0.0
            id_to_result[r.id] = r
        
        # Max pool or Sum pool? Typically sum for linear combination
        fused_scores[r.id] += weighted_score

    fused: List[RetrievalResult] = []
    for rid, score in fused_scores.items():
        orig = id_to_result[rid]
        fused.append(RetrievalResult(
            id=orig.id,
            content=orig.content,
            score=score,
            metadata=orig.metadata,
            source="hybrid:linear",
        ))
        
    fused.sort(key=lambda x: x.score, reverse=True)
    return fused
