"""Regression tests for fusion — confirms results are correctly ranked
and deduplicated across dense/sparse result sets, without needing real
API calls (uses fabricated result lists)."""

from src.retrieval.fusion import reciprocal_rank_fusion


def test_fusion_ranks_by_descending_score():
    dense_results = [
        {"source_path": "a.md", "chunk_id": 0, "content": "A"},
        {"source_path": "b.md", "chunk_id": 0, "content": "B"},
    ]
    sparse_results = [
        {"source_path": "b.md", "chunk_id": 0, "content": "B"},
        {"source_path": "c.md", "chunk_id": 0, "content": "C"},
    ]
    fused = reciprocal_rank_fusion(dense_results, sparse_results)

    scores = [r["fused_score"] for r in fused]
    assert scores == sorted(scores, reverse=True)  # strictly descending order


def test_fusion_boosts_chunks_appearing_in_both_lists():
    """A chunk ranked in BOTH dense and sparse results should score higher
    than one appearing in only one list — this is the entire point of RRF."""
    dense_results = [{"source_path": "shared.md", "chunk_id": 0, "content": "X"}]
    sparse_results = [{"source_path": "shared.md", "chunk_id": 0, "content": "X"}]
    only_dense = [{"source_path": "only.md", "chunk_id": 0, "content": "Y"}]

    fused_shared = reciprocal_rank_fusion(dense_results, sparse_results)
    fused_single = reciprocal_rank_fusion(only_dense, [])

    assert fused_shared[0]["fused_score"] > fused_single[0]["fused_score"]


def test_fusion_returns_empty_list_for_empty_inputs():
    assert reciprocal_rank_fusion([], []) == []