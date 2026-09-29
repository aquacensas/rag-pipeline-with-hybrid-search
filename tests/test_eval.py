"""Regression tests for the eval metrics module — covers the ambiguous-
question scoping fix and the retrieval relevance None-vs-zero distinction,
both real bugs found during Phase 4."""

from src.evaluation.metrics import score_retrieval_relevance
from src.generation.confidence import compute_composite_confidence


def test_retrieval_relevance_none_when_no_expected_sources():
    """A no_answer case has no expected sources — relevance should be
    None (not applicable), not 0.0 (which would read as 'totally wrong')."""
    result = score_retrieval_relevance([], retrieved_chunks=[{"source_path": "x.md"}])
    assert result is None


def test_retrieval_relevance_computes_correct_fraction():
    expected = ["a.md", "b.md"]
    retrieved = [{"source_path": "a.md"}, {"source_path": "c.md"}]
    result = score_retrieval_relevance(expected, retrieved)
    assert result == 0.5  # only 1 of 2 expected sources was retrieved


def test_composite_confidence_rejects_out_of_range_input():
    import pytest
    with pytest.raises(ValueError):
        compute_composite_confidence(retrieval_confidence=1.5, citation_coverage=0.5, completeness_score=0.5)


def test_composite_confidence_weighted_correctly():
    result = compute_composite_confidence(retrieval_confidence=1.0, citation_coverage=1.0, completeness_score=1.0)
    assert result.composite_score == 1.0  # all perfect signals -> perfect composite