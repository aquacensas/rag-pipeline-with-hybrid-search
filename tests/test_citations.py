"""Regression tests for citation verification — includes the exact
'no citations made' bug we found and fixed during Phase 4, so it can
never silently regress."""

from src.generation.citations import verify_citations, extract_citation_number, split_into_claims


def test_verify_citations_raises_on_empty_answer():
    import pytest
    with pytest.raises(ValueError):
        verify_citations("", [{"source_path": "x.md", "content": "irrelevant"}])


def test_no_citations_made_returns_none_coverage_not_zero():
    """Regression test for the real bug found in Phase 4: an honest 'I
    don't know' answer with zero citations must report coverage=None,
    not coverage=0.0 — conflating 'no claims made' with 'claims made but
    all wrong' silently corrupted the faithfulness metric."""
    answer = "The context does not contain enough information to answer this question."
    result = verify_citations(answer, retrieved_chunks=[{"source_path": "x.md", "content": "unrelated"}])
    assert result.total_citations == 0
    assert result.coverage is None  # NOT 0.0 — this is the exact regression to guard against


def test_extract_citation_number_finds_all_numbers():
    assert extract_citation_number("This is supported [1][3].") == [1, 3]
    assert extract_citation_number("No citations here.") == []


def test_split_into_claims_separates_sentences():
    claims = split_into_claims("First sentence [1]. Second sentence [2].")
    assert len(claims) == 2