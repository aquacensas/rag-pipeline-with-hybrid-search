"""Regression tests for the three chunking strategies — confirms each
produces valid, correctly-tagged output on a simple known input."""

import pytest
from src.ingestion.loaders import RawDocument
from src.ingestion.chunking import chunk_fixed, chunk_structural, chunk_document


SAMPLE_DOC = RawDocument(
    source_path="test.md",
    file_type="md",
    content="# Heading One\n\nSome content here about topic A.\n\n# Heading Two\n\nMore content about topic B.",
)


def test_chunk_fixed_produces_nonempty_chunks():
    chunks = chunk_fixed(SAMPLE_DOC, chunk_size=50, overlap=10)
    assert len(chunks) > 0
    assert all(c.strategy == "fixed" for c in chunks)
    assert all(c.source_path == "test.md" for c in chunks)


def test_chunk_structural_preserves_headings():
    chunks = chunk_structural(SAMPLE_DOC, max_chunk_size=1000)
    assert len(chunks) == 2  # two headings -> two sections
    headings = {c.section_heading for c in chunks}
    assert "Heading One" in headings
    assert "Heading Two" in headings


def test_chunk_document_dispatch_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        chunk_document(SAMPLE_DOC, strategy="not_a_real_strategy")


def test_chunk_fixed_on_empty_content_returns_no_chunks():
    empty_doc = RawDocument(source_path="empty.md", file_type="md", content="")
    # loaders.py already filters empty docs before they reach chunking,
    # but chunking itself should still handle it gracefully if ever called directly
    chunks = chunk_fixed(empty_doc)
    assert chunks == []