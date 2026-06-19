"""Unit tests for the parent-child chunker — synthetic text, no PDF needed."""
from app.ingestion.parser import ParsedPage
from app.ingestion.chunker import chunk_pages


def _long_text(n_sentences: int = 60) -> str:
    return ". ".join(f"This is sentence number {i}" for i in range(n_sentences)) + "."


def test_chunk_pages_produces_multiple_children():
    page = ParsedPage(page_number=1, text=_long_text())
    chunks = chunk_pages([page], source="SBP", document_title="Test Circular")

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.source == "SBP"
        assert chunk.document_title == "Test Circular"
        assert chunk.page_number == 1


def test_children_reference_a_shared_parent():
    page = ParsedPage(page_number=1, text=_long_text())
    chunks = chunk_pages([page], source="FBR", document_title="Test Doc")

    assert len({c.parent_id for c in chunks}) >= 1
    for c in chunks:
        assert c.text in c.parent_text  # child text must come from its own parent


def test_empty_page_is_skipped():
    page = ParsedPage(page_number=1, text="   ")
    assert chunk_pages([page], source="PSX", document_title="Empty") == []