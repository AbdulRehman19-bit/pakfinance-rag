"""Unit tests for the BM25 sparse index."""
from app.indexing.sparse_index import build_bm25_index, tokenize


def _make_record(chunk_id, text, **extra):
    return {
        "chunk_id": chunk_id, "text": text, "source": "SBP", "category": "general",
        "document_title": "Test Doc", "page_number": 1, "parent_id": "p1",
        "parent_text": text, **extra,
    }


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("SBP's Policy Rate: 22%!") == ["sbp", "s", "policy", "rate", "22"]


def test_bm25_ranks_exact_keyword_match_highest():
    records = [
        _make_record("c1", "The withholding tax rate under SRO 1213 is 15 percent."),
        _make_record("c2", "Monetary policy committee discussed inflation outlook."),
        _make_record("c3", "SRO 1213 amends the income tax ordinance schedule."),
    ]
    index = build_bm25_index(records)
    results = index.search("SRO 1213", top_k=2)

    assert len(results) > 0
    top_chunk_ids = [record["chunk_id"] for record, score in results]
    assert "c1" in top_chunk_ids or "c3" in top_chunk_ids


def test_search_with_no_matches_returns_empty():
    records = [_make_record("c1", "Completely unrelated content about holidays.")]
    index = build_bm25_index(records)
    assert index.search("cryptocurrency blockchain") == []


def test_save_and_load_roundtrip_handles_small_corpus(tmp_path):
    """Regression test: a single-document corpus can produce a *negative* BM25
    score (IDF goes negative when a term is in nearly every doc). The index
    must still surface that match — only exact-zero scores get filtered."""
    records = [_make_record("c1", "Foreign exchange remittance circular details.")]
    index = build_bm25_index(records)

    save_path = tmp_path / "bm25_test.pkl"
    index.save(save_path)
    loaded = index.__class__.load(save_path)

    assert loaded.records[0]["chunk_id"] == "c1"
    results = loaded.search("remittance")
    assert len(results) == 1
    assert results[0][0]["chunk_id"] == "c1"