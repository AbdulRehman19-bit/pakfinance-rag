"""Unit tests for Reciprocal Rank Fusion — pure algorithm, no external services."""
from app.retrieval.fusion import reciprocal_rank_fusion


def test_item_in_both_lists_outranks_single_list_top_item():
    bm25_ids = ["a", "b", "c"]
    dense_ids = ["c", "d", "e"]
    fused = reciprocal_rank_fusion(bm25_ids, dense_ids, k=60)

    assert fused[0].chunk_id == "c"  # cross-retriever agreement wins
    assert fused[0].bm25_rank == 3
    assert fused[0].dense_rank == 1


def test_items_from_either_list_are_preserved():
    fused = reciprocal_rank_fusion(["a"], ["b"], k=60)
    assert {r.chunk_id for r in fused} == {"a", "b"}


def test_score_matches_rrf_formula_exactly():
    fused = reciprocal_rank_fusion(["x"], [], k=60)
    assert abs(fused[0].rrf_score - (1.0 / 61)) < 1e-9


def test_empty_inputs_return_empty_list():
    assert reciprocal_rank_fusion([], []) == []