"""
Reciprocal Rank Fusion (RRF) — merges the BM25 ranked list and the dense
ranked list into one ranked list, using only rank position (not raw scores,
which aren't comparable across BM25 and cosine similarity).

score(doc) = sum over retrievers of  1 / (k + rank_in_that_retriever)

A chunk that shows up in both lists — even at a mediocre rank in each —
outscores a chunk that's #1 in only one list. That's the whole point: it
rewards cross-retriever agreement over single-retriever overconfidence.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FusedResult:
    chunk_id: str
    rrf_score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None


def reciprocal_rank_fusion(
    bm25_ranked_ids: list[str],
    dense_ranked_ids: list[str],
    k: int = 60,
) -> list[FusedResult]:
    """k is RRF's damping constant — higher k flattens the gap between rank 1
    and rank 20; 60 is the standard default from the original RRF paper."""
    scores: dict[str, float] = {}
    bm25_ranks: dict[str, int] = {}
    dense_ranks: dict[str, int] = {}

    for rank, chunk_id in enumerate(bm25_ranked_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
        bm25_ranks[chunk_id] = rank

    for rank, chunk_id in enumerate(dense_ranked_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
        dense_ranks[chunk_id] = rank

    fused = [
        FusedResult(
            chunk_id=chunk_id,
            rrf_score=score,
            bm25_rank=bm25_ranks.get(chunk_id),
            dense_rank=dense_ranks.get(chunk_id),
        )
        for chunk_id, score in scores.items()
    ]
    fused.sort(key=lambda r: r.rrf_score, reverse=True)
    return fused