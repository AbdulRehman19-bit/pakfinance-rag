"""
Cross-encoder reranking via Cohere's Rerank API — the precision pass before
chunks reach the LLM. BM25 and dense search never actually read the query
and a chunk together; a cross-encoder does, which is why it catches cases
both retrievers miss (e.g. a chunk that shares keywords/topic but answers a
different question than the one asked).
"""
from __future__ import annotations

import cohere

from app.config import get_settings


def rerank(query: str, candidates: list[dict], top_n: int | None = None) -> list[tuple[dict, float]]:
    """
    candidates: chunk record dicts, each with a "text" field.
    Returns (record, rerank_score) pairs, best-first, length top_n.
    Falls back to RRF order (no rerank) if no Cohere key is configured, so
    the pipeline still works end-to-end before you've signed up for a key.
    """
    settings = get_settings()
    if not candidates:
        return []

    top_n = top_n or settings.rerank_top_n

    if not settings.cohere_api_key:
        print("  ! COHERE_API_KEY not set — skipping rerank, using RRF order")
        return [(c, 0.0) for c in candidates[:top_n]]

    client = cohere.Client(settings.cohere_api_key)
    documents = [c["text"] for c in candidates]

    try:
        response = client.rerank(
            model=settings.rerank_model,
            query=query,
            documents=documents,
            top_n=min(top_n, len(documents)),
        )
    except Exception as exc:
        print(f"  ! Cohere rerank failed ({exc}) — falling back to RRF order")
        return [(c, 0.0) for c in candidates[:top_n]]

    return [(candidates[result.index], result.relevance_score) for result in response.results]