"""
Hybrid retrieval orchestrator — the three-stage pipeline:
  1. classify query -> category -> load that category's BM25 + Chroma index
  2. BM25 search + dense search, run independently
  3. RRF fuse -> Cohere rerank -> return final chunks

If the predicted category has no index yet (e.g. you haven't scraped any
documents for it), this falls back to the "all" index automatically.
"""
from __future__ import annotations

from pathlib import Path

from app.config import get_settings
from app.indexing.classifier import classify_query, Category
from app.indexing.sparse_index import BM25Index
from app.indexing.dense_index import get_chroma_client, get_embedding_function
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import rerank
from app.tracing import traceable

settings = get_settings()
BM25_DIR = Path(settings.bm25_index_path).parent
CHROMA_DIR = settings.chroma_persist_dir

_bm25_cache: dict[str, BM25Index] = {}  # avoid re-reading the pickle every query


def _load_bm25(category: str) -> BM25Index | None:
    if category in _bm25_cache:
        return _bm25_cache[category]
    path = BM25_DIR / f"bm25_{category}.pkl"
    if not path.exists():
        return None
    index = BM25Index.load(path)
    _bm25_cache[category] = index
    return index


def _dense_search(category: str, query: str, top_k: int) -> list[str]:
    client = get_chroma_client(CHROMA_DIR)
    try:
        collection = client.get_collection(f"chunks_{category}", embedding_function=get_embedding_function())
    except Exception:
        return []
    results = collection.query(query_texts=[query], n_results=top_k)
    return results["ids"][0] if results["ids"] else []


@traceable(name="hybrid_retrieve")
def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Full hybrid pipeline for one query. Returns final reranked chunk
    records, each carrying bm25_rank/dense_rank/rrf_score/rerank_score."""
    predicted_category = classify_query(query)

    search_category = predicted_category.value
    bm25_index = _load_bm25(search_category)
    if bm25_index is None:
        search_category = "all"  # predicted category has no index yet
        bm25_index = _load_bm25("all")
    if bm25_index is None:
        return []  # indexes not built yet — run scripts/build_indexes.py first

    bm25_results = bm25_index.search(query, top_k=settings.bm25_top_k)
    bm25_ranked_ids = [record["chunk_id"] for record, _score in bm25_results]
    records_by_id = {record["chunk_id"]: record for record, _score in bm25_results}

    dense_ranked_ids = _dense_search(search_category, query, settings.dense_top_k)

    fused = reciprocal_rank_fusion(bm25_ranked_ids, dense_ranked_ids, k=settings.rrf_k)

    # Backfill anything the dense retriever found that BM25 didn't — pull its
    # text/metadata straight from Chroma (cheap, ids-only round trip).
    missing_ids = [r.chunk_id for r in fused if r.chunk_id not in records_by_id]
    if missing_ids:
        client = get_chroma_client(CHROMA_DIR)
        collection = client.get_collection(f"chunks_{search_category}", embedding_function=get_embedding_function())
        fetched = collection.get(ids=missing_ids)
        for cid, text, meta in zip(fetched["ids"], fetched["documents"], fetched["metadatas"]):
            records_by_id[cid] = {"chunk_id": cid, "text": text, "parent_text": text, **meta}

    candidate_cap = settings.bm25_top_k + settings.dense_top_k
    candidates = []
    for r in fused[:candidate_cap]:
        record = dict(records_by_id[r.chunk_id])
        record["bm25_rank"] = r.bm25_rank
        record["dense_rank"] = r.dense_rank
        record["rrf_score"] = r.rrf_score
        candidates.append(record)

    reranked = rerank(query, candidates, top_n=top_k)

    final = []
    for record, score in reranked:
        record["rerank_score"] = score
        final.append(record)
    return final