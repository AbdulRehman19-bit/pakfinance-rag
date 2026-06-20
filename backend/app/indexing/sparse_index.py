"""
BM25 sparse index.

One BM25Okapi instance per category, so a category-routed query only scans
that category's chunks instead of the whole corpus.

Note on the score filter: BM25's IDF term can go *negative* when a token
appears in most/all documents of a small corpus — exactly the situation
small categories can hit. We filter out scores that are exactly 0 (meaning
none of the query's tokens exist anywhere in this index's vocabulary at
all) but keep negative scores, since relative rank order is still
meaningful and that's all RRF fusion (next commit) actually uses.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass
class BM25Index:
    bm25: BM25Okapi
    records: list[dict]  # records[i] is the chunk dict backing bm25 doc i

    def search(self, query: str, top_k: int = 20) -> list[tuple[dict, float]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(self.records[i], float(scores[i])) for i in ranked_idx[:top_k] if scores[i] != 0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "BM25Index":
        with path.open("rb") as f:
            return pickle.load(f)


def build_bm25_index(chunk_records: list[dict]) -> BM25Index:
    """chunk_records: list of dicts as returned by chunk_store.fetch_chunks() —
    each must have a "text" field."""
    tokenized_corpus = [tokenize(record["text"]) for record in chunk_records]
    bm25 = BM25Okapi(tokenized_corpus)
    return BM25Index(bm25=bm25, records=chunk_records)