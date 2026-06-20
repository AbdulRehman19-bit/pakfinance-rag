"""
Build (or rebuild) the BM25 + Chroma indexes from Supabase.

Indexes are split by category — one BM25 pickle + one Chroma collection per
category, plus an "all" pair as a fallback — so a category-routed query
searches a small slice of the corpus instead of everything.

Run:
    python scripts/build_indexes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.storage.chunk_store import fetch_chunks
from app.indexing.sparse_index import build_bm25_index
from app.indexing.dense_index import get_chroma_client, build_chroma_collection
from app.indexing.classifier import Category

settings = get_settings()
BM25_DIR = Path(settings.bm25_index_path).parent  # backend/indexes/
CHROMA_DIR = settings.chroma_persist_dir


def build_for_category(client, category: str, chunk_records: list[dict]) -> None:
    if not chunk_records:
        print(f"  (skipping {category} — no chunks)")
        return

    bm25_index = build_bm25_index(chunk_records)
    bm25_index.save(BM25_DIR / f"bm25_{category}.pkl")
    print(f"  [BM25] {category}: {len(chunk_records)} chunks indexed")

    build_chroma_collection(client, f"chunks_{category}", chunk_records)


def main() -> None:
    client = get_chroma_client(CHROMA_DIR)

    print("Fetching all chunks from Supabase...")
    all_chunks = fetch_chunks()
    print(f"  {len(all_chunks)} chunks total\n")

    # "all" index — fallback for low-confidence query classification
    build_for_category(client, "all", all_chunks)

    # one index per category
    for category in Category:
        category_chunks = [c for c in all_chunks if c["category"] == category.value]
        build_for_category(client, category.value, category_chunks)

    print("\nDone. Indexes written to:")
    print(f"  BM25:   {BM25_DIR}/bm25_<category>.pkl")
    print(f"  Chroma: {CHROMA_DIR}/  (collections: chunks_<category>)")


if __name__ == "__main__":
    main()