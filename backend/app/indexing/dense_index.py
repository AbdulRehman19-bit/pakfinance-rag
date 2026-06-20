"""
Dense index — ChromaDB collections backed by a local embedding model.

Uses Chroma's built-in default embedding function (all-MiniLM-L6-v2 via
ONNX runtime) — no API key, no torch/sentence-transformers dependency, model
downloads automatically (~80MB) on first use and is cached afterward. This
keeps the whole pipeline free, matching the Netlify + Hugging Face Spaces
free-tier deploy.

One collection per category (matching the BM25 split), plus an "all"
collection as a fallback for low-confidence query classification.
"""
from __future__ import annotations

import chromadb
from chromadb.utils import embedding_functions

EMBED_BATCH_SIZE = 100


def get_chroma_client(persist_dir: str):
    return chromadb.PersistentClient(path=persist_dir)


def get_embedding_function():
    """Local, free, no API key — all-MiniLM-L6-v2 via Chroma's bundled ONNX runtime."""
    return embedding_functions.DefaultEmbeddingFunction()


def build_chroma_collection(client, collection_name: str, chunk_records: list[dict]) -> None:
    """(Re)build one Chroma collection from scratch with the given chunk records."""
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass  # collection didn't exist yet — fine

    collection = client.create_collection(
        name=collection_name,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )

    for i in range(0, len(chunk_records), EMBED_BATCH_SIZE):
        batch = chunk_records[i : i + EMBED_BATCH_SIZE]
        collection.add(
            ids=[record["chunk_id"] for record in batch],
            documents=[record["text"] for record in batch],
            metadatas=[
                {
                    "source": record["source"],
                    "category": record["category"],
                    "document_title": record["document_title"],
                    "page_number": record.get("page_number") or 0,
                    "parent_id": record["parent_id"],
                }
                for record in batch
            ],
        )
    print(f"  [Chroma] {collection_name}: {len(chunk_records)} chunks indexed")