"""
Dense index — ChromaDB collections backed by OpenAI embeddings.

One collection per category (matching the BM25 split), plus an "all"
collection as a fallback for low-confidence query classification.
"""
from __future__ import annotations

import chromadb
from chromadb.utils import embedding_functions

from app.config import get_settings

EMBED_BATCH_SIZE = 100


def get_chroma_client(persist_dir: str):
    return chromadb.PersistentClient(path=persist_dir)


def get_embedding_function():
    settings = get_settings()
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=settings.openai_api_key,
        model_name=settings.embedding_model,
    )


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