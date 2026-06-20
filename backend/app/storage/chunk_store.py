"""Read/write chunks to Supabase — the persistent source of truth that the
BM25 and dense indexes are (re)built from."""
from __future__ import annotations

import hashlib

from app.ingestion.chunker import Chunk
from app.storage.supabase_client import get_supabase_client

DOCUMENTS_TABLE = "documents"
CHUNKS_TABLE = "chunks"


def _document_id(document_title: str) -> str:
    return hashlib.sha1(document_title.encode("utf-8")).hexdigest()[:16]


def upsert_document(
    title: str,
    source: str,
    category: str,
    source_url: str | None = None,
    page_count: int | None = None,
) -> str:
    """Upsert the parent document row; returns its document_id."""
    client = get_supabase_client()
    document_id = _document_id(title)
    client.table(DOCUMENTS_TABLE).upsert(
        {
            "document_id": document_id,
            "source": source,
            "title": title,
            "category": category,
            "source_url": source_url,
            "page_count": page_count,
        }
    ).execute()
    return document_id


def push_chunks(chunks: list[Chunk], document_id: str, category: str, batch_size: int = 200) -> int:
    """Upsert chunks into Supabase in batches. Returns the number of rows written."""
    client = get_supabase_client()
    rows = [
        {
            "chunk_id": c.chunk_id,
            "parent_id": c.parent_id,
            "document_id": document_id,
            "source": c.source,
            "category": c.metadata.get("category", category),
            "document_title": c.document_title,
            "page_number": c.page_number,
            "text": c.text,
            "parent_text": c.parent_text,
            "metadata": c.metadata,
        }
        for c in chunks
    ]

    written = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        client.table(CHUNKS_TABLE).upsert(batch).execute()
        written += len(batch)
    return written


def fetch_chunks(category: str | None = None, source: str | None = None) -> list[dict]:
    """Pull chunks back out of Supabase — this is what build_indexes.py reads from."""
    client = get_supabase_client()
    query = client.table(CHUNKS_TABLE).select("*")
    if category:
        query = query.eq("category", category)
    if source:
        query = query.eq("source", source)

    all_rows: list[dict] = []
    page_size = 1000
    offset = 0
    while True:
        page = query.range(offset, offset + page_size - 1).execute()
        rows = page.data
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        offset += page_size
    return all_rows