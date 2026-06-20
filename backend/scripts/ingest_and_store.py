"""
End-to-end ingestion: parse every PDF under data/raw/<source>/, chunk it,
classify it, and push the result into Supabase.

Run after the scrapers have downloaded PDFs:
    python scripts/ingest_and_store.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `app.*` imports

from app.ingestion.parser import parse_pdf
from app.ingestion.chunker import chunk_pages
from app.indexing.classifier import classify_document, label_chunks
from app.storage.chunk_store import upsert_document, push_chunks

RAW_DIR = Path("data/raw")
SOURCES = ["psx", "sbp", "fbr"]


def ingest_pdf(pdf_path: Path, source: str) -> int:
    title = pdf_path.stem.replace("-", " ").title()
    pages = parse_pdf(pdf_path)
    sample_text = " ".join(p.text for p in pages[:2])  # first 2 pages is enough signal to classify
    category = classify_document(title, sample_text)

    chunks = chunk_pages(pages, source=source.upper(), document_title=title)
    label_chunks(chunks, category)

    document_id = upsert_document(title=title, source=source.upper(), category=category.value, page_count=len(pages))
    written = push_chunks(chunks, document_id=document_id, category=category.value)
    print(f"  [{source.upper()}] {title}: {len(pages)} pages -> {written} chunks ({category.value})")
    return written


def main() -> None:
    total = 0
    for source in SOURCES:
        source_dir = RAW_DIR / source
        if not source_dir.exists():
            continue
        for pdf_path in sorted(source_dir.glob("*.pdf")):
            total += ingest_pdf(pdf_path, source)
    print(f"\nDone. {total} chunks written to Supabase.")


if __name__ == "__main__":
    main()