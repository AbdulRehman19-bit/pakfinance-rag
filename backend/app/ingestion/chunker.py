"""
Parent-child chunking.

Two passes of LangChain's RecursiveCharacterTextSplitter:
  1. Parent splitter — large chunks (~2000 chars) that preserve broad context.
  2. Child splitter — small chunks (~400 chars) carved out of each parent.

Child chunks are what get embedded and indexed (small = precise retrieval).
Each child stores a reference back to its parent_id, so at generation time we
can expand the retrieved child back to its full parent context — better recall
on the small chunk, better context for the LLM on the parent.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.parser import ParsedPage

PARENT_CHUNK_SIZE = 2000
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE = 400
CHILD_CHUNK_OVERLAP = 50


@dataclass
class Chunk:
    chunk_id: str
    parent_id: str
    text: str          # the small child text — what gets embedded/indexed
    parent_text: str   # full parent text — what the LLM actually reads
    source: str         # "PSX" | "SBP" | "FBR"
    document_title: str
    page_number: int | None = None
    metadata: dict = field(default_factory=dict)


def _make_id(*parts: str) -> str:
    raw = "|".join(parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def chunk_pages(
    pages: list[ParsedPage],
    source: str,
    document_title: str,
) -> list[Chunk]:
    """
    Turn a parsed document's pages into parent-child chunks.

    Parents are built per-page first (so page_number stays accurate), then
    split further into children for indexing.
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []

    for page in pages:
        if not page.full_text.strip():
            continue

        parent_texts = parent_splitter.split_text(page.full_text)

        for parent_idx, parent_text in enumerate(parent_texts):
            parent_id = _make_id(document_title, str(page.page_number), str(parent_idx))
            child_texts = child_splitter.split_text(parent_text)

            for child_idx, child_text in enumerate(child_texts):
                chunk_id = _make_id(parent_id, str(child_idx))
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        parent_id=parent_id,
                        text=child_text,
                        parent_text=parent_text,
                        source=source,
                        document_title=document_title,
                        page_number=page.page_number,
                        metadata={
                            "parent_index": parent_idx,
                            "child_index": child_idx,
                        },
                    )
                )

    return chunks