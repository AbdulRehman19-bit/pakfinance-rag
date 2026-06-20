"""Prompt templates for grounded generation over retrieved chunks."""
from __future__ import annotations

SYSTEM_PROMPT = """You are a financial/regulatory research assistant for Pakistan, answering \
questions using ONLY the provided context from PSX, SBP, and FBR documents.

Rules:
- Answer using only the information in the context below. Do not use outside knowledge.
- If the context doesn't contain enough information to answer, say so plainly — don't guess.
- Cite every claim with its source in the format [Source: <document_title>, p.<page_number>].
- Be concise and precise. Use exact figures, dates, and section/circular numbers as written.
- If sources disagree or are ambiguous, point that out rather than picking one silently.
"""


def build_context_block(chunks: list[dict]) -> str:
    """Render retrieved chunks into a numbered context block the LLM can cite from."""
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        title = chunk.get("document_title", "Unknown document")
        page = chunk.get("page_number")
        page_str = f", p.{page}" if page else ""
        source = chunk.get("source", "")
        text = chunk.get("parent_text") or chunk.get("text", "")
        blocks.append(f"[{i}] {source} — {title}{page_str}\n{text.strip()}")
    return "\n\n".join(blocks)


def build_user_prompt(query: str, chunks: list[dict]) -> str:
    context = build_context_block(chunks)
    return f"""Context:
{context}

Question: {query}

Answer the question using only the context above, with citations in the format [Source: <document_title>, p.<page_number>]."""