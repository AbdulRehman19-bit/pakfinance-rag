"""
PDF parsing layer.

PyMuPDF (fitz) is used for fast text extraction (page-by-page).
pdfplumber is used specifically for table extraction, since it handles
tabular layouts (common in SBP circulars, FBR tax tables, PSX financial
statements) far better than PyMuPDF's raw text extraction.

Tables are rendered as markdown and appended to the page text, so a single
chunk of text carries both prose and tabular data for the LLM to read.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber


@dataclass
class ParsedPage:
    page_number: int  # 1-indexed
    text: str
    tables_markdown: list[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """Page prose followed by any tables rendered as markdown."""
        if not self.tables_markdown:
            return self.text
        tables_block = "\n\n".join(self.tables_markdown)
        return f"{self.text}\n\n{tables_block}".strip()


def _extract_text_pages(pdf_path: Path) -> dict[int, str]:
    """Fast page-level text extraction via PyMuPDF."""
    pages: dict[int, str] = {}
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            pages[i] = page.get_text("text").strip()
    return pages


def _table_to_markdown(table: list[list[str | None]]) -> str:
    """Render a pdfplumber table (list of rows) as a markdown table."""
    if not table or not table[0]:
        return ""
    rows = [[cell if cell is not None else "" for cell in row] for row in table]
    header, *body = rows
    md = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    md += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(md)


def _extract_tables_pages(pdf_path: Path) -> dict[int, list[str]]:
    """Per-page table extraction via pdfplumber, rendered as markdown."""
    tables_by_page: dict[int, list[str]] = {}
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            if tables:
                tables_by_page[i] = [_table_to_markdown(t) for t in tables if t]
    return tables_by_page


def parse_pdf(pdf_path: str | Path) -> list[ParsedPage]:
    """
    Parse a single PDF into a list of ParsedPage objects, one per page,
    each carrying prose text plus any tables found on that page as markdown.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    text_pages = _extract_text_pages(pdf_path)
    table_pages = _extract_tables_pages(pdf_path)

    parsed = [
        ParsedPage(
            page_number=page_num,
            text=text,
            tables_markdown=table_pages.get(page_num, []),
        )
        for page_num, text in text_pages.items()
    ]
    return parsed