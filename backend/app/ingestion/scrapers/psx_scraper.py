"""
PSX scraper.

PSX's data portal (dps.psx.com.pk) is a JavaScript single-page app — its
listing/financial-reports pages don't return useful HTML to a plain GET
request, unlike SBP/FBR's static pages. Two modes are supported:

1. `CATEGORY_PAGE_URLS` — for psx.com.pk's *Regulations -> Rule Book*
   pages, IF they turn out to be server-rendered HTML tables of PDF links
   (verify this yourself in a browser's "view source" before trusting it).
2. `MANUAL_PSX_PDF_URLS` — fallback: paste direct PDF URLs you've copied by
   hand from the PSX site. This is the reliable option until/unless mode 1
   is confirmed.
"""
from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from app.ingestion.scrapers.base import (
    ScrapedDocument,
    append_manifest,
    download_pdf,
    make_session,
    polite_delay,
)

RAW_DIR = Path("data/raw/psx")
MANIFEST_PATH = Path("data/processed/psx_manifest.jsonl")

# Fill in once you've confirmed a server-rendered listing URL on psx.com.pk.
CATEGORY_PAGE_URLS: list[str] = [
    # "https://www.psx.com.pk/psx/regulations/rule-book",
]

# Manual fallback — (title, direct_pdf_url) pairs copied by hand.
MANUAL_PSX_PDF_URLS: list[tuple[str, str]] = [
    # ("PSX Rule Book 2024", "https://www.psx.com.pk/.../rule-book-2024.pdf"),
]


def _pdf_links(page_url: str, session) -> list[tuple[str, str]]:
    response = session.get(page_url, timeout=30)
    if response.status_code != 200:
        return []
    soup = BeautifulSoup(response.text, "html.parser")
    return [
        (a.get_text(strip=True) or a["href"].split("/")[-1], urljoin(page_url, a["href"]))
        for a in soup.find_all("a", href=True)
        if a["href"].lower().endswith(".pdf")
    ]


def scrape(limit: int | None = None) -> list[ScrapedDocument]:
    session = make_session()
    downloaded: list[ScrapedDocument] = []

    pairs: list[tuple[str, str]] = list(MANUAL_PSX_PDF_URLS)
    for page_url in CATEGORY_PAGE_URLS:
        print(f"[PSX] scanning {page_url}")
        pairs.extend(_pdf_links(page_url, session))
        polite_delay()

    for title, pdf_url in pairs:
        if limit and len(downloaded) >= limit:
            break
        local_path = download_pdf(session, pdf_url, RAW_DIR, title)
        if local_path:
            doc = ScrapedDocument(
                source="PSX", title=title, url=pdf_url,
                local_path=str(local_path), downloaded_at=time.time(),
            )
            append_manifest(MANIFEST_PATH, doc)
            downloaded.append(doc)
            print(f"  + {title}")

    return downloaded


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scrape/download PSX regulatory PDFs.")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    docs = scrape(limit=args.limit)
    print(f"\nDownloaded {len(docs)} PSX documents.")