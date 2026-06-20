"""
FBR scraper.

FBR groups primary-law documents into category listing pages that are plain
server-rendered HTML tables of titled PDF links, e.g.:
  https://www.fbr.gov.pk/categ/income-tax-ordinance/326   <- confirmed working

FBR's SRO *search* page (https://www.fbr.gov.pk/ShowSROs) renders its results
table via JavaScript and returns an empty table to a static GET — this
scraper intentionally targets the category pages instead.

Only `income-tax-ordinance: 326` is confirmed. The other slugs/ids below are
my best guess at FBR's URL pattern — verify them by browsing fbr.gov.pk's
left-hand navigation (Sales Tax / Federal Excise / Customs -> Acts) and
correcting the id if a category 404s.
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

BASE_URL = "https://www.fbr.gov.pk"
CATEGORY_PAGES = {
    "income-tax-ordinance": 326,   # confirmed
    "sales-tax-act": 327,          # unverified — check and fix if needed
    "federal-excise-act": 328,     # unverified — check and fix if needed
    "customs-act": 329,            # unverified — check and fix if needed
}
RAW_DIR = Path("data/raw/fbr")
MANIFEST_PATH = Path("data/processed/fbr_manifest.jsonl")


def _pdf_links(category_url: str, session) -> list[tuple[str, str]]:
    response = session.get(category_url, timeout=30)
    if response.status_code != 200:
        return []
    soup = BeautifulSoup(response.text, "html.parser")
    return [
        (a.get_text(strip=True) or a["href"].split("/")[-1], urljoin(category_url, a["href"]))
        for a in soup.find_all("a", href=True)
        if a["href"].lower().endswith(".pdf")
    ]


def scrape(categories: dict[str, int] | None = None, limit: int | None = None) -> list[ScrapedDocument]:
    categories = categories or CATEGORY_PAGES
    session = make_session()
    downloaded: list[ScrapedDocument] = []

    for slug, cat_id in categories.items():
        category_url = f"{BASE_URL}/categ/{slug}/{cat_id}"
        print(f"[FBR] scanning {category_url}")

        for title, pdf_url in _pdf_links(category_url, session):
            if limit and len(downloaded) >= limit:
                return downloaded

            local_path = download_pdf(session, pdf_url, RAW_DIR, f"{slug}-{title}")
            if local_path:
                doc = ScrapedDocument(
                    source="FBR", title=title, url=pdf_url,
                    local_path=str(local_path), downloaded_at=time.time(),
                )
                append_manifest(MANIFEST_PATH, doc)
                downloaded.append(doc)
                print(f"  + {title}")
        polite_delay()

    return downloaded


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape FBR ordinance/act category pages.")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    docs = scrape(limit=args.limit)
    print(f"\nDownloaded {len(docs)} FBR documents.")