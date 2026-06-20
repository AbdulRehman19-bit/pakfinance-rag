"""
SBP scraper.

SBP publishes circulars/notifications as static HTML, organized by
department and year, e.g.:
  https://www.sbp.org.pk/bprd/2024/index.htm   <- year index, lists circulars
  https://www.sbp.org.pk/bprd/2024/C3.htm      <- one circular's page, embeds PDF link(s)

This walks a year index, follows each circular page, and downloads any PDFs
linked from it. Confirmed working against the BPRD department for 2022/2025
circular pages.
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

BASE_URL = "https://www.sbp.org.pk"
# Departments known to publish circulars under /<dept>/<year>/index.htm
DEPARTMENTS = ["bprd", "smefd", "epd", "bpd"]
RAW_DIR = Path("data/raw/sbp")
MANIFEST_PATH = Path("data/processed/sbp_manifest.jsonl")


def _circular_links(index_url: str, session) -> list[str]:
    """Pull every link to a circular sub-page from a department/year index page."""
    response = session.get(index_url, timeout=30)
    if response.status_code != 200:
        return []
    soup = BeautifulSoup(response.text, "html.parser")
    links = [
        urljoin(index_url, a["href"])
        for a in soup.find_all("a", href=True)
        if a["href"].lower().endswith(".htm")
    ]
    return list(dict.fromkeys(links))  # de-dupe, preserve order


def _pdf_links(circular_url: str, session) -> list[tuple[str, str]]:
    """Return (title, pdf_url) pairs found on a single circular page."""
    response = session.get(circular_url, timeout=30)
    if response.status_code != 200:
        return []
    soup = BeautifulSoup(response.text, "html.parser")
    return [
        (a.get_text(strip=True) or a["href"].split("/")[-1], urljoin(circular_url, a["href"]))
        for a in soup.find_all("a", href=True)
        if a["href"].lower().endswith(".pdf")
    ]


def scrape(years: list[int], departments: list[str] | None = None, limit: int | None = None) -> list[ScrapedDocument]:
    """Scrape SBP circulars for the given years across the given departments.
    `limit` caps the number of PDFs downloaded — useful for a quick test run."""
    departments = departments or DEPARTMENTS
    session = make_session()
    downloaded: list[ScrapedDocument] = []

    for dept in departments:
        for year in years:
            index_url = f"{BASE_URL}/{dept}/{year}/index.htm"
            print(f"[SBP] scanning {index_url}")
            circular_pages = _circular_links(index_url, session)
            polite_delay()

            for page_url in circular_pages:
                if limit and len(downloaded) >= limit:
                    return downloaded

                for title, pdf_url in _pdf_links(page_url, session):
                    local_path = download_pdf(session, pdf_url, RAW_DIR, f"{dept}-{year}-{title}")
                    if local_path:
                        doc = ScrapedDocument(
                            source="SBP", title=title, url=pdf_url,
                            local_path=str(local_path), downloaded_at=time.time(),
                        )
                        append_manifest(MANIFEST_PATH, doc)
                        downloaded.append(doc)
                        print(f"  + {title}")
                polite_delay(0.5)

    return downloaded


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape SBP circulars/notifications.")
    parser.add_argument("--years", type=int, nargs="+", default=[2024, 2025])
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    docs = scrape(years=args.years, limit=args.limit)
    print(f"\nDownloaded {len(docs)} SBP documents.")