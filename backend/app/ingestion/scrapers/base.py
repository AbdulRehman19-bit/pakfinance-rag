"""Shared scraping utilities: retrying HTTP downloads, safe filenames, manifest logging."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

USER_AGENT = "PakFinanceRAG-Bot/1.0 (+https://github.com/AbdulRehma19-bit/pakfinance-rag)"
REQUEST_TIMEOUT = 30


@dataclass
class ScrapedDocument:
    source: str          # "PSX" | "SBP" | "FBR"
    title: str
    url: str
    local_path: str
    downloaded_at: float


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def slugify(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    text = re.sub(r"[\s_]+", "-", text)
    return text[:max_len] or "document"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((requests.RequestException,)),
)
def fetch(session: requests.Session, url: str) -> requests.Response:
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response


def download_pdf(session: requests.Session, url: str, dest_dir: Path, title: str) -> Path | None:
    """Download a PDF if not already present locally. Returns the local path, or None on failure."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{slugify(title)}.pdf"

    if dest_path.exists():
        return dest_path  # already downloaded — skip re-fetching

    try:
        response = fetch(session, url)
    except Exception as exc:
        print(f"  ! failed to download {url}: {exc}")
        return None

    content_type = response.headers.get("Content-Type", "").lower()
    if "pdf" not in content_type and not url.lower().endswith(".pdf"):
        print(f"  ! skipped non-PDF content at {url}")
        return None

    dest_path.write_bytes(response.content)
    return dest_path


def append_manifest(manifest_path: Path, doc: ScrapedDocument) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(doc)) + "\n")


def polite_delay(seconds: float = 1.0) -> None:
    """Be a good citizen — don't hammer government servers."""
    time.sleep(seconds)