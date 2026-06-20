"""
Document/query classifier — coarse topical labeling so retrieval can filter
down to a relevant slice of the corpus before running BM25/dense search,
instead of scanning every chunk on every query.

Rule-based by design: free, instant, deterministic — no extra LLM call or
model load needed at query time. Keyword lists are small and editable.
"""
from __future__ import annotations

from collections import Counter
from enum import Enum


class Category(str, Enum):
    TAXATION = "taxation"
    MONETARY_POLICY = "monetary_policy"
    BANKING_REGULATION = "banking_regulation"
    FOREIGN_EXCHANGE = "foreign_exchange"
    CAPITAL_MARKETS = "capital_markets"
    CORPORATE_DISCLOSURE = "corporate_disclosure"
    GENERAL = "general"


CATEGORY_KEYWORDS: dict[Category, list[str]] = {
    Category.TAXATION: [
        "income tax", "sales tax", "withholding tax", "sro", "tax ordinance",
        "federal excise", "customs duty", "tax credit", "taxpayer", "fbr",
    ],
    Category.MONETARY_POLICY: [
        "policy rate", "monetary policy", "discount rate",
        "open market operation", "inflation target", "mpc",
    ],
    Category.BANKING_REGULATION: [
        "prudential regulation", "banking companies ordinance", "aml", "cft",
        "kyc", "capital adequacy", "basel", "bprd",
    ],
    Category.FOREIGN_EXCHANGE: [
        "foreign exchange", "fe circular", "exchange company", "remittance",
        "forex", "epd",
    ],
    Category.CAPITAL_MARKETS: [
        "listing regulations", "rule book", "trading regulations", "kse",
        "listed company", "brokerage", "psx", "trec",
    ],
    Category.CORPORATE_DISCLOSURE: [
        "annual report", "financial statements", "notice to shareholders",
        "board of directors", "disclosure", "material information",
    ],
}


def classify_text(text: str, min_score: int = 1) -> Category:
    """Score keyword hits per category on a lowercased haystack; return the best match."""
    haystack = text.lower()
    scores: Counter = Counter()

    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            scores[category] += haystack.count(keyword)

    if not scores or scores.most_common(1)[0][1] < min_score:
        return Category.GENERAL
    return scores.most_common(1)[0][0]


def classify_document(title: str, sample_text: str = "") -> Category:
    """Classify once per document (not per chunk) — title carries the strongest
    signal, so it's weighted 3x against a sample of body text."""
    combined = (title + " ") * 3 + sample_text[:2000]
    return classify_text(combined)


def classify_query(query: str) -> Category:
    """Same scorer, used at retrieval time to predict which category a question belongs to."""
    return classify_text(query, min_score=1)


def label_chunks(chunks: list, category: Category) -> list:
    """Stamp a category onto every chunk's metadata dict in place. Returns the same list for chaining."""
    for chunk in chunks:
        chunk.metadata["category"] = category.value
    return chunks