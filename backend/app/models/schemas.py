"""Pydantic request/response models shared across the API layer."""
from typing import Literal
from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """A single chunk surfaced by the hybrid retriever, after rerank."""
    chunk_id: str
    text: str
    source: str  # e.g. "PSX", "SBP", "FBR"
    document_title: str
    page_number: int | None = None
    bm25_rank: int | None = None
    dense_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[RetrievedChunk]
    latency_ms: float


class FeedbackRequest(BaseModel):
    query: str
    answer: str
    rating: Literal["up", "down"]
    comment: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    env: str