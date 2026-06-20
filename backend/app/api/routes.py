"""API routes — /query now runs the real hybrid retrieval + generation pipeline."""
import time
from fastapi import APIRouter

from app.config import get_settings
from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
    FeedbackRequest,
    HealthResponse,
)
from app.retrieval.retriever import retrieve
from app.generation.llm import generate_answer

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(status="ok", env=settings.app_env)


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    start = time.perf_counter()

    chunks = retrieve(request.query, top_k=request.top_k)
    answer = generate_answer(request.query, chunks)

    sources = [
        RetrievedChunk(
            chunk_id=c["chunk_id"],
            text=c["text"],
            source=c["source"],
            document_title=c["document_title"],
            page_number=c.get("page_number"),
            bm25_rank=c.get("bm25_rank"),
            dense_rank=c.get("dense_rank"),
            rrf_score=c.get("rrf_score"),
            rerank_score=c.get("rerank_score"),
        )
        for c in chunks
    ]

    elapsed_ms = (time.perf_counter() - start) * 1000
    return QueryResponse(
        query=request.query,
        answer=answer,
        sources=sources,
        latency_ms=round(elapsed_ms, 2),
    )


@router.post("/feedback")
def feedback(request: FeedbackRequest) -> dict[str, str]:
    # TODO: persist feedback to Supabase for later eval
    return {"status": "received"}