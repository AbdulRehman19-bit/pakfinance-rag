"""API routes. /query is wired to the real retrieval+generation pipeline in a later commit —
for now it returns a stub so the frontend has a stable contract to build against."""
import time
from fastapi import APIRouter

from app.config import get_settings
from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    FeedbackRequest,
    HealthResponse,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(status="ok", env=settings.app_env)


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    start = time.perf_counter()
    # TODO: replace with real hybrid retrieval + rerank + generation pipeline
    elapsed_ms = (time.perf_counter() - start) * 1000
    return QueryResponse(
        query=request.query,
        answer="Retrieval pipeline not wired up yet — coming in the next commits.",
        sources=[],
        latency_ms=round(elapsed_ms, 2),
    )


@router.post("/feedback")
def feedback(request: FeedbackRequest) -> dict[str, str]:
    # TODO: persist feedback (e.g. to a simple log or DB) for later eval
    return {"status": "received"}