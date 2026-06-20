"""FastAPI entrypoint. Run locally with: uvicorn app.main:app --reload --port 8000"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.api.routes import router
from app.tracing import configure_tracing

settings = get_settings()
configure_tracing()

app = FastAPI(
    title="PakFinance RAG API",
    description="Hybrid BM25 + dense retrieval RAG over PSX/SBP/FBR documents.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "PakFinance RAG API is running. See /docs for the API reference."}