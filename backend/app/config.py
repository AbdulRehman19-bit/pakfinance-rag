"""
Centralized settings, loaded from environment variables / .env file.
Every other module reads config from here — never call os.getenv() elsewhere.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- App ---
    app_env: str = "development"
    cors_origins: str = "http://localhost:3000"

    # --- Supabase ---
    supabase_url: str = ""
    supabase_key: str = ""

    # --- LLM ---
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"


    # --- Reranker ---
    cohere_api_key: str = ""
    rerank_model: str = "rerank-multilingual-v3.0"
    rerank_top_n: int = 5

    # --- Retrieval ---
    bm25_top_k: int = 20
    dense_top_k: int = 20
    rrf_k: int = 60  # RRF damping constant

    # --- Tracing ---
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "pakfinance-rag"

    # --- Paths ---
    chroma_persist_dir: str = "backend/indexes/chroma_db"
    bm25_index_path: str = "backend/indexes/bm25_index.pkl"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Cached singleton — settings are read from env once per process."""
    return Settings()