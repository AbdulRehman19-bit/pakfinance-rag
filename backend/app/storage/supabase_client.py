"""Supabase client singleton. Supabase is the durable source of truth for
chunk text + metadata — BM25 and Chroma indexes are rebuilt from it at
startup, since the Hugging Face Space filesystem isn't guaranteed to persist
across redeploys."""
from functools import lru_cache

from supabase import create_client, Client

from app.config import get_settings


@lru_cache
def get_supabase_client() -> Client:
    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_key:
        raise RuntimeError(
            "SUPABASE_URL / SUPABASE_KEY not set — add them to your .env (see .env.example)."
        )
    return create_client(settings.supabase_url, settings.supabase_key)