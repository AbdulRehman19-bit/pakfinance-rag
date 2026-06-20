"""
LangSmith tracing setup.

configure_tracing() reads our typed Settings and sets the env vars
LangSmith's SDK actually looks for (LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY,
LANGCHAIN_PROJECT) — called once at app startup (see app/main.py).

`traceable` is re-exported from here so every other module imports it from
one place: `from app.tracing import traceable`.
"""
from __future__ import annotations

import os

from langsmith import traceable  # re-exported for convenience

from app.config import get_settings


def configure_tracing() -> None:
    """Call once at app startup. No-ops cleanly if tracing isn't configured."""
    settings = get_settings()

    if not settings.langchain_tracing_v2:
        print("  [LangSmith] tracing disabled (LANGCHAIN_TRACING_V2 not set)")
        return

    if not settings.langchain_api_key:
        print("  ! LANGCHAIN_TRACING_V2 is true but LANGCHAIN_API_KEY is not set — tracing disabled")
        return

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    print(f"  [LangSmith] tracing enabled -> project '{settings.langchain_project}'")


__all__ = ["configure_tracing", "traceable"]