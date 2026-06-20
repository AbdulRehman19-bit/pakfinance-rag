"""Gemini 1.5 Flash wrapper — turns retrieved chunks + a question into a grounded answer."""
from __future__ import annotations

import google.generativeai as genai

from app.config import get_settings
from app.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from app.tracing import traceable

_configured = False


def _ensure_configured() -> None:
    global _configured
    if not _configured:
        settings = get_settings()
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY not set — add it to your .env (see .env.example).")
        genai.configure(api_key=settings.gemini_api_key)
        _configured = True


@traceable(name="generate_answer")
def generate_answer(query: str, chunks: list[dict]) -> str:
    """Generate a grounded answer from the query and reranked chunks. Returns
    a plain "no context" message if nothing was retrieved — never lets the
    model improvise without sources."""
    if not chunks:
        return (
            "I couldn't find anything in the indexed PSX/SBP/FBR documents to answer "
            "this question. Try rephrasing, or it may genuinely be outside the corpus."
        )

    _ensure_configured()
    settings = get_settings()
    model = genai.GenerativeModel(
        model_name=settings.gemini_model,
        system_instruction=SYSTEM_PROMPT,
    )

    user_prompt = build_user_prompt(query, chunks)
    response = model.generate_content(
        user_prompt,
        generation_config={"temperature": 0.1, "max_output_tokens": 1024},
    )
    return response.text.strip()