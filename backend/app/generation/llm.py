"""
Gemini wrapper — turns retrieved chunks + a question into a grounded answer.

Uses Google's current `google-genai` SDK. The old `google-generativeai`
package was deprecated by Google on 2025-08-31 (legacy/maintenance mode, no
active development) — this is its client-based replacement.
"""
from __future__ import annotations

from google import genai
from google.genai import types

from app.config import get_settings
from app.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from app.tracing import traceable

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        settings = get_settings()
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY not set — add it to your .env (see .env.example).")
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


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

    client = _get_client()
    settings = get_settings()
    user_prompt = build_user_prompt(query, chunks)

    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.1,
            max_output_tokens=1024,
        ),
    )
    return response.text.strip()