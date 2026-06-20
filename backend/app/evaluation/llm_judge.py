"""
Custom LLM-as-judge — a second, independent quality check alongside RAGAS.
Asks Gemini directly to score each (question, answer, context) triple on
groundedness and relevance. Uses the current `google-genai` SDK — see
app/generation/llm.py for why (google-generativeai is deprecated).
"""
from __future__ import annotations

import json
from pathlib import Path

from google import genai
from google.genai import types

from app.config import get_settings
from app.generation.llm import generate_answer
from app.retrieval.retriever import retrieve

JUDGE_PROMPT = """You are evaluating a RAG system's answer for a Pakistani financial/regulatory \
question-answering assistant. Score it on two dimensions, 1 (worst) to 5 (best):

- groundedness: does the answer ONLY state things actually supported by the context? \
(5 = every claim is backed by the context; 1 = mostly made up / contradicts the context)
- relevance: does the answer actually address the question asked? \
(5 = directly and completely answers it; 1 = off-topic or non-responsive)

Context:
{context}

Question: {question}

Answer: {answer}

Respond with ONLY a JSON object, no other text, no markdown fences:
{{"groundedness": <1-5>, "relevance": <1-5>, "justification": "<one or two sentences>"}}
"""

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        settings = get_settings()
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY not set — add it to your .env.")
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def _parse_judge_response(raw: str) -> dict:
    """Gemini sometimes wraps JSON in ```json fences despite instructions —
    strip defensively, and degrade gracefully rather than crashing on garbage."""
    cleaned = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "groundedness": None,
            "relevance": None,
            "justification": f"(unparseable judge output: {cleaned[:200]})",
        }


def judge(question: str, answer: str, chunks: list[dict]) -> dict:
    """Returns {"groundedness": int|None, "relevance": int|None, "justification": str}."""
    client = _get_client()
    settings = get_settings()

    context = "\n\n".join(c.get("text", "") for c in chunks) or "(no context retrieved)"
    prompt = JUDGE_PROMPT.format(context=context, question=question, answer=answer)

    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0),
    )
    return _parse_judge_response(response.text.strip())


def judge_golden_set(top_k: int = 5) -> list[dict]:
    """Runs the LLM-as-judge over every item in the golden Q&A set, using the
    real retrieve+generate pipeline (same as ragas_eval.run_pipeline_on_golden_set)."""
    golden_path = Path("data/eval/golden_qa.json")
    golden = json.loads(golden_path.read_text(encoding="utf-8"))

    results = []
    for item in golden:
        question = item["question"]
        chunks = retrieve(question, top_k=top_k)
        answer = generate_answer(question, chunks)
        verdict = judge(question, answer, chunks)
        results.append({"question": question, "answer": answer, **verdict})
        print(f"  [{verdict.get('groundedness')}/{verdict.get('relevance')}] {question[:60]}...")

    return results


if __name__ == "__main__":
    results = judge_golden_set()
    scored = [r for r in results if r["groundedness"] is not None]
    if scored:
        avg_g = sum(r["groundedness"] for r in scored) / len(scored)
        avg_r = sum(r["relevance"] for r in scored) / len(scored)
        print(f"\nAverage groundedness: {avg_g:.2f}/5")
        print(f"Average relevance: {avg_r:.2f}/5")