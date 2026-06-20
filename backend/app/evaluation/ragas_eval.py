"""
RAGAS evaluation — runs the retrieval+generation pipeline against a labeled
question/ground_truth set and computes standard RAG quality metrics:

  - faithfulness:       does the answer only state things the context supports?
  - answer_relevancy:   does the answer actually address the question asked?
  - context_precision:  are the retrieved chunks relevant, and ranked well?
  - context_recall:     did retrieval find what's needed to answer correctly?

RAGAS defaults to OpenAI for its judge LLM and embeddings — we deliberately
removed OpenAI from this project, so both are swapped here: Gemini (via
langchain-google-genai) as the judge LLM, and the same local ONNX MiniLM
embeddings the rest of the app already uses (via a tiny adapter — no new
heavy dependency like sentence-transformers/torch needed for this).

Needs a small labeled set in data/eval/golden_qa.json — copy
data/eval/golden_qa.example.json and fill in real Q&A pairs first.
"""
from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from app.config import get_settings
from app.generation.llm import generate_answer
from app.indexing.dense_index import get_embedding_function
from app.retrieval.retriever import retrieve

GOLDEN_SET_PATH = Path("data/eval/golden_qa.json")


class _ChromaEmbeddingsAdapter:
    """Wraps our existing local ONNX MiniLM embedding function in the plain
    embed_documents/embed_query interface RAGAS expects — reuses the same
    free embeddings as the rest of the app instead of adding a new dependency."""

    def __init__(self) -> None:
        self._ef = get_embedding_function()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._ef(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._ef([text])[0]


def _configured_metrics() -> list:
    settings = get_settings()
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY not set — RAGAS needs it as the judge LLM.")

    chat = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0,
    )
    llm = LangchainLLMWrapper(chat)
    embeddings = LangchainEmbeddingsWrapper(_ChromaEmbeddingsAdapter())

    for metric in (faithfulness, answer_relevancy, context_precision, context_recall):
        metric.llm = llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = embeddings

    return [faithfulness, answer_relevancy, context_precision, context_recall]


def _load_golden_set() -> list[dict]:
    if not GOLDEN_SET_PATH.exists():
        raise FileNotFoundError(
            f"{GOLDEN_SET_PATH} not found — copy data/eval/golden_qa.example.json "
            "to data/eval/golden_qa.json and fill in real Q&A pairs first."
        )
    return json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))


def run_pipeline_on_golden_set(top_k: int = 5) -> Dataset:
    """Runs every golden question through the real retrieve+generate pipeline
    and assembles a RAGAS-shaped dataset from the results."""
    golden = _load_golden_set()

    questions, contexts, answers, ground_truths = [], [], [], []
    for item in golden:
        question = item["question"]
        chunks = retrieve(question, top_k=top_k)
        answer = generate_answer(question, chunks)

        questions.append(question)
        contexts.append([c["text"] for c in chunks] or ["(no context retrieved)"])
        answers.append(answer)
        ground_truths.append(item.get("ground_truth", ""))
        print(f"  ran: {question[:60]}...")

    return Dataset.from_dict(
        {"question": questions, "contexts": contexts, "answer": answers, "ground_truth": ground_truths}
    )


def run_ragas_eval(top_k: int = 5) -> list[dict]:
    dataset = run_pipeline_on_golden_set(top_k=top_k)
    result = evaluate(dataset, metrics=_configured_metrics())
    return result.to_pandas().to_dict(orient="records")


if __name__ == "__main__":
    for row in run_ragas_eval():
        print(row)