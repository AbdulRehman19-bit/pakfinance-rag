"""Unit tests for prompt-building — pure string formatting, no API calls."""
from app.generation.prompts import build_context_block, build_user_prompt


def test_build_context_block_includes_source_and_page():
    chunks = [{
        "document_title": "BPRD Circular No. 3", "page_number": 2,
        "source": "SBP", "parent_text": "Some regulatory text.",
    }]
    block = build_context_block(chunks)
    assert "[1] SBP — BPRD Circular No. 3, p.2" in block
    assert "Some regulatory text." in block


def test_build_context_block_handles_missing_page_number():
    chunks = [{"document_title": "Income Tax Ordinance", "source": "FBR", "text": "Tax text"}]
    block = build_context_block(chunks)
    assert ", p." not in block
    assert "Tax text" in block


def test_build_user_prompt_includes_query_and_context():
    chunks = [{"document_title": "Doc", "source": "PSX", "text": "Listing rule text"}]
    prompt = build_user_prompt("What are listing rules?", chunks)
    assert "What are listing rules?" in prompt
    assert "Listing rule text" in prompt