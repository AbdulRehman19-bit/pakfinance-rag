"""Unit tests for the LLM-judge response parser — pure string/JSON logic, no API call."""
from app.evaluation.llm_judge import _parse_judge_response


def test_parses_clean_json():
    result = _parse_judge_response('{"groundedness": 4, "relevance": 5, "justification": "Good answer."}')
    assert result == {"groundedness": 4, "relevance": 5, "justification": "Good answer."}


def test_strips_markdown_json_fences():
    raw = '```json\n{"groundedness": 3, "relevance": 3, "justification": "OK."}\n```'
    result = _parse_judge_response(raw)
    assert result["groundedness"] == 3


def test_handles_unparseable_output_gracefully():
    result = _parse_judge_response("not valid json at all")
    assert result["groundedness"] is None
    assert "unparseable" in result["justification"]