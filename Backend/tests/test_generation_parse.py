"""Model JSON parsing does not call a provider."""

from app.generation import INSUFFICIENT_ANSWER, parse_model_output


def test_parse_strips_code_fence() -> None:
    raw = """```json
{"answer": "The limit is 10.", "used_passage_ids": ["abc"], "sufficient": true}
```"""
    parsed = parse_model_output(raw)
    assert parsed == {
        "answer": "The limit is 10.",
        "used_passage_ids": ["abc"],
        "sufficient": True,
    }


def test_parse_invalid_json_is_insufficient() -> None:
    parsed = parse_model_output("this is not json")
    assert parsed["answer"] == INSUFFICIENT_ANSWER
    assert parsed["used_passage_ids"] == []
    assert parsed["sufficient"] is False
