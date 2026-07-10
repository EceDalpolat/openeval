# Robustness tests for the Judge — it must not crash on malformed/missing judge output.

import pytest

from openeval.connectors.base import ModelResponse
from openeval.judge.judge import Judge, coerce_dimension, extract_json
from openeval.judge.schemas import EvalCase

CASE = EvalCase(question="What is RAG?", answer="It adds external knowledge.")

VALID = """{
  "faithfulness": {"score": 0.9, "reasoning": "correct"},
  "relevance":    {"score": 0.8, "reasoning": "relevant"},
  "clarity":      {"score": 0.7, "reasoning": "clear"},
  "safety":       {"score": 1.0, "reasoning": "safe"},
  "consistency":  {"score": 0.9, "reasoning": "consistent"}
}"""


class FakeConnector:
    """Returns the next response each time generate() is called; raises if the response is an Exception."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate(self, prompt, system=""):
        item = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return ModelResponse(content=item, model="fake", input_tokens=1, output_tokens=1)


# --- extract_json: stripping fences + preamble ---

def test_extract_json_strips_fences_and_preamble():
    raw = 'Here is the result:\n```json\n{"a": 1}\n```\nHope this helps.'
    assert extract_json(raw) == {"a": 1}


def test_extract_json_raises_when_no_json():
    with pytest.raises(ValueError):
        extract_json("there is no json here at all")


# --- coerce_dimension: safe default for missing/malformed dimension ---

def test_coerce_missing_dimension_defaults_to_zero():
    dim = coerce_dimension({}, "faithfulness")
    assert dim.score == 0.0 and "missing" in dim.reasoning


def test_coerce_normalizes_score_over_one():
    dim = coerce_dimension({"clarity": {"score": 90, "reasoning": "x"}}, "clarity")
    assert dim.score == 0.9


# --- Judge: end-to-end robustness ---

def test_judge_parses_valid_response():
    judge = Judge(FakeConnector([VALID]))
    result = judge.evaluate(CASE)
    assert result.faithfulness.score == 0.9
    assert result.safety.score == 1.0


def test_judge_recovers_from_garbage_without_crashing():
    judge = Judge(FakeConnector(["completely unrelated text, not JSON"]))
    result = judge.evaluate(CASE)  # must not crash
    assert result.overall_score == 0.0  # all neutral defaults


def test_judge_handles_partial_json():
    partial = '{"faithfulness": {"score": 0.8, "reasoning": "ok"}}'
    result = Judge(FakeConnector([partial])).evaluate(CASE)
    assert result.faithfulness.score == 0.8   # the provided dimension was read
    assert result.relevance.score == 0.0      # missing dimension defaults


def test_judge_retries_on_transient_error_then_succeeds():
    conn = FakeConnector([RuntimeError("network down"), RuntimeError("still down"), VALID])
    result = Judge(conn).evaluate(CASE)
    assert conn.calls == 3           # two failures + one success
    assert result.faithfulness.score == 0.9


def test_judge_gives_up_after_max_retries():
    conn = FakeConnector([RuntimeError("down")])
    with pytest.raises(RuntimeError, match="failed to produce a response"):
        Judge(conn, max_retries=3).evaluate(CASE)
    assert conn.calls == 3
