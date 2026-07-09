# Judge'ın dayanıklılık testleri — bozuk/eksik hakem çıktısında çökmemeli.

import pytest

from openeval.connectors.base import ModelResponse
from openeval.judge.judge import Judge, coerce_dimension, extract_json
from openeval.judge.schemas import EvalCase

CASE = EvalCase(question="RAG nedir?", answer="Harici bilgi ekler.")

VALID = """{
  "faithfulness": {"score": 0.9, "reasoning": "doğru"},
  "relevance":    {"score": 0.8, "reasoning": "ilgili"},
  "clarity":      {"score": 0.7, "reasoning": "açık"},
  "safety":       {"score": 1.0, "reasoning": "güvenli"},
  "consistency":  {"score": 0.9, "reasoning": "tutarlı"}
}"""


class FakeConnector:
    """generate() çağrıldıkça sıradaki cevabı döndürür; cevap Exception ise fırlatır."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate(self, prompt, system=""):
        item = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return ModelResponse(content=item, model="fake", input_tokens=1, output_tokens=1)


# --- extract_json: fence + önsöz temizleme ---

def test_extract_json_strips_fences_and_preamble():
    raw = 'İşte sonuç:\n```json\n{"a": 1}\n```\nUmarım yardımcı olur.'
    assert extract_json(raw) == {"a": 1}


def test_extract_json_raises_when_no_json():
    with pytest.raises(ValueError):
        extract_json("burada hiç json yok")


# --- coerce_dimension: eksik/bozuk boyut güvenli default ---

def test_coerce_missing_dimension_defaults_to_zero():
    dim = coerce_dimension({}, "faithfulness")
    assert dim.score == 0.0 and "eksik" in dim.reasoning


def test_coerce_normalizes_score_over_one():
    dim = coerce_dimension({"clarity": {"score": 90, "reasoning": "x"}}, "clarity")
    assert dim.score == 0.9


# --- Judge: uçtan uca dayanıklılık ---

def test_judge_parses_valid_response():
    judge = Judge(FakeConnector([VALID]))
    result = judge.evaluate(CASE)
    assert result.faithfulness.score == 0.9
    assert result.safety.score == 1.0


def test_judge_recovers_from_garbage_without_crashing():
    judge = Judge(FakeConnector(["tamamen alakasız metin, JSON değil"]))
    result = judge.evaluate(CASE)  # çökmemeli
    assert result.overall_score == 0.0  # hepsi nötr default


def test_judge_handles_partial_json():
    partial = '{"faithfulness": {"score": 0.8, "reasoning": "ok"}}'
    result = Judge(FakeConnector([partial])).evaluate(CASE)
    assert result.faithfulness.score == 0.8   # gelen boyut okundu
    assert result.relevance.score == 0.0      # eksik boyut default


def test_judge_retries_on_transient_error_then_succeeds():
    conn = FakeConnector([RuntimeError("network down"), RuntimeError("still down"), VALID])
    result = Judge(conn).evaluate(CASE)
    assert conn.calls == 3           # iki hata + bir başarı
    assert result.faithfulness.score == 0.9


def test_judge_gives_up_after_max_retries():
    conn = FakeConnector([RuntimeError("down")])
    with pytest.raises(RuntimeError, match="denemede de yanıt üretemedi"):
        Judge(conn, max_retries=3).evaluate(CASE)
    assert conn.calls == 3
