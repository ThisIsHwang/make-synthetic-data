from __future__ import annotations

from gemma27_rl.config import MQMConfig
from gemma27_rl.rewards import (
    OpenAICompatibleMQMScorer,
    gemba_mqm_extract_error_spans,
    gemba_mqm_parse_errors,
    gemba_mqm_score,
)
from gemma27_rl.types import SampleForScoring


def test_gemba_mqm_parse_and_score() -> None:
    raw = """Critical:
accuracy/mistranslation - "x"
Major:
fluency/grammar - "y"
Minor:
style/awkward - "z"
"""
    parsed = gemba_mqm_parse_errors(raw)
    assert len(parsed["critical"]) == 1
    assert len(parsed["major"]) == 1
    assert len(parsed["minor"]) == 1
    assert gemba_mqm_score(raw) == -25


def test_openai_mqm_predict_fn_path() -> None:
    captured: list[list[dict[str, str]]] = []

    def fake_predict(rows: list[list[dict[str, str]]]) -> list[float]:
        captured.extend(rows)
        return [-5.0 for _ in rows]

    scorer = OpenAICompatibleMQMScorer(cfg=MQMConfig(enabled=True), predict_fn=fake_predict)
    out = scorer.score_batch([SampleForScoring(src="hello", mt="안녕", ref=None)])

    assert out.sequence_scores == [-5.0]
    assert out.metadata["error_spans"] == [[]]
    assert len(captured) == 1
    assert captured[0][-1]["role"] == "user"
    assert "hello" in captured[0][-1]["content"]
    assert "안녕" in captured[0][-1]["content"]


def test_gemba_mqm_extract_error_spans_maps_quoted_text() -> None:
    mt = "나는 학교에 갔다."
    raw = """Critical:
accuracy/mistranslation - "학교"
Major:
fluency/grammar - "갔다"
Minor:
no-error
"""
    spans = gemba_mqm_extract_error_spans(raw, mt)
    assert len(spans) == 2
    assert spans[0]["severity"] == "CRITICAL"
    assert spans[0]["text"] == "학교"
    assert spans[0]["start"] < spans[0]["end"]
    assert spans[1]["severity"] == "MAJOR"
