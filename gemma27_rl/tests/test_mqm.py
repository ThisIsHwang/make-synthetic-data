from __future__ import annotations

import json

import pytest

from gemma27_rl.config import MQMConfig
import gemma27_rl.rewards as rewards_mod
from gemma27_rl.rewards import (
    GembaParseError,
    OpenAICompatibleMQMScorer,
    gemba_mqm_extract_error_spans,
    gemba_mqm_parse_errors,
    gemba_mqm_score,
)
from gemma27_rl.rl_types import SampleForScoring


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


def test_gemba_mqm_parse_errors_rejects_unstructured_output() -> None:
    with pytest.raises(ValueError, match="structured errors|unparseable"):
        gemba_mqm_parse_errors("The translation looks mostly fine to me.")


def test_openai_mqm_retries_until_output_is_parseable(monkeypatch: pytest.MonkeyPatch) -> None:
    scorer = OpenAICompatibleMQMScorer(
        cfg=MQMConfig(
            enabled=True,
            base_url="http://localhost:8000/v1",
            max_retries=0,
        )
    )
    sample = SampleForScoring(src="hello", mt="안녕", ref=None)
    calls = iter(
        (["Looks fine overall.", "still bad"] * 9)
        + ['Major:\naccuracy/mistranslation - "안녕"']
    )
    call_count = {"n": 0}

    def _fake_call(messages, max_tokens=None, chat_template_kwargs_override=None):
        call_count["n"] += 1
        return next(calls)

    monkeypatch.setattr(scorer, "_call_openai_compatible_api", _fake_call)

    score, raw_text, spans = scorer._score_one_sample(
        sample,
        [{"role": "user", "content": "test"}],
    )

    assert call_count["n"] == 19
    assert score == -5.0
    assert raw_text == 'Major:\naccuracy/mistranslation - "안녕"'
    assert len(spans) == 1
    assert spans[0]["text"] == "안녕"


def test_openai_mqm_parse_failures_do_not_fallback_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    scorer = OpenAICompatibleMQMScorer(
        cfg=MQMConfig(
            enabled=True,
            base_url="http://localhost:8000/v1",
            max_retries=0,
            error_policy="zero",
        )
    )
    monkeypatch.setattr(
        scorer,
        "_call_openai_compatible_api",
        lambda messages, max_tokens=None, chat_template_kwargs_override=None: "Looks fine overall.",
    )

    with pytest.raises(GembaParseError, match="unparseable"):
        scorer._score_one_sample(
            SampleForScoring(src="hello", mt="안녕", ref=None),
            [{"role": "user", "content": "test"}],
        )


def test_openai_mqm_repairs_unparseable_output_before_retrying(monkeypatch: pytest.MonkeyPatch) -> None:
    scorer = OpenAICompatibleMQMScorer(
        cfg=MQMConfig(
            enabled=True,
            base_url="http://localhost:8000/v1",
            max_retries=0,
        )
    )
    sample = SampleForScoring(src="hello", mt="안녕", ref=None)
    responses = iter(
        [
            "This translation has a major mistranslation around 안녕.",
            'Major:\naccuracy/mistranslation - "안녕"\nMinor:\nno-error',
        ]
    )
    captured: list[str] = []

    def _fake_call(messages, max_tokens=None, chat_template_kwargs_override=None):
        captured.append(messages[0]["content"])
        return next(responses)

    monkeypatch.setattr(scorer, "_call_openai_compatible_api", _fake_call)

    score, raw_text, spans = scorer._score_one_sample(sample, [{"role": "user", "content": "test"}])

    assert len(captured) == 2
    assert score == -5.0
    assert raw_text == 'Major:\naccuracy/mistranslation - "안녕"\nMinor:\nno-error'
    assert spans[0]["text"] == "안녕"


def test_openai_mqm_enables_thinking_after_first_two_failed_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    scorer = OpenAICompatibleMQMScorer(
        cfg=MQMConfig(
            enabled=True,
            base_url="http://localhost:8000/v1",
            max_retries=0,
        )
    )
    sample = SampleForScoring(src="hello", mt="안녕", ref=None)
    calls = {"n": 0}
    seen_thinking: list[bool] = []

    def _fake_call(messages, max_tokens=None, chat_template_kwargs_override=None):
        calls["n"] += 1
        seen_thinking.append(bool((chat_template_kwargs_override or {}).get("enable_thinking")))
        if calls["n"] <= 4:
            return "bad"
        return 'Major:\naccuracy/mistranslation - "안녕"'

    monkeypatch.setattr(scorer, "_call_openai_compatible_api", _fake_call)

    score, _, _ = scorer._score_one_sample(sample, [{"role": "user", "content": "test"}])

    assert score == -5.0
    assert seen_thinking[:4] == [False] * 4
    assert seen_thinking[-1] is True


def test_openai_mqm_score_batch_skips_sample_after_all_attempts_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    scorer = OpenAICompatibleMQMScorer(
        cfg=MQMConfig(
            enabled=True,
            base_url="http://localhost:8000/v1",
            max_retries=0,
        )
    )
    monkeypatch.setattr(
        scorer,
        "_call_openai_compatible_api",
        lambda messages, max_tokens=None, chat_template_kwargs_override=None: "bad",
    )

    out = scorer.score_batch([SampleForScoring(src="hello", mt="안녕", ref=None)])

    assert out.sequence_scores == [0.0]
    assert out.metadata["skipped_rows"] == [True]
    assert out.metadata["error_spans"] == [[]]


def test_openai_mqm_request_includes_reasoning_parser(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"choices": [{"message": {"content": "Minor:\nno-error"}}]}).encode("utf-8")

    class _FakeOpener:
        def open(self, req, timeout=None):
            captured["timeout"] = timeout
            captured["payload"] = json.loads(req.data.decode("utf-8"))
            return _FakeResponse()

    monkeypatch.setattr(rewards_mod, "_temporarily_unset_proxy_env", lambda: (lambda: None))
    monkeypatch.setattr(rewards_mod.urllib_request, "build_opener", lambda *args, **kwargs: _FakeOpener())

    scorer = OpenAICompatibleMQMScorer(
        cfg=MQMConfig(
            enabled=True,
            base_url="http://localhost:8000/v1",
            reasoning_parser="qwen3",
        )
    )
    raw = scorer._call_openai_compatible_api([{"role": "user", "content": "test"}])

    assert raw == "Minor:\nno-error"
    assert captured["timeout"] == 120.0
    assert captured["payload"]["reasoning_parser"] == "qwen3"
