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


def test_openai_mqm_predict_fn_uses_sample_language_pair() -> None:
    captured: list[list[dict[str, str]]] = []

    def fake_predict(rows: list[list[dict[str, str]]]) -> list[float]:
        captured.extend(rows)
        return [-5.0 for _ in rows]

    scorer = OpenAICompatibleMQMScorer(cfg=MQMConfig(enabled=True), predict_fn=fake_predict)
    out = scorer.score_batch(
        [SampleForScoring(src="안녕", mt="hello", ref=None, source_lang="Korean", target_lang="English")]
    )

    assert out.sequence_scores == [-5.0]
    assert "Korean source:" in captured[0][-1]["content"]
    assert "English translation:" in captured[0][-1]["content"]


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
        [
            "Looks fine overall.",
            'Major:\naccuracy/mistranslation - "안녕"',
        ]
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

    assert call_count["n"] == 2
    assert score == -5.0
    assert raw_text == 'Major:\naccuracy/mistranslation - "안녕"'
    assert len(spans) == 1
    assert spans[0]["text"] == "안녕"


def test_openai_mqm_parse_failures_raise_when_score_never_parses(monkeypatch: pytest.MonkeyPatch) -> None:
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
        lambda messages, max_tokens=None, chat_template_kwargs_override=None: "Looks fine overall.",
    )

    with pytest.raises(GembaParseError, match="score parse returned None"):
        scorer._score_one_sample(
            SampleForScoring(src="hello", mt="안녕", ref=None),
            [{"role": "user", "content": "test"}],
        )


def test_openai_mqm_parse_failures_are_recorded_to_jsonl(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    log_path = tmp_path / "mqm_parse_failures.jsonl"
    scorer = OpenAICompatibleMQMScorer(
        cfg=MQMConfig(
            enabled=True,
            base_url="http://localhost:8000/v1",
        ),
        parse_failure_log_path=log_path,
    )

    calls = iter(
        [
            "Looks fine overall.",
            'Major:\naccuracy/mistranslation - "안녕"',
        ]
    )

    monkeypatch.setattr(
        scorer,
        "_call_openai_compatible_api",
        lambda messages, max_tokens=None, chat_template_kwargs_override=None: next(calls),
    )

    score, raw_text, spans = scorer._score_one_sample(
        SampleForScoring(src="hello", mt="안녕", ref=None),
        [{"role": "user", "content": "test"}],
    )

    assert score == -5.0
    assert raw_text == 'Major:\naccuracy/mistranslation - "안녕"'
    assert len(spans) == 1
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["scorer"] == "mqm"
    assert rows[0]["stage"] == "raw_output_parse_failed"
    assert rows[0]["raw_text"] == "Looks fine overall."
    assert rows[0]["mt"] == "안녕"


def test_openai_mqm_enables_thinking_after_first_failed_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
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
        if calls["n"] <= 2:
            return "bad"
        return 'Major:\naccuracy/mistranslation - "안녕"'

    monkeypatch.setattr(scorer, "_call_openai_compatible_api", _fake_call)

    score, _, _ = scorer._score_one_sample(sample, [{"role": "user", "content": "test"}])

    assert score == -5.0
    assert seen_thinking == [False, False, True]
    assert seen_thinking[-1] is True


def test_openai_mqm_starts_with_thinking_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    scorer = OpenAICompatibleMQMScorer(
        cfg=MQMConfig(
            enabled=True,
            base_url="http://localhost:8000/v1",
            max_retries=0,
            chat_template_kwargs={"enable_thinking": True},
        )
    )
    sample = SampleForScoring(src="hello", mt="안녕", ref=None)
    seen_thinking: list[bool] = []

    def _fake_call(messages, max_tokens=None, chat_template_kwargs_override=None):
        seen_thinking.append(bool((chat_template_kwargs_override or {}).get("enable_thinking")))
        return 'Major:\naccuracy/mistranslation - "안녕"'

    monkeypatch.setattr(scorer, "_call_openai_compatible_api", _fake_call)

    score, _, _ = scorer._score_one_sample(sample, [{"role": "user", "content": "test"}])

    assert score == -5.0
    assert seen_thinking == [True]


def test_openai_mqm_allows_empty_spans_when_score_parses(monkeypatch: pytest.MonkeyPatch) -> None:
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
        lambda messages, max_tokens=None, chat_template_kwargs_override=None: 'Major:\naccuracy/mistranslation - "hello"',
    )

    score, raw_text, spans = scorer._score_one_sample(
        SampleForScoring(src="hello", mt="안녕", ref=None),
        [{"role": "user", "content": "test"}],
    )

    assert score == -5.0
    assert raw_text == 'Major:\naccuracy/mistranslation - "hello"'
    assert spans == []


def test_openai_mqm_score_batch_falls_back_without_skip_when_score_parse_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    assert out.metadata["skipped_rows"] == [False]
    assert out.metadata["error_spans"] == [[]]


def test_openai_mqm_request_omits_reasoning_parser(monkeypatch) -> None:
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
    assert "reasoning_parser" not in captured["payload"]


def test_openai_mqm_accepts_message_content_text_parts(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": [
                                    {"type": "text", "text": 'Major:\naccuracy/mistranslation - "안녕"'},
                                ]
                            }
                        }
                    ]
                }
            ).encode("utf-8")

    class _FakeOpener:
        def open(self, req, timeout=None):
            return _FakeResponse()

    monkeypatch.setattr(rewards_mod, "_temporarily_unset_proxy_env", lambda: (lambda: None))
    monkeypatch.setattr(rewards_mod.urllib_request, "build_opener", lambda *args, **kwargs: _FakeOpener())

    scorer = OpenAICompatibleMQMScorer(
        cfg=MQMConfig(
            enabled=True,
            base_url="http://localhost:8000/v1",
        )
    )
    raw = scorer._call_openai_compatible_api(
        [{"role": "user", "content": "test"}],
        chat_template_kwargs_override={"enable_thinking": True},
    )

    assert raw == 'Major:\naccuracy/mistranslation - "안녕"'


def test_openai_mqm_rejects_non_message_content_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"choices": [{"text": 'Major:\naccuracy/mistranslation - "안녕"'}]}).encode("utf-8")

    class _FakeOpener:
        def open(self, req, timeout=None):
            return _FakeResponse()

    monkeypatch.setattr(rewards_mod, "_temporarily_unset_proxy_env", lambda: (lambda: None))
    monkeypatch.setattr(rewards_mod.urllib_request, "build_opener", lambda *args, **kwargs: _FakeOpener())

    scorer = OpenAICompatibleMQMScorer(
        cfg=MQMConfig(
            enabled=True,
            base_url="http://localhost:8000/v1",
        )
    )

    with pytest.raises(RuntimeError, match=r"expected choices\[0\]\.message\.content"):
        scorer._call_openai_compatible_api(
            [{"role": "user", "content": "test"}],
            chat_template_kwargs_override={"enable_thinking": True},
        )
