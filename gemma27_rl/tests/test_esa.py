from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from gemma27_rl.config import ESAConfig, load_config
import gemma27_rl.rewards as rewards_mod
from gemma27_rl.rewards import (
    GembaParseError,
    OpenAICompatibleESAScorer,
    build_gemba_esa_error_messages,
    gemba_esa_format_error_spans,
    gemba_esa_parse_errors,
    gemba_esa_parse_score,
)
from gemma27_rl.rl_types import SampleForScoring


def test_gemba_esa_parse_and_score() -> None:
    raw = """Major:
accuracy/mistranslation - \"x\"
Minor:
fluency/grammar - \"y\"
"""
    parsed = gemba_esa_parse_errors(raw)
    assert len(parsed["major"]) == 1
    assert len(parsed["minor"]) == 1
    assert "Major:" in gemba_esa_format_error_spans(raw)
    assert gemba_esa_parse_score("Score (0-100): 83") == 83.0
    assert gemba_esa_parse_score("The quality is 82/100 overall.") == 82.0
    assert gemba_esa_parse_score("**Score: 81 out of 100**") == 81.0
    assert gemba_esa_parse_score("[79]") == 79.0


def test_gemba_esa_parse_errors_rejects_unstructured_output() -> None:
    with pytest.raises(ValueError, match="structured errors|unparseable"):
        gemba_esa_parse_errors("I do not see major issues here.")


def test_gemba_esa_build_messages_without_fewshot() -> None:
    messages = build_gemba_esa_error_messages(
        source_lang="English",
        target_lang="Korean",
        source_seg="hello",
        target_seg="안녕",
        use_fewshot=False,
    )
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert "hello" in messages[-1]["content"]
    assert "안녕" in messages[-1]["content"]


def test_openai_esa_predict_fn_path() -> None:
    captured: list[SampleForScoring] = []

    def fake_predict(samples: list[SampleForScoring]) -> list[float]:
        captured.extend(samples)
        return [77.5 for _ in samples]

    scorer = OpenAICompatibleESAScorer(cfg=ESAConfig(enabled=True), predict_fn=fake_predict)
    out = scorer.score_batch([SampleForScoring(src="hello", mt="안녕", ref=None)])

    assert out.sequence_scores == [77.5]
    assert len(captured) == 1
    assert captured[0].src == "hello"
    assert captured[0].mt == "안녕"


def test_load_config_allows_esa_only_reward(tmp_path: Path) -> None:
    cfg_path = tmp_path / "esa_only.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  hf_dataset_name: dummy/dataset",
                "reward:",
                "  metricx:",
                "    enabled: false",
                "  xcomet:",
                "    enabled: false",
                "  mqm:",
                "    enabled: false",
                "  esa:",
                "    enabled: true",
                "    base_url: http://localhost:8000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)
    assert cfg.reward.esa.enabled is True
    assert cfg.reward.esa.base_url == "http://localhost:8000"


def test_openai_esa_request_includes_reasoning_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"choices": [{"message": {"content": "Score (0-100): 83"}}]}).encode("utf-8")

    class _FakeOpener:
        def open(self, req, timeout=None):
            captured["timeout"] = timeout
            captured["payload"] = json.loads(req.data.decode("utf-8"))
            return _FakeResponse()

    monkeypatch.setattr(rewards_mod, "_temporarily_unset_proxy_env", lambda: (lambda: None))
    monkeypatch.setattr(rewards_mod.urllib_request, "build_opener", lambda *args, **kwargs: _FakeOpener())

    scorer = OpenAICompatibleESAScorer(
        cfg=ESAConfig(
            enabled=True,
            base_url="http://localhost:8000/v1",
            reasoning_parser="qwen3",
        )
    )
    raw = scorer._call_openai_compatible_api(
        [{"role": "user", "content": "test"}],
        max_tokens=64,
        chat_template_kwargs_override={"enable_thinking": True},
    )

    assert raw == "Score (0-100): 83"
    assert captured["timeout"] == 120.0
    assert captured["payload"]["reasoning_parser"] == "qwen3"
    assert captured["payload"]["chat_template_kwargs"] == {"enable_thinking": True}


def test_openai_esa_retries_until_error_annotations_are_parseable(monkeypatch: pytest.MonkeyPatch) -> None:
    scorer = OpenAICompatibleESAScorer(
        cfg=ESAConfig(
            enabled=True,
            base_url="http://localhost:8000/v1",
            max_retries=0,
        )
    )
    sample = SampleForScoring(src="hello", mt="안녕", ref=None)
    calls = iter(
        (["Looks okay overall.", "still bad"] * 9)
        + ['Major:\naccuracy/mistranslation - "안녕"', "Score (0-100): 83"]
    )
    call_count = {"n": 0}

    def _fake_call(messages, max_tokens, chat_template_kwargs_override=None):
        call_count["n"] += 1
        return next(calls)

    monkeypatch.setattr(scorer, "_call_openai_compatible_api", _fake_call)

    score, raw_error_text, raw_score_text = scorer._score_one_sample(sample)

    assert call_count["n"] == 20
    assert score == 83.0
    assert raw_error_text == 'Major:\naccuracy/mistranslation - "안녕"'
    assert raw_score_text == "Score (0-100): 83"


def test_openai_esa_parse_failures_do_not_fallback_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    scorer = OpenAICompatibleESAScorer(
        cfg=ESAConfig(
            enabled=True,
            base_url="http://localhost:8000/v1",
            max_retries=0,
            error_policy="zero",
        )
    )
    monkeypatch.setattr(
        scorer,
        "_call_openai_compatible_api",
        lambda messages, max_tokens, chat_template_kwargs_override=None: "Looks okay overall.",
    )

    with pytest.raises(GembaParseError, match="unparseable"):
        scorer._score_one_sample(SampleForScoring(src="hello", mt="안녕", ref=None))


def test_openai_esa_repairs_error_annotations_and_extracts_score(monkeypatch: pytest.MonkeyPatch) -> None:
    scorer = OpenAICompatibleESAScorer(
        cfg=ESAConfig(
            enabled=True,
            base_url="http://localhost:8000/v1",
            max_retries=0,
        )
    )
    sample = SampleForScoring(src="hello", mt="안녕", ref=None)
    responses = iter(
        [
            "There seems to be one major issue around 안녕.",
            'Major:\naccuracy/mistranslation - "안녕"\nMinor:\nno-error',
            "I would give this translation a fairly strong result overall.",
            "81",
        ]
    )
    captured: list[str] = []

    def _fake_call(messages, max_tokens, chat_template_kwargs_override=None):
        captured.append(messages[0]["content"])
        return next(responses)

    monkeypatch.setattr(scorer, "_call_openai_compatible_api", _fake_call)

    score, raw_error_text, raw_score_text = scorer._score_one_sample(sample)

    assert len(captured) == 4
    assert score == 81.0
    assert raw_error_text == 'Major:\naccuracy/mistranslation - "안녕"\nMinor:\nno-error'
    assert raw_score_text == "81"


def test_openai_esa_enables_thinking_after_first_ten_failed_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    scorer = OpenAICompatibleESAScorer(
        cfg=ESAConfig(
            enabled=True,
            base_url="http://localhost:8000/v1",
            max_retries=0,
        )
    )
    seen_thinking: list[bool] = []
    call_count = {"n": 0}

    def _fake_call(messages, max_tokens, chat_template_kwargs_override=None):
        call_count["n"] += 1
        seen_thinking.append(bool((chat_template_kwargs_override or {}).get("enable_thinking")))
        if call_count["n"] <= 20:
            return "bad"
        if call_count["n"] == 21:
            return 'Major:\naccuracy/mistranslation - "안녕"'
        return "81"

    monkeypatch.setattr(scorer, "_call_openai_compatible_api", _fake_call)

    score, _, _ = scorer._score_one_sample(SampleForScoring(src="hello", mt="안녕", ref=None))

    assert score == 81.0
    assert seen_thinking[:20] == [False] * 20
    assert seen_thinking[-1] is True


def test_openai_esa_score_batch_skips_sample_after_all_attempts_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    scorer = OpenAICompatibleESAScorer(
        cfg=ESAConfig(
            enabled=True,
            base_url="http://localhost:8000/v1",
            max_retries=0,
        )
    )
    monkeypatch.setattr(
        scorer,
        "_call_openai_compatible_api",
        lambda messages, max_tokens, chat_template_kwargs_override=None: "bad",
    )

    out = scorer.score_batch([SampleForScoring(src="hello", mt="안녕", ref=None)])

    assert out.sequence_scores == [0.0]
    assert out.metadata["skipped_rows"] == [True]
