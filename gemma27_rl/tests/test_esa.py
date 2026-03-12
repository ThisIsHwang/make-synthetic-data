from __future__ import annotations

import json
from pathlib import Path
import threading

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
    assert gemba_esa_parse_score("1. Check adequacy 2. Check fluency 7. Final note") is None
    assert gemba_esa_parse_score("The answer mentions 5 issues but gives no score.") is None


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


def test_openai_esa_uses_sample_language_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    scorer = OpenAICompatibleESAScorer(
        cfg=ESAConfig(
            enabled=True,
            base_url="http://localhost:8000/v1",
            max_retries=0,
        )
    )
    captured: list[str] = []

    def _fake_call(messages, max_tokens, chat_template_kwargs_override=None):
        captured.append(messages[-1]["content"])
        if len(captured) == 1:
            return 'Major:\naccuracy/mistranslation - "hello"'
        return "81"

    monkeypatch.setattr(scorer, "_call_openai_compatible_api", _fake_call)

    score, _, _ = scorer._score_one_sample(
        SampleForScoring(src="안녕", mt="hello", ref=None, source_lang="Korean", target_lang="English")
    )

    assert score == 81.0
    assert "Korean source:" in captured[0]
    assert "English translation:" in captured[0]
    assert "Korean source:" in captured[1]
    assert "English translation:" in captured[1]


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


def test_openai_esa_request_omits_reasoning_parser(monkeypatch: pytest.MonkeyPatch) -> None:
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
    assert "reasoning_parser" not in captured["payload"]
    assert captured["payload"]["chat_template_kwargs"] == {"enable_thinking": True}


def test_openai_esa_accepts_message_content_text_parts(monkeypatch: pytest.MonkeyPatch) -> None:
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
                                    {"type": "text", "text": "Score (0-100): 83"},
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

    scorer = OpenAICompatibleESAScorer(
        cfg=ESAConfig(
            enabled=True,
            base_url="http://localhost:8000/v1",
        )
    )
    raw = scorer._call_openai_compatible_api(
        [{"role": "user", "content": "test"}],
        max_tokens=64,
        chat_template_kwargs_override={"enable_thinking": True},
    )

    assert raw == "Score (0-100): 83"


def test_openai_esa_score_batch_refills_inflight_window(monkeypatch: pytest.MonkeyPatch) -> None:
    scorer = OpenAICompatibleESAScorer(
        cfg=ESAConfig(
            enabled=True,
            base_url="http://localhost:8000/v1",
            batch_size=2,
            max_retries=0,
        )
    )
    slow_started = threading.Event()
    slow_release = threading.Event()
    third_started = threading.Event()
    third_started_while_slow_blocked = {"value": False}
    holder: dict[str, object] = {}

    samples = [
        SampleForScoring(src="slow", mt="mt-slow", ref=None),
        SampleForScoring(src="fast", mt="mt-fast", ref=None),
        SampleForScoring(src="third", mt="mt-third", ref=None),
    ]

    def _fake_score_one_sample(sample: SampleForScoring) -> tuple[float, str, str]:
        if sample.src == "slow":
            slow_started.set()
            assert slow_release.wait(timeout=5.0)
            return (61.0, "slow-errors", "61")
        if sample.src == "fast":
            return (72.0, "fast-errors", "72")
        third_started_while_slow_blocked["value"] = not slow_release.is_set()
        third_started.set()
        return (83.0, "third-errors", "83")

    monkeypatch.setattr(scorer, "_score_one_sample", _fake_score_one_sample)

    def _run() -> None:
        try:
            holder["out"] = scorer.score_batch(samples)
        except Exception as exc:  # pragma: no cover - assertion relay
            holder["exc"] = exc

    thread = threading.Thread(target=_run, name="esa-bounded-concurrency-test")
    thread.start()

    assert slow_started.wait(timeout=2.0)
    assert third_started.wait(timeout=2.0)
    assert third_started_while_slow_blocked["value"] is True

    slow_release.set()
    thread.join(timeout=5.0)
    assert not thread.is_alive()
    assert "exc" not in holder

    out = holder["out"]
    assert isinstance(out, rewards_mod.RewardOutput)
    assert out.sequence_scores == [61.0, 72.0, 83.0]
    assert out.metadata["raw_error_outputs"] == ["slow-errors", "fast-errors", "third-errors"]
    assert out.metadata["raw_score_outputs"] == ["61", "72", "83"]


def test_openai_esa_rejects_non_message_content_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"choices": [{"output_text": "Score (0-100): 83"}]}).encode("utf-8")

    class _FakeOpener:
        def open(self, req, timeout=None):
            return _FakeResponse()

    monkeypatch.setattr(rewards_mod, "_temporarily_unset_proxy_env", lambda: (lambda: None))
    monkeypatch.setattr(rewards_mod.urllib_request, "build_opener", lambda *args, **kwargs: _FakeOpener())

    scorer = OpenAICompatibleESAScorer(
        cfg=ESAConfig(
            enabled=True,
            base_url="http://localhost:8000/v1",
        )
    )

    with pytest.raises(RuntimeError, match=r"expected choices\[0\]\.message\.content"):
        scorer._call_openai_compatible_api(
            [{"role": "user", "content": "test"}],
            max_tokens=64,
            chat_template_kwargs_override={"enable_thinking": True},
        )


def test_openai_esa_retries_until_score_is_parseable(monkeypatch: pytest.MonkeyPatch) -> None:
    scorer = OpenAICompatibleESAScorer(
        cfg=ESAConfig(
            enabled=True,
            base_url="http://localhost:8000/v1",
            max_retries=0,
        )
    )
    sample = SampleForScoring(src="hello", mt="안녕", ref=None)
    calls = iter(["Looks okay overall.", "still bad", "Looks okay overall.", "Score (0-100): 83"])
    call_count = {"n": 0}

    def _fake_call(messages, max_tokens, chat_template_kwargs_override=None):
        call_count["n"] += 1
        return next(calls)

    monkeypatch.setattr(scorer, "_call_openai_compatible_api", _fake_call)

    score, raw_error_text, raw_score_text = scorer._score_one_sample(sample)

    assert call_count["n"] == 4
    assert score == 83.0
    assert raw_error_text == "Looks okay overall."
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

    with pytest.raises(GembaParseError, match="score parse returned None"):
        scorer._score_one_sample(SampleForScoring(src="hello", mt="안녕", ref=None))


def test_openai_esa_parse_failures_are_recorded_to_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log_path = tmp_path / "esa_parse_failures.jsonl"
    scorer = OpenAICompatibleESAScorer(
        cfg=ESAConfig(
            enabled=True,
            base_url="http://localhost:8000/v1",
        ),
        parse_failure_log_path=log_path,
    )
    sample = SampleForScoring(src="hello", mt="안녕", ref=None)
    monkeypatch.setattr(rewards_mod, "_esa_score_phase_specs", lambda _: ((False, 1),))

    calls = iter(
        [
            'Major:\naccuracy/mistranslation - "안녕"',
            "안녕하세요, 이건 점수 형식이 아닙니다.",
        ]
    )

    def _fake_call(messages, max_tokens, chat_template_kwargs_override=None):
        return next(calls)

    monkeypatch.setattr(scorer, "_call_openai_compatible_api", _fake_call)

    with pytest.raises(GembaParseError, match="score parse returned None"):
        scorer._score_one_sample(sample)

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["scorer"] == "esa"
    assert rows[0]["stage"] == "score_parse_failed"
    assert rows[0]["raw_error_text"] == 'Major:\naccuracy/mistranslation - "안녕"'
    assert rows[0]["raw_score_text"] == "안녕하세요, 이건 점수 형식이 아닙니다."
    assert rows[0]["mt"] == "안녕"


def test_openai_esa_ignores_unparseable_error_annotations_and_retries_score(monkeypatch: pytest.MonkeyPatch) -> None:
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
            "I would give this translation a fairly strong result overall.",
            "There seems to be one major issue around 안녕.",
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
    assert raw_error_text == "There seems to be one major issue around 안녕."
    assert raw_score_text == "81"


def test_openai_esa_enables_thinking_after_first_failed_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
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
        if call_count["n"] == 1:
            return "annotation"
        if call_count["n"] == 2:
            return "bad"
        if call_count["n"] == 3:
            return "annotation"
        if call_count["n"] == 4:
            return "81"
        raise AssertionError(f"unexpected call count: {call_count['n']}")

    monkeypatch.setattr(scorer, "_call_openai_compatible_api", _fake_call)

    score, _, _ = scorer._score_one_sample(SampleForScoring(src="hello", mt="안녕", ref=None))

    assert score == 81.0
    assert seen_thinking == [False, False, True, True]


def test_openai_esa_starts_with_thinking_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    scorer = OpenAICompatibleESAScorer(
        cfg=ESAConfig(
            enabled=True,
            base_url="http://localhost:8000/v1",
            max_retries=0,
            chat_template_kwargs={"enable_thinking": True},
        )
    )
    seen_thinking: list[bool] = []

    def _fake_call(messages, max_tokens, chat_template_kwargs_override=None):
        seen_thinking.append(bool((chat_template_kwargs_override or {}).get("enable_thinking")))
        return "annotation" if len(seen_thinking) == 1 else "81"

    monkeypatch.setattr(scorer, "_call_openai_compatible_api", _fake_call)

    score, _, _ = scorer._score_one_sample(SampleForScoring(src="hello", mt="안녕", ref=None))

    assert score == 81.0
    assert seen_thinking == [True, True]


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
