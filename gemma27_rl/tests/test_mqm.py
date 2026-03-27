from __future__ import annotations

import json
import os
import threading

import pytest

from gemma27_rl.config import MQMConfig
import gemma27_rl.rewards as rewards_mod
from gemma27_rl.rewards import (
    _compute_mqm_unanchored_seq_penalty,
    GembaParseError,
    OpenAICompatibleMQMScorer,
    build_gemba_mqm_messages,
    gemba_mqm_extract_error_annotations,
    gemba_mqm_extract_error_spans,
    gemba_mqm_parse_errors,
    gemba_mqm_parse_structured_errors,
    gemba_mqm_score,
)
from gemma27_rl.rl_types import SampleForScoring


def _mqm_json_errors(errors: list[dict[str, object]]) -> str:
    return json.dumps({"errors": errors}, ensure_ascii=False, indent=2)


def test_gemba_mqm_parse_and_score() -> None:
    raw = _mqm_json_errors(
        [
            {"severity": "critical", "type": "accuracy/mistranslation", "target_span": "x", "source_span": None, "confidence": 0.97},
            {"severity": "major", "type": "fluency/grammar", "target_span": "y", "source_span": None, "confidence": 0.91},
            {"severity": "minor", "type": "style/awkward", "target_span": "z", "source_span": None, "confidence": 0.83},
        ]
    )
    parsed = gemba_mqm_parse_errors(raw)
    assert len(parsed["critical"]) == 1
    assert len(parsed["major"]) == 1
    assert len(parsed["minor"]) == 1
    assert gemba_mqm_score(raw) == -25


def test_gemba_mqm_parse_structured_errors_repairs_unescaped_inner_quotes_in_json_spans() -> None:
    raw = """{
  "errors": [
    {
      "severity": "minor",
      "type": "fluency/grammar",
      "target_span": "referred to as the "Center"",
      "source_span": ""센터"이라 한다",
      "confidence": 0.85
    }
  ]
}"""

    parsed = gemba_mqm_parse_structured_errors(raw)

    assert len(parsed) == 1
    assert parsed[0]["severity"] == "minor"
    assert parsed[0]["type"] == "fluency/grammar"
    assert parsed[0]["target_span"] == 'referred to as the "Center"'
    assert parsed[0]["source_span"] == '"센터"이라 한다'
    assert parsed[0]["confidence"] == pytest.approx(0.85)


def test_gemba_mqm_parse_structured_errors_repairs_mid_string_unescaped_quotes_in_source_span() -> None:
    raw = """{
  "errors": [
    {
      "severity": "major",
      "type": "accuracy/mistranslation",
      "target_span": "collaboration with relevant organizations, institutions, and experts, including Seongdong-gu residents",
      "source_span": "성동구민(이하"구민"이라 한다) 을 포함한 관련 단체·기관 및 전문가가 참여하여 협력할 수 있는",
      "confidence": 0.95
    }
  ]
}"""

    parsed = gemba_mqm_parse_structured_errors(raw)

    assert len(parsed) == 1
    assert parsed[0]["severity"] == "major"
    assert parsed[0]["type"] == "accuracy/mistranslation"
    assert parsed[0]["target_span"] == (
        "collaboration with relevant organizations, institutions, and experts, including Seongdong-gu residents"
    )
    assert parsed[0]["source_span"] == '성동구민(이하"구민"이라 한다) 을 포함한 관련 단체·기관 및 전문가가 참여하여 협력할 수 있는'
    assert parsed[0]["confidence"] == pytest.approx(0.95)


def test_gemba_mqm_parse_structured_errors_repairs_mixed_escaped_and_unescaped_quotes_across_fields() -> None:
    raw = """{
  "errors": [
    {
      "severity": "major",
      "type": "accuracy/mistranslation",
      "target_span": "구민(이하\\"구민\\")",
      "source_span": "Geumcheon-gu residents (hereinafter referred to as"Gu residents")",
      "confidence": 0.95
    },
    {
      "severity": "minor",
      "type": "accuracy/omission",
      "target_span": null,
      "source_span": "highly regarded by",
      "confidence": 0.88
    },
    {
      "severity": "minor",
      "type": "fluency/grammar",
      "target_span": "추진하는 사항 관련자가",
      "source_span": "persons related to matters to be promoted",
      "confidence": 0.85
    }
  ]
}"""

    parsed = gemba_mqm_parse_structured_errors(raw)

    assert len(parsed) == 3
    assert parsed[0]["target_span"] == '구민(이하"구민")'
    assert parsed[0]["source_span"] == 'Geumcheon-gu residents (hereinafter referred to as"Gu residents")'
    assert parsed[1]["target_span"] is None
    assert parsed[1]["source_span"] == "highly regarded by"
    assert parsed[2]["target_span"] == "추진하는 사항 관련자가"
    assert parsed[2]["source_span"] == "persons related to matters to be promoted"


def test_gemba_mqm_parse_structured_errors_repairs_quote_after_comma_in_source_span() -> None:
    raw = """{
  "errors": [
    {
      "severity": "major",
      "type": "accuracy/mistranslation",
      "target_span": "개통되었다",
      "source_span": "completely connected",
      "confidence": 0.95
    },
    {
      "severity": "minor",
      "type": "fluency/grammar",
      "target_span": null,
      "source_span": "Gyeongchun Line Forest Park,"which runs from...",
      "confidence": 0.88
    }
  ]
}"""

    parsed = gemba_mqm_parse_structured_errors(raw)

    assert len(parsed) == 2
    assert parsed[0]["target_span"] == "개통되었다"
    assert parsed[0]["source_span"] == "completely connected"
    assert parsed[1]["target_span"] is None
    assert parsed[1]["source_span"] == 'Gyeongchun Line Forest Park,"which runs from...'
    assert parsed[1]["confidence"] == pytest.approx(0.88)


def test_gemba_mqm_parse_structured_errors_handles_all_logged_malformed_quote_patterns() -> None:
    raw = """{
  "errors": [
    {
      "severity": "major",
      "type": "accuracy/mistranslation",
      "target_span": "구민(이하"구민\\")",
      "source_span": "Geumcheon-gu residents (hereinafter referred to as"Gu residents")",
      "confidence": 0.95
    },
    {
      "severity": "minor",
      "type": "accuracy/omission",
      "target_span": null,
      "source_span": "highly regarded by",
      "confidence": 0.88
    },
    {
      "severity": "minor",
      "type": "fluency/grammar",
      "target_span": "추진하는 사항 관련자가",
      "source_span": "persons related to matters to be promoted",
      "confidence": 0.85
    }
  ]
}"""

    parsed = gemba_mqm_parse_structured_errors(raw)

    assert parsed == [
        {
            "severity": "major",
            "type": "accuracy/mistranslation",
            "target_span": '구민(이하"구민")',
            "source_span": 'Geumcheon-gu residents (hereinafter referred to as"Gu residents")',
            "confidence": pytest.approx(0.95),
        },
        {
            "severity": "minor",
            "type": "accuracy/omission",
            "target_span": None,
            "source_span": "highly regarded by",
            "confidence": pytest.approx(0.88),
        },
        {
            "severity": "minor",
            "type": "fluency/grammar",
            "target_span": "추진하는 사항 관련자가",
            "source_span": "persons related to matters to be promoted",
            "confidence": pytest.approx(0.85),
        },
    ]


def test_openai_mqm_predict_fn_path() -> None:
    captured: list[list[dict[str, str]]] = []

    def fake_predict(rows: list[list[dict[str, str]]]) -> list[float]:
        captured.extend(rows)
        return [-5.0 for _ in rows]

    scorer = OpenAICompatibleMQMScorer(cfg=MQMConfig(enabled=True), predict_fn=fake_predict)
    out = scorer.score_batch([SampleForScoring(src="hello", mt="안녕", ref=None)])

    assert out.sequence_scores == [-5.0]
    assert out.metadata["error_spans"] == [[]]
    assert out.metadata["unanchored_errors"] == [[]]
    assert out.metadata["failure_rows"] == [False]
    assert len(captured) == 1
    assert captured[0][-1]["role"] == "user"
    assert "hello" in captured[0][-1]["content"]
    assert "안녕" in captured[0][-1]["content"]
    assert '"errors"' in captured[0][-1]["content"]
    assert "confidence" in captured[0][-1]["content"]


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


def test_build_gemba_mqm_messages_fewshot_outputs_json() -> None:
    messages = build_gemba_mqm_messages(
        source_lang="English",
        target_lang="Korean",
        source_seg="hello",
        target_seg="안녕",
    )

    assistant_contents = [message["content"] for message in messages if message["role"] == "assistant"]
    assert assistant_contents
    assert assistant_contents[0].lstrip().startswith("{")
    assert '"errors"' in assistant_contents[0]
    assert "confidence" in assistant_contents[0]


def test_gemba_mqm_build_messages_generic_preserves_existing_behavior() -> None:
    messages = build_gemba_mqm_messages(
        source_lang="English",
        target_lang="Korean",
        source_seg="hello",
        target_seg="안녕",
    )
    explicit_generic = build_gemba_mqm_messages(
        source_lang="English",
        target_lang="Korean",
        source_seg="hello",
        target_seg="안녕",
        use_fewshot=True,
        prompt_pack="generic",
    )

    assert messages == explicit_generic
    assert len(messages) == 8
    assert messages[0]["content"] == rewards_mod.GEMBA_SYSTEM_PROMPT
    assert messages[1]["content"] == rewards_mod.GEMBA_FEWSHOT_USER_1
    assert messages[2]["content"] == rewards_mod.GEMBA_FEWSHOT_ASSISTANT_1


def test_gemba_mqm_build_messages_prompt_pack_enterprise_contains_extra_guidance() -> None:
    messages = build_gemba_mqm_messages(
        source_lang="Korean",
        target_lang="English",
        source_seg="원문",
        target_seg="translation",
        prompt_pack="ko_en_enterprise_v1",
    )
    combined = "\n".join(message["content"] for message in messages)

    assert "Translation evaluation only. Do not repair or normalize the source." in combined
    assert "assistant, asks for more input, or explains instead of translating" in combined
    assert "애드인 서버 knox/brity 반영 예정입니다." in combined
    assert "프로님 안녕하세요~ 사랑회의실 몇시부터 이용하시나요?? ^^;;" in combined
    assert "Please provide the Korean text you would like translated." in combined
    assert "This is a no-reply email address." not in combined


def test_gemba_mqm_build_messages_use_fewshot_false_omits_fewshot_turns() -> None:
    messages = build_gemba_mqm_messages(
        source_lang="Korean",
        target_lang="English",
        source_seg="원문",
        target_seg="translation",
        use_fewshot=False,
        prompt_pack="ko_en_enterprise_v1",
    )

    assert len(messages) == 2
    assert [message["role"] for message in messages] == ["system", "user"]
    assert "애드인 서버 knox/brity 반영 예정입니다." not in messages[-1]["content"]
    assert "Translation evaluation only. Do not repair or normalize the source." in messages[-1]["content"]


def test_gemba_mqm_extract_error_spans_maps_quoted_text() -> None:
    mt = "나는 학교에 갔다."
    raw = _mqm_json_errors(
        [
            {"severity": "critical", "type": "accuracy/mistranslation", "target_span": "학교", "source_span": None, "confidence": 0.92},
            {"severity": "major", "type": "fluency/grammar", "target_span": "갔다", "source_span": None, "confidence": 0.76},
        ]
    )
    spans = gemba_mqm_extract_error_spans(raw, mt)
    assert len(spans) == 2
    assert spans[0]["severity"] == "CRITICAL"
    assert spans[0]["text"] == "학교"
    assert spans[0]["start"] < spans[0]["end"]
    assert spans[0]["confidence"] == pytest.approx(0.92)
    assert spans[1]["severity"] == "MAJOR"


def test_gemba_mqm_extract_error_spans_returns_all_detected_spans() -> None:
    mt = "a b c d e f"
    raw = _mqm_json_errors(
        [
            {"severity": "major", "type": "accuracy/mistranslation", "target_span": "a", "source_span": None, "confidence": 0.9},
            {"severity": "major", "type": "accuracy/mistranslation", "target_span": "b", "source_span": None, "confidence": 0.9},
            {"severity": "major", "type": "accuracy/mistranslation", "target_span": "c", "source_span": None, "confidence": 0.9},
            {"severity": "major", "type": "accuracy/mistranslation", "target_span": "d", "source_span": None, "confidence": 0.9},
            {"severity": "major", "type": "accuracy/mistranslation", "target_span": "e", "source_span": None, "confidence": 0.9},
            {"severity": "major", "type": "accuracy/mistranslation", "target_span": "f", "source_span": None, "confidence": 0.9},
        ]
    )

    spans = gemba_mqm_extract_error_spans(raw, mt)

    assert [span["text"] for span in spans] == ["a", "b", "c", "d", "e", "f"]


def test_gemba_mqm_extract_error_spans_sets_error_type() -> None:
    mt = "나는 학교에 갔다."
    raw = _mqm_json_errors(
        [
            {"severity": "major", "type": "accuracy/mistranslation", "target_span": "학교", "source_span": None, "confidence": 0.92},
        ]
    )

    spans = gemba_mqm_extract_error_spans(raw, mt)

    assert len(spans) == 1
    assert spans[0]["error_type"] == "accuracy/mistranslation"
    assert spans[0]["type"] == "accuracy/mistranslation"


def test_gemba_mqm_extract_error_annotations_returns_unanchored_omission() -> None:
    raw = _mqm_json_errors(
        [
            {
                "severity": "major",
                "type": "accuracy/omission",
                "target_span": None,
                "source_span": "행안부 검증",
                "confidence": 0.92,
            }
        ]
    )

    anchored_spans, unanchored_errors = gemba_mqm_extract_error_annotations(raw, "정부 발표")

    assert anchored_spans == []
    assert unanchored_errors == [
        {
            "severity": "MAJOR",
            "source": "mqm",
            "label": 'accuracy/omission - source: "행안부 검증"',
            "error_type": "accuracy/omission",
            "detail_text": "행안부 검증",
        }
    ]


def test_gemba_mqm_extract_error_spans_preserves_old_behavior() -> None:
    raw = _mqm_json_errors(
        [
            {
                "severity": "major",
                "type": "accuracy/omission",
                "target_span": None,
                "source_span": "행안부 검증",
                "confidence": 0.92,
            },
            {
                "severity": "minor",
                "type": "fluency/grammar",
                "target_span": "정부",
                "source_span": None,
                "confidence": 0.76,
            },
        ]
    )

    anchored_spans, unanchored_errors = gemba_mqm_extract_error_annotations(raw, "정부 발표")
    spans = gemba_mqm_extract_error_spans(raw, "정부 발표")

    assert len(anchored_spans) == 1
    assert anchored_spans == spans
    assert len(unanchored_errors) == 1
    assert unanchored_errors[0]["error_type"] == "accuracy/omission"


def test_compute_mqm_unanchored_seq_penalty_filters_allowed_types_and_applies_scale() -> None:
    penalty = _compute_mqm_unanchored_seq_penalty(
        unanchored_errors=[
            {"severity": "MAJOR", "error_type": "accuracy/omission"},
            {"severity": "MINOR", "error_type": "style/awkward"},
        ],
        severity_weights={"MINOR": -1.0, "MAJOR": -5.0, "CRITICAL": -10.0},
        type_weights={"accuracy/omission": 2.0, "style/awkward": 0.5},
        allowed_types=["accuracy"],
        scale=0.75,
    )

    assert penalty == pytest.approx(-7.5)


def test_gemba_mqm_parse_errors_rejects_unstructured_output() -> None:
    with pytest.raises(ValueError, match="structured errors|unparseable"):
        gemba_mqm_parse_errors("The translation looks mostly fine to me.")


def test_gemba_mqm_parse_accepts_unquoted_and_punctuation_only_details() -> None:
    raw = (
        "Critical:\n"
        "no-error\n"
        "Major:\n"
        "terminology/inappropriate for context - 자유 공화당 (Freedom Caucus)\n"
        "Minor:\n"
        "accuracy/omission - 통과시키려는\n"
        'fluency/punctuation - """\n'
    )
    parsed = gemba_mqm_parse_errors(raw)

    assert parsed["critical"] == []
    assert parsed["major"] == ['terminology/inappropriate for context - 자유 공화당 (Freedom Caucus)']
    assert parsed["minor"] == ['accuracy/omission - 통과시키려는', 'fluency/punctuation - """']
    assert gemba_mqm_score(raw) == -7


def test_gemba_mqm_parse_accepts_multiline_quoted_details() -> None:
    raw = (
        "Critical:\n"
        "no-error\n"
        "Major:\n"
        "no-error\n"
        "Minor:\n"
        'accuracy/omission - "천안함"\n'
        'fluency/other - "장관\n'
        '정경두는"\n'
        'fluency/punctuation - "9 일, 천안함"\n'
    )
    parsed = gemba_mqm_parse_errors(raw)

    assert parsed["major"] == []
    assert parsed["minor"] == [
        'accuracy/omission - "천안함"',
        'fluency/other - "장관 정경두는"',
        'fluency/punctuation - "9 일, 천안함"',
    ]
    assert gemba_mqm_score(raw) == -3


def test_gemba_mqm_extract_error_spans_tolerates_unquoted_details_and_unmatched_punctuation() -> None:
    mt = "아버지가 동생이 짓궂게 굴어서 혼냈다."
    raw = (
        "Critical:\n"
        "no-error\n"
        "Major:\n"
        "accuracy/mistranslation - 동생\n"
        "Minor:\n"
        "accuracy/omission - 짓궂게 굴어서\n"
        'fluency/punctuation - """\n'
    )
    spans = gemba_mqm_extract_error_spans(raw, mt)

    assert [span["text"] for span in spans] == ["동생", "짓궂게 굴어서"]
    assert [span["severity"] for span in spans] == ["MAJOR", "MINOR"]


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
            _mqm_json_errors(
                [
                    {
                        "severity": "major",
                        "type": "accuracy/mistranslation",
                        "target_span": "안녕",
                        "source_span": None,
                        "confidence": 0.95,
                    }
                ]
            ),
        ]
    )
    call_count = {"n": 0}

    def _fake_call(messages, max_tokens=None, chat_template_kwargs_override=None):
        call_count["n"] += 1
        return next(calls)

    monkeypatch.setattr(scorer, "_call_openai_compatible_api", _fake_call)

    score, raw_text, spans, unanchored = scorer._score_one_sample(
        sample,
        [{"role": "user", "content": "test"}],
    )

    assert call_count["n"] == 2
    assert score == -5.0
    assert raw_text == _mqm_json_errors(
        [
            {
                "severity": "major",
                "type": "accuracy/mistranslation",
                "target_span": "안녕",
                "source_span": None,
                "confidence": 0.95,
            }
        ]
    )
    assert len(spans) == 1
    assert spans[0]["text"] == "안녕"
    assert unanchored == []


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

    with pytest.raises(GembaParseError, match="unparseable lines"):
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
            _mqm_json_errors(
                [
                    {
                        "severity": "major",
                        "type": "accuracy/mistranslation",
                        "target_span": "안녕",
                        "source_span": None,
                        "confidence": 0.95,
                    }
                ]
            ),
        ]
    )

    monkeypatch.setattr(
        scorer,
        "_call_openai_compatible_api",
        lambda messages, max_tokens=None, chat_template_kwargs_override=None: next(calls),
    )

    score, raw_text, spans, unanchored = scorer._score_one_sample(
        SampleForScoring(src="hello", mt="안녕", ref=None),
        [{"role": "user", "content": "test"}],
    )

    assert score == -5.0
    assert raw_text == _mqm_json_errors(
        [
            {
                "severity": "major",
                "type": "accuracy/mistranslation",
                "target_span": "안녕",
                "source_span": None,
                "confidence": 0.95,
            }
        ]
    )
    assert len(spans) == 1
    assert unanchored == []
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["scorer"] == "mqm"
    assert rows[0]["stage"] == "raw_output_parse_failed"
    assert rows[0]["raw_text"] == "Looks fine overall."
    assert rows[0]["mt"] == "안녕"


def test_openai_mqm_score_batch_refills_inflight_window(monkeypatch: pytest.MonkeyPatch) -> None:
    scorer = OpenAICompatibleMQMScorer(
        cfg=MQMConfig(
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

    def _fake_score_one_sample(
        sample: SampleForScoring,
        messages: list[dict[str, str]],
    ) -> tuple[float, str, list[dict[str, object]], list[dict[str, object]]]:
        assert messages[-1]["role"] == "user"
        if sample.src == "slow":
            slow_started.set()
            assert slow_release.wait(timeout=5.0)
            return (-1.0, "slow", [], [])
        if sample.src == "fast":
            return (-2.0, "fast", [], [])
        third_started_while_slow_blocked["value"] = not slow_release.is_set()
        third_started.set()
        return (-3.0, "third", [], [])

    monkeypatch.setattr(scorer, "_score_one_sample", _fake_score_one_sample)

    def _run() -> None:
        try:
            holder["out"] = scorer.score_batch(samples)
        except Exception as exc:  # pragma: no cover - assertion relay
            holder["exc"] = exc

    thread = threading.Thread(target=_run, name="mqm-bounded-concurrency-test")
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
    assert out.sequence_scores == [-1.0, -2.0, -3.0]
    assert out.metadata["raw_outputs"] == ["slow", "fast", "third"]


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
        return _mqm_json_errors(
            [
                {
                    "severity": "major",
                    "type": "accuracy/mistranslation",
                    "target_span": "안녕",
                    "source_span": None,
                    "confidence": 0.95,
                }
            ]
        )

    monkeypatch.setattr(scorer, "_call_openai_compatible_api", _fake_call)

    score, _, _, _ = scorer._score_one_sample(sample, [{"role": "user", "content": "test"}])

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
        return _mqm_json_errors(
            [
                {
                    "severity": "major",
                    "type": "accuracy/mistranslation",
                    "target_span": "안녕",
                    "source_span": None,
                    "confidence": 0.95,
                }
            ]
        )

    monkeypatch.setattr(scorer, "_call_openai_compatible_api", _fake_call)

    score, _, _, _ = scorer._score_one_sample(sample, [{"role": "user", "content": "test"}])

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
        lambda messages, max_tokens=None, chat_template_kwargs_override=None: _mqm_json_errors(
            [
                {
                    "severity": "major",
                    "type": "accuracy/mistranslation",
                    "target_span": "hello",
                    "source_span": None,
                    "confidence": 0.82,
                }
            ]
        ),
    )

    score, raw_text, spans, unanchored = scorer._score_one_sample(
        SampleForScoring(src="hello", mt="안녕", ref=None),
        [{"role": "user", "content": "test"}],
    )

    assert score == -5.0
    assert raw_text == _mqm_json_errors(
        [
            {
                "severity": "major",
                "type": "accuracy/mistranslation",
                "target_span": "hello",
                "source_span": None,
                "confidence": 0.82,
            }
        ]
    )
    assert spans == []
    assert len(unanchored) == 1
    assert unanchored[0]["error_type"] == "accuracy/mistranslation"
    assert unanchored[0]["detail_text"] == "hello"


def test_openai_mqm_score_batch_returns_unanchored_errors_metadata(
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
        lambda messages, max_tokens=None, chat_template_kwargs_override=None: _mqm_json_errors(
            [
                {
                    "severity": "major",
                    "type": "accuracy/omission",
                    "target_span": None,
                    "source_span": "행안부 검증",
                    "confidence": 0.95,
                }
            ]
        ),
    )

    out = scorer.score_batch([SampleForScoring(src="hello", mt="안녕", ref=None)])

    assert len(out.metadata["unanchored_errors"]) == 1
    assert out.metadata["error_spans"] == [[]]
    assert out.metadata["unanchored_errors"][0][0]["error_type"] == "accuracy/omission"


def test_openai_mqm_score_batch_failure_policy_neutral_zero_sets_failure_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = OpenAICompatibleMQMScorer(
        cfg=MQMConfig(
            enabled=True,
            base_url="http://localhost:8000/v1",
            max_retries=0,
            failure_policy="neutral_zero",
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
    assert out.metadata["failure_rows"] == [True]


def test_openai_mqm_score_batch_failure_policy_worst_score_uses_score_min(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = OpenAICompatibleMQMScorer(
        cfg=MQMConfig(
            enabled=True,
            base_url="http://localhost:8000/v1",
            max_retries=0,
            failure_policy="worst_score",
            score_min=-25.0,
        )
    )
    monkeypatch.setattr(
        scorer,
        "_call_openai_compatible_api",
        lambda messages, max_tokens=None, chat_template_kwargs_override=None: "bad",
    )

    out = scorer.score_batch([SampleForScoring(src="hello", mt="안녕", ref=None)])

    assert out.sequence_scores == [-25.0]
    assert out.metadata["failure_rows"] == [True]


def test_openai_mqm_score_batch_failure_policy_raise_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    scorer = OpenAICompatibleMQMScorer(
        cfg=MQMConfig(
            enabled=True,
            base_url="http://localhost:8000/v1",
            max_retries=0,
            failure_policy="raise",
        )
    )
    monkeypatch.setattr(
        scorer,
        "_call_openai_compatible_api",
        lambda messages, max_tokens=None, chat_template_kwargs_override=None: "bad",
    )

    with pytest.raises(GembaParseError, match="unparseable lines"):
        _ = scorer.score_batch([SampleForScoring(src="hello", mt="안녕", ref=None)])


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

    monkeypatch.setenv("HTTP_PROXY", "http://proxy.local:8080")
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
    assert os.environ["HTTP_PROXY"] == "http://proxy.local:8080"


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
