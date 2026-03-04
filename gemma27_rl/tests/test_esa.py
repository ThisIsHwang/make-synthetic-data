from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from gemma27_rl.config import ESAConfig, load_config
from gemma27_rl.rewards import (
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
