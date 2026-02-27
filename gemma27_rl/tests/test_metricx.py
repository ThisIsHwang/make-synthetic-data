from __future__ import annotations

import pytest

from gemma27_rl.config import MetricXConfig
from gemma27_rl.rewards import MetricXQEScorer, metricx_qe_input, metricx_score_to_reward
from gemma27_rl.rl_types import SampleForScoring


def test_metricx_qe_input_format() -> None:
    assert metricx_qe_input("src", "mt") == "source: src candidate: mt"


def test_metricx_reward_conversion() -> None:
    assert metricx_score_to_reward(3.25, offset=5.0) == pytest.approx(1.75)


def test_metricx_scorer_predict_fn_path() -> None:
    captured: list[str] = []

    def fake_predict(inputs: list[str]) -> list[float]:
        captured.extend(inputs)
        return [1.5 for _ in inputs]

    scorer = MetricXQEScorer(cfg=MetricXConfig(enabled=True), predict_fn=fake_predict)
    out = scorer.score_batch([SampleForScoring(src="hello", mt="world", ref=None)])

    assert captured == ["source: hello candidate: world"]
    assert out.sequence_scores == [1.5]


def test_metricx_default_tokenizer_name() -> None:
    cfg = MetricXConfig()
    assert cfg.tokenizer_name == "google/mt5-xl"
