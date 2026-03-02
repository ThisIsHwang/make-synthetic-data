from __future__ import annotations

import pytest

import re

from distributed_rl.config import RemoteLLMRewardConfig
from distributed_rl.rewards.base import (
    RewardModel,
    RewardOutput,
    SampleForScoring,
    make_trl_reward_func,
)
from distributed_rl.rewards.metricx import metricx_qe_input, metricx_ref_input, metricx_score_to_reward
from distributed_rl.rewards.remote_llm import RemoteLLMReward


class TestSampleForScoring:
    def test_basic_creation(self):
        s = SampleForScoring(src="hello", mt="world")
        assert s.src == "hello"
        assert s.mt == "world"
        assert s.ref is None

    def test_with_ref(self):
        s = SampleForScoring(src="hello", mt="world", ref="welt")
        assert s.ref == "welt"


class TestMetricXHelpers:
    def test_qe_input(self):
        result = metricx_qe_input("src text", "mt text")
        assert result == "source: src text candidate: mt text"

    def test_ref_input(self):
        result = metricx_ref_input("src", "mt", "ref")
        assert result == "source: src candidate: mt reference: ref"

    def test_score_to_reward(self):
        assert metricx_score_to_reward(3.25, offset=5.0) == 1.75
        assert metricx_score_to_reward(0.0, offset=5.0) == 5.0
        assert metricx_score_to_reward(5.0, offset=5.0) == 0.0


class _MockReward(RewardModel):
    """Mock reward model for testing."""

    def __init__(self, fixed_scores: list[float]):
        self._scores = fixed_scores

    def score(self, samples: list[SampleForScoring]) -> RewardOutput:
        scores = self._scores[:len(samples)]
        return RewardOutput(sequence_scores=scores)


class TestMakeTrlRewardFunc:
    def test_basic_wrapping(self):
        model = _MockReward([1.0, 2.0, 3.0])
        func = make_trl_reward_func(model, weight=1.0, offset=0.0)
        result = func(
            completions=["a", "b", "c"],
            src_text=["s1", "s2", "s3"],
        )
        assert result == [1.0, 2.0, 3.0]

    def test_with_weight_and_offset(self):
        model = _MockReward([3.0, 4.0])
        func = make_trl_reward_func(model, weight=1.0, offset=5.0)
        # reward = weight * (offset - raw_score) = 1.0 * (5.0 - 3.0) = 2.0
        result = func(completions=["a", "b"], src_text=["s1", "s2"])
        assert result == [2.0, 1.0]

    def test_with_custom_weight(self):
        model = _MockReward([1.0])
        func = make_trl_reward_func(model, weight=2.0, offset=0.0)
        result = func(completions=["a"], src_text=["s1"])
        assert result == [2.0]

    def test_missing_src_column_uses_empty(self):
        model = _MockReward([1.0])
        func = make_trl_reward_func(model, weight=1.0, offset=0.0)
        # No src_text in kwargs → defaults to empty strings
        result = func(completions=["a"])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# RemoteLLMReward tests
# ---------------------------------------------------------------------------

class TestRemoteLLMReward:
    """Tests for RemoteLLMReward using predict_fn injection (no API calls)."""

    def test_judge_mode_basic(self):
        cfg = RemoteLLMRewardConfig(enabled=True, mode="judge")
        model = RemoteLLMReward(cfg, predict_fn=lambda samples: [0.85, 0.6])
        result = model.score([
            SampleForScoring(src="hello", mt="안녕하세요"),
            SampleForScoring(src="world", mt="세계"),
        ])
        assert result.sequence_scores == [0.85, 0.6]

    def test_scoring_mode_basic(self):
        cfg = RemoteLLMRewardConfig(enabled=True, mode="scoring")
        model = RemoteLLMReward(cfg, predict_fn=lambda samples: [0.9])
        result = model.score([SampleForScoring(src="hi", mt="안녕")])
        assert len(result.sequence_scores) == 1
        assert result.sequence_scores[0] == 0.9

    def test_empty_samples(self):
        cfg = RemoteLLMRewardConfig(enabled=True)
        model = RemoteLLMReward(cfg, predict_fn=lambda s: [])
        result = model.score([])
        assert result.sequence_scores == []

    def test_disabled_model(self):
        cfg = RemoteLLMRewardConfig(enabled=False)
        # Should not raise even without openai installed.
        model = RemoteLLMReward(cfg)
        assert model._client is None

    def test_multiple_samples(self):
        scores = [0.1, 0.5, 0.9, 0.3]
        cfg = RemoteLLMRewardConfig(enabled=True, mode="judge")
        model = RemoteLLMReward(cfg, predict_fn=lambda samples: scores[:len(samples)])
        samples = [
            SampleForScoring(src=f"src{i}", mt=f"mt{i}") for i in range(4)
        ]
        result = model.score(samples)
        assert result.sequence_scores == scores

    def test_with_trl_reward_func(self):
        cfg = RemoteLLMRewardConfig(enabled=True, mode="judge")
        model = RemoteLLMReward(cfg, predict_fn=lambda samples: [0.8, 0.5])
        func = make_trl_reward_func(model, weight=2.0, offset=0.0)
        result = func(
            completions=["안녕하세요", "세계"],
            src_text=["hello", "world"],
        )
        assert result == [1.6, 1.0]  # weight=2.0 * [0.8, 0.5]


class TestRemoteLLMScoreParsing:
    """Tests for score parsing and normalization (no API calls)."""

    def _make_model(self, **kwargs):
        """Helper to create a RemoteLLMReward with score parsing ready."""
        defaults = {"enabled": True, "mode": "judge"}
        defaults.update(kwargs)
        cfg = RemoteLLMRewardConfig(**defaults)
        model = RemoteLLMReward(cfg, predict_fn=lambda s: [])
        model._score_pattern = re.compile(cfg.score_regex)
        return model

    # --- _parse_score tests ---

    def test_parse_score_standard(self):
        model = self._make_model()
        assert model._parse_score("Good translation. Score: 8.5") == 8.5

    def test_parse_score_integer(self):
        model = self._make_model()
        assert model._parse_score("Score: 10") == 10.0

    def test_parse_score_equals(self):
        model = self._make_model()
        assert model._parse_score("score=7") == 7.0

    def test_parse_score_fraction_fallback(self):
        model = self._make_model()
        # Default regex won't match "8/10", but fallback pattern will.
        assert model._parse_score("I rate this 8/10") == 8.0

    def test_parse_score_last_number_fallback(self):
        model = self._make_model()
        # No "Score:" prefix, no "/10" → falls back to last number in range.
        assert model._parse_score("The quality is 7") == 7.0

    def test_parse_score_fails(self):
        model = self._make_model()
        with pytest.raises(ValueError, match="Could not parse score"):
            model._parse_score("No numbers here at all.")

    def test_parse_score_out_of_range_rejected(self):
        model = self._make_model(score_min=0.0, score_max=10.0)
        # "500" is outside [0, 10], so the last-number fallback skips it.
        with pytest.raises(ValueError, match="Could not parse score"):
            model._parse_score("Something something 500")

    # --- _normalize_score tests ---

    def test_normalize_zero(self):
        model = self._make_model(score_min=0.0, score_max=10.0)
        assert model._normalize_score(0.0) == 0.0

    def test_normalize_max(self):
        model = self._make_model(score_min=0.0, score_max=10.0)
        assert model._normalize_score(10.0) == 1.0

    def test_normalize_mid(self):
        model = self._make_model(score_min=0.0, score_max=10.0)
        assert model._normalize_score(5.0) == 0.5

    def test_normalize_clamp_above(self):
        model = self._make_model(score_min=0.0, score_max=10.0)
        assert model._normalize_score(15.0) == 1.0

    def test_normalize_clamp_below(self):
        model = self._make_model(score_min=0.0, score_max=10.0)
        assert model._normalize_score(-2.0) == 0.0

    def test_normalize_custom_range(self):
        model = self._make_model(score_min=1.0, score_max=5.0)
        assert model._normalize_score(3.0) == 0.5
