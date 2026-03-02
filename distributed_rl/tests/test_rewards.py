from __future__ import annotations

import pytest

from distributed_rl.rewards.base import (
    RewardModel,
    RewardOutput,
    SampleForScoring,
    make_trl_reward_func,
)
from distributed_rl.rewards.metricx import metricx_qe_input, metricx_ref_input, metricx_score_to_reward


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
