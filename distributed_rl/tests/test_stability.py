"""Tests for MiniRL stability features (arxiv 2512.01374).

Tests cover:
  - StabilityConfig defaults and YAML parsing
  - StabilityMonitorCallback (entropy floor, KL ceiling, trends, halt)
  - Reward outlier clipping in make_trl_reward_func()
  - Asymmetric epsilon passthrough
  - Backward compatibility (no stability section in YAML)
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from distributed_rl.config import (
    DistributedRLConfig,
    StabilityConfig,
    _coerce_dataclass,
    load_config,
)
from distributed_rl.rewards.base import (
    RewardModel,
    RewardOutput,
    SampleForScoring,
    make_trl_reward_func,
)
from distributed_rl.stability import StabilityMonitorCallback, StabilityState


# ---------------------------------------------------------------------------
# StabilityConfig tests
# ---------------------------------------------------------------------------


class TestStabilityConfig:
    def test_defaults_all_disabled(self):
        cfg = StabilityConfig()
        assert cfg.enable_monitoring is False
        assert cfg.entropy_floor == 0.0
        assert cfg.kl_ceiling == 0.0
        assert cfg.halt_on_collapse_steps == 0
        assert cfg.reward_clip_value == 0.0
        assert cfg.epsilon_high == 0.0
        assert cfg.epsilon_low == 0.0
        assert cfg.routing_replay == "none"
        assert cfg.monitor_router_entropy is False

    def test_coerce_from_dict(self):
        payload = {
            "stability": {
                "enable_monitoring": True,
                "entropy_floor": 0.5,
                "kl_ceiling": 10.0,
                "halt_on_collapse_steps": 20,
                "reward_clip_value": 8.0,
                "epsilon_high": 0.27,
                "epsilon_low": 0.2,
            }
        }
        cfg = _coerce_dataclass(DistributedRLConfig, payload)
        assert cfg.stability.enable_monitoring is True
        assert cfg.stability.entropy_floor == 0.5
        assert cfg.stability.kl_ceiling == 10.0
        assert cfg.stability.halt_on_collapse_steps == 20
        assert cfg.stability.reward_clip_value == 8.0
        assert cfg.stability.epsilon_high == 0.27
        assert cfg.stability.epsilon_low == 0.2

    def test_backward_compat_no_stability_section(self):
        payload = {
            "model": {"policy_name_or_path": "test-model"},
            "training": {"loss_type": "grpo"},
        }
        cfg = _coerce_dataclass(DistributedRLConfig, payload)
        # Stability defaults should be preserved.
        assert cfg.stability.enable_monitoring is False
        assert cfg.stability.reward_clip_value == 0.0

    def test_yaml_roundtrip(self, tmp_path: Path):
        config_data = {
            "data": {"hf_dataset_name": "test"},
            "reward": {"metricx": {"enabled": True}},
            "stability": {
                "enable_monitoring": True,
                "entropy_floor": 0.3,
                "epsilon_high": 0.27,
            },
        }
        cfg_path = tmp_path / "stability_test.yaml"
        cfg_path.write_text(yaml.safe_dump(config_data))

        cfg = load_config(cfg_path)
        assert cfg.stability.enable_monitoring is True
        assert cfg.stability.entropy_floor == 0.3
        assert cfg.stability.epsilon_high == 0.27
        # Defaults preserved for unset fields.
        assert cfg.stability.epsilon_low == 0.0


class TestStabilityValidation:
    def _make_config(self, tmp_path: Path, stability: dict) -> Path:
        config_data = {
            "data": {"hf_dataset_name": "test"},
            "reward": {"metricx": {"enabled": True}},
            "stability": stability,
        }
        cfg_path = tmp_path / "val_test.yaml"
        cfg_path.write_text(yaml.safe_dump(config_data))
        return cfg_path

    def test_rejects_invalid_routing_replay(self, tmp_path: Path):
        cfg_path = self._make_config(tmp_path, {"routing_replay": "invalid"})
        with pytest.raises(ValueError, match="routing_replay"):
            load_config(cfg_path)

    def test_rejects_routing_without_router_logits(self, tmp_path: Path):
        cfg_path = self._make_config(tmp_path, {"routing_replay": "r2"})
        with pytest.raises(ValueError, match="output_router_logits"):
            load_config(cfg_path)

    def test_rejects_negative_epsilon(self, tmp_path: Path):
        cfg_path = self._make_config(tmp_path, {"epsilon_high": -0.1})
        with pytest.raises(ValueError, match="epsilon"):
            load_config(cfg_path)

    def test_rejects_negative_clip_value(self, tmp_path: Path):
        cfg_path = self._make_config(tmp_path, {"reward_clip_value": -5.0})
        with pytest.raises(ValueError, match="reward_clip_value"):
            load_config(cfg_path)


# ---------------------------------------------------------------------------
# StabilityMonitorCallback tests
# ---------------------------------------------------------------------------


class TestStabilityMonitorCallback:
    def _make_callback(self, **overrides) -> StabilityMonitorCallback:
        cfg = StabilityConfig(enable_monitoring=True, **overrides)
        return StabilityMonitorCallback(cfg)

    def _mock_args_state_control(self):
        args = MagicMock()
        state = MagicMock()
        state.global_step = 1
        control = MagicMock()
        control.should_training_stop = False
        return args, state, control

    def test_entropy_floor_warning(self):
        cb = self._make_callback(entropy_floor=1.0)
        args, state, control = self._mock_args_state_control()

        # Entropy below floor.
        logs = {"entropy": 0.3}
        cb.on_log(args, state, control, logs=logs)
        assert cb.state.consecutive_below_entropy_floor == 1

        # Entropy above floor resets counter.
        logs = {"entropy": 1.5}
        cb.on_log(args, state, control, logs=logs)
        assert cb.state.consecutive_below_entropy_floor == 0

    def test_kl_ceiling_warning(self):
        cb = self._make_callback(kl_ceiling=5.0)
        args, state, control = self._mock_args_state_control()

        logs = {"kl": 8.0}
        cb.on_log(args, state, control, logs=logs)
        assert cb.state.consecutive_above_kl_ceiling == 1

        logs = {"kl": 3.0}
        cb.on_log(args, state, control, logs=logs)
        assert cb.state.consecutive_above_kl_ceiling == 0

    def test_halt_on_collapse(self):
        cb = self._make_callback(entropy_floor=1.0, halt_on_collapse_steps=3)
        args, state, control = self._mock_args_state_control()

        # Simulate 3 consecutive steps below floor.
        for _ in range(3):
            cb.on_log(args, state, control, logs={"entropy": 0.1})

        assert cb.state.consecutive_below_entropy_floor == 3

        # on_step_end should trigger halt.
        cb.on_step_end(args, state, control)
        assert control.should_training_stop is True

    def test_no_halt_when_disabled(self):
        cb = self._make_callback(entropy_floor=1.0, halt_on_collapse_steps=0)
        args, state, control = self._mock_args_state_control()

        for _ in range(10):
            cb.on_log(args, state, control, logs={"entropy": 0.1})

        cb.on_step_end(args, state, control)
        # halt_on_collapse_steps=0 means no halt.
        assert control.should_training_stop is False

    def test_trend_analysis(self):
        cb = self._make_callback()
        args, state, control = self._mock_args_state_control()

        # Feed 10 entropy values: 5 high then 5 low → negative trend.
        for val in [2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0]:
            logs: dict = {"entropy": val}
            cb.on_log(args, state, control, logs=logs)

        # The last call should have added entropy_trend.
        assert "stability/entropy_trend" in logs
        assert logs["stability/entropy_trend"] < 0  # Declining entropy.

    def test_monitoring_interval(self):
        cb = self._make_callback(monitoring_interval=5)
        args, state, control = self._mock_args_state_control()

        # Step 1: not on interval, should skip.
        state.global_step = 1
        logs = {"entropy": 0.5}
        cb.on_log(args, state, control, logs=logs)
        assert len(cb.state.entropy_history) == 0

        # Step 5: on interval, should process.
        state.global_step = 5
        logs = {"entropy": 0.5}
        cb.on_log(args, state, control, logs=logs)
        assert len(cb.state.entropy_history) == 1

    def test_none_logs_safe(self):
        cb = self._make_callback()
        args, state, control = self._mock_args_state_control()
        # Should not raise.
        cb.on_log(args, state, control, logs=None)


# ---------------------------------------------------------------------------
# Reward clipping tests
# ---------------------------------------------------------------------------


class _FakeRewardModel(RewardModel):
    """Reward model that returns predetermined scores for testing."""

    def __init__(self, scores: list[float]):
        self._scores = scores

    def score(self, samples: list[SampleForScoring]) -> RewardOutput:
        return RewardOutput(sequence_scores=self._scores[: len(samples)])


class TestRewardClipping:
    def test_no_clipping_by_default(self):
        model = _FakeRewardModel([0.5, 10.0, 25.0])
        func = make_trl_reward_func(model, offset=5.0, weight=1.0)
        rewards = func(["a", "b", "c"], src_text=["x", "x", "x"])
        # reward = 1*(5-score): 4.5, -5.0, -20.0
        assert rewards == pytest.approx([4.5, -5.0, -20.0])

    def test_clipping_bounds_rewards(self):
        model = _FakeRewardModel([0.5, 10.0, 25.0])
        func = make_trl_reward_func(model, offset=5.0, weight=1.0, clip_value=5.0)
        rewards = func(["a", "b", "c"], src_text=["x", "x", "x"])
        # Without clip: 4.5, -5.0, -20.0
        # With clip_value=5.0: 4.5, -5.0, -5.0
        assert rewards == pytest.approx([4.5, -5.0, -5.0])

    def test_clipping_both_directions(self):
        model = _FakeRewardModel([0.0, 25.0])
        func = make_trl_reward_func(model, offset=5.0, weight=1.0, clip_value=3.0)
        rewards = func(["a", "b"], src_text=["x", "x"])
        # Without clip: 5.0, -20.0
        # With clip_value=3.0: 3.0, -3.0
        assert rewards == pytest.approx([3.0, -3.0])

    def test_clipping_zero_disables(self):
        model = _FakeRewardModel([0.0])
        func = make_trl_reward_func(model, offset=5.0, weight=1.0, clip_value=0.0)
        rewards = func(["a"], src_text=["x"])
        assert rewards == pytest.approx([5.0])

    def test_clipping_with_xcomet_style(self):
        model = _FakeRewardModel([0.95, 0.1])
        func = make_trl_reward_func(model, offset=0.0, weight=1.0, clip_value=0.5)
        rewards = func(["a", "b"], src_text=["x", "x"])
        # Without clip: 0.95, 0.1
        # With clip_value=0.5: 0.5, 0.1
        assert rewards == pytest.approx([0.5, 0.1])
