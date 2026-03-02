from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from distributed_rl.config import (
    DistributedRLConfig,
    _coerce_dataclass,
    load_config,
)


class TestCoerceDataclass:
    def test_empty_payload_returns_defaults(self):
        cfg = _coerce_dataclass(DistributedRLConfig, {})
        assert cfg.model.policy_name_or_path == "Qwen/Qwen3.5-35B-A3B"
        assert cfg.training.loss_type == "grpo"
        assert cfg.misc.seed == 42

    def test_partial_override(self):
        payload = {
            "model": {"policy_name_or_path": "my-model"},
            "training": {"loss_type": "dapo", "lr": 5e-7},
        }
        cfg = _coerce_dataclass(DistributedRLConfig, payload)
        assert cfg.model.policy_name_or_path == "my-model"
        assert cfg.training.loss_type == "dapo"
        assert cfg.training.lr == 5e-7
        # Defaults preserved
        assert cfg.training.epsilon == 0.2
        assert cfg.misc.dtype == "bfloat16"

    def test_nested_reward_config(self):
        payload = {
            "reward": {
                "w_metricx": 2.0,
                "metricx": {"model_name": "custom-model", "gpu_id": 3},
            },
        }
        cfg = _coerce_dataclass(DistributedRLConfig, payload)
        assert cfg.reward.w_metricx == 2.0
        assert cfg.reward.metricx.model_name == "custom-model"
        assert cfg.reward.metricx.gpu_id == 3
        # XComet defaults
        assert cfg.reward.xcomet.enabled is True


class TestLoadConfig:
    def test_load_from_yaml(self, tmp_path: Path):
        config_data = {
            "model": {"policy_name_or_path": "Qwen/Qwen2-0.5B-Instruct"},
            "data": {"hf_dataset_name": "google/wmt24pp", "hf_dataset_config_name": "en-ko_KR"},
            "training": {"loss_type": "grpo", "max_steps": 10},
            "reward": {"metricx": {"enabled": True}},
        }
        cfg_path = tmp_path / "test_config.yaml"
        cfg_path.write_text(yaml.safe_dump(config_data))

        cfg = load_config(cfg_path)
        assert cfg.model.policy_name_or_path == "Qwen/Qwen2-0.5B-Instruct"
        assert cfg.training.max_steps == 10

    def test_validation_rejects_invalid_loss_type(self, tmp_path: Path):
        config_data = {
            "data": {"hf_dataset_name": "test"},
            "training": {"loss_type": "invalid"},
            "reward": {"metricx": {"enabled": True}},
        }
        cfg_path = tmp_path / "bad_config.yaml"
        cfg_path.write_text(yaml.safe_dump(config_data))

        with pytest.raises(ValueError, match="loss_type"):
            load_config(cfg_path)

    def test_validation_rejects_no_reward_models(self, tmp_path: Path):
        config_data = {
            "data": {"hf_dataset_name": "test"},
            "reward": {
                "metricx": {"enabled": False},
                "xcomet": {"enabled": False},
            },
        }
        cfg_path = tmp_path / "no_reward.yaml"
        cfg_path.write_text(yaml.safe_dump(config_data))

        with pytest.raises(ValueError, match="At least one reward"):
            load_config(cfg_path)

    def test_validation_rejects_same_gpu_ids(self, tmp_path: Path):
        config_data = {
            "data": {"hf_dataset_name": "test"},
            "reward": {
                "metricx": {"enabled": True, "gpu_id": 6},
                "xcomet": {"enabled": True, "gpu_id": 6},
            },
        }
        cfg_path = tmp_path / "same_gpu.yaml"
        cfg_path.write_text(yaml.safe_dump(config_data))

        with pytest.raises(ValueError, match="different GPUs"):
            load_config(cfg_path)


class TestRemoteLLMRewardConfig:
    def test_validation_rejects_invalid_mode(self, tmp_path: Path):
        config_data = {
            "data": {"hf_dataset_name": "test"},
            "reward": {
                "metricx": {"enabled": False},
                "xcomet": {"enabled": False},
                "remote_llm": {"enabled": True, "mode": "invalid"},
            },
        }
        cfg_path = tmp_path / "bad_remote.yaml"
        cfg_path.write_text(yaml.safe_dump(config_data))
        with pytest.raises(ValueError, match="mode"):
            load_config(cfg_path)

    def test_remote_llm_as_only_reward(self, tmp_path: Path):
        """Remote LLM alone should pass 'at least one reward' validation."""
        config_data = {
            "data": {"hf_dataset_name": "test"},
            "reward": {
                "metricx": {"enabled": False},
                "xcomet": {"enabled": False},
                "remote_llm": {
                    "enabled": True,
                    "mode": "judge",
                    "base_url": "http://localhost:8000/v1",
                },
            },
        }
        cfg_path = tmp_path / "remote_only.yaml"
        cfg_path.write_text(yaml.safe_dump(config_data))
        cfg = load_config(cfg_path)
        assert cfg.reward.remote_llm.enabled is True

    def test_validation_rejects_bad_score_range(self, tmp_path: Path):
        config_data = {
            "data": {"hf_dataset_name": "test"},
            "reward": {
                "metricx": {"enabled": False},
                "xcomet": {"enabled": False},
                "remote_llm": {
                    "enabled": True,
                    "score_min": 10.0,
                    "score_max": 5.0,
                },
            },
        }
        cfg_path = tmp_path / "bad_range.yaml"
        cfg_path.write_text(yaml.safe_dump(config_data))
        with pytest.raises(ValueError, match="score_max"):
            load_config(cfg_path)

    def test_default_disabled(self):
        """RemoteLLMRewardConfig is disabled by default."""
        cfg = _coerce_dataclass(DistributedRLConfig, {})
        assert cfg.reward.remote_llm.enabled is False
        assert cfg.reward.w_remote_llm == 0.0


class TestExistingConfigs:
    def test_toy_config_loads(self, configs_dir: Path):
        cfg_path = configs_dir / "train_toy.yaml"
        if cfg_path.exists():
            cfg = load_config(cfg_path)
            assert cfg.training.loss_type == "grpo"
            assert cfg.training.max_steps == 10
