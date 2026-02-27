from __future__ import annotations

import pytest

from qwen35_moe_rl.config import (
    DistributedConfig,
    RLConfig,
    RLPostTrainConfig,
)


def test_distributed_config_defaults() -> None:
    cfg = DistributedConfig()
    assert cfg.enabled is False
    assert cfg.backend == "deepspeed"
    assert cfg.zero_stage == 2
    assert cfg.offload_optimizer is False
    assert cfg.offload_param is False
    assert cfg.deepspeed_config_path is None
    assert cfg.fsdp_sharding_strategy == "FULL_SHARD"


def test_distributed_config_in_rl_post_train() -> None:
    cfg = RLPostTrainConfig()
    assert hasattr(cfg, "distributed")
    assert isinstance(cfg.distributed, DistributedConfig)
    assert cfg.distributed.enabled is False


def test_rl_config_algorithm_gspo_default() -> None:
    cfg = RLConfig()
    assert cfg.algorithm == "gspo"
    assert cfg.output_router_logits is True
    assert cfg.router_aux_loss_coef == pytest.approx(0.001)
    assert cfg.monitor_expert_utilization is True
