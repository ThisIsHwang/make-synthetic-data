from __future__ import annotations

import pytest

pytest.importorskip("torch")

from gemma27_rl.config import RLPostTrainConfig
from gemma27_rl import trainer as trainer_mod
from gemma27_rl.trainer import _validate_deepspeed_partition_strict


def _base_cfg() -> RLPostTrainConfig:
    cfg = RLPostTrainConfig()
    cfg.rl.backend = "deepspeed"
    cfg.model.policy_gpu_ids = [0]
    cfg.model.reference_gpu_ids = []
    cfg.reward.metricx.enabled = False
    cfg.reward.xcomet.enabled = False
    return cfg


def test_deepspeed_partition_world_size_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _base_cfg()
    cfg.model.policy_gpu_ids = [0, 1]
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    with pytest.raises(RuntimeError, match="WORLD_SIZE == len\\(model.policy_gpu_ids\\)"):
        _validate_deepspeed_partition_strict(cfg)


def test_deepspeed_partition_mapping_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _base_cfg()
    cfg.model.policy_gpu_ids = [0]
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(RuntimeError, match="partition mismatch"):
        _validate_deepspeed_partition_strict(cfg)


def test_deepspeed_partition_reserved_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _base_cfg()
    cfg.model.policy_gpu_ids = [0]
    cfg.model.reference_gpu_ids = [0]
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    with pytest.raises(RuntimeError, match="must not overlap reserved"):
        _validate_deepspeed_partition_strict(cfg)


def test_assign_disjoint_keeps_physical_reserved_ids_under_deepspeed_include(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _base_cfg()
    cfg.model.policy_gpu_ids = [0, 1, 2, 3, 4, 5]
    cfg.model.reference_gpu_ids = [6]
    cfg.reward.metricx.enabled = True
    cfg.reward.metricx.device = "cuda:7"
    cfg.reward.xcomet.enabled = False
    cfg.misc.device = "cuda:0"

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5")
    monkeypatch.setattr(trainer_mod.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(trainer_mod.torch.cuda, "device_count", lambda: 6)

    trainer_mod._assign_disjoint_gpu_devices(cfg)

    assert cfg.model.policy_gpu_ids == [0, 1, 2, 3, 4, 5]
    assert cfg.model.reference_gpu_ids == [6]
    assert cfg.reward.metricx.device == "cuda:7"
