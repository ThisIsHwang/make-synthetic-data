from __future__ import annotations

import os

import pytest

pytest.importorskip("torch")

from gemma27_rl.config import RLPostTrainConfig
from gemma27_rl import trainer as trainer_mod
from gemma27_rl.trainer import _configure_nccl_heartbeat_timeout, _validate_deepspeed_partition_strict


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
    with pytest.raises(RuntimeError, match="WORLD_SIZE to be a multiple"):
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


def test_deepspeed_partition_remote_reference_overlap_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _base_cfg()
    cfg.model.policy_gpu_ids = [0]
    cfg.model.reference_gpu_ids = [0]
    cfg.model.reference_worker_host = "aux-node-1"
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    _validate_deepspeed_partition_strict(cfg)


def test_deepspeed_partition_colocated_reference_overlap_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _base_cfg()
    cfg.model.policy_gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    cfg.model.reference_gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    cfg.model.reference_runtime = "colocate"
    cfg.model.policy_runtime_mode = "colocate"
    cfg.model.lora.enabled = True
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
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


def test_configure_nccl_heartbeat_timeout_sets_default(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _base_cfg()
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.delenv("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", raising=False)
    _configure_nccl_heartbeat_timeout(cfg)
    assert os.environ.get("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC") == "7200"


def test_configure_nccl_heartbeat_timeout_keeps_user_value(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _base_cfg()
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "900")
    _configure_nccl_heartbeat_timeout(cfg)
    assert os.environ.get("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC") == "900"


def test_configure_nccl_heartbeat_timeout_ignored_for_native(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _base_cfg()
    cfg.rl.backend = "native"
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.delenv("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", raising=False)
    _configure_nccl_heartbeat_timeout(cfg)
    assert os.environ.get("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC") is None


def test_save_deepspeed_checkpoint_wrapper_passes_hf_model(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def _fake_save(**kwargs):
        captured.update(kwargs)
        return kwargs["ckpt_dir"]

    monkeypatch.setattr(trainer_mod, "_save_deepspeed_checkpoint_to_dir", _fake_save)

    hf_model = object()
    result = trainer_mod._save_deepspeed_checkpoint(
        output_dir=tmp_path,
        update_idx=7,
        engine=object(),
        tokenizer=object(),
        hf_model=hf_model,
        trainer_state={"update_idx": 7},
    )

    assert result == tmp_path / "checkpoint-7"
    assert captured["hf_model"] is hf_model


def test_save_deepspeed_resume_checkpoint_wrapper_passes_hf_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    captured: dict[str, object] = {}

    def _fake_save(**kwargs):
        captured.update(kwargs)
        return kwargs["ckpt_dir"]

    monkeypatch.setattr(trainer_mod, "_save_deepspeed_checkpoint_to_dir", _fake_save)

    hf_model = object()
    result = trainer_mod._save_deepspeed_resume_checkpoint(
        output_dir=tmp_path,
        update_idx=9,
        engine=object(),
        tokenizer=object(),
        hf_model=hf_model,
        trainer_state={"best_eval_update": 8},
    )

    assert result == tmp_path / "resume_latest"
    assert captured["hf_model"] is hf_model
    trainer_state = captured["trainer_state"]
    assert isinstance(trainer_state, dict)
    assert trainer_state["update_idx"] == 9
