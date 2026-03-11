from __future__ import annotations

import datetime
from contextlib import nullcontext
import os
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from gemma27_rl.config import RLPostTrainConfig
from gemma27_rl import trainer as trainer_mod
from gemma27_rl.trainer import (
    _build_zero3_peft_state_dict,
    _configure_nccl_heartbeat_timeout,
    _init_deepspeed_distributed,
    _is_deepspeed_resume_shard_mismatch_error,
    _register_qwen35_zero3_external_parameters,
    _load_policy_model,
    _validate_deepspeed_partition_strict,
)


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


def test_init_deepspeed_distributed_uses_safe_default_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _base_cfg()
    captured: dict[str, object] = {}

    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.delenv("TORCH_DISTRIBUTED_TIMEOUT_SEC", raising=False)
    monkeypatch.setattr(trainer_mod, "deepspeed", SimpleNamespace(init_distributed=lambda **kwargs: captured.update(kwargs)))
    monkeypatch.setattr(trainer_mod, "_is_distributed_initialized", lambda: False)

    _init_deepspeed_distributed(cfg)

    assert captured["timeout"] == datetime.timedelta(seconds=7200)


def test_init_deepspeed_distributed_respects_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _base_cfg()
    captured: dict[str, object] = {}

    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("TORCH_DISTRIBUTED_TIMEOUT_SEC", "5400")
    monkeypatch.setattr(trainer_mod, "deepspeed", SimpleNamespace(init_distributed=lambda **kwargs: captured.update(kwargs)))
    monkeypatch.setattr(trainer_mod, "_is_distributed_initialized", lambda: False)

    _init_deepspeed_distributed(cfg)

    assert captured["timeout"] == datetime.timedelta(seconds=5400)


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


class _FakeColocatedPolicyModel(nn.Module):
    def __init__(self, vocab_size: int = 16, hidden_size: int = 8) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size)

    def disable_adapter(self):
        return nullcontext()

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        hidden = self.emb(input_ids)
        return SimpleNamespace(logits=self.proj(hidden))


def test_create_colocated_reference_logprob_batch_fn_scores_without_name_error() -> None:
    cfg = _base_cfg()
    cfg.model.reference_runtime = "colocate"
    cfg.model.policy_runtime_mode = "colocate"
    cfg.model.lora.enabled = True
    cfg.model.reference_logprob_micro_batch_size = 2

    batch_fn, model_device = trainer_mod._create_colocated_reference_logprob_batch_fn(
        cfg,
        _FakeColocatedPolicyModel(),
        device="cpu",
    )
    rows = batch_fn([([1, 2], [3]), ([4], [5, 6])])

    assert model_device == "cpu"
    assert len(rows) == 2
    assert len(rows[0]) == 1
    assert len(rows[1]) == 2


class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1d = nn.Conv1d(4, 4, kernel_size=3, groups=4, bias=False)


class _FakeQwen35PolicyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = Qwen3_5GatedDeltaNet()
        self.other = nn.Linear(4, 4)


def test_register_qwen35_zero3_external_parameters_registers_conv_weight(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _base_cfg()
    cfg.rl.deepspeed_zero_stage = 3

    captured: list[tuple[nn.Module, torch.nn.Parameter]] = []

    class _FakeZero:
        @staticmethod
        def register_external_parameter(module: nn.Module, parameter: torch.nn.Parameter) -> None:
            captured.append((module, parameter))

    monkeypatch.setattr(trainer_mod, "deepspeed", SimpleNamespace(zero=_FakeZero()))

    model = _FakeQwen35PolicyModel()
    _register_qwen35_zero3_external_parameters(model, cfg)

    assert captured == [(model.block, model.block.conv1d.weight)]


def test_register_qwen35_zero3_external_parameters_skips_non_zero3(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _base_cfg()
    cfg.rl.deepspeed_zero_stage = 2

    captured: list[tuple[nn.Module, torch.nn.Parameter]] = []

    class _FakeZero:
        @staticmethod
        def register_external_parameter(module: nn.Module, parameter: torch.nn.Parameter) -> None:
            captured.append((module, parameter))

    monkeypatch.setattr(trainer_mod, "deepspeed", SimpleNamespace(zero=_FakeZero()))

    _register_qwen35_zero3_external_parameters(_FakeQwen35PolicyModel(), cfg)

    assert captured == []


class _FakePeftModelBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora = nn.Parameter(torch.ones(2, 3))
        self.base = nn.Parameter(torch.zeros(4, 5), requires_grad=False)


def test_build_zero3_peft_state_dict_gathers_trainable_params(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeGatheredParameters:
        def __init__(self, params, modifier_rank=0):  # type: ignore[no-untyped-def]
            captured["params"] = list(params)
            captured["modifier_rank"] = modifier_rank

        def __enter__(self):  # type: ignore[no-untyped-def]
            captured["entered"] = True
            return self

        def __exit__(self, exc_type, exc, tb):  # type: ignore[no-untyped-def]
            captured["exited"] = True
            return False

    monkeypatch.setattr(trainer_mod, "PeftModel", _FakePeftModelBase)
    monkeypatch.setattr(
        trainer_mod,
        "get_peft_model_state_dict",
        lambda model, state_dict=None: {"lora": state_dict["lora"].clone()},
    )
    monkeypatch.setattr(
        trainer_mod,
        "deepspeed",
        SimpleNamespace(zero=SimpleNamespace(GatheredParameters=_FakeGatheredParameters)),
    )
    monkeypatch.setattr(trainer_mod, "_distributed_world_size", lambda: 8)
    monkeypatch.setattr(trainer_mod, "_is_rank0", lambda: True)

    model = _FakePeftModelBase()
    state_dict = _build_zero3_peft_state_dict(model)

    assert isinstance(state_dict, dict)
    assert tuple(state_dict["lora"].shape) == (2, 3)
    params = captured["params"]
    assert isinstance(params, list)
    assert params == [model.lora]
    assert captured["modifier_rank"] == 0


def test_load_policy_model_falls_back_when_resume_adapter_artifacts_are_zero_sized(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    cfg = _base_cfg()
    cfg.model.lora.enabled = True
    cfg.model.policy_name_or_path = "base-model"

    resume_dir = tmp_path / "resume_latest"
    resume_dir.mkdir()
    (resume_dir / "adapter_config.json").write_text('{"base_model_name_or_path": "base-model"}', encoding="utf-8")
    (resume_dir / "latest").write_text("state", encoding="utf-8")

    base_model = nn.Linear(2, 2)
    fallback_model = object()
    captured: dict[str, object] = {}

    monkeypatch.setattr(trainer_mod, "_load_causal_lm", lambda **kwargs: base_model)
    monkeypatch.setattr(trainer_mod, "_require_peft_for_lora", lambda: None)
    monkeypatch.setattr(trainer_mod, "_register_qwen35_zero3_external_parameters", lambda model, cfg: None)
    monkeypatch.setattr(trainer_mod, "_log_trainable_parameter_summary", lambda model, tag: None)
    monkeypatch.setattr(trainer_mod, "_build_lora_config", lambda cfg: "LORA_CFG")
    def _fake_get_peft_model(model, config):  # type: ignore[no-untyped-def]
        captured["fallback_args"] = (model, config)
        return fallback_model

    monkeypatch.setattr(trainer_mod, "get_peft_model", _fake_get_peft_model)
    monkeypatch.setattr(
        trainer_mod,
        "PeftModel",
        SimpleNamespace(
            from_pretrained=lambda model, source, is_trainable=True: (_ for _ in ()).throw(
                RuntimeError("size mismatch for base_model ... copying a param with shape torch.Size([0])")
            )
        ),
    )

    model = _load_policy_model(cfg, device="cpu", model_name_or_path=str(resume_dir))

    assert model is fallback_model
    assert captured["fallback_args"] == (base_model, "LORA_CFG")


def test_is_deepspeed_resume_shard_mismatch_error_matches_text() -> None:
    exc = AssertionError("assert len(self.ckpt_list) > 0")
    assert _is_deepspeed_resume_shard_mismatch_error(exc) is True


def test_is_deepspeed_resume_shard_mismatch_error_matches_state_dict_factory_traceback() -> None:
    namespace: dict[str, object] = {}
    code = compile(
        "def check_ckpt_list():\n    assert len(self.ckpt_list) > 0\n\ncheck_ckpt_list()\n",
        "state_dict_factory.py",
        "exec",
    )
    try:
        exec(code, {"self": SimpleNamespace(ckpt_list=[])}, namespace)
        raise AssertionError("expected AssertionError")
    except AssertionError as exc:
        assert _is_deepspeed_resume_shard_mismatch_error(exc) is True
