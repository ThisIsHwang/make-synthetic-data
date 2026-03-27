from __future__ import annotations

import datetime
import json
from contextlib import nullcontext
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from gemma27_rl.config import RLPostTrainConfig
from gemma27_rl.rl_types import Rollout
from gemma27_rl import trainer as trainer_mod
from gemma27_rl.trainer import (
    _build_zero3_peft_state_dict,
    _configure_nccl_heartbeat_timeout,
    _init_deepspeed_distributed,
    _is_deepspeed_resume_shard_mismatch_error,
    _load_policy_model,
    _read_local_adapter_lora_rank,
    _save_deepspeed_checkpoint_to_dir,
    _resolve_vllm_server_lora_rank,
    _resolve_vllm_runtime_host,
    _resolve_vllm_current_adapter_target,
    _rewrite_peft_adapter_config_for_vllm,
    _rewrite_peft_state_dict_for_vllm,
    _register_qwen35_zero3_external_parameters,
    _sync_vllm_rollout_adapter,
    _validate_exported_vllm_adapter_dir,
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


def _rollout(example_id: str, *, prompt_ids: list[int] | None = None, completion_ids: list[int] | None = None) -> Rollout:
    prompt = list(prompt_ids or [1, 2])
    completion = list(completion_ids or [3])
    return Rollout(
        example_id=example_id,
        prompt_text=f"prompt::{example_id}",
        prompt_input_ids=prompt,
        completion_text=f"completion::{example_id}",
        completion_token_ids=completion,
        old_logprobs=[-0.1 for _ in completion],
        ref_logprobs=None,
        token_char_offsets=[(0, 1) for _ in completion],
        src_text=f"src::{example_id}",
        src_lang="English",
        tgt_lang="Korean",
        src_lang_code="en",
        tgt_lang_code="ko",
        ref_text=f"ref::{example_id}",
        raw_completion_token_ids=list(completion),
        completion_raw_text=f"completion::{example_id}",
        completion_clean_text=f"completion::{example_id}",
    )


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


def test_resolve_vllm_runtime_host_uses_master_addr_for_multinode_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _base_cfg()
    cfg.model.policy_gpu_ids = [0, 1, 2, 3]
    cfg.vllm.host = "127.0.0.1"
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.8")

    resolved = _resolve_vllm_runtime_host(cfg)

    assert resolved == "10.0.0.8"


def test_resolve_vllm_runtime_host_raises_for_multinode_loopback_without_reachable_master_addr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _base_cfg()
    cfg.model.policy_gpu_ids = [0, 1, 2, 3]
    cfg.vllm.host = "localhost"
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")

    with pytest.raises(ValueError, match="Multi-node DeepSpeed with vLLM requires vllm.host"):
        _resolve_vllm_runtime_host(cfg)


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


def test_load_causal_lm_raises_clear_error_when_transformers_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trainer_mod, "AutoModelForCausalLM", None)
    monkeypatch.setattr(trainer_mod, "AutoTokenizer", None)
    monkeypatch.setattr(trainer_mod, "_TRANSFORMERS_IMPORT_ERROR", ModuleNotFoundError("transformers"))

    with pytest.raises(RuntimeError, match="Loading the policy model requires the `transformers` package"):
        trainer_mod._load_causal_lm(
            model_name_or_path="dummy",
            kwargs={},
            single_device="cpu",
            gpu_ids=[0],
            component_name="policy",
        )


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


def test_save_deepspeed_checkpoint_to_dir_preserves_previous_checkpoint_on_metadata_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _Engine:
        def save_checkpoint(self, path: str, tag: str = "") -> None:
            ckpt_path = Path(path)
            ckpt_path.mkdir(parents=True, exist_ok=True)
            (ckpt_path / f"{tag or 'state'}.txt").write_text("new-shard\n", encoding="utf-8")

    class _Tokenizer:
        def save_pretrained(self, path: Path) -> None:
            raise RuntimeError("tokenizer failed")

    ckpt_dir = tmp_path / "resume_latest"
    ckpt_dir.mkdir()
    (ckpt_dir / "sentinel.txt").write_text("old\n", encoding="utf-8")

    monkeypatch.setattr(trainer_mod, "_is_rank0", lambda: True)
    monkeypatch.setattr(trainer_mod, "_dist_barrier", lambda: None)
    monkeypatch.setattr(trainer_mod, "_all_gather_object", lambda obj: [obj])
    monkeypatch.setattr(trainer_mod, "_broadcast_object_list", lambda payload, src=0: payload)

    with pytest.raises(RuntimeError, match="tokenizer failed"):
        _save_deepspeed_checkpoint_to_dir(
            ckpt_dir=ckpt_dir,
            engine=_Engine(),
            tokenizer=_Tokenizer(),
            hf_model=None,
            trainer_state=None,
        )

    assert (ckpt_dir / "sentinel.txt").read_text(encoding="utf-8") == "old\n"
    assert not (ckpt_dir / "state.txt").exists()


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


def test_rewrite_peft_state_dict_for_vllm_keeps_only_gemma3_language_model_tensors() -> None:
    model = SimpleNamespace(config=SimpleNamespace(model_type="gemma3"))
    state_dict = {
        "base_model.model.model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(1),
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(1),
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(1),
    }

    rewritten = _rewrite_peft_state_dict_for_vllm(model, state_dict)

    assert sorted(rewritten) == [
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight",
    ]


def test_rewrite_peft_state_dict_for_vllm_raises_when_gemma3_multimodal_adapter_has_no_language_keys() -> None:
    model = SimpleNamespace(config=SimpleNamespace(model_type="gemma3"))
    state_dict = {
        "base_model.model.model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(1),
    }

    with pytest.raises(RuntimeError, match="no language_model LoRA weights remain"):
        _rewrite_peft_state_dict_for_vllm(model, state_dict)


def test_rewrite_peft_adapter_config_for_vllm_keeps_gemma3_target_modules_unchanged(tmp_path) -> None:  # type: ignore[no-untyped-def]
    model = SimpleNamespace(config=SimpleNamespace(model_type="gemma3"))
    cfg_path = tmp_path / "adapter_config.json"
    cfg_path.write_text(
        """{
  "target_modules": ["down_proj", "v_proj", "gate_proj", "q_proj", "o_proj", "k_proj", "up_proj"]
}
""",
        encoding="utf-8",
    )

    _rewrite_peft_adapter_config_for_vllm(model, tmp_path)

    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert payload["target_modules"] == ["down_proj", "v_proj", "gate_proj", "q_proj", "o_proj", "k_proj", "up_proj"]


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


def test_save_pretrained_model_preserves_filtered_lora_keys_in_saved_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _FakePeftConfig:
        def __init__(self) -> None:
            self.base_model_name_or_path = "base-model"
            self.inference_mode = False
            self.task_type = "CAUSAL_LM"
            self.is_prompt_learning = False
            self.target_modules = ["q_proj"]

        def save_pretrained(self, output_dir: str, auto_mapping_dict=None) -> None:  # type: ignore[no-untyped-def]
            del auto_mapping_dict
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            (Path(output_dir) / "adapter_config.json").write_text(
                json.dumps(
                    {
                        "base_model_name_or_path": self.base_model_name_or_path,
                        "target_modules": list(self.target_modules),
                    }
                )
                + "\n",
                encoding="utf-8",
            )

    class _FakeExportModel(_FakePeftModelBase):
        def __init__(self) -> None:
            super().__init__()
            self.active_adapter = "default"
            self.peft_config = {"default": _FakePeftConfig()}

        def create_or_update_model_card(self, output_dir: str) -> None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            (Path(output_dir) / "README.md").write_text("test\n", encoding="utf-8")

    state_dict = {
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(1),
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(1),
    }
    monkeypatch.setattr(trainer_mod, "PeftModel", _FakeExportModel)
    monkeypatch.setattr(trainer_mod, "_build_zero3_peft_state_dict", lambda model: dict(state_dict))
    monkeypatch.setattr(trainer_mod, "_all_gather_object", lambda value: [value])
    monkeypatch.setattr(trainer_mod, "_broadcast_object_list", lambda payload, src=0: payload)
    monkeypatch.setattr(trainer_mod, "_is_rank0", lambda: True)

    model = _FakeExportModel()
    trainer_mod._save_pretrained_model(model, tmp_path, require_peft_adapter=True)

    _validate_exported_vllm_adapter_dir(tmp_path)
    from safetensors import safe_open

    with safe_open(str(tmp_path / "adapter_model.safetensors"), framework="pt", device="cpu") as handle:
        assert sorted(handle.keys()) == sorted(state_dict)


def test_validate_exported_vllm_adapter_dir_requires_weight_artifact(tmp_path: Path) -> None:
    (tmp_path / "adapter_config.json").write_text('{"base_model_name_or_path": "base"}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="missing adapter weights"):
        _validate_exported_vllm_adapter_dir(tmp_path)


def test_validate_exported_vllm_adapter_dir_requires_lora_tensor_keys(tmp_path: Path) -> None:
    (tmp_path / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "base", "target_modules": ["q_proj"]}',
        encoding="utf-8",
    )
    torch.save({"base_model.model.model.layers.0.self_attn.q_proj.weight": torch.ones(1)}, tmp_path / "adapter_model.bin")

    with pytest.raises(RuntimeError, match="contains no LoRA tensors"):
        _validate_exported_vllm_adapter_dir(tmp_path)


def test_validate_exported_vllm_adapter_dir_rejects_target_module_mismatch(tmp_path: Path) -> None:
    (tmp_path / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "base", "target_modules": ["qkv_proj", "gate_up_proj"]}',
        encoding="utf-8",
    )
    torch.save(
        {
            "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(1),
            "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(1),
        },
        tmp_path / "adapter_model.bin",
    )

    with pytest.raises(RuntimeError, match="target_modules do not match saved LoRA tensors"):
        _validate_exported_vllm_adapter_dir(tmp_path)


def test_read_local_adapter_lora_rank_reads_rank_from_adapter_config(tmp_path: Path) -> None:
    (tmp_path / "adapter_config.json").write_text('{"base_model_name_or_path": "base", "r": 96}', encoding="utf-8")

    assert _read_local_adapter_lora_rank(tmp_path) == 96


def test_resolve_vllm_server_lora_rank_uses_larger_adapter_rank(tmp_path: Path) -> None:
    cfg = _base_cfg()
    cfg.model.lora.enabled = True
    cfg.model.lora.r = 64
    (tmp_path / "adapter_config.json").write_text('{"base_model_name_or_path": "base", "r": 128}', encoding="utf-8")

    assert _resolve_vllm_server_lora_rank(cfg, str(tmp_path)) == 128


def test_sync_vllm_rollout_adapter_promotes_candidate_after_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter_root = tmp_path / "vllm_adapters"
    current_dir = adapter_root / "current"
    old_dir = adapter_root / "candidate-5"
    old_dir.mkdir(parents=True)
    (old_dir / "adapter_config.json").write_text('{"base_model_name_or_path": "base"}', encoding="utf-8")
    (old_dir / "adapter_model.safetensors").write_text("old", encoding="utf-8")
    current_dir.symlink_to(old_dir, target_is_directory=True)

    saved_paths: list[Path] = []
    load_calls: list[Path] = []
    unload_calls: list[str] = []

    class _FakeClient:
        adapter_dir = current_dir

        def unload_adapter(self) -> None:
            unload_calls.append("unload")

        def load_adapter(self, adapter_path: Path) -> None:
            load_calls.append(adapter_path)

    def _fake_save(model, output_dir: Path, *, require_peft_adapter: bool = False) -> None:  # type: ignore[no-untyped-def]
        assert require_peft_adapter is True
        saved_paths.append(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "adapter_config.json").write_text('{"base_model_name_or_path": "base"}', encoding="utf-8")
        torch.save(
            {
                "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(1),
                "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(1),
            },
            output_dir / "adapter_model.bin",
        )

    monkeypatch.setattr(trainer_mod, "_save_pretrained_model", _fake_save)
    monkeypatch.setattr(trainer_mod, "_dist_barrier", lambda: None)
    monkeypatch.setattr(trainer_mod, "_broadcast_object_list", lambda payload, src=0: payload)
    monkeypatch.setattr(trainer_mod, "_is_rank0", lambda: True)

    _sync_vllm_rollout_adapter(update_idx=6, vllm_rollout_client=_FakeClient(), policy_model=object())

    candidate_dir = adapter_root / "candidate-6"
    assert saved_paths == [candidate_dir]
    assert unload_calls == ["unload"]
    assert load_calls == [candidate_dir]
    assert current_dir.is_symlink()
    assert _resolve_vllm_current_adapter_target(current_dir) == candidate_dir
    assert candidate_dir.exists()
    assert not old_dir.exists()


def test_sync_vllm_rollout_adapter_restores_previous_candidate_on_load_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter_root = tmp_path / "vllm_adapters"
    current_dir = adapter_root / "current"
    old_dir = adapter_root / "candidate-11"
    old_dir.mkdir(parents=True)
    (old_dir / "adapter_config.json").write_text('{"base_model_name_or_path": "base"}', encoding="utf-8")
    (old_dir / "adapter_model.safetensors").write_text("old", encoding="utf-8")
    current_dir.symlink_to(old_dir, target_is_directory=True)

    load_calls: list[Path] = []
    unload_calls: list[str] = []

    class _FakeClient:
        adapter_dir = current_dir

        def unload_adapter(self) -> None:
            unload_calls.append("unload")

        def load_adapter(self, adapter_path: Path) -> None:
            load_calls.append(adapter_path)
            if adapter_path.name == "candidate-12":
                raise RuntimeError("vLLM returned 500")

    def _fake_save(model, output_dir: Path, *, require_peft_adapter: bool = False) -> None:  # type: ignore[no-untyped-def]
        assert require_peft_adapter is True
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "adapter_config.json").write_text('{"base_model_name_or_path": "base"}', encoding="utf-8")
        torch.save(
            {
                "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(1),
                "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(1),
            },
            output_dir / "adapter_model.bin",
        )

    monkeypatch.setattr(trainer_mod, "_save_pretrained_model", _fake_save)
    monkeypatch.setattr(trainer_mod, "_dist_barrier", lambda: None)
    monkeypatch.setattr(trainer_mod, "_broadcast_object_list", lambda payload, src=0: payload)
    monkeypatch.setattr(trainer_mod, "_is_rank0", lambda: True)

    with pytest.raises(RuntimeError, match="failed to refresh vLLM rollout adapter: RuntimeError: vLLM returned 500"):
        _sync_vllm_rollout_adapter(update_idx=12, vllm_rollout_client=_FakeClient(), policy_model=object())

    candidate_dir = adapter_root / "candidate-12"
    assert unload_calls == ["unload"]
    assert load_calls == [candidate_dir, old_dir]
    assert current_dir.is_symlink()
    assert _resolve_vllm_current_adapter_target(current_dir) == old_dir
    assert old_dir.exists()
    assert candidate_dir.exists()


def test_sync_vllm_rollout_adapter_raises_clear_error_when_candidate_rank_exceeds_server_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter_root = tmp_path / "vllm_adapters"
    current_dir = adapter_root / "current"
    old_dir = adapter_root / "candidate-20"
    old_dir.mkdir(parents=True)
    (old_dir / "adapter_config.json").write_text('{"base_model_name_or_path": "base", "r": 64}', encoding="utf-8")
    (old_dir / "adapter_model.safetensors").write_text("old", encoding="utf-8")
    current_dir.symlink_to(old_dir, target_is_directory=True)

    class _FakeClient:
        adapter_dir = current_dir
        max_lora_rank = 64

        def unload_adapter(self) -> None:
            raise AssertionError("unload_adapter should not be reached when rank validation fails")

        def load_adapter(self, adapter_path: Path) -> None:
            raise AssertionError("load_adapter should not be reached when rank validation fails")

    def _fake_save(model, output_dir: Path, *, require_peft_adapter: bool = False) -> None:  # type: ignore[no-untyped-def]
        assert require_peft_adapter is True
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "adapter_config.json").write_text('{"base_model_name_or_path": "base", "r": 128}', encoding="utf-8")
        torch.save(
            {
                "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(1),
                "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(1),
            },
            output_dir / "adapter_model.bin",
        )

    monkeypatch.setattr(trainer_mod, "_save_pretrained_model", _fake_save)
    monkeypatch.setattr(trainer_mod, "_dist_barrier", lambda: None)
    monkeypatch.setattr(trainer_mod, "_broadcast_object_list", lambda payload, src=0: payload)
    monkeypatch.setattr(trainer_mod, "_is_rank0", lambda: True)

    with pytest.raises(RuntimeError, match="adapter_r=128 max_lora_rank=64"):
        _sync_vllm_rollout_adapter(update_idx=21, vllm_rollout_client=_FakeClient(), policy_model=object())


def test_fill_missing_reference_logprobs_raises_when_batch_scoring_still_leaves_missing_kl() -> None:
    cfg = RLPostTrainConfig()
    cfg.model.use_reference_model = True
    cfg.rl.kl_coef = 0.1
    rollouts = [_rollout("ok"), _rollout("bad", prompt_ids=[9, 9])]

    def _score(items):  # type: ignore[no-untyped-def]
        if len(items) > 1:
            raise RuntimeError("temporary batch failure")
        prompt_ids, completion_ids = items[0]
        if prompt_ids == [9, 9]:
            raise RuntimeError("reference worker 500")
        return [[-0.2 for _ in completion_ids]]

    with pytest.raises(RuntimeError, match="refusing to skip KL"):
        trainer_mod._fill_missing_reference_logprobs(
            merged_rollouts=rollouts,
            cfg=cfg,
            update_idx=17,
            ref_logprob_batch_fn=_score,
            ref_logprob_client=None,
            ref_model=None,
            ref_device=None,
            device="cpu",
        )

    assert rollouts[0].ref_logprobs == [-0.2]
    assert rollouts[1].ref_logprobs is None


def test_fill_missing_reference_logprobs_raises_when_token_count_mismatches_completion() -> None:
    cfg = RLPostTrainConfig()
    cfg.model.use_reference_model = True
    cfg.rl.kl_coef = 0.1
    rollouts = [_rollout("bad", completion_ids=[3, 4])]

    with pytest.raises(RuntimeError, match="token count mismatch: expected=2 returned=1"):
        trainer_mod._fill_missing_reference_logprobs(
            merged_rollouts=rollouts,
            cfg=cfg,
            update_idx=19,
            ref_logprob_batch_fn=lambda items: [[-0.2]],
            ref_logprob_client=None,
            ref_model=None,
            ref_device=None,
            device="cpu",
        )

    assert rollouts[0].ref_logprobs is None


def test_fill_missing_reference_logprobs_distributed_colocate_raises_when_gathered_rows_are_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = RLPostTrainConfig()
    cfg.model.use_reference_model = True
    cfg.model.reference_runtime = "colocate"
    cfg.rl.kl_coef = 0.1
    merged_rollouts = [_rollout("ok"), _rollout("bad")]
    broadcast_payloads: list[list[object]] = []

    def _fake_broadcast(payload, src=0):  # type: ignore[no-untyped-def]
        del src
        broadcast_payloads.append(list(payload))
        return payload

    monkeypatch.setattr(trainer_mod, "_is_rank0", lambda: True)
    monkeypatch.setattr(trainer_mod, "_distributed_world_size", lambda: 1)
    monkeypatch.setattr(trainer_mod, "_broadcast_object_list", _fake_broadcast)
    monkeypatch.setattr(trainer_mod, "_scatter_object_from_rank0", lambda payload, rank=0: list((payload or [[]])[0]))
    monkeypatch.setattr(
        trainer_mod,
        "_score_reference_requests_with_batch_fn",
        lambda **kwargs: ({0: [-0.3]}, 1, 1, {1: "RuntimeError: reference worker 500"}),
    )
    monkeypatch.setattr(
        trainer_mod,
        "_gather_object_to_rank0",
        lambda payload: [payload],
    )

    with pytest.raises(RuntimeError, match="refusing to skip KL"):
        trainer_mod._fill_missing_reference_logprobs_distributed_colocate(
            merged_rollouts=merged_rollouts,
            cfg=cfg,
            update_idx=18,
            ref_logprob_batch_fn=lambda items: [],
            rank=0,
        )

    assert merged_rollouts[0].ref_logprobs == [-0.3]
    assert merged_rollouts[1].ref_logprobs is None
    assert any("refusing to skip KL" in str(item[0]) for item in broadcast_payloads if item and item[0])


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
