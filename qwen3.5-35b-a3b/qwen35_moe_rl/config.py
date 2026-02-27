from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .prompting import DEFAULT_TRANSLATION_PROMPT_TEMPLATE


@dataclass
class ModelConfig:
    policy_name_or_path: str = "Qwen/Qwen3.5-35B-A3B"
    reference_name_or_path: str | None = None
    tokenizer_name_or_path: str | None = None
    trust_remote_code: bool = True
    attn_implementation: str | None = "auto"
    use_fast_tokenizer: bool = True
    reference_device: str | None = None
    policy_gpu_ids: list[int] = field(default_factory=list)
    reference_gpu_ids: list[int] = field(default_factory=list)
    # MoE memory optimization.
    gradient_checkpointing: bool = False
    gradient_checkpointing_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"use_reentrant": False}
    )


@dataclass
class DataConfig:
    train_file: str | None = "../runs/exp001/final_dataset.jsonl"
    eval_file: str | None = None
    hf_dataset_name: str | None = None
    hf_dataset_config_name: str | None = None
    hf_train_split: str = "train"
    hf_eval_split: str | None = None
    hf_revision: str | None = None
    hf_streaming: bool = False

    split_field: str | None = None
    train_split: str | None = None
    eval_split: str | None = None
    limit: int | None = None
    eval_limit: int | None = None

    id_field: str = "id"
    src_text_field: str = "src_text"
    src_lang_field: str = "src_lang"
    tgt_lang_field: str = "tgt_lang"
    src_lang_code_field: str = "src_lang_code"
    tgt_lang_code_field: str = "tgt_lang_code"
    ref_text_field: str = "ref_text"
    is_bad_source_field: str = "is_bad_source"
    skip_bad_source: bool = False

    default_src_lang: str = "English"
    default_tgt_lang: str = "Korean"
    default_src_lang_code: str = "en"
    default_tgt_lang_code: str = "ko"


@dataclass
class PromptConfig:
    template: str = DEFAULT_TRANSLATION_PROMPT_TEMPLATE


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    num_samples_per_prompt: int = 2
    do_sample: bool = True
    repetition_penalty: float = 1.0


@dataclass
class MetricXConfig:
    enabled: bool = True
    model_name: str = "google/metricx-24-hybrid-large-v2p6"
    tokenizer_name: str | None = "google/mt5-xl"
    use_reference: bool = False
    batch_size: int = 4
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_input_length: int = 2048
    overflow_policy: str = "truncate"  # truncate|skip
    offset: float = 5.0


@dataclass
class XCometConfig:
    enabled: bool = True
    model_name: str = "Unbabel/XCOMET-XL"
    batch_size: int = 2
    device: str = "cuda"
    use_reference: bool = False


@dataclass
class RewardConfig:
    w_metricx: float = 1.0
    w_xcomet_seq: float = 0.0
    xcomet_seq_scale: float = 1.0
    severity_weights: dict[str, float] = field(
        default_factory=lambda: {
            "MINOR": -1.0,
            "MAJOR": -5.0,
            "CRITICAL": -10.0,
        }
    )
    overlap_policy: str = "any_overlap"  # any_overlap|majority_overlap
    majority_threshold: float = 0.5
    use_confidence: bool = False
    span_combine_policy: str = "sum"  # sum|min|max
    cache_enabled: bool = True

    metricx: MetricXConfig = field(default_factory=MetricXConfig)
    xcomet: XCometConfig = field(default_factory=XCometConfig)


@dataclass
class RLConfig:
    algorithm: str = "gspo"  # grpo|reinforce|gspo
    lr: float = 1e-6
    weight_decay: float = 0.0
    batch_size: int = 2
    grad_accum: int = 1
    clip_eps: float = 0.2
    kl_coef: float = 0.01
    entropy_coef: float = 0.0
    max_grad_norm: float = 1.0
    ppo_epochs: int = 1
    updates: int = 20
    normalize_advantage: bool = True
    group_normalize: bool = True
    group_advantage_coef: float = 1.0
    eps: float = 1e-8
    # MoE-specific fields.
    output_router_logits: bool = True
    router_aux_loss_coef: float = 0.001
    monitor_expert_utilization: bool = True


@dataclass
class EvalConfig:
    eval_every_n_updates: int = 5
    eval_limit: int = 64
    run_before_train: bool = True


@dataclass
class LoggingConfig:
    output_dir: str = "./outputs/qwen35-gspo"
    jsonl_name: str = "train_log.jsonl"
    rollout_jsonl_name: str = "train_rollouts.jsonl"
    save_rollouts: bool = False
    eval_output_jsonl_name: str = "eval_outputs.jsonl"
    save_eval_outputs: bool = False
    save_every_n_updates: int = 10


@dataclass
class DistributedConfig:
    enabled: bool = False
    backend: str = "deepspeed"  # deepspeed | fsdp | ddp
    zero_stage: int = 2  # 0, 1, 2, 3 (DeepSpeed only)
    offload_optimizer: bool = False  # optimizer state → CPU (ZeRO-Offload)
    offload_param: bool = False  # parameters → CPU (ZeRO-3 only)
    deepspeed_config_path: str | None = None  # external JSON override
    fsdp_sharding_strategy: str = "FULL_SHARD"  # FULL_SHARD | SHARD_GRAD_OP | NO_SHARD


@dataclass
class MiscConfig:
    seed: int = 42
    device: str = "cuda"
    dtype: str = "bfloat16"
    huggingface_cache_dir: str | None = "/media/sdd3"
    huggingface_token: str | None = None
    huggingface_token_env: str = "HF_TOKEN"


@dataclass
class RLPostTrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    misc: MiscConfig = field(default_factory=MiscConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)


def _coerce_dataclass(cls: type[Any], payload: dict[str, Any]) -> Any:
    defaults = cls()
    kwargs: dict[str, Any] = {}
    for field_info in cls.__dataclass_fields__.values():  # type: ignore[attr-defined]
        key = field_info.name
        if key not in payload:
            continue
        raw = payload[key]
        default_value = getattr(defaults, key)
        if hasattr(default_value, "__dataclass_fields__") and isinstance(raw, dict):
            kwargs[key] = _coerce_dataclass(type(default_value), raw)
        else:
            kwargs[key] = raw
    return cls(**kwargs)


def _resolve_optional_path(value: str | None, base_dir: Path) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return value
    path = Path(text).expanduser()
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _resolve_model_name_or_path(value: str | None, base_dir: Path) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return value
    path = Path(text).expanduser()
    if path.is_absolute():
        return str(path)
    if text.startswith(".") or text.startswith("~") or path.exists():
        return str((base_dir / path).resolve())
    # Keep probable model IDs (e.g., Qwen/Qwen3.5-35B-A3B) unchanged.
    return text


def _normalize_gpu_ids(raw_ids: list[int], field_name: str) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for idx in raw_ids:
        try:
            value = int(idx)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"{field_name} contains non-integer value: {idx!r}") from exc
        if value < 0:
            raise ValueError(f"{field_name} must contain non-negative GPU indices.")
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _parse_cuda_index(device: str | None) -> int | None:
    if device is None:
        return None
    text = str(device).strip().lower()
    if not text.startswith("cuda"):
        return None
    if text == "cuda":
        return None
    if text.startswith("cuda:"):
        suffix = text.split(":", 1)[1].strip()
        if suffix.isdigit():
            return int(suffix)
    return None


def _validate_config(cfg: RLPostTrainConfig) -> None:
    use_hf_dataset = bool(cfg.data.hf_dataset_name and str(cfg.data.hf_dataset_name).strip())
    if use_hf_dataset:
        if not cfg.data.hf_train_split.strip():
            raise ValueError("data.hf_train_split must not be empty when data.hf_dataset_name is set.")
    else:
        if not cfg.data.train_file or not Path(cfg.data.train_file).exists():
            raise FileNotFoundError(f"data.train_file not found: {cfg.data.train_file}")
        if cfg.data.eval_file and not Path(cfg.data.eval_file).exists():
            raise FileNotFoundError(f"data.eval_file not found: {cfg.data.eval_file}")
    if cfg.generation.num_samples_per_prompt <= 0:
        raise ValueError("generation.num_samples_per_prompt must be > 0")
    if cfg.rl.batch_size <= 0:
        raise ValueError("rl.batch_size must be > 0")
    if cfg.rl.grad_accum <= 0:
        raise ValueError("rl.grad_accum must be > 0")
    if cfg.rl.updates <= 0:
        raise ValueError("rl.updates must be > 0")
    if cfg.rl.algorithm not in {"grpo", "reinforce", "gspo"}:
        raise ValueError("rl.algorithm must be one of: grpo, reinforce, gspo")
    if cfg.rl.router_aux_loss_coef < 0:
        raise ValueError("rl.router_aux_loss_coef must be >= 0")
    if cfg.reward.overlap_policy not in {"any_overlap", "majority_overlap"}:
        raise ValueError("reward.overlap_policy must be any_overlap or majority_overlap")
    if cfg.reward.span_combine_policy not in {"sum", "min", "max"}:
        raise ValueError("reward.span_combine_policy must be sum|min|max")
    if not cfg.reward.metricx.enabled and not cfg.reward.xcomet.enabled:
        raise ValueError("At least one reward model must be enabled.")

    if cfg.distributed.enabled:
        if cfg.distributed.backend not in {"deepspeed", "fsdp", "ddp"}:
            raise ValueError("distributed.backend must be one of: deepspeed, fsdp, ddp")
        if cfg.distributed.zero_stage not in {0, 1, 2, 3}:
            raise ValueError("distributed.zero_stage must be 0, 1, 2, or 3")
        if cfg.distributed.offload_param and cfg.distributed.zero_stage < 3:
            import logging as _logging

            _logging.getLogger(__name__).warning(
                "distributed.offload_param is only effective with zero_stage=3. "
                "Current zero_stage=%s; offload_param will be ignored.",
                cfg.distributed.zero_stage,
            )

    cfg.model.policy_gpu_ids = _normalize_gpu_ids(cfg.model.policy_gpu_ids, "model.policy_gpu_ids")
    cfg.model.reference_gpu_ids = _normalize_gpu_ids(cfg.model.reference_gpu_ids, "model.reference_gpu_ids")

    policy_set = set(cfg.model.policy_gpu_ids)
    ref_set = set(cfg.model.reference_gpu_ids)
    overlap = sorted(policy_set & ref_set)
    if overlap:
        raise ValueError(
            "model.policy_gpu_ids and model.reference_gpu_ids must be disjoint. "
            f"overlap={overlap}"
        )

    reserved = set(policy_set) | set(ref_set)
    metricx_idx = _parse_cuda_index(cfg.reward.metricx.device if cfg.reward.metricx.enabled else None)
    xcomet_idx = _parse_cuda_index(cfg.reward.xcomet.device if cfg.reward.xcomet.enabled else None)

    if metricx_idx is not None and metricx_idx in reserved:
        raise ValueError(
            "reward.metricx.device overlaps policy/reference GPU allocation. "
            f"metricx_gpu={metricx_idx} reserved={sorted(reserved)}"
        )
    if xcomet_idx is not None and xcomet_idx in reserved:
        raise ValueError(
            "reward.xcomet.device overlaps policy/reference GPU allocation. "
            f"xcomet_gpu={xcomet_idx} reserved={sorted(reserved)}"
        )
    if metricx_idx is not None and xcomet_idx is not None and metricx_idx == xcomet_idx:
        raise ValueError("reward.metricx.device and reward.xcomet.device must be different GPUs.")


def load_config(path: str | Path) -> RLPostTrainConfig:
    cfg_path = Path(path)
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    cfg = _coerce_dataclass(RLPostTrainConfig, payload)

    base_dir = cfg_path.parent.resolve()
    cfg.data.train_file = _resolve_optional_path(cfg.data.train_file, base_dir)
    cfg.data.eval_file = _resolve_optional_path(cfg.data.eval_file, base_dir)
    cfg.model.policy_name_or_path = (
        _resolve_model_name_or_path(cfg.model.policy_name_or_path, base_dir)
        or cfg.model.policy_name_or_path
    )
    cfg.model.reference_name_or_path = _resolve_model_name_or_path(cfg.model.reference_name_or_path, base_dir)
    cfg.model.tokenizer_name_or_path = _resolve_model_name_or_path(cfg.model.tokenizer_name_or_path, base_dir)
    cfg.logging.output_dir = _resolve_optional_path(cfg.logging.output_dir, base_dir) or cfg.logging.output_dir
    cfg.misc.huggingface_cache_dir = _resolve_optional_path(cfg.misc.huggingface_cache_dir, base_dir)
    cfg.distributed.deepspeed_config_path = _resolve_optional_path(
        cfg.distributed.deepspeed_config_path, base_dir
    )

    _validate_config(cfg)
    return cfg


def dump_config(cfg: RLPostTrainConfig, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(asdict(cfg), sort_keys=False), encoding="utf-8")
