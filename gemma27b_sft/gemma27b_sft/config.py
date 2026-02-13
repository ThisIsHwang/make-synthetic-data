from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from string import Formatter
from typing import Any

import yaml


DEFAULT_TRANSLATION_PROMPT_TEMPLATE = (
    "You are a professional {source_lang} ({src_lang_code}) to {target_lang}\n"
    "({tgt_lang_code}) translator. Your goal is to accurately convey the meaning and\n"
    "nuances of the original {source_lang} text while adhering to {target_lang} grammar,\n"
    "vocabulary, and cultural sensitivities. Produce only the {target_lang}\n"
    "translation, without any additional explanations or commentary. Please translate\n"
    "the following {source_lang} text into {target_lang}:\n\n\n{text}"
)


@dataclass
class ModelConfig:
    name_or_path: str = "google/gemma-3-27b-it"
    trust_remote_code: bool = False
    attn_implementation: str | None = "auto"
    freeze_output_embeddings: bool = True


@dataclass
class DataConfig:
    train_file: str = "../runs/exp001/final_dataset.jsonl"
    eval_file: str | None = None
    source_field: str = "source_text"
    target_field: str = "target_text"
    source_lang_name: str = "auto"
    target_lang_name: str = "auto"
    source_lang_code: str = "en"
    target_lang_code: str = "ko"
    source_lang_name_field: str | None = None
    target_lang_name_field: str | None = None
    source_lang_code_field: str | None = None
    target_lang_code_field: str | None = None
    prompt_template: str = DEFAULT_TRANSLATION_PROMPT_TEMPLATE
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    preprocessing_num_workers: int = 4
    log_text_samples: int = 3
    log_text_max_chars: int = 1200


@dataclass
class TrainConfig:
    output_dir: str = "./outputs/gemma3-27b-it-sft"
    seed: int = 42
    num_train_epochs: float = 1.0
    max_steps: int = -1
    global_batch_size: int = 16
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.0
    max_seq_length: int = 2048
    bf16: bool = True
    tf32: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    report_to: list[str] = field(default_factory=list)
    resume_from_checkpoint: str | None = None
    ddp_find_unused_parameters: bool = False
    expected_world_size: int | None = None
    fsdp: str | None = None
    fsdp_transformer_layer_cls_to_wrap: str = "auto"
    fsdp_backward_prefetch: str = "BACKWARD_PRE"
    fsdp_forward_prefetch: bool = False
    fsdp_cpu_offload: bool = False
    fsdp_use_orig_params: bool = True
    fsdp_limit_all_gathers: bool = True
    fsdp_activation_checkpointing: bool = True
    fsdp_sync_module_states: bool = True
    fsdp_cpu_ram_efficient_loading: bool = True


@dataclass
class SFTConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _coerce_dataclass(cls: type[Any], data: dict[str, Any]) -> Any:
    kwargs: dict[str, Any] = {}
    defaults = cls()
    for field_info in cls.__dataclass_fields__.values():  # type: ignore[attr-defined]
        name = field_info.name
        if name not in data:
            continue
        value = data[name]
        default_value = getattr(defaults, name)
        if hasattr(default_value, "__dataclass_fields__") and isinstance(value, dict):
            kwargs[name] = _coerce_dataclass(type(default_value), value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _resolve_optional_path(value: str | None, base_dir: Path) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return value
    p = Path(raw).expanduser()
    if p.is_absolute():
        return str(p)
    return str((base_dir / p).resolve())


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def compute_gradient_accumulation_steps(cfg: SFTConfig) -> int:
    world_size = _world_size()
    micro_global = cfg.train.per_device_train_batch_size * world_size
    if micro_global <= 0:
        raise ValueError("per_device_train_batch_size * WORLD_SIZE must be > 0")
    if cfg.train.global_batch_size % micro_global != 0:
        raise ValueError(
            "global_batch_size must be divisible by per_device_train_batch_size * WORLD_SIZE. "
            f"got global_batch_size={cfg.train.global_batch_size}, "
            f"per_device_train_batch_size={cfg.train.per_device_train_batch_size}, WORLD_SIZE={world_size}"
        )
    return cfg.train.global_batch_size // micro_global


def _template_fields(template: str) -> set[str]:
    fields: set[str] = set()
    for _, field_name, _, _ in Formatter().parse(template):
        if not field_name:
            continue
        normalized = field_name.split(".", 1)[0].split("[", 1)[0]
        if normalized:
            fields.add(normalized)
    return fields


def load_config(path: str | Path) -> SFTConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    cfg = _coerce_dataclass(SFTConfig, payload)
    base_dir = config_path.parent.resolve()

    cfg.data.train_file = _resolve_optional_path(cfg.data.train_file, base_dir) or cfg.data.train_file
    cfg.data.eval_file = _resolve_optional_path(cfg.data.eval_file, base_dir)
    cfg.train.output_dir = _resolve_optional_path(cfg.train.output_dir, base_dir) or cfg.train.output_dir
    if cfg.train.fsdp is not None and not str(cfg.train.fsdp).strip():
        cfg.train.fsdp = None

    if not Path(cfg.data.train_file).exists():
        raise FileNotFoundError(f"data.train_file not found: {cfg.data.train_file}")
    if cfg.data.eval_file and not Path(cfg.data.eval_file).exists():
        raise FileNotFoundError(f"data.eval_file not found: {cfg.data.eval_file}")
    if cfg.train.learning_rate != 1e-4:
        raise ValueError("This project enforces Adafactor learning_rate=1e-4.")
    if cfg.train.global_batch_size <= 0:
        raise ValueError("train.global_batch_size must be > 0.")
    if cfg.train.max_seq_length <= 0:
        raise ValueError("train.max_seq_length must be > 0")
    if cfg.train.expected_world_size is not None and cfg.train.expected_world_size <= 0:
        raise ValueError("train.expected_world_size must be > 0 when set.")
    if cfg.data.preprocessing_num_workers < 0:
        raise ValueError("data.preprocessing_num_workers must be >= 0")
    if cfg.data.log_text_samples < 0:
        raise ValueError("data.log_text_samples must be >= 0")
    if cfg.data.log_text_max_chars <= 0:
        raise ValueError("data.log_text_max_chars must be > 0")
    if cfg.data.max_train_samples is not None and cfg.data.max_train_samples <= 0:
        raise ValueError("data.max_train_samples must be > 0 when set.")
    if cfg.data.max_eval_samples is not None and cfg.data.max_eval_samples <= 0:
        raise ValueError("data.max_eval_samples must be > 0 when set.")
    if not cfg.data.source_lang_code_field and not str(cfg.data.source_lang_code).strip():
        raise ValueError("Set data.source_lang_code or data.source_lang_code_field.")
    if not cfg.data.target_lang_code_field and not str(cfg.data.target_lang_code).strip():
        raise ValueError("Set data.target_lang_code or data.target_lang_code_field.")
    if not cfg.data.prompt_template.strip():
        raise ValueError("data.prompt_template must not be empty.")
    required_template_keys = {"source_lang", "src_lang_code", "target_lang", "tgt_lang_code", "text"}
    found_keys = _template_fields(cfg.data.prompt_template)
    missing_keys = sorted(required_template_keys - found_keys)
    if missing_keys:
        raise ValueError(
            "data.prompt_template is missing required placeholders: "
            + ", ".join(missing_keys)
        )

    compute_gradient_accumulation_steps(cfg)
    return cfg


def dump_config(cfg: SFTConfig, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(asdict(cfg), sort_keys=False), encoding="utf-8")
