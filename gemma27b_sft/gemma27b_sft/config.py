from __future__ import annotations

from collections.abc import Mapping
import json
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
ALLOWED_PROMPT_TEMPLATE_KEYS = {
    "source_lang",
    "src_lang_code",
    "target_lang",
    "tgt_lang_code",
    "text",
}


@dataclass
class ModelConfig:
    name_or_path: str = "google/gemma-3-27b-it"
    tokenizer_name_or_path: str | None = None
    trust_remote_code: bool = False
    attn_implementation: str | None = "auto"
    freeze_input_embeddings: bool = True
    freeze_output_embeddings: bool = True
    freeze_vision_encoder: bool = True


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
    log_text_max_chars: int = 0
    log_chat_template_text: bool = False


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
    deepspeed: str | dict[str, Any] | None = None
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


def _child_config_path(parent: str, name: str) -> str:
    if not parent or parent == "config":
        return name
    return f"{parent}.{name}"


def _coerce_dataclass(cls: type[Any], data: Any, path: str = "config") -> Any:
    label = path or cls.__name__
    if not isinstance(data, Mapping):
        raise ValueError(f"Config section {label} must be a mapping/object.")

    field_map = cls.__dataclass_fields__  # type: ignore[attr-defined]
    known = set(field_map)
    unknown = sorted(str(key) for key in data.keys() if key not in known)
    if unknown:
        raise ValueError(f"Unknown config keys in {label}: {', '.join(unknown)}")

    kwargs: dict[str, Any] = {}
    defaults = cls()
    for field_info in field_map.values():
        name = field_info.name
        if name not in data:
            continue
        value = data[name]
        default_value = getattr(defaults, name)
        if hasattr(default_value, "__dataclass_fields__"):
            kwargs[name] = _coerce_dataclass(
                type(default_value),
                value,
                path=_child_config_path(path, name),
            )
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _resolve_optional_path(value: str | None, base_dir: Path) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    if p.is_absolute():
        return str(p)
    return str((base_dir / p).resolve())


def _validate_jsonl_file(path: str, label: str) -> None:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not file_path.is_file():
        raise ValueError(f"{label} must be a file: {path}")
    if file_path.suffix.lower() != ".jsonl":
        raise ValueError(f"{label} must point to a .jsonl file: {path}")

    with file_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{label} must be newline-delimited JSON objects. "
                    f"First non-empty line is invalid JSON at line {line_no}: {path}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"{label} must contain JSON objects per line. "
                    f"First non-empty line is {type(record).__name__} at line {line_no}: {path}"
                )
            return

    raise ValueError(f"{label} must contain at least one JSON object line: {path}")


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
    config_path = Path(path).expanduser()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload is None:
        payload = {}
    cfg = _coerce_dataclass(SFTConfig, payload, path="config")
    base_dir = config_path.parent.resolve()

    cfg.data.train_file = _resolve_optional_path(cfg.data.train_file, base_dir) or cfg.data.train_file
    cfg.data.eval_file = _resolve_optional_path(cfg.data.eval_file, base_dir)
    cfg.train.output_dir = _resolve_optional_path(cfg.train.output_dir, base_dir) or cfg.train.output_dir
    cfg.train.resume_from_checkpoint = _resolve_optional_path(cfg.train.resume_from_checkpoint, base_dir)
    if isinstance(cfg.train.deepspeed, str):
        cfg.train.deepspeed = _resolve_optional_path(cfg.train.deepspeed, base_dir) or cfg.train.deepspeed
    if cfg.train.fsdp is not None and not str(cfg.train.fsdp).strip():
        cfg.train.fsdp = None
    if not str(cfg.data.train_file).strip():
        raise ValueError("data.train_file must not be empty.")
    if not str(cfg.train.output_dir).strip():
        raise ValueError("train.output_dir must not be empty.")
    if isinstance(cfg.train.deepspeed, str) and not str(cfg.train.deepspeed).strip():
        cfg.train.deepspeed = None

    _validate_jsonl_file(cfg.data.train_file, "data.train_file")
    if cfg.data.eval_file:
        _validate_jsonl_file(cfg.data.eval_file, "data.eval_file")
    if isinstance(cfg.train.deepspeed, str) and not Path(cfg.train.deepspeed).exists():
        raise FileNotFoundError(f"train.deepspeed config not found: {cfg.train.deepspeed}")
    if cfg.train.deepspeed is not None and cfg.train.fsdp:
        raise ValueError("Set only one backend: train.deepspeed or train.fsdp (not both).")
    if cfg.train.learning_rate <= 0.0:
        raise ValueError("train.learning_rate must be > 0.")
    if cfg.train.num_train_epochs <= 0:
        raise ValueError("train.num_train_epochs must be > 0.")
    if cfg.train.global_batch_size <= 0:
        raise ValueError("train.global_batch_size must be > 0.")
    if cfg.train.max_steps < -1 or cfg.train.max_steps == 0:
        raise ValueError("train.max_steps must be -1 or > 0.")
    if cfg.train.max_seq_length <= 0:
        raise ValueError("train.max_seq_length must be > 0")
    if cfg.train.weight_decay < 0.0:
        raise ValueError("train.weight_decay must be >= 0.")
    if cfg.train.dataloader_num_workers < 0:
        raise ValueError("train.dataloader_num_workers must be >= 0.")
    if cfg.train.logging_steps <= 0:
        raise ValueError("train.logging_steps must be > 0.")
    if cfg.train.save_steps <= 0:
        raise ValueError("train.save_steps must be > 0.")
    if cfg.train.eval_steps <= 0:
        raise ValueError("train.eval_steps must be > 0.")
    if cfg.train.expected_world_size is not None and cfg.train.expected_world_size <= 0:
        raise ValueError("train.expected_world_size must be > 0 when set.")
    if cfg.train.resume_from_checkpoint is not None and not Path(cfg.train.resume_from_checkpoint).exists():
        raise FileNotFoundError(
            f"train.resume_from_checkpoint not found: {cfg.train.resume_from_checkpoint}"
        )
    if cfg.data.preprocessing_num_workers < 0:
        raise ValueError("data.preprocessing_num_workers must be >= 0")
    if cfg.data.log_text_samples < 0:
        raise ValueError("data.log_text_samples must be >= 0")
    if cfg.data.log_text_max_chars < 0:
        raise ValueError("data.log_text_max_chars must be >= 0 (0 disables truncation).")
    if cfg.data.max_train_samples is not None and cfg.data.max_train_samples <= 0:
        raise ValueError("data.max_train_samples must be > 0 when set.")
    if cfg.data.max_eval_samples is not None and cfg.data.max_eval_samples <= 0:
        raise ValueError("data.max_eval_samples must be > 0 when set.")
    if not str(cfg.data.source_field).strip():
        raise ValueError("data.source_field must not be empty.")
    if not str(cfg.data.target_field).strip():
        raise ValueError("data.target_field must not be empty.")
    if not cfg.data.source_lang_code_field and not str(cfg.data.source_lang_code).strip():
        raise ValueError("Set data.source_lang_code or data.source_lang_code_field.")
    if not cfg.data.target_lang_code_field and not str(cfg.data.target_lang_code).strip():
        raise ValueError("Set data.target_lang_code or data.target_lang_code_field.")
    if not cfg.data.prompt_template.strip():
        raise ValueError("data.prompt_template must not be empty.")
    found_keys = _template_fields(cfg.data.prompt_template)
    unknown_keys = sorted(found_keys - ALLOWED_PROMPT_TEMPLATE_KEYS)
    if unknown_keys:
        raise ValueError(
            "data.prompt_template has unknown placeholders: "
            + ", ".join(unknown_keys)
            + ". Allowed placeholders: "
            + ", ".join(sorted(ALLOWED_PROMPT_TEMPLATE_KEYS))
        )
    if "text" not in found_keys:
        raise ValueError(
            "data.prompt_template must include {text}."
        )

    compute_gradient_accumulation_steps(cfg)
    return cfg


def dump_config(cfg: SFTConfig, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(asdict(cfg), sort_keys=False), encoding="utf-8")
