from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RunConfig:
    out_dir: str = "./runs/exp001"
    seed: int = 1234
    randomize_seed_each_run: bool = True
    log_level: str = "INFO"


@dataclass
class DataConfig:
    madlad_dataset: str = "allenai/MADLAD-400"
    madlad_split: str = "clean"
    src_lang: str = "en"
    tgt_lang: str = "ko"
    src_lang_name: str = "English"
    tgt_lang_name: str = "Korean"
    hf_token_env: str = "HF_TOKEN"
    madlad_revision: str | None = None
    trust_remote_code: bool = True
    local_data_glob: str | None = None
    target_examples_total: int = 10_000
    sample_pool_size: int = 1_000_000
    text_field: str = "text"
    streaming: bool = True
    stop_when_pool_ready: bool = True
    max_scan_docs: int | None = 5_000_000
    sentence_ratio: float = 0.5
    blob_ratio: float = 0.5


@dataclass
class SegmentationConfig:
    mode: str = "auto"
    min_chars: int = 20
    max_chars: int = 5000
    merge_short_lines: bool = True
    short_line_threshold: int = 12
    split_punctuation_regex: str = r"(?<=[.!?。！？])\s+"
    drop_noise: bool = True


@dataclass
class BucketingConfig:
    boundaries: list[int] = field(
        default_factory=lambda: [0, 10, 20, 40, 80, 120, 200, 400, 800, 999_999]
    )
    measure: str = "approx_tokens"
    per_bucket_quota: str = "auto"
    bucket_oversample_factor: float = 2.0


@dataclass
class RetryConfig:
    max_attempts: int = 6
    backoff_s: list[float] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])


@dataclass
class TeacherGenerationConfig:
    max_tokens: int = 512
    top_p: float = 1.0
    greedy_temperature: float = 0.0
    sample_temperature: float = 1.0
    final_temperature: float = 1.0
    seed: int | None = None


@dataclass
class TeacherConfig:
    backend: str = "openai_compatible"
    base_url: str = "https://api.example.com/v1"
    api_key_env: str = "QWEN_API_KEY"
    model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    request_timeout_s: float = 120.0
    sdk_max_retries: int = 0
    unset_proxy_env: bool = True
    max_concurrency: int = 32
    retry: RetryConfig = field(default_factory=RetryConfig)
    generation: TeacherGenerationConfig = field(default_factory=TeacherGenerationConfig)


@dataclass
class BlobConfig:
    enabled: bool = True
    blob_ratio: float = 0.5
    blob_max_tokens: int = 512


@dataclass
class FinalGenerationConfig:
    num_candidates: int = 128
    store_top_k: int = 1
    strategy: str = "auto"
    blob: BlobConfig = field(default_factory=BlobConfig)


@dataclass
class MetricXConfig:
    checkpoint: str = "google/metricx-24-hybrid-large-v2p6"
    batch_size: int = 1
    device: str = "cuda"
    cache_db: str = "./runs/exp001/metricx_cache.sqlite"
    backend: str = "metricx24_cli"
    persistent_worker: bool = True
    worker_start_timeout_s: int = 600
    worker_response_timeout_s: int = 1800
    python_bin: str = ""
    module: str = "metricx24.predict"
    repo_dir: str = ""
    tokenizer: str = "google/mt5-xl"
    max_input_length: int = 1536


@dataclass
class JudgeConfig:
    enabled: bool = True
    model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    temperature: float = 0.0
    max_tokens: int = 128
    fail_policy: str = "conservative"


@dataclass
class FiltersConfig:
    rule_based: bool = True
    min_chars: int = 1
    max_chars: int = 20000
    length_ratio_min: float = 0.2
    length_ratio_max: float = 4.0
    max_copy_overlap: float = 0.8
    blocked_substrings: list[str] = field(
        default_factory=lambda: [
            "Here is the translation",
            "I will translate",
            "번역:",
            "assistant:",
            "user:",
            "system:",
            "<think>",
            "</think>",
            "```",
        ]
    )
    llm_judge: JudgeConfig = field(default_factory=JudgeConfig)


@dataclass
class PipelineConfig:
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    bucketing: BucketingConfig = field(default_factory=BucketingConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    final_generation: FinalGenerationConfig = field(default_factory=FinalGenerationConfig)
    metricx: MetricXConfig = field(default_factory=MetricXConfig)
    filters: FiltersConfig = field(default_factory=FiltersConfig)


_ALLOWED_SPLITS = {"clean", "noisy"}
_ALLOWED_STAGES = {
    "sample_sources",
    "prefilter_score",
    "select_sources",
    "generate_128",
    "score_select_best",
    "score_128_select_best",
    "format_filter",
    "export",
    "all",
}


def _coerce_dataclass(cls: type[Any], data: dict[str, Any]) -> Any:
    kwargs: dict[str, Any] = {}
    for field_info in cls.__dataclass_fields__.values():  # type: ignore[attr-defined]
        name = field_info.name
        if name not in data:
            continue
        value = data[name]
        field_type = field_info.type
        default_value = getattr(cls(), name)
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
    expanded = Path(raw).expanduser()
    if expanded.is_absolute():
        return str(expanded)
    return str((base_dir / expanded).resolve())


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    cfg = _coerce_dataclass(PipelineConfig, payload)
    config_dir = config_path.parent.resolve()

    # Backward-compatible aliases from older specs/examples.
    lang_alias = {
        "eng": "en",
        "kor": "ko",
    }
    cfg.data.src_lang = lang_alias.get(cfg.data.src_lang, cfg.data.src_lang)
    cfg.data.tgt_lang = lang_alias.get(cfg.data.tgt_lang, cfg.data.tgt_lang)

    # Resolve file paths relative to config file location.
    cfg.metricx.python_bin = _resolve_optional_path(cfg.metricx.python_bin, config_dir) or ""
    cfg.metricx.repo_dir = _resolve_optional_path(cfg.metricx.repo_dir, config_dir) or ""
    cfg.data.local_data_glob = _resolve_optional_path(cfg.data.local_data_glob, config_dir)
    # Force MetricX batch size to 1 for stable/compatible execution.
    cfg.metricx.batch_size = 1

    dataset_name = cfg.data.madlad_dataset.strip()
    if "allendai/" in dataset_name.lower():
        raise ValueError(
            "data.madlad_dataset has a typo: 'allendai'. "
            "Use 'allenai/MADLAD-400'."
        )
    if cfg.data.madlad_split not in _ALLOWED_SPLITS:
        raise ValueError(f"data.madlad_split must be one of {_ALLOWED_SPLITS}")
    if cfg.data.sample_pool_size <= 0:
        raise ValueError("data.sample_pool_size must be > 0")
    if cfg.data.target_examples_total <= 0:
        raise ValueError("data.target_examples_total must be > 0")
    if cfg.data.max_scan_docs is not None and cfg.data.max_scan_docs <= 0:
        raise ValueError("data.max_scan_docs must be > 0 when set")
    if cfg.teacher.max_concurrency <= 0:
        raise ValueError("teacher.max_concurrency must be > 0")
    if cfg.teacher.sdk_max_retries < 0:
        raise ValueError("teacher.sdk_max_retries must be >= 0")
    if cfg.metricx.worker_start_timeout_s <= 0:
        raise ValueError("metricx.worker_start_timeout_s must be > 0")
    if cfg.metricx.worker_response_timeout_s <= 0:
        raise ValueError("metricx.worker_response_timeout_s must be > 0")
    if cfg.final_generation.num_candidates <= 0:
        raise ValueError("final_generation.num_candidates must be > 0")
    if cfg.metricx.python_bin and not Path(cfg.metricx.python_bin).exists():
        raise ValueError(
            f"metricx.python_bin does not exist: {cfg.metricx.python_bin} "
            "(path is resolved relative to the config file location)."
        )

    return cfg


def validate_stage(stage: str) -> str:
    if stage not in _ALLOWED_STAGES:
        raise ValueError(f"Unknown stage: {stage}. Allowed: {sorted(_ALLOWED_STAGES)}")
    return stage


def dump_config(cfg: PipelineConfig, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(asdict(cfg), sort_keys=False), encoding="utf-8")


def resolve_metricx_cache_path(cfg: PipelineConfig) -> Path:
    path = Path(cfg.metricx.cache_db)
    if not path.is_absolute():
        path = Path(cfg.run.out_dir) / path
    return path
