from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any

from transformers import AutoTokenizer

from .cli import _install_exception_logging, _setup_logging
from .config import DataConfig, GenerationConfig, RLPostTrainConfig, load_config
from .data import load_examples
from .rl_types import Example
from .rollout import compute_prompt_token_lengths
from .utils import configure_huggingface_cache, resolve_huggingface_token


logger = logging.getLogger(__name__)
_PROMPT_LENGTH_CACHE_VERSION = 1


@dataclass(frozen=True)
class PromptLengthCacheInfo:
    path: Path
    cache_key: str
    cache_hit: bool


def _stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _normalize_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _normalize_jsonable(raw) for key, raw in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_jsonable(item) for item in value]
    return str(value)


def _resolve_source_hint(cfg: DataConfig, split: str) -> tuple[str | None, str | None]:
    if split == "train":
        if cfg.train_dir:
            return "dir", cfg.train_dir
        if cfg.train_file:
            return "file", cfg.train_file
        if cfg.hf_dataset_name:
            return "hf", cfg.hf_dataset_name
        return None, None

    if cfg.eval_dir:
        return "dir", cfg.eval_dir
    if cfg.eval_file:
        return "file", cfg.eval_file
    if cfg.train_dir:
        return "dir", cfg.train_dir
    if cfg.train_file:
        return "file", cfg.train_file
    if cfg.hf_dataset_name:
        return "hf", cfg.hf_dataset_name
    return None, None


def resolve_preprocess_cache_dir(cfg: DataConfig, *, split: str) -> Path:
    if cfg.preprocess_cache_dir:
        return Path(cfg.preprocess_cache_dir).resolve()
    if cfg.split_cache_dir:
        return Path(cfg.split_cache_dir).resolve() / "prompt_lengths"

    source_kind, source_value = _resolve_source_hint(cfg, split)
    if source_kind == "dir" and source_value:
        source_path = Path(source_value).resolve()
        return source_path.parent / ".gemma27_preprocess_cache" / source_path.name
    if source_kind == "file" and source_value:
        source_path = Path(source_value).resolve()
        return source_path.parent / ".gemma27_preprocess_cache" / source_path.stem

    dataset_name = str(source_value or cfg.hf_dataset_name or "dataset").strip().replace("/", "__") or "dataset"
    return Path.cwd().resolve() / ".gemma27_preprocess_cache" / dataset_name


def _example_cache_payload(example: Example) -> dict[str, Any]:
    return {
        "example_id": str(example.example_id),
        "src_text": str(example.src_text),
        "src_lang": str(example.src_lang),
        "tgt_lang": str(example.tgt_lang),
        "src_lang_code": example.src_lang_code,
        "tgt_lang_code": example.tgt_lang_code,
        "ref_text": example.ref_text,
        "domain": example.domain,
        "teacher_path": example.teacher_path,
        "input_file_path": example.input_file_path,
    }


def build_examples_fingerprint(examples: list[Example]) -> str:
    digest = hashlib.sha1()
    for example in examples:
        digest.update(_stable_json_dumps(_example_cache_payload(example)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_tokenizer_signature(tokenizer: Any) -> dict[str, Any]:
    try:
        tokenizer_len = int(len(tokenizer))
    except Exception:
        tokenizer_len = None

    return {
        "class_name": tokenizer.__class__.__name__,
        "class_module": tokenizer.__class__.__module__,
        "is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "length": tokenizer_len,
        "vocab_size": _normalize_jsonable(getattr(tokenizer, "vocab_size", None)),
        "pad_token_id": _normalize_jsonable(getattr(tokenizer, "pad_token_id", None)),
        "bos_token_id": _normalize_jsonable(getattr(tokenizer, "bos_token_id", None)),
        "eos_token_id": _normalize_jsonable(getattr(tokenizer, "eos_token_id", None)),
        "special_tokens_map": _normalize_jsonable(getattr(tokenizer, "special_tokens_map", None)),
        "chat_template": str(getattr(tokenizer, "chat_template", "")) or None,
    }


def resolve_prompt_length_cache_path(
    *,
    data_cfg: DataConfig,
    split: str,
    examples: list[Example],
    tokenizer: Any,
    template: str,
    gen_cfg: GenerationConfig,
    limit: int | None,
) -> tuple[Path, str, dict[str, Any]]:
    examples_fingerprint = build_examples_fingerprint(examples)
    tokenizer_signature = build_tokenizer_signature(tokenizer)
    payload = {
        "version": _PROMPT_LENGTH_CACHE_VERSION,
        "split": str(split),
        "limit": int(limit) if limit is not None else None,
        "examples_fingerprint": examples_fingerprint,
        "tokenizer_signature": tokenizer_signature,
        "prompt_template": str(template),
        "chat_template_kwargs": _normalize_jsonable(getattr(gen_cfg, "chat_template_kwargs", None) or {}),
    }
    cache_key = hashlib.sha1(_stable_json_dumps(payload).encode("utf-8")).hexdigest()[:24]
    cache_root = resolve_preprocess_cache_dir(data_cfg, split=split)
    return cache_root / split / f"{cache_key}.json", cache_key, payload


def load_cached_prompt_token_lengths(
    *,
    data_cfg: DataConfig,
    split: str,
    examples: list[Example],
    tokenizer: Any,
    template: str,
    gen_cfg: GenerationConfig,
    limit: int | None,
) -> tuple[list[int] | None, PromptLengthCacheInfo]:
    cache_path, cache_key, expected_metadata = resolve_prompt_length_cache_path(
        data_cfg=data_cfg,
        split=split,
        examples=examples,
        tokenizer=tokenizer,
        template=template,
        gen_cfg=gen_cfg,
        limit=limit,
    )
    info = PromptLengthCacheInfo(path=cache_path, cache_key=cache_key, cache_hit=False)
    if not cache_path.exists():
        return None, info

    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Ignoring unreadable prompt-length cache %s: %s", cache_path, exc)
        return None, info

    prompt_lengths = payload.get("prompt_token_lengths")
    if not isinstance(prompt_lengths, list):
        logger.warning("Ignoring malformed prompt-length cache without prompt_token_lengths: %s", cache_path)
        return None, info

    normalized_lengths: list[int] = []
    try:
        for raw in prompt_lengths:
            normalized_lengths.append(int(raw))
    except Exception:
        logger.warning("Ignoring malformed prompt-length cache with non-integer lengths: %s", cache_path)
        return None, info

    if len(normalized_lengths) != len(examples):
        logger.warning(
            "Ignoring prompt-length cache with mismatched count: cache=%s examples=%s path=%s",
            len(normalized_lengths),
            len(examples),
            cache_path,
        )
        return None, info

    cached_metadata = payload.get("metadata")
    if cached_metadata != expected_metadata:
        logger.warning("Ignoring stale prompt-length cache with mismatched metadata: %s", cache_path)
        return None, info

    return normalized_lengths, PromptLengthCacheInfo(path=cache_path, cache_key=cache_key, cache_hit=True)


def write_prompt_token_lengths_cache(
    *,
    cache_path: Path,
    metadata: dict[str, Any],
    prompt_lengths: list[int],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "count": len(prompt_lengths),
        "prompt_token_length_min": min(prompt_lengths) if prompt_lengths else 0,
        "prompt_token_length_max": max(prompt_lengths) if prompt_lengths else 0,
        "prompt_token_lengths": [int(length) for length in prompt_lengths],
        "version": _PROMPT_LENGTH_CACHE_VERSION,
    }
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(cache_path)


def prepare_prompt_token_lengths(
    *,
    cfg: RLPostTrainConfig,
    split: str,
    examples: list[Example],
    tokenizer: Any,
    limit: int | None,
    force_recompute: bool = False,
) -> tuple[list[int], PromptLengthCacheInfo]:
    cached_lengths, cache_info = load_cached_prompt_token_lengths(
        data_cfg=cfg.data,
        split=split,
        examples=examples,
        tokenizer=tokenizer,
        template=cfg.prompt.template,
        gen_cfg=cfg.generation,
        limit=limit,
    )
    if cached_lengths is not None and not force_recompute:
        return cached_lengths, cache_info

    prompt_lengths = compute_prompt_token_lengths(
        examples=examples,
        tokenizer=tokenizer,
        template=cfg.prompt.template,
        gen_cfg=cfg.generation,
        batch_size=cfg.data.prompt_length_batch_size,
    )
    cache_path, cache_key, metadata = resolve_prompt_length_cache_path(
        data_cfg=cfg.data,
        split=split,
        examples=examples,
        tokenizer=tokenizer,
        template=cfg.prompt.template,
        gen_cfg=cfg.generation,
        limit=limit,
    )
    write_prompt_token_lengths_cache(
        cache_path=cache_path,
        metadata=metadata,
        prompt_lengths=prompt_lengths,
    )
    return prompt_lengths, PromptLengthCacheInfo(path=cache_path, cache_key=cache_key, cache_hit=False)


def _should_prepare_prompt_lengths(cfg: RLPostTrainConfig, *, force_prompt_lengths: bool) -> bool:
    if force_prompt_lengths:
        return True
    batching_strategy = str(cfg.data.batching_strategy or "direction").strip().lower() or "direction"
    return batching_strategy == "direction_domain_length"


def load_preprocess_tokenizer(cfg: RLPostTrainConfig) -> Any:
    tokenizer_name = cfg.model.tokenizer_name_or_path or cfg.model.policy_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=cfg.model.use_fast_tokenizer)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def prepare_dataset_artifacts(
    cfg: RLPostTrainConfig,
    *,
    tokenizer: Any | None = None,
    force_recompute: bool = False,
    force_prompt_lengths: bool = False,
) -> dict[str, Any]:
    train_examples = load_examples(cfg.data, split="train", limit=cfg.data.limit)
    eval_limit = cfg.eval.eval_limit if cfg.eval.eval_limit is not None else cfg.data.eval_limit
    eval_examples = load_examples(cfg.data, split="eval", limit=eval_limit)

    result: dict[str, Any] = {
        "train_count": len(train_examples),
        "eval_count": len(eval_examples),
        "prepared_prompt_lengths": False,
    }

    if _should_prepare_prompt_lengths(cfg, force_prompt_lengths=force_prompt_lengths):
        prep_tokenizer = tokenizer if tokenizer is not None else load_preprocess_tokenizer(cfg)
        prompt_lengths, cache_info = prepare_prompt_token_lengths(
            cfg=cfg,
            split="train",
            examples=train_examples,
            tokenizer=prep_tokenizer,
            limit=cfg.data.limit,
            force_recompute=force_recompute,
        )
        result.update(
            {
                "prepared_prompt_lengths": True,
                "prompt_length_count": len(prompt_lengths),
                "prompt_length_batch_size": int(cfg.data.prompt_length_batch_size),
                "prompt_length_cache_hit": cache_info.cache_hit,
                "prompt_length_cache_path": str(cache_info.path),
                "prompt_length_min": min(prompt_lengths) if prompt_lengths else 0,
                "prompt_length_max": max(prompt_lengths) if prompt_lengths else 0,
            }
        )

    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gemma 27B RL data preprocessing")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute prompt-length caches even when a matching cache already exists",
    )
    parser.add_argument(
        "--prepare-prompt-lengths",
        action="store_true",
        help="Precompute train prompt lengths even when data.batching_strategy is not direction_domain_length",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _setup_logging()
    parser = _build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    log_file = Path(cfg.logging.output_dir) / "preprocess.log"
    _setup_logging(log_file)
    _install_exception_logging()
    logger.info("logging to %s", log_file)

    try:
        hf_token = resolve_huggingface_token(
            explicit_token=cfg.misc.huggingface_token,
            token_env_name=cfg.misc.huggingface_token_env,
        )
        configure_huggingface_cache(cfg.misc.huggingface_cache_dir, token=hf_token)

        summary = prepare_dataset_artifacts(
            cfg,
            force_recompute=bool(args.force),
            force_prompt_lengths=bool(args.prepare_prompt_lengths),
        )
        logger.info("preprocess summary=%s", summary)
        return 0
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except Exception:
        logger.exception("Fatal error in gemma27_rl preprocess")
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
