from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase

from .config import DataConfig, SFTConfig


logger = logging.getLogger(__name__)


_WMT_LANGUAGE_NAMES: dict[str, str] = {
    "ar": "Arabic",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "gu": "Gujarati",
    "ha": "Hausa",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "kk": "Kazakh",
    "km": "Khmer",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mr": "Marathi",
    "ms": "Malay",
    "mt": "Maltese",
    "ne": "Nepali",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "ps": "Pashto",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sr": "Serbian",
    "sv": "Swedish",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "zh": "Chinese",
}


def _normalize_code(code: str) -> str:
    return code.strip().replace("_", "-").lower()


def _resolve_language_name(name: str, code: str) -> str:
    if name and name.strip() and name.strip().lower() != "auto":
        return name.strip()
    normalized_code = _normalize_code(code)
    if normalized_code in _WMT_LANGUAGE_NAMES:
        return _WMT_LANGUAGE_NAMES[normalized_code]
    base_code = normalized_code.split("-", 1)[0]
    if base_code in _WMT_LANGUAGE_NAMES:
        return _WMT_LANGUAGE_NAMES[base_code]
    return normalized_code


def _example_value(example: dict[str, Any], field_name: str | None) -> str | None:
    if not field_name:
        return None
    if field_name not in example:
        return None
    value = str(example[field_name]).strip()
    return value or None


def _resolve_languages(data_cfg: DataConfig, example: dict[str, Any]) -> tuple[str, str, str, str]:
    src_code = _example_value(example, data_cfg.source_lang_code_field) or str(data_cfg.source_lang_code).strip()
    tgt_code = _example_value(example, data_cfg.target_lang_code_field) or str(data_cfg.target_lang_code).strip()
    if not src_code or not tgt_code:
        raise ValueError("source/target language code is empty. Set code fields or fixed codes in config.")
    src_code = _normalize_code(src_code)
    tgt_code = _normalize_code(tgt_code)

    src_name_raw = _example_value(example, data_cfg.source_lang_name_field) or data_cfg.source_lang_name
    tgt_name_raw = _example_value(example, data_cfg.target_lang_name_field) or data_cfg.target_lang_name
    src_name = _resolve_language_name(src_name_raw, src_code)
    tgt_name = _resolve_language_name(tgt_name_raw, tgt_code)
    return src_name, src_code, tgt_name, tgt_code


def _build_prompt(data_cfg: DataConfig, source_text: str, source_lang: str, src_lang_code: str, target_lang: str, tgt_lang_code: str) -> str:
    variables = {
        "source_lang": source_lang,
        "src_lang_code": src_lang_code,
        "target_lang": target_lang,
        "tgt_lang_code": tgt_lang_code,
        "text": source_text,
    }
    try:
        return data_cfg.prompt_template.format(**variables)
    except KeyError as exc:
        missing = exc.args[0]
        raise ValueError(
            f"Unknown placeholder in data.prompt_template: {missing}. "
            "Allowed placeholders: source_lang, src_lang_code, target_lang, tgt_lang_code, text."
        ) from exc


def _messages(data_cfg: DataConfig, example: dict[str, Any], source_text: str, target_text: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    source_lang, src_lang_code, target_lang, tgt_lang_code = _resolve_languages(data_cfg, example)
    user = _build_prompt(
        data_cfg=data_cfg,
        source_text=source_text,
        source_lang=source_lang,
        src_lang_code=src_lang_code,
        target_lang=target_lang,
        tgt_lang_code=tgt_lang_code,
    )
    prompt_messages = [{"role": "user", "content": user}]
    full_messages = prompt_messages + [{"role": "assistant", "content": target_text}]
    return prompt_messages, full_messages


def _apply_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    add_generation_prompt: bool,
    max_seq_length: int,
    truncate: bool = True,
) -> list[int]:
    if getattr(tokenizer, "chat_template", None):
        kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": add_generation_prompt,
            "truncation": truncate,
        }
        if truncate:
            kwargs["max_length"] = max_seq_length
        ids = tokenizer.apply_chat_template(messages, **kwargs)
        return list(ids)

    parts = []
    for msg in messages:
        role = msg["role"].upper()
        parts.append(f"{role}: {msg['content']}")
    if add_generation_prompt:
        parts.append("ASSISTANT:")
    text = "\n\n".join(parts)
    tok_kwargs: dict[str, Any] = {
        "truncation": truncate,
        "add_special_tokens": True,
    }
    if truncate:
        tok_kwargs["max_length"] = max_seq_length
    return list(tokenizer(text, **tok_kwargs)["input_ids"])


def _build_tokenize_fn(cfg: SFTConfig, tokenizer: PreTrainedTokenizerBase):
    data_cfg = cfg.data
    max_len = cfg.train.max_seq_length

    def _tokenize(example: dict[str, Any]) -> dict[str, Any]:
        source_text_original = str(example[data_cfg.source_field])
        source_text = source_text_original
        target_text = str(example[data_cfg.target_field])
        source_shrink_steps = 0
        prompt_ids: list[int] = []
        full_ids: list[int] = []
        labels: list[int] = []
        non_ignored = 0

        target_ids_raw = list(
            tokenizer(
                target_text,
                truncation=False,
                add_special_tokens=False,
            )["input_ids"]
        )
        has_target = len(target_ids_raw) > 0 and bool(target_text.strip())

        # Build training sample as: [prompt-with-generation-prefix] + [target tokens].
        # This avoids template-diff corner cases that can produce zero target labels.
        while True:
            prompt_messages, _ = _messages(data_cfg, example, source_text, target_text)
            prompt_ids_raw = _apply_chat_template(
                tokenizer=tokenizer,
                messages=prompt_messages,
                add_generation_prompt=True,
                max_seq_length=max_len,
                truncate=False,
            )

            if not has_target:
                prompt_ids = prompt_ids_raw[:max_len]
                target_ids = []
            else:
                if len(prompt_ids_raw) + len(target_ids_raw) <= max_len:
                    prompt_ids = prompt_ids_raw
                    target_ids = target_ids_raw
                else:
                    target_reserve = min(len(target_ids_raw), max(32, max_len // 8))
                    prompt_budget = max(0, max_len - target_reserve)
                    prompt_ids = prompt_ids_raw[:prompt_budget]
                    target_budget = max(0, max_len - len(prompt_ids))
                    target_ids = target_ids_raw[:target_budget]

            full_ids = prompt_ids + target_ids
            labels = ([-100] * len(prompt_ids)) + target_ids
            non_ignored = len(target_ids)

            if non_ignored > 0 or not has_target or len(source_text) <= 1 or source_shrink_steps >= 6:
                break
            source_text = source_text[: max(1, len(source_text) // 2)]
            source_shrink_steps += 1

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
            "num_target_tokens": non_ignored,
            "prompt_tokens": len(prompt_ids),
            "full_tokens": len(full_ids),
            "source_chars": len(source_text),
            "source_chars_original": len(source_text_original),
            "source_shrink_steps": source_shrink_steps,
            "target_chars": len(target_text),
        }

    return _tokenize


@dataclass
class CompletionDataCollator:
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int | None = 8

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch_inputs = [
            {"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]}
            for f in features
        ]
        padded = self.tokenizer.pad(
            batch_inputs,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        seq_len = int(padded["input_ids"].shape[1])
        labels = []
        for feature in features:
            raw = feature["labels"]
            labels.append(raw + [-100] * (seq_len - len(raw)))
        padded["labels"] = torch.tensor(labels, dtype=torch.long)
        return padded


def _load_json_dataset(path: str) -> Dataset:
    ds = load_dataset("json", data_files=path, split="train")
    assert isinstance(ds, Dataset)
    return ds


def _safe_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _required_json_fields(cfg: DataConfig) -> list[str]:
    fields = [cfg.source_field, cfg.target_field]
    for name in [
        cfg.source_lang_name_field,
        cfg.target_lang_name_field,
        cfg.source_lang_code_field,
        cfg.target_lang_code_field,
    ]:
        if name and name not in fields:
            fields.append(name)
    return fields


def _safe_load_json_dataset(path: str, cfg: DataConfig) -> Dataset:
    required_fields = _required_json_fields(cfg)
    rows: list[dict[str, str]] = []
    bad_json = 0
    total_lines = 0
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            total_lines += 1
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                bad_json += 1
                if bad_json <= 3:
                    logger.warning("Skipping invalid JSON line=%s in %s", line_no, path)
                continue
            if not isinstance(record, dict):
                continue
            out: dict[str, str] = {}
            for field in required_fields:
                out[field] = _safe_string(record.get(field))
            rows.append(out)
    if bad_json > 0:
        logger.warning("Ignored invalid JSON lines=%s while reading %s", bad_json, path)
    logger.info("Safe JSON loader rows=%s total_lines=%s required_fields=%s", len(rows), total_lines, required_fields)
    return Dataset.from_list(rows)


def _load_json_dataset_resilient(path: str, cfg: DataConfig) -> Dataset:
    try:
        return _load_json_dataset(path)
    except Exception as exc:  # pylint: disable=broad-except
        msg = str(exc).lower()
        if "arrowtypeerror" in type(exc).__name__.lower() or "expected bytes" in msg:
            logger.warning(
                "Falling back to safe JSON loader due to Arrow type mismatch: %s: %s",
                type(exc).__name__,
                exc,
            )
            return _safe_load_json_dataset(path, cfg)
        raise


def _summarize_tokenization(ds: Dataset, max_seq_length: int, sample_limit: int = 4096) -> dict[str, int | float]:
    sample_size = min(len(ds), sample_limit)
    if sample_size == 0:
        return {
            "sample_size": 0,
            "zero_target_count": 0,
            "empty_target_text_count": 0,
            "prompt_ge_max_count": 0,
            "full_ge_max_count": 0,
            "max_prompt_tokens": 0,
            "max_full_tokens": 0,
            "mean_target_tokens": 0.0,
            "source_shrunk_count": 0,
            "max_source_shrink_steps": 0,
        }

    sampled = ds.select(range(sample_size))
    num_target_tokens = sampled["num_target_tokens"]
    prompt_tokens = sampled["prompt_tokens"]
    full_tokens = sampled["full_tokens"]
    target_chars = sampled["target_chars"]
    source_shrink_steps = sampled["source_shrink_steps"]

    zero_target_count = sum(1 for v in num_target_tokens if int(v) <= 0)
    empty_target_text_count = sum(1 for v in target_chars if int(v) <= 0)
    prompt_ge_max_count = sum(1 for v in prompt_tokens if int(v) >= max_seq_length)
    full_ge_max_count = sum(1 for v in full_tokens if int(v) >= max_seq_length)
    max_prompt_tokens = max((int(v) for v in prompt_tokens), default=0)
    max_full_tokens = max((int(v) for v in full_tokens), default=0)
    mean_target_tokens = float(sum(int(v) for v in num_target_tokens)) / float(sample_size)
    source_shrunk_count = sum(1 for v in source_shrink_steps if int(v) > 0)
    max_source_shrink_steps = max((int(v) for v in source_shrink_steps), default=0)

    return {
        "sample_size": sample_size,
        "zero_target_count": zero_target_count,
        "empty_target_text_count": empty_target_text_count,
        "prompt_ge_max_count": prompt_ge_max_count,
        "full_ge_max_count": full_ge_max_count,
        "max_prompt_tokens": max_prompt_tokens,
        "max_full_tokens": max_full_tokens,
        "mean_target_tokens": mean_target_tokens,
        "source_shrunk_count": source_shrunk_count,
        "max_source_shrink_steps": max_source_shrink_steps,
    }


def _raise_empty_train_dataset_error(cfg: SFTConfig, raw_rows: int, token_stats: dict[str, int | float]) -> None:
    raise ValueError(
        "All training rows were filtered out (num_target_tokens == 0).\n"
        f"- data.train_file={cfg.data.train_file}\n"
        f"- rows_before_filter={raw_rows}\n"
        f"- sampled_rows={token_stats['sample_size']} sampled_zero_target={token_stats['zero_target_count']}\n"
        f"- sampled_empty_target_text={token_stats['empty_target_text_count']}\n"
        f"- sampled_prompt_ge_max_seq={token_stats['prompt_ge_max_count']} (max_seq_length={cfg.train.max_seq_length})\n"
        f"- sampled_source_shrunk={token_stats['source_shrunk_count']} max_shrink_steps={token_stats['max_source_shrink_steps']}\n"
        f"- sampled_max_prompt_tokens={token_stats['max_prompt_tokens']} sampled_max_full_tokens={token_stats['max_full_tokens']}\n"
        "Likely causes:\n"
        "1) data.target_field points to empty/wrong column.\n"
        "2) Prompt + source text is too long and truncates away assistant tokens.\n"
        "3) train.max_seq_length is too small for this prompt template.\n"
        "Try increasing train.max_seq_length, shortening source/prompt, or checking source/target fields."
    )


def build_datasets(cfg: SFTConfig, tokenizer: PreTrainedTokenizerBase) -> tuple[Dataset, Dataset | None]:
    data_cfg: DataConfig = cfg.data
    train_ds = _load_json_dataset_resilient(data_cfg.train_file, data_cfg)
    raw_train_rows = len(train_ds)
    logger.info("Loaded train dataset rows=%s path=%s", raw_train_rows, data_cfg.train_file)
    if raw_train_rows == 0:
        raise ValueError(f"Train dataset is empty: {data_cfg.train_file}")
    if data_cfg.max_train_samples is not None:
        train_ds = train_ds.select(range(min(len(train_ds), data_cfg.max_train_samples)))
        logger.info("Applied data.max_train_samples=%s -> train rows=%s", data_cfg.max_train_samples, len(train_ds))

    eval_ds = None
    if data_cfg.eval_file:
        eval_ds = _load_json_dataset_resilient(data_cfg.eval_file, data_cfg)
        logger.info("Loaded eval dataset rows=%s path=%s", len(eval_ds), data_cfg.eval_file)
        if data_cfg.max_eval_samples is not None:
            eval_ds = eval_ds.select(range(min(len(eval_ds), data_cfg.max_eval_samples)))
            logger.info("Applied data.max_eval_samples=%s -> eval rows=%s", data_cfg.max_eval_samples, len(eval_ds))

    logger.info(
        "Tokenization language setup src_code=%s tgt_code=%s src_code_field=%s tgt_code_field=%s",
        cfg.data.source_lang_code,
        cfg.data.target_lang_code,
        cfg.data.source_lang_code_field,
        cfg.data.target_lang_code_field,
    )
    tokenize_fn = _build_tokenize_fn(cfg, tokenizer)
    train_mapped = train_ds.map(
        tokenize_fn,
        num_proc=data_cfg.preprocessing_num_workers or None,
        desc="Tokenizing train dataset",
    )
    train_stats = _summarize_tokenization(train_mapped, cfg.train.max_seq_length)
    logger.info(
        "Train tokenization stats sample_size=%s zero_target=%s empty_target=%s prompt_ge_max=%s source_shrunk=%s max_shrink_steps=%s max_prompt=%s max_full=%s mean_target_tokens=%.2f",
        train_stats["sample_size"],
        train_stats["zero_target_count"],
        train_stats["empty_target_text_count"],
        train_stats["prompt_ge_max_count"],
        train_stats["source_shrunk_count"],
        train_stats["max_source_shrink_steps"],
        train_stats["max_prompt_tokens"],
        train_stats["max_full_tokens"],
        train_stats["mean_target_tokens"],
    )
    train_filtered = train_mapped.filter(
        lambda ex: ex["num_target_tokens"] > 0,
        num_proc=data_cfg.preprocessing_num_workers or None,
        desc="Filtering empty-train labels",
    )
    train_kept = len(train_filtered)
    logger.info("Train filtering kept=%s dropped=%s", train_kept, len(train_mapped) - train_kept)
    if train_kept == 0:
        _raise_empty_train_dataset_error(cfg, len(train_mapped), train_stats)
    train_ds = train_filtered.remove_columns(
        [c for c in train_filtered.column_names if c not in {"input_ids", "attention_mask", "labels"}]
    )

    if eval_ds is not None:
        eval_mapped = eval_ds.map(
            tokenize_fn,
            num_proc=data_cfg.preprocessing_num_workers or None,
            desc="Tokenizing eval dataset",
        )
        eval_stats = _summarize_tokenization(eval_mapped, cfg.train.max_seq_length)
        logger.info(
            "Eval tokenization stats sample_size=%s zero_target=%s empty_target=%s prompt_ge_max=%s source_shrunk=%s max_shrink_steps=%s max_prompt=%s max_full=%s mean_target_tokens=%.2f",
            eval_stats["sample_size"],
            eval_stats["zero_target_count"],
            eval_stats["empty_target_text_count"],
            eval_stats["prompt_ge_max_count"],
            eval_stats["source_shrunk_count"],
            eval_stats["max_source_shrink_steps"],
            eval_stats["max_prompt_tokens"],
            eval_stats["max_full_tokens"],
            eval_stats["mean_target_tokens"],
        )
        eval_filtered = eval_mapped.filter(
            lambda ex: ex["num_target_tokens"] > 0,
            num_proc=data_cfg.preprocessing_num_workers or None,
            desc="Filtering empty-eval labels",
        )
        eval_kept = len(eval_filtered)
        logger.info("Eval filtering kept=%s dropped=%s", eval_kept, len(eval_mapped) - eval_kept)
        if eval_kept == 0:
            logger.warning("Eval dataset became empty after filtering; training will continue without eval.")
            eval_ds = None
        else:
            eval_ds = eval_filtered.remove_columns(
                [c for c in eval_filtered.column_names if c not in {"input_ids", "attention_mask", "labels"}]
            )

    logger.info(
        "Final prepared datasets train=%s eval=%s",
        len(train_ds),
        len(eval_ds) if eval_ds is not None else 0,
    )

    return train_ds, eval_ds
