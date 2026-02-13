from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, Features, Value
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


def _coerce_token_ids(value: Any, field_name: str) -> list[int]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        # Handle dict-like containers such as transformers.BatchEncoding.
        for key in (field_name, "input_ids", "ids", "labels", "attention_mask"):
            if key in value:
                return _coerce_token_ids(value[key], field_name)
        if len(value) == 1:
            return _coerce_token_ids(next(iter(value.values())), field_name)
        raise ValueError(f"{field_name} mapping has no token-id-like key: {list(value.keys())[:8]}")
    data = getattr(value, "data", None)
    if isinstance(data, Mapping):
        return _coerce_token_ids(data, field_name)
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return _coerce_token_ids(value.tolist(), field_name)
        except Exception:  # pylint: disable=broad-except
            pass
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, list):
        if not value:
            return []
        if isinstance(value[0], list):
            flat: list[int] = []
            for item in value:
                flat.extend(_coerce_token_ids(item, field_name))
            return flat
        try:
            return [int(v) for v in value]
        except Exception as exc:  # pylint: disable=broad-except
            raise ValueError(f"Invalid {field_name} list values: {value[:8]}") from exc
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            decoded = json.loads(raw)
            if isinstance(decoded, list):
                return [int(v) for v in decoded]
            if isinstance(decoded, int):
                return [int(decoded)]
        except Exception:  # pylint: disable=broad-except
            pass
        # Fallback: allow whitespace-separated token id strings.
        parts = raw.split()
        if parts and all(part.lstrip("-").isdigit() for part in parts):
            return [int(part) for part in parts]
        raise ValueError(f"{field_name} is a non-token string: {raw[:120]!r}")
    raise ValueError(f"type of {field_name} is unknown: {type(value)}")


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
        # Some tokenizer/template combinations can return non-list types.
        if isinstance(ids, str):
            tok_kwargs: dict[str, Any] = {"add_special_tokens": False, "truncation": truncate}
            if truncate:
                tok_kwargs["max_length"] = max_seq_length
            ids = tokenizer(ids, **tok_kwargs)["input_ids"]
        return _coerce_token_ids(ids, "input_ids")

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
    return _coerce_token_ids(tokenizer(text, **tok_kwargs)["input_ids"], "input_ids")


def _common_prefix_len(left: list[int], right: list[int]) -> int:
    n = min(len(left), len(right))
    i = 0
    while i < n and left[i] == right[i]:
        i += 1
    return i


def _common_suffix_len(left: list[int], right: list[int], left_stop: int, right_stop: int) -> int:
    i = 0
    while left_stop - i - 1 >= 0 and right_stop - i - 1 >= 0:
        if left[left_stop - i - 1] != right[right_stop - i - 1]:
            break
        i += 1
    return i


def _extract_template_target_span(
    tokenizer: PreTrainedTokenizerBase,
    prompt_messages: list[dict[str, str]],
    target_text: str,
    max_seq_length: int,
) -> tuple[list[int], list[int]] | None:
    if not getattr(tokenizer, "chat_template", None):
        return None

    full_with_target = prompt_messages + [{"role": "assistant", "content": target_text}]
    full_with_empty = prompt_messages + [{"role": "assistant", "content": ""}]

    target_ids_full = _apply_chat_template(
        tokenizer=tokenizer,
        messages=full_with_target,
        add_generation_prompt=False,
        max_seq_length=max_seq_length,
        truncate=False,
    )
    empty_ids_full = _apply_chat_template(
        tokenizer=tokenizer,
        messages=full_with_empty,
        add_generation_prompt=False,
        max_seq_length=max_seq_length,
        truncate=False,
    )
    if not target_ids_full:
        return None

    prefix_len = _common_prefix_len(empty_ids_full, target_ids_full)
    suffix_len = _common_suffix_len(
        empty_ids_full,
        target_ids_full,
        left_stop=len(empty_ids_full),
        right_stop=len(target_ids_full),
    )
    # Prevent over-trimming when target text happens to share tail tokens.
    max_wrapper_suffix = max(0, len(empty_ids_full) - prefix_len)
    suffix_len = min(suffix_len, max_wrapper_suffix)
    target_end = max(prefix_len, len(target_ids_full) - suffix_len)
    target_span = target_ids_full[prefix_len:target_end]
    if not target_span:
        return None
    prompt_prefix = target_ids_full[:prefix_len]
    return prompt_prefix, target_span


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

        target_ids_fallback = list(
            tokenizer(
                target_text,
                truncation=False,
                add_special_tokens=False,
            )["input_ids"]
        )
        target_ids_fallback = _coerce_token_ids(target_ids_fallback, "target_ids")
        has_target = len(target_ids_fallback) > 0 and bool(target_text.strip())

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
            target_ids_effective = target_ids_fallback
            template_span = _extract_template_target_span(
                tokenizer=tokenizer,
                prompt_messages=prompt_messages,
                target_text=target_text,
                max_seq_length=max_len,
            )
            if template_span is not None:
                template_prompt_ids, template_target_ids = template_span
                if template_target_ids:
                    prompt_ids_raw = template_prompt_ids
                    target_ids_effective = template_target_ids

            if not has_target:
                prompt_ids = prompt_ids_raw[:max_len]
                target_ids = []
            else:
                if len(prompt_ids_raw) + len(target_ids_effective) <= max_len:
                    prompt_ids = prompt_ids_raw
                    target_ids = target_ids_effective
                else:
                    target_reserve = min(len(target_ids_effective), max(32, max_len // 8))
                    prompt_budget = max(0, max_len - target_reserve)
                    prompt_ids = prompt_ids_raw[:prompt_budget]
                    target_budget = max(0, max_len - len(prompt_ids))
                    target_ids = target_ids_effective[:target_budget]

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
        normalized_features = []
        for idx, feature in enumerate(features):
            try:
                input_ids = _coerce_token_ids(feature.get("input_ids"), "input_ids")
                attention_mask = _coerce_token_ids(feature.get("attention_mask"), "attention_mask")
                labels = _coerce_token_ids(feature.get("labels"), "labels")
                token_type_ids = _coerce_token_ids(feature.get("token_type_ids"), "token_type_ids")
                if not token_type_ids:
                    token_type_ids = [0] * len(input_ids)
                elif len(token_type_ids) < len(input_ids):
                    token_type_ids = token_type_ids + ([0] * (len(input_ids) - len(token_type_ids)))
                elif len(token_type_ids) > len(input_ids):
                    token_type_ids = token_type_ids[: len(input_ids)]
            except Exception as exc:  # pylint: disable=broad-except
                raise ValueError(
                    f"Feature normalization failed at batch_index={idx}: {exc}. "
                    f"feature_keys={list(feature.keys())}"
                ) from exc
            normalized_features.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "token_type_ids": token_type_ids,
                }
            )

        batch_inputs = [
            {
                "input_ids": f["input_ids"],
                "attention_mask": f["attention_mask"],
                "token_type_ids": f["token_type_ids"],
            }
            for f in normalized_features
        ]
        padded = self.tokenizer.pad(
            batch_inputs,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        if "token_type_ids" not in padded:
            padded["token_type_ids"] = torch.zeros_like(padded["input_ids"], dtype=torch.long)
        else:
            padded["token_type_ids"] = padded["token_type_ids"].to(dtype=torch.long)
        seq_len = int(padded["input_ids"].shape[1])
        labels = []
        for feature in normalized_features:
            raw = feature["labels"]
            labels.append(raw + [-100] * (seq_len - len(raw)))
        padded["labels"] = torch.tensor(labels, dtype=torch.long)
        return padded


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


def _dataset_features(cfg: DataConfig) -> Features:
    return Features({field: Value("string") for field in _required_json_fields(cfg)})


def _safe_load_json_dataset(path: str, cfg: DataConfig) -> Dataset:
    required_fields = _required_json_fields(cfg)
    stats: dict[str, Any] = {
        "bad_json": 0,
        "total_lines": 0,
        "rows": 0,
        "non_string_counts": {field: 0 for field in required_fields},
        "missing_counts": {field: 0 for field in required_fields},
        "sample_cast_logs": 0,
    }

    def _generator():
        with Path(path).open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                stats["total_lines"] += 1
                raw = line.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError:
                    stats["bad_json"] += 1
                    if stats["bad_json"] <= 3:
                        logger.warning("Skipping invalid JSON line=%s in %s", line_no, path)
                    continue
                if not isinstance(record, dict):
                    continue
                out: dict[str, str] = {}
                for field in required_fields:
                    value = record.get(field)
                    if value is None:
                        stats["missing_counts"][field] += 1
                    elif not isinstance(value, str):
                        stats["non_string_counts"][field] += 1
                        if stats["sample_cast_logs"] < 8:
                            logger.warning(
                                "Casting non-string field=%s line=%s type=%s -> string",
                                field,
                                line_no,
                                type(value).__name__,
                            )
                            stats["sample_cast_logs"] += 1
                    out[field] = _safe_string(value)
                stats["rows"] += 1
                if stats["rows"] % 50000 == 0:
                    logger.info("JSON->datasets progress path=%s rows=%s", path, stats["rows"])
                yield out

    ds = Dataset.from_generator(
        _generator,
        features=_dataset_features(cfg),
    )
    if stats["bad_json"] > 0:
        logger.warning("Ignored invalid JSON lines=%s while reading %s", stats["bad_json"], path)
    logger.info(
        "datasets JSON loader rows=%s total_lines=%s required_fields=%s non_string_counts=%s missing_counts=%s",
        len(ds),
        stats["total_lines"],
        required_fields,
        stats["non_string_counts"],
        stats["missing_counts"],
    )
    return ds


def _load_json_dataset_resilient(path: str, cfg: DataConfig) -> Dataset:
    # Use `datasets` with a controlled generator + explicit string features.
    # This keeps the training path in HF datasets while preventing mixed-type
    # Arrow inference issues from raw JSONL.
    return _safe_load_json_dataset(path, cfg)


def _summarize_tokenization(dataset: Dataset, max_seq_length: int, sample_limit: int = 4096) -> dict[str, int | float]:
    sample_size = min(len(dataset), sample_limit)
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

    sampled = dataset.select(range(sample_size))
    num_target_tokens = [int(v) for v in sampled["num_target_tokens"]]
    prompt_tokens = [int(v) for v in sampled["prompt_tokens"]]
    full_tokens = [int(v) for v in sampled["full_tokens"]]
    target_chars = [int(v) for v in sampled["target_chars"]]
    source_shrink_steps = [int(v) for v in sampled["source_shrink_steps"]]

    zero_target_count = sum(1 for v in num_target_tokens if v <= 0)
    empty_target_text_count = sum(1 for v in target_chars if v <= 0)
    prompt_ge_max_count = sum(1 for v in prompt_tokens if v >= max_seq_length)
    full_ge_max_count = sum(1 for v in full_tokens if v >= max_seq_length)
    max_prompt_tokens = max(prompt_tokens, default=0)
    max_full_tokens = max(full_tokens, default=0)
    mean_target_tokens = float(sum(num_target_tokens)) / float(sample_size)
    source_shrunk_count = sum(1 for v in source_shrink_steps if v > 0)
    max_source_shrink_steps = max(source_shrink_steps, default=0)

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


def _tokenize_dataset(
    dataset: Dataset,
    tokenize_fn,
    split_name: str,
    num_workers: int,
) -> Dataset:
    map_kwargs: dict[str, Any] = {"desc": f"{split_name} tokenization"}
    if num_workers > 1:
        map_kwargs["num_proc"] = num_workers
    try:
        return dataset.map(tokenize_fn, **map_kwargs)
    except Exception as exc:
        if "num_proc" in map_kwargs:
            logger.warning(
                "%s tokenization with num_proc=%s failed (%s). Retrying with num_proc=1.",
                split_name,
                num_workers,
                exc,
            )
            map_kwargs.pop("num_proc", None)
            return dataset.map(tokenize_fn, **map_kwargs)
        raise


def _to_training_dataset(dataset: Dataset) -> Dataset:
    keep_cols = {"input_ids", "attention_mask", "labels"}
    drop_cols = [name for name in dataset.column_names if name not in keep_cols]
    if drop_cols:
        dataset = dataset.remove_columns(drop_cols)
    return dataset


def _truncate_for_log(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    extra = len(text) - max_chars
    return f"{text[:max_chars]} ... [truncated {extra} chars]"


def _log_pre_tokenization_samples(dataset: Dataset, cfg: SFTConfig, split_name: str) -> None:
    sample_limit = min(len(dataset), cfg.data.log_text_samples)
    if sample_limit <= 0:
        return

    logger.info("%s pre-tokenization preview sample_count=%s", split_name, sample_limit)
    sampled = dataset.select(range(sample_limit))
    max_chars = cfg.data.log_text_max_chars

    for idx, row in enumerate(sampled):
        try:
            source_text = _safe_string(row.get(cfg.data.source_field))
            target_text = _safe_string(row.get(cfg.data.target_field))
            source_lang, src_lang_code, target_lang, tgt_lang_code = _resolve_languages(cfg.data, row)
            prompt_messages, _ = _messages(cfg.data, row, source_text, target_text)
            prompt_text = prompt_messages[0]["content"] if prompt_messages else ""
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("%s pre-tokenization preview failed idx=%s: %s", split_name, idx, exc)
            continue

        source_log = _truncate_for_log(source_text, max_chars)
        prompt_log = _truncate_for_log(prompt_text, max_chars)
        target_log = _truncate_for_log(target_text, max_chars)
        logger.info(
            "%s pre-tokenization sample idx=%s src=%s(%s) tgt=%s(%s) chars(source/prompt/target)=%s/%s/%s\n"
            "SOURCE>>> %s\n"
            "PROMPT>>> %s\n"
            "TARGET>>> %s",
            split_name,
            idx,
            source_lang,
            src_lang_code,
            target_lang,
            tgt_lang_code,
            len(source_text),
            len(prompt_text),
            len(target_text),
            source_log,
            prompt_log,
            target_log,
        )


def build_datasets(cfg: SFTConfig, tokenizer: PreTrainedTokenizerBase) -> tuple[Dataset, Dataset | None]:
    data_cfg: DataConfig = cfg.data
    train_rows = _load_json_dataset_resilient(data_cfg.train_file, data_cfg)
    raw_train_rows = len(train_rows)
    logger.info("Loaded train dataset rows=%s path=%s", raw_train_rows, data_cfg.train_file)
    if raw_train_rows == 0:
        raise ValueError(f"Train dataset is empty: {data_cfg.train_file}")
    if data_cfg.max_train_samples is not None:
        sample_count = min(len(train_rows), data_cfg.max_train_samples)
        train_rows = train_rows.select(range(sample_count))
        logger.info("Applied data.max_train_samples=%s -> train rows=%s", data_cfg.max_train_samples, sample_count)

    eval_rows: Dataset | None = None
    if data_cfg.eval_file:
        eval_rows = _load_json_dataset_resilient(data_cfg.eval_file, data_cfg)
        logger.info("Loaded eval dataset rows=%s path=%s", len(eval_rows), data_cfg.eval_file)
        if data_cfg.max_eval_samples is not None:
            sample_count = min(len(eval_rows), data_cfg.max_eval_samples)
            eval_rows = eval_rows.select(range(sample_count))
            logger.info("Applied data.max_eval_samples=%s -> eval rows=%s", data_cfg.max_eval_samples, sample_count)

    logger.info(
        "Tokenization language setup src_code=%s tgt_code=%s src_code_field=%s tgt_code_field=%s",
        cfg.data.source_lang_code,
        cfg.data.target_lang_code,
        cfg.data.source_lang_code_field,
        cfg.data.target_lang_code_field,
    )
    _log_pre_tokenization_samples(train_rows, cfg, "Train")
    if eval_rows is not None:
        _log_pre_tokenization_samples(eval_rows, cfg, "Eval")
    tokenize_fn = _build_tokenize_fn(cfg, tokenizer)
    train_mapped = _tokenize_dataset(
        train_rows,
        tokenize_fn=tokenize_fn,
        split_name="Train",
        num_workers=data_cfg.preprocessing_num_workers,
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
        lambda row: int(row.get("num_target_tokens", 0)) > 0,
        desc="Train filtering non-empty target",
    )
    train_kept = len(train_filtered)
    logger.info("Train filtering kept=%s dropped=%s", train_kept, len(train_mapped) - train_kept)
    if train_kept == 0:
        _raise_empty_train_dataset_error(cfg, len(train_mapped), train_stats)
    train_ds = _to_training_dataset(train_filtered)

    eval_ds: Dataset | None = None
    if eval_rows is not None:
        eval_mapped = _tokenize_dataset(
            eval_rows,
            tokenize_fn=tokenize_fn,
            split_name="Eval",
            num_workers=data_cfg.preprocessing_num_workers,
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
            lambda row: int(row.get("num_target_tokens", 0)) > 0,
            desc="Eval filtering non-empty target",
        )
        eval_kept = len(eval_filtered)
        logger.info("Eval filtering kept=%s dropped=%s", eval_kept, len(eval_mapped) - eval_kept)
        if eval_kept == 0:
            logger.warning("Eval dataset became empty after filtering; training will continue without eval.")
            eval_ds = None
        else:
            eval_ds = _to_training_dataset(eval_filtered)

    logger.info(
        "Final prepared datasets train=%s eval=%s",
        len(train_ds),
        len(eval_ds) if eval_ds is not None else 0,
    )

    return train_ds, eval_ds
