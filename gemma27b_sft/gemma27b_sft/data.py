from __future__ import annotations

import json
import logging
import re
import hashlib
import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, Features, Value
from transformers import PreTrainedTokenizerBase

from .config import DataConfig, SFTConfig


logger = logging.getLogger(__name__)
_ESCAPED_NEWLINE_RE = re.compile(r"\\r\\n|\\n|\\r")


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


def _tail_contains_token(token_ids: list[int], token_id: int | None, window: int = 8) -> bool:
    if token_id is None or not token_ids:
        return False
    tail = token_ids[-max(1, window) :]
    return int(token_id) in tail


def _generation_prompt_suffix_ids(
    tokenizer: PreTrainedTokenizerBase,
    prompt_messages: list[dict[str, str]],
    max_seq_length: int,
    prompt_with_generation: list[int] | None = None,
) -> list[int]:
    prompt_ids_with_generation = (
        list(prompt_with_generation)
        if prompt_with_generation is not None
        else _apply_chat_template(
            tokenizer=tokenizer,
            messages=prompt_messages,
            add_generation_prompt=True,
            max_seq_length=max_seq_length,
            truncate=False,
        )
    )
    prompt_ids_without_generation = _apply_chat_template(
        tokenizer=tokenizer,
        messages=prompt_messages,
        add_generation_prompt=False,
        max_seq_length=max_seq_length,
        truncate=False,
    )
    prefix_len = _common_prefix_len(prompt_ids_without_generation, prompt_ids_with_generation)
    return prompt_ids_with_generation[prefix_len:]


def _truncate_prompt_keep_generation_suffix(
    prompt_ids: list[int],
    prompt_budget: int,
    required_suffix_ids: list[int],
) -> tuple[list[int], bool]:
    if prompt_budget <= 0:
        # A target-bearing row without any prompt tokens becomes unconditional LM.
        return [], False
    if len(prompt_ids) <= prompt_budget:
        return list(prompt_ids), True

    truncated = prompt_ids[-prompt_budget:]
    if not required_suffix_ids:
        return truncated, True
    if len(truncated) < len(required_suffix_ids):
        return truncated, False
    return truncated, truncated[-len(required_suffix_ids) :] == required_suffix_ids


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
    prompt_with_generation = _apply_chat_template(
        tokenizer=tokenizer,
        messages=prompt_messages,
        add_generation_prompt=True,
        max_seq_length=max_seq_length,
        truncate=False,
    )

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

    # Prefer direct generation-prefix split: [prompt+generation_prefix] + [assistant_text + turn suffix]
    prefix_from_generation = _common_prefix_len(prompt_with_generation, target_ids_full)
    if prefix_from_generation == len(prompt_with_generation):
        target_span = target_ids_full[prefix_from_generation:]
        if target_span:
            return prompt_with_generation, target_span

    prefix_len = _common_prefix_len(empty_ids_full, target_ids_full)
    target_span = target_ids_full[prefix_len:]
    if not target_span:
        return None
    prompt_prefix = target_ids_full[:prefix_len]
    return prompt_prefix, target_span


def _build_tokenize_fn(cfg: SFTConfig, tokenizer: PreTrainedTokenizerBase):
    data_cfg = cfg.data
    max_len = cfg.train.max_seq_length
    eos_token_id = tokenizer.eos_token_id
    eot_token_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    model_input_names = set(getattr(tokenizer, "model_input_names", []) or [])
    model_name_lower = cfg.model.name_or_path.lower()
    emit_token_type_ids = (
        "token_type_ids" in model_input_names
        or "gemma-3" in model_name_lower
        or "gemma3" in model_name_lower
    )
    if eot_token_id is None or eot_token_id == tokenizer.unk_token_id:
        eot_token_id = None

    def _tokenize(example: dict[str, Any]) -> dict[str, Any]:
        source_text_original = str(example[data_cfg.source_field])
        source_text = source_text_original
        target_text = str(example[data_cfg.target_field])
        source_has_content = bool(source_text_original.strip())
        source_shrink_steps = 0
        raw_prompt_tokens = 0
        prompt_ids: list[int] = []
        full_ids: list[int] = []
        labels: list[int] = []
        non_ignored = 0
        use_template_target = False
        template_target_ends_with_eot = False
        prompt_was_truncated = False
        target_was_truncated = False
        any_truncation_seen = False

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
            generation_suffix_ids = _generation_prompt_suffix_ids(
                tokenizer=tokenizer,
                prompt_messages=prompt_messages,
                max_seq_length=max_len,
                prompt_with_generation=prompt_ids_raw,
            )
            target_ids_effective = target_ids_fallback
            prompt_was_truncated = False
            target_was_truncated = False
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
                    use_template_target = True
                    template_target_ends_with_eot = bool(
                        eot_token_id is not None and template_target_ids[-1] == eot_token_id
                    )
                    generation_suffix_ids = _generation_prompt_suffix_ids(
                        tokenizer=tokenizer,
                        prompt_messages=prompt_messages,
                        max_seq_length=max_len,
                        prompt_with_generation=prompt_ids_raw,
                    )
            if source_shrink_steps == 0:
                raw_prompt_tokens = len(prompt_ids_raw)

            if not has_target:
                prompt_ids, _ = _truncate_prompt_keep_generation_suffix(
                    prompt_ids=prompt_ids_raw,
                    prompt_budget=max_len,
                    required_suffix_ids=generation_suffix_ids,
                )
                prompt_was_truncated = len(prompt_ids) < len(prompt_ids_raw)
                target_ids = []
            else:
                prompt_has_generation_suffix = True
                if len(prompt_ids_raw) + len(target_ids_effective) <= max_len:
                    prompt_ids = prompt_ids_raw
                    target_ids = list(target_ids_effective)
                else:
                    target_reserve = min(len(target_ids_effective), max(32, max_len // 8))
                    prompt_budget = max(0, max_len - target_reserve)
                    prompt_ids, prompt_has_generation_suffix = _truncate_prompt_keep_generation_suffix(
                        prompt_ids=prompt_ids_raw,
                        prompt_budget=prompt_budget,
                        required_suffix_ids=generation_suffix_ids,
                    )
                    prompt_was_truncated = len(prompt_ids) < len(prompt_ids_raw)
                    if prompt_has_generation_suffix:
                        target_budget = max(0, max_len - len(prompt_ids))
                        target_ids = target_ids_effective[:target_budget]
                        target_was_truncated = len(target_ids) < len(target_ids_effective)
                    else:
                        target_ids = []
                        target_was_truncated = True

                if prompt_has_generation_suffix:
                    if use_template_target and template_target_ends_with_eot and eot_token_id is not None:
                        if not target_ids:
                            if prompt_ids:
                                prompt_ids = prompt_ids[:-1]
                                target_ids = [int(eot_token_id)]
                        elif target_ids[-1] != eot_token_id:
                            if len(prompt_ids) + len(target_ids) < max_len:
                                target_ids.append(int(eot_token_id))
                            else:
                                target_ids[-1] = int(eot_token_id)
                    elif not use_template_target and eos_token_id is not None:
                        # For template-derived targets (Qwen/Gemma chat templates), do not force-append EOS.
                        # They usually already include turn-end markers and forced EOS can duplicate terminal tokens.
                        eos_in_tail = _tail_contains_token(target_ids, int(eos_token_id), window=8)
                        if not eos_in_tail:
                            if len(prompt_ids) + len(target_ids) < max_len:
                                target_ids.append(int(eos_token_id))
                            elif target_ids:
                                target_ids[-1] = int(eos_token_id)
                            elif prompt_ids:
                                prompt_ids = prompt_ids[:-1]
                                target_ids = [int(eos_token_id)]
            any_truncation_seen = any_truncation_seen or prompt_was_truncated or target_was_truncated

            full_ids = prompt_ids + target_ids
            labels = ([-100] * len(prompt_ids)) + target_ids
            non_ignored = len(target_ids)

            # Retry with a shorter source whenever truncation is still degrading supervision.
            should_retry_with_shorter_source = (
                has_target
                and len(source_text) > 1
                and source_shrink_steps < 6
                and (non_ignored <= 0 or prompt_was_truncated or target_was_truncated)
            )
            if not should_retry_with_shorter_source:
                break
            next_source_text = source_text[: max(1, len(source_text) // 2)]
            if next_source_text == source_text:
                break
            source_text = next_source_text
            source_shrink_steps += 1

        output = {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
            "num_target_tokens": non_ignored,
            "prompt_tokens": len(prompt_ids),
            "raw_prompt_tokens": raw_prompt_tokens,
            "prompt_truncated": int(prompt_was_truncated),
            "target_truncated": int(target_was_truncated),
            "truncation_seen": int(any_truncation_seen),
            "full_tokens": len(full_ids),
            "source_chars": len(source_text),
            "source_has_content": int(source_has_content),
            "source_chars_original": len(source_text_original),
            "source_shrink_steps": source_shrink_steps,
            "target_chars": len(target_text),
        }
        if emit_token_type_ids:
            output["token_type_ids"] = [0] * len(full_ids)
        return output

    return _tokenize


def _safe_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _restore_escaped_newlines(text: str) -> tuple[str, int]:
    if "\\" not in text:
        return text, 0
    restored, replaced = _ESCAPED_NEWLINE_RE.subn("\n", text)
    return restored, replaced


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


def _dataset_fingerprint(path: str, cfg: DataConfig) -> str:
    """
    Build a stable fingerprint that changes when the source JSONL file content/metadata
    or relevant loader schema changes. This prevents stale `datasets` generator cache
    reuse when users overwrite the same file path with new rows.
    """
    file_path = Path(path).expanduser().resolve()
    stat = file_path.stat()
    payload = {
        "path": str(file_path),
        "size": int(stat.st_size),
        "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))),
        "required_fields": _required_json_fields(cfg),
        "source_field": cfg.source_field,
        "target_field": cfg.target_field,
        "source_lang_name_field": cfg.source_lang_name_field,
        "target_lang_name_field": cfg.target_lang_name_field,
        "source_lang_code_field": cfg.source_lang_code_field,
        "target_lang_code_field": cfg.target_lang_code_field,
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _safe_load_json_dataset(path: str, cfg: DataConfig) -> Dataset:
    required_fields = _required_json_fields(cfg)
    cache_buster = _dataset_fingerprint(path, cfg)
    stats: dict[str, Any] = {
        "bad_json": 0,
        "total_lines": 0,
        "rows": 0,
        "non_string_counts": {field: 0 for field in required_fields},
        "missing_counts": {field: 0 for field in required_fields},
        "source_escaped_newline_rows": 0,
        "source_escaped_newline_replacements": 0,
        "target_escaped_newline_rows": 0,
        "target_escaped_newline_replacements": 0,
        "sample_cast_logs": 0,
    }

    def _generator(_cache_buster: str = ""):
        # `_cache_buster` is intentionally unused and only exists so we can
        # vary `gen_kwargs` to invalidate stale generator caches across
        # different datasets versions.
        _ = _cache_buster
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
                    text_value = _safe_string(value)
                    if field == cfg.source_field:
                        text_value, replaced = _restore_escaped_newlines(text_value)
                        if replaced > 0:
                            stats["source_escaped_newline_rows"] += 1
                            stats["source_escaped_newline_replacements"] += replaced
                    elif field == cfg.target_field:
                        text_value, replaced = _restore_escaped_newlines(text_value)
                        if replaced > 0:
                            stats["target_escaped_newline_rows"] += 1
                            stats["target_escaped_newline_replacements"] += replaced
                    out[field] = text_value
                stats["rows"] += 1
                if stats["rows"] % 50000 == 0:
                    logger.info("JSON->datasets progress path=%s rows=%s", path, stats["rows"])
                yield out

    from_generator_kwargs: dict[str, Any] = {
        "features": _dataset_features(cfg),
    }
    signature = inspect.signature(Dataset.from_generator)
    if "gen_kwargs" in signature.parameters:
        from_generator_kwargs["gen_kwargs"] = {"_cache_buster": cache_buster}
    if "fingerprint" in signature.parameters:
        from_generator_kwargs["fingerprint"] = cache_buster
    ds = Dataset.from_generator(_generator, **from_generator_kwargs)
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
    if stats["source_escaped_newline_rows"] > 0:
        logger.info(
            "Restored escaped newlines in source field rows=%s replacements=%s",
            stats["source_escaped_newline_rows"],
            stats["source_escaped_newline_replacements"],
        )
    if stats["target_escaped_newline_rows"] > 0:
        logger.info(
            "Restored escaped newlines in target field rows=%s replacements=%s",
            stats["target_escaped_newline_rows"],
            stats["target_escaped_newline_replacements"],
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
            "empty_source_text_count": 0,
            "prompt_ge_max_count": 0,
            "prompt_truncated_count": 0,
            "target_truncated_count": 0,
            "truncation_seen_count": 0,
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
    raw_prompt_tokens = [int(v) for v in sampled["raw_prompt_tokens"]]
    prompt_truncated = [int(v) for v in sampled["prompt_truncated"]]
    target_truncated = [int(v) for v in sampled["target_truncated"]]
    truncation_seen = [int(v) for v in sampled["truncation_seen"]]
    full_tokens = [int(v) for v in sampled["full_tokens"]]
    target_chars = [int(v) for v in sampled["target_chars"]]
    source_has_content = [int(v) for v in sampled["source_has_content"]]
    source_shrink_steps = [int(v) for v in sampled["source_shrink_steps"]]

    zero_target_count = sum(1 for v in num_target_tokens if v <= 0)
    empty_target_text_count = sum(1 for v in target_chars if v <= 0)
    empty_source_text_count = sum(1 for v in source_has_content if v <= 0)
    prompt_ge_max_count = sum(1 for v in raw_prompt_tokens if v >= max_seq_length)
    prompt_truncated_count = sum(1 for v in prompt_truncated if v > 0)
    target_truncated_count = sum(1 for v in target_truncated if v > 0)
    truncation_seen_count = sum(1 for v in truncation_seen if v > 0)
    full_ge_max_count = sum(1 for v in full_tokens if v >= max_seq_length)
    max_prompt_tokens = max(raw_prompt_tokens, default=0)
    max_full_tokens = max(full_tokens, default=0)
    mean_target_tokens = float(sum(num_target_tokens)) / float(sample_size)
    source_shrunk_count = sum(1 for v in source_shrink_steps if v > 0)
    max_source_shrink_steps = max(source_shrink_steps, default=0)

    return {
        "sample_size": sample_size,
        "zero_target_count": zero_target_count,
        "empty_target_text_count": empty_target_text_count,
        "empty_source_text_count": empty_source_text_count,
        "prompt_ge_max_count": prompt_ge_max_count,
        "prompt_truncated_count": prompt_truncated_count,
        "target_truncated_count": target_truncated_count,
        "truncation_seen_count": truncation_seen_count,
        "full_ge_max_count": full_ge_max_count,
        "max_prompt_tokens": max_prompt_tokens,
        "max_full_tokens": max_full_tokens,
        "mean_target_tokens": mean_target_tokens,
        "source_shrunk_count": source_shrunk_count,
        "max_source_shrink_steps": max_source_shrink_steps,
    }


def _warn_tokenization_risks(split_name: str, stats: dict[str, int | float], max_seq_length: int) -> None:
    sample_size = int(stats.get("sample_size", 0) or 0)
    if sample_size <= 0:
        return

    prompt_ge_max = int(stats.get("prompt_ge_max_count", 0) or 0)
    prompt_truncated = int(stats.get("prompt_truncated_count", 0) or 0)
    target_truncated = int(stats.get("target_truncated_count", 0) or 0)
    truncation_seen = int(stats.get("truncation_seen_count", 0) or 0)
    full_ge_max = int(stats.get("full_ge_max_count", 0) or 0)
    source_shrunk = int(stats.get("source_shrunk_count", 0) or 0)
    empty_source = int(stats.get("empty_source_text_count", 0) or 0)
    mean_target = float(stats.get("mean_target_tokens", 0.0) or 0.0)

    prompt_ge_max_ratio = float(prompt_ge_max) / float(sample_size)
    prompt_truncated_ratio = float(prompt_truncated) / float(sample_size)
    target_truncated_ratio = float(target_truncated) / float(sample_size)
    truncation_seen_ratio = float(truncation_seen) / float(sample_size)
    full_ge_max_ratio = float(full_ge_max) / float(sample_size)
    source_shrunk_ratio = float(source_shrunk) / float(sample_size)
    empty_source_ratio = float(empty_source) / float(sample_size)

    if prompt_ge_max_ratio >= 0.20:
        logger.warning(
            "%s tokenization risk: %.1f%% raw prompts reach/exceed max_seq_length=%s "
            "(prompt_ge_max_count=%s/%s). Source/prompt text may need shortening.",
            split_name,
            prompt_ge_max_ratio * 100.0,
            max_seq_length,
            prompt_ge_max,
            sample_size,
        )
    if truncation_seen_ratio >= 0.10:
        logger.warning(
            "%s tokenization risk: truncation was observed in %.1f%% samples "
            "(truncation_seen_count=%s/%s) before any source shrink recovery.",
            split_name,
            truncation_seen_ratio * 100.0,
            truncation_seen,
            sample_size,
        )
    if prompt_truncated_ratio > 0.0:
        logger.warning(
            "%s tokenization risk: final prompt remains truncated in %.1f%% samples "
            "(prompt_truncated_count=%s/%s).",
            split_name,
            prompt_truncated_ratio * 100.0,
            prompt_truncated,
            sample_size,
        )
    if target_truncated_ratio > 0.0:
        logger.warning(
            "%s tokenization risk: final target remains truncated in %.1f%% samples "
            "(target_truncated_count=%s/%s).",
            split_name,
            target_truncated_ratio * 100.0,
            target_truncated,
            sample_size,
        )
    if full_ge_max_ratio >= 0.30:
        logger.warning(
            "%s tokenization risk: %.1f%% full sequences hit max_seq_length=%s "
            "(full_ge_max_count=%s/%s).",
            split_name,
            full_ge_max_ratio * 100.0,
            max_seq_length,
            full_ge_max,
            sample_size,
        )
    if source_shrunk_ratio >= 0.05:
        logger.warning(
            "%s tokenization risk: source text had to be shrunk in %.1f%% samples "
            "(source_shrunk_count=%s/%s).",
            split_name,
            source_shrunk_ratio * 100.0,
            source_shrunk,
            sample_size,
        )
    if empty_source_ratio > 0.0:
        logger.warning(
            "%s tokenization risk: source text is empty in %.1f%% samples "
            "(empty_source_text_count=%s/%s). Check data.source_field and input JSONL.",
            split_name,
            empty_source_ratio * 100.0,
            empty_source,
            sample_size,
        )
    if mean_target < 16.0:
        logger.warning(
            "%s tokenization risk: mean_target_tokens is low (%.2f). "
            "Dataset may provide weak supervision.",
            split_name,
            mean_target,
        )


def _raise_empty_train_dataset_error(cfg: SFTConfig, raw_rows: int, token_stats: dict[str, int | float]) -> None:
    raise ValueError(
        "All training rows were filtered out (num_target_tokens == 0).\n"
        f"- data.train_file={cfg.data.train_file}\n"
        f"- rows_before_filter={raw_rows}\n"
        f"- sampled_rows={token_stats['sample_size']} sampled_zero_target={token_stats['zero_target_count']}\n"
        f"- sampled_empty_target_text={token_stats['empty_target_text_count']}\n"
        f"- sampled_empty_source_text={token_stats['empty_source_text_count']}\n"
        f"- sampled_prompt_ge_max_seq={token_stats['prompt_ge_max_count']} (raw prompts, max_seq_length={cfg.train.max_seq_length})\n"
        f"- sampled_prompt_truncated={token_stats['prompt_truncated_count']} sampled_target_truncated={token_stats['target_truncated_count']}\n"
        f"- sampled_source_shrunk={token_stats['source_shrunk_count']} max_shrink_steps={token_stats['max_source_shrink_steps']}\n"
        f"- sampled_max_prompt_tokens={token_stats['max_prompt_tokens']} sampled_max_full_tokens={token_stats['max_full_tokens']}\n"
        "Likely causes:\n"
        "1) data.source_field points to empty/wrong column.\n"
        "2) data.target_field points to empty/wrong column.\n"
        "3) Prompt + source text is too long and truncates away assistant tokens.\n"
        "4) train.max_seq_length is too small for this prompt template.\n"
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
    keep_cols = {"input_ids", "attention_mask", "labels", "token_type_ids"}
    drop_cols = [name for name in dataset.column_names if name not in keep_cols]
    if drop_cols:
        dataset = dataset.remove_columns(drop_cols)
    return dataset


def _truncate_for_log(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    extra = len(text) - max_chars
    return f"{text[:max_chars]} ... [truncated {extra} chars]"


def _render_chat_template_text(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    add_generation_prompt: bool,
) -> str:
    if getattr(tokenizer, "chat_template", None):
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            if isinstance(rendered, str):
                return rendered
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to render chat template text: %s", exc)

    parts = []
    for msg in messages:
        parts.append(f"{msg['role'].upper()}: {msg['content']}")
    if add_generation_prompt:
        parts.append("ASSISTANT:")
    return "\n\n".join(parts)


def _log_pre_tokenization_samples(
    dataset: Dataset,
    cfg: SFTConfig,
    tokenizer: PreTrainedTokenizerBase,
    split_name: str,
) -> None:
    sample_limit = min(len(dataset), cfg.data.log_text_samples)
    if sample_limit <= 0:
        return

    logger.info("%s pre-tokenization preview sample_count=%s", split_name, sample_limit)
    sampled = dataset.select(range(sample_limit))
    max_chars = cfg.data.log_text_max_chars
    include_chat_template = bool(cfg.data.log_chat_template_text)

    for idx, row in enumerate(sampled):
        try:
            source_text = _safe_string(row.get(cfg.data.source_field))
            target_text = _safe_string(row.get(cfg.data.target_field))
            source_lang, src_lang_code, target_lang, tgt_lang_code = _resolve_languages(cfg.data, row)
            prompt_messages, full_messages = _messages(cfg.data, row, source_text, target_text)
            prompt_text = prompt_messages[0]["content"] if prompt_messages else ""
            chat_prompt_text = ""
            chat_full_text = ""
            if include_chat_template:
                chat_prompt_text = _render_chat_template_text(
                    tokenizer=tokenizer,
                    messages=prompt_messages,
                    add_generation_prompt=True,
                )
                chat_full_text = _render_chat_template_text(
                    tokenizer=tokenizer,
                    messages=full_messages,
                    add_generation_prompt=False,
                )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("%s pre-tokenization preview failed idx=%s: %s", split_name, idx, exc)
            continue

        source_log = _truncate_for_log(source_text, max_chars)
        prompt_log = _truncate_for_log(prompt_text, max_chars)
        target_log = _truncate_for_log(target_text, max_chars)
        if include_chat_template:
            chat_prompt_log = _truncate_for_log(chat_prompt_text, max_chars)
            chat_full_log = _truncate_for_log(chat_full_text, max_chars)
            logger.info(
                "%s pre-tokenization sample idx=%s src=%s(%s) tgt=%s(%s) "
                "chars(source/prompt/target/chat_prompt/chat_full)=%s/%s/%s/%s/%s\n"
                "SOURCE>>> %s\n"
                "PROMPT>>> %s\n"
                "TARGET>>> %s\n"
                "CHAT_TEMPLATE_PROMPT>>> %s\n"
                "CHAT_TEMPLATE_FULL>>> %s",
                split_name,
                idx,
                source_lang,
                src_lang_code,
                target_lang,
                tgt_lang_code,
                len(source_text),
                len(prompt_text),
                len(target_text),
                len(chat_prompt_text),
                len(chat_full_text),
                source_log,
                prompt_log,
                target_log,
                chat_prompt_log,
                chat_full_log,
            )
        else:
            logger.info(
                "%s pre-tokenization sample idx=%s src=%s(%s) tgt=%s(%s) "
                "chars(source/prompt/target)=%s/%s/%s\n"
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
    _log_pre_tokenization_samples(train_rows, cfg, tokenizer, "Train")
    if eval_rows is not None:
        _log_pre_tokenization_samples(eval_rows, cfg, tokenizer, "Eval")
    tokenize_fn = _build_tokenize_fn(cfg, tokenizer)
    train_mapped = _tokenize_dataset(
        train_rows,
        tokenize_fn=tokenize_fn,
        split_name="Train",
        num_workers=data_cfg.preprocessing_num_workers,
    )
    train_stats = _summarize_tokenization(train_mapped, cfg.train.max_seq_length)
    logger.info(
        "Train tokenization stats sample_size=%s zero_target=%s empty_target=%s empty_source=%s prompt_ge_max=%s truncation_seen=%s prompt_truncated=%s target_truncated=%s source_shrunk=%s max_shrink_steps=%s max_prompt=%s max_full=%s mean_target_tokens=%.2f",
        train_stats["sample_size"],
        train_stats["zero_target_count"],
        train_stats["empty_target_text_count"],
        train_stats["empty_source_text_count"],
        train_stats["prompt_ge_max_count"],
        train_stats["truncation_seen_count"],
        train_stats["prompt_truncated_count"],
        train_stats["target_truncated_count"],
        train_stats["source_shrunk_count"],
        train_stats["max_source_shrink_steps"],
        train_stats["max_prompt_tokens"],
        train_stats["max_full_tokens"],
        train_stats["mean_target_tokens"],
    )
    _warn_tokenization_risks("Train", train_stats, cfg.train.max_seq_length)
    train_filtered = train_mapped.filter(
        lambda row: int(row.get("num_target_tokens", 0)) > 0 and int(row.get("source_has_content", 0)) > 0,
        desc="Train filtering non-empty source/target",
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
            "Eval tokenization stats sample_size=%s zero_target=%s empty_target=%s empty_source=%s prompt_ge_max=%s truncation_seen=%s prompt_truncated=%s target_truncated=%s source_shrunk=%s max_shrink_steps=%s max_prompt=%s max_full=%s mean_target_tokens=%.2f",
            eval_stats["sample_size"],
            eval_stats["zero_target_count"],
            eval_stats["empty_target_text_count"],
            eval_stats["empty_source_text_count"],
            eval_stats["prompt_ge_max_count"],
            eval_stats["truncation_seen_count"],
            eval_stats["prompt_truncated_count"],
            eval_stats["target_truncated_count"],
            eval_stats["source_shrunk_count"],
            eval_stats["max_source_shrink_steps"],
            eval_stats["max_prompt_tokens"],
            eval_stats["max_full_tokens"],
            eval_stats["mean_target_tokens"],
        )
        _warn_tokenization_risks("Eval", eval_stats, cfg.train.max_seq_length)
        eval_filtered = eval_mapped.filter(
            lambda row: int(row.get("num_target_tokens", 0)) > 0 and int(row.get("source_has_content", 0)) > 0,
            desc="Eval filtering non-empty source/target",
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
