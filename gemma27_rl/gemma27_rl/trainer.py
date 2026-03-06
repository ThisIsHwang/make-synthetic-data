from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import math
import os
from pathlib import Path
import random
import select
import shutil
import socket
from statistics import mean
import subprocess
import sys
from typing import Any, Callable

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import deepspeed
except Exception:  # pragma: no cover - optional dependency
    deepspeed = None  # type: ignore[assignment]

try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model, get_peft_model_state_dict
except Exception:  # pragma: no cover - optional dependency
    LoraConfig = None  # type: ignore[assignment]
    PeftModel = None  # type: ignore[assignment]
    TaskType = None  # type: ignore[assignment]
    get_peft_model = None  # type: ignore[assignment]
    get_peft_model_state_dict = None  # type: ignore[assignment]

from .advantage import (
    apply_group_relative_advantage,
    broadcast_sequence_reward,
    build_sequence_rewards,
    combine_advantages,
    normalize_advantages,
)
from .config import RLPostTrainConfig, dump_config
from .data import load_examples
from .eval import evaluate_on_dataset
from .grpo import update_policy
from .prompting import (
    collect_tokenizer_special_token_strings as _collect_special_token_strings_shared,
    sanitize_text_for_scoring as _sanitize_text_for_scoring_shared,
)
from .rewards import (
    OpenAICompatibleESAScorer,
    OpenAICompatibleMQMScorer,
    MetricXQEScorer,
    XCometXLScorer,
    metricx_score_to_reward,
    spans_to_token_rewards,
)
from .rollout import compute_completion_logprobs, compute_completion_logprobs_batch, generate_rollouts
from .rl_types import Rollout, SampleForScoring
from .utils import (
    build_worker_launch_command,
    collect_huggingface_worker_env,
    configure_huggingface_cache,
    merge_env_overrides,
    resolve_device,
    resolve_huggingface_token,
    resolve_torch_dtype,
    set_seed,
)


logger = logging.getLogger(__name__)
_ESA_ALL_ZERO_WARNED = False
_FORBIDDEN_THINK_TAGS: tuple[str, ...] = ("<think>", "</think>")
_DEFAULT_THINK_TAG_TOKEN_PENALTY = -100.0
_DEFAULT_THINK_TAG_SEQUENCE_PENALTY = -30.0
_DEFAULT_REPEAT_TOKEN_PENALTY = -2.0
_DEFAULT_REPEAT_SEQUENCE_PENALTY = -0.5
_DEFAULT_NGRAM_TOKEN_PENALTY = -1.0
_DEFAULT_NGRAM_SEQUENCE_PENALTY = -0.5
_DEFAULT_SPECIAL_TOKEN_PENALTY = -50.0
_DEFAULT_SPECIAL_SEQUENCE_PENALTY = -10.0


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return max(minimum, int(default))
    try:
        value = int(raw.strip())
    except Exception:
        return max(minimum, int(default))
    return max(minimum, value)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        return float(raw.strip())
    except Exception:
        return float(default)


def _span_overlap_chars(start: int, end: int, tok_s: int, tok_e: int) -> int:
    if end <= start or tok_e <= tok_s:
        return 0
    return max(0, min(tok_e, end) - max(tok_s, start))


def _span_applies_to_token(
    span: dict[str, Any],
    tok_s: int,
    tok_e: int,
    overlap_policy: str,
    majority_threshold: float,
) -> bool:
    try:
        start = int(span.get("start", 0))
        end = int(span.get("end", 0))
    except Exception:
        return False
    overlap = _span_overlap_chars(start, end, tok_s, tok_e)
    if overlap <= 0:
        return False
    if overlap_policy == "any_overlap":
        return True
    ratio = overlap / max(1, tok_e - tok_s)
    return ratio >= majority_threshold


def _short_text(text: str, limit: int = 120) -> str:
    clean = text.replace("\n", "\\n")
    if len(clean) <= limit:
        return clean
    return clean[:limit] + "..."


def _find_forbidden_think_tag_spans(text: str) -> list[tuple[int, int, str]]:
    text_lc = text.lower()
    matches: list[tuple[int, int, str]] = []
    for tag in _FORBIDDEN_THINK_TAGS:
        tag_lc = tag.lower()
        start = 0
        while True:
            idx = text_lc.find(tag_lc, start)
            if idx < 0:
                break
            matches.append((idx, idx + len(tag), tag))
            start = idx + len(tag_lc)
    matches.sort(key=lambda row: row[0])
    return matches


def _apply_forbidden_think_tag_penalty(
    *,
    completion_text: str,
    token_char_offsets: list[tuple[int, int]],
    token_rewards: list[float],
    seq_reward: float,
    token_penalty: float,
    seq_penalty_per_match: float,
) -> tuple[list[float], float, int, int]:
    matches = _find_forbidden_think_tag_spans(completion_text)
    if not matches:
        return token_rewards, float(seq_reward), 0, 0

    token_hits: set[int] = set()
    max_tokens = min(len(token_rewards), len(token_char_offsets))
    for start, end, _ in matches:
        for tok_idx in range(max_tokens):
            tok_s, tok_e = token_char_offsets[tok_idx]
            if _span_overlap_chars(start, end, tok_s, tok_e) > 0:
                token_rewards[tok_idx] += float(token_penalty)
                token_hits.add(tok_idx)

    adjusted_seq_reward = float(seq_reward) + (float(seq_penalty_per_match) * float(len(matches)))
    return token_rewards, adjusted_seq_reward, len(matches), len(token_hits)


def _find_repeated_token_positions(
    completion_token_ids: list[int],
    *,
    min_repeat_run_length: int,
    max_repeat_pattern_length: int = 4,
) -> tuple[list[int], int]:
    if min_repeat_run_length < 2:
        min_repeat_run_length = 2
    if max_repeat_pattern_length < 1:
        max_repeat_pattern_length = 1
    token_count = len(completion_token_ids)
    if token_count < min_repeat_run_length:
        return [], 0

    repeated_positions: set[int] = set()
    repeated_runs = 0

    # 1) Consecutive repeats: A A A
    run_len = 1
    for idx in range(1, token_count):
        if int(completion_token_ids[idx]) == int(completion_token_ids[idx - 1]):
            run_len += 1
            if run_len == min_repeat_run_length:
                repeated_runs += 1
                repeated_positions.add(idx)
            elif run_len > min_repeat_run_length:
                repeated_positions.add(idx)
        else:
            run_len = 1

    # 2) Periodic non-consecutive repeats: A B A B A B, A B C A B C, ...
    max_period = min(max_repeat_pattern_length, token_count // 2)
    min_period_repeats = max(2, min_repeat_run_length)
    for period in range(2, max_period + 1):
        idx = 0
        while idx + (2 * period) <= token_count:
            left = completion_token_ids[idx:idx + period]
            right = completion_token_ids[idx + period:idx + (2 * period)]
            if left != right:
                idx += 1
                continue
            # Skip degenerate periodic runs like A A A A from period=2;
            # those are already covered by consecutive repeat detection.
            if len({int(v) for v in left}) <= 1:
                idx += 1
                continue

            end = idx + (2 * period)
            while end + period <= token_count:
                prev_chunk = completion_token_ids[end - period:end]
                next_chunk = completion_token_ids[end:end + period]
                if prev_chunk != next_chunk:
                    break
                end += period

            total_repeats = (end - idx) // period
            if total_repeats >= min_period_repeats:
                repeated_runs += 1
                for pos in range(idx + period, end):
                    repeated_positions.add(pos)
            idx = max(idx + 1, end - period + 1)

    return sorted(repeated_positions), repeated_runs


def _apply_repeated_token_penalty(
    *,
    completion_token_ids: list[int],
    token_rewards: list[float],
    seq_reward: float,
    token_penalty: float,
    seq_penalty_per_repeat: float,
    min_repeat_run_length: int,
    max_repeat_pattern_length: int = 4,
) -> tuple[list[float], float, int, int]:
    repeated_positions, repeated_runs = _find_repeated_token_positions(
        completion_token_ids,
        min_repeat_run_length=min_repeat_run_length,
        max_repeat_pattern_length=max_repeat_pattern_length,
    )
    if not repeated_positions:
        return token_rewards, float(seq_reward), 0, 0

    max_tokens = len(token_rewards)
    for idx in repeated_positions:
        if idx < max_tokens:
            token_rewards[idx] += float(token_penalty)
    adjusted_seq_reward = float(seq_reward) + (float(seq_penalty_per_repeat) * float(len(repeated_positions)))
    return token_rewards, adjusted_seq_reward, len(repeated_positions), repeated_runs


def _find_repeated_ngram_positions(
    completion_token_ids: list[int],
    *,
    ngram_size: int,
    min_occurrences: int = 2,
) -> tuple[list[int], int]:
    if ngram_size < 2:
        ngram_size = 2
    if min_occurrences < 2:
        min_occurrences = 2
    token_count = len(completion_token_ids)
    if token_count < ngram_size:
        return [], 0

    counts: dict[tuple[int, ...], int] = {}
    repeated_positions: set[int] = set()
    repeated_occurrences = 0

    max_start = token_count - ngram_size
    for start in range(max_start + 1):
        ngram = tuple(int(v) for v in completion_token_ids[start:start + ngram_size])
        next_count = int(counts.get(ngram, 0)) + 1
        counts[ngram] = next_count
        if next_count < min_occurrences:
            continue
        repeated_occurrences += 1
        for pos in range(start, start + ngram_size):
            repeated_positions.add(pos)

    return sorted(repeated_positions), repeated_occurrences


def _apply_ngram_repeat_penalty(
    *,
    completion_token_ids: list[int],
    token_rewards: list[float],
    seq_reward: float,
    token_penalty: float,
    seq_penalty_per_repeat: float,
    ngram_size: int,
    min_occurrences: int = 2,
) -> tuple[list[float], float, int, int]:
    repeated_positions, repeated_occurrences = _find_repeated_ngram_positions(
        completion_token_ids,
        ngram_size=ngram_size,
        min_occurrences=min_occurrences,
    )
    if not repeated_positions:
        return token_rewards, float(seq_reward), 0, 0

    max_tokens = len(token_rewards)
    for idx in repeated_positions:
        if idx < max_tokens:
            token_rewards[idx] += float(token_penalty)
    adjusted_seq_reward = float(seq_reward) + (float(seq_penalty_per_repeat) * float(repeated_occurrences))
    return token_rewards, adjusted_seq_reward, len(repeated_positions), repeated_occurrences


def _zero_token_rewards_on_special_token_ids(
    *,
    token_rewards: list[float],
    completion_token_ids: list[int],
    special_token_ids: set[int],
) -> int:
    if not token_rewards or not completion_token_ids or not special_token_ids:
        return 0
    max_tokens = min(len(token_rewards), len(completion_token_ids))
    masked = 0
    for tok_idx in range(max_tokens):
        if int(completion_token_ids[tok_idx]) not in special_token_ids:
            continue
        if float(token_rewards[tok_idx]) == 0.0:
            continue
        token_rewards[tok_idx] = 0.0
        masked += 1
    return masked


def _collect_tokenizer_special_token_strings(tokenizer: Any | None) -> list[str]:
    return _collect_special_token_strings_shared(tokenizer)


def _collect_tokenizer_special_token_ids(tokenizer: Any | None) -> set[int]:
    if tokenizer is None:
        return set()
    out: set[int] = set()
    raw_ids = getattr(tokenizer, "all_special_ids", None)
    if isinstance(raw_ids, (list, tuple, set)):
        for tok_id in raw_ids:
            try:
                out.add(int(tok_id))
            except Exception:
                continue
    return out


def _build_special_token_id_label_map(tokenizer: Any | None, special_token_ids: set[int]) -> dict[int, str]:
    if tokenizer is None or not special_token_ids:
        return {}
    labels: dict[int, str] = {}
    converter = getattr(tokenizer, "convert_ids_to_tokens", None)
    if callable(converter):
        for tok_id in sorted(special_token_ids):
            try:
                token_text = converter(int(tok_id))
            except Exception:
                continue
            if isinstance(token_text, str) and token_text:
                labels[int(tok_id)] = token_text
    return labels


def _format_top_special_id_counts(
    id_counts: dict[int, int],
    *,
    id_label_map: dict[int, str],
    limit: int = 8,
) -> str:
    if not id_counts:
        return "-"
    rows = sorted(id_counts.items(), key=lambda item: (-int(item[1]), int(item[0])))
    shown = rows[: max(1, int(limit))]
    return ", ".join(
        f"{id_label_map.get(tok_id, str(tok_id))}(id={tok_id}):{count}"
        for tok_id, count in shown
    )


def _format_top_special_text_counts(text_counts: dict[str, int], *, limit: int = 8) -> str:
    if not text_counts:
        return "-"
    rows = sorted(text_counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
    shown = rows[: max(1, int(limit))]
    return ", ".join(f"{_short_text(text, limit=60)}:{count}" for text, count in shown)


def _find_special_token_text_spans(text: str, special_tokens: list[str]) -> list[tuple[int, int, str]]:
    if not text or not special_tokens:
        return []
    matches: list[tuple[int, int, str]] = []
    for token in special_tokens:
        tok = str(token or "")
        if not tok:
            continue
        start = 0
        while True:
            idx = text.find(tok, start)
            if idx < 0:
                break
            matches.append((idx, idx + len(tok), tok))
            start = idx + len(tok)
    matches.sort(key=lambda row: row[0])
    return matches


def _looks_like_end_of_turn_marker(token_text: str) -> bool:
    text = str(token_text or "").strip().lower()
    if not text:
        return False
    return any(hint in text for hint in ("end_of_turn", "eot", "im_end", "endofturn", "eos"))


def _collect_exempt_final_end_of_turn_markers(
    tokenizer: Any | None,
    *,
    special_token_strings: list[str],
    special_token_ids: set[int],
) -> tuple[set[int], set[str]]:
    exempt_strings: set[str] = set()
    exempt_ids: set[int] = set()

    for token in special_token_strings:
        if _looks_like_end_of_turn_marker(token):
            exempt_strings.add(str(token))

    eos_token = getattr(tokenizer, "eos_token", None) if tokenizer is not None else None
    if isinstance(eos_token, str) and eos_token.strip():
        exempt_strings.add(eos_token.strip())

    eos_token_id = getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None
    if isinstance(eos_token_id, int):
        exempt_ids.add(int(eos_token_id))
    elif isinstance(eos_token_id, (list, tuple, set)):
        for tok_id in eos_token_id:
            try:
                exempt_ids.add(int(tok_id))
            except Exception:
                continue

    converter = getattr(tokenizer, "convert_tokens_to_ids", None) if tokenizer is not None else None
    if callable(converter):
        for token in exempt_strings:
            try:
                tok_id = converter(token)
            except Exception:
                continue
            try:
                tok_id_int = int(tok_id)
            except Exception:
                continue
            exempt_ids.add(tok_id_int)

    if special_token_ids:
        exempt_ids = {tok_id for tok_id in exempt_ids if tok_id in special_token_ids}
    return exempt_ids, exempt_strings


def _count_special_token_id_occurrences(
    *,
    token_ids: list[int],
    special_token_ids: set[int],
    exempt_final_token_ids: set[int] | None = None,
    hit_counter: dict[int, int] | None = None,
) -> int:
    if not token_ids or not special_token_ids:
        return 0
    exempt_ids = exempt_final_token_ids or set()
    final_idx = len(token_ids) - 1
    occurrences = 0
    for tok_idx, tok_id_raw in enumerate(token_ids):
        tok_id = int(tok_id_raw)
        if tok_id not in special_token_ids:
            continue
        if tok_idx == final_idx and tok_id in exempt_ids:
            continue
        occurrences += 1
        if hit_counter is not None:
            hit_counter[tok_id] = int(hit_counter.get(tok_id, 0)) + 1
    return occurrences


def _apply_special_token_penalty(
    *,
    completion_text: str,
    completion_token_ids: list[int],
    penalty_token_ids: list[int] | None = None,
    token_char_offsets: list[tuple[int, int]],
    token_rewards: list[float],
    seq_reward: float,
    special_token_ids: set[int],
    special_token_strings: list[str],
    token_penalty: float,
    seq_penalty_per_occurrence: float,
    exempt_final_token_ids: set[int] | None = None,
    exempt_final_token_strings: set[str] | None = None,
    id_hit_counter: dict[int, int] | None = None,
    text_hit_counter: dict[str, int] | None = None,
) -> tuple[list[float], float, int, int]:
    exempt_ids = exempt_final_token_ids or set()
    exempt_strings_lc = {str(tok).lower() for tok in (exempt_final_token_strings or set()) if str(tok)}
    special_occurrences = 0
    token_hits: set[int] = set()
    completion_id_occurrences = 0

    max_tokens = min(len(token_rewards), len(completion_token_ids))
    final_token_idx = max_tokens - 1
    count_id_hits_in_completion = penalty_token_ids is None
    for tok_idx in range(max_tokens):
        token_id = int(completion_token_ids[tok_idx])
        if token_id not in special_token_ids:
            continue
        if tok_idx == final_token_idx and token_id in exempt_ids:
            continue
        completion_id_occurrences += 1
        if count_id_hits_in_completion and id_hit_counter is not None:
            id_hit_counter[token_id] = int(id_hit_counter.get(token_id, 0)) + 1
        token_hits.add(tok_idx)
        token_rewards[tok_idx] += float(token_penalty)

    seq_id_source = penalty_token_ids if penalty_token_ids is not None else completion_token_ids
    if penalty_token_ids is None:
        special_occurrences += completion_id_occurrences
    else:
        special_occurrences += _count_special_token_id_occurrences(
            token_ids=seq_id_source,
            special_token_ids=special_token_ids,
            exempt_final_token_ids=exempt_ids,
            hit_counter=id_hit_counter,
        )

    text_matches = _find_special_token_text_spans(completion_text, special_token_strings)
    max_with_offsets = min(len(token_rewards), len(token_char_offsets))
    text_end = len(completion_text.rstrip())
    seen_text_spans: set[tuple[int, int]] = set()
    for start, end, matched_token_text in text_matches:
        matched_text_lc = completion_text[start:end].lower() if end > start else ""
        if end == text_end and matched_text_lc in exempt_strings_lc:
            continue
        span_key = (int(start), int(end))
        if span_key in seen_text_spans:
            continue
        seen_text_spans.add(span_key)
        overlapped_indices: list[int] = []
        for tok_idx in range(max_with_offsets):
            tok_s, tok_e = token_char_offsets[tok_idx]
            if _span_overlap_chars(start, end, tok_s, tok_e) > 0:
                overlapped_indices.append(tok_idx)
        has_overlap = len(overlapped_indices) > 0
        all_overlaps_special_id = has_overlap and all(
            tok_idx < len(completion_token_ids) and int(completion_token_ids[tok_idx]) in special_token_ids
            for tok_idx in overlapped_indices
        )
        # Avoid double-counting the same special marker when it's already counted by ID.
        should_count_occurrence = not all_overlaps_special_id
        if should_count_occurrence:
            special_occurrences += 1
            if text_hit_counter is not None:
                matched_text = completion_text[start:end] if end > start else ""
                if not matched_text:
                    matched_text = str(matched_token_text)
                text_hit_counter[matched_text] = int(text_hit_counter.get(matched_text, 0)) + 1
        for tok_idx in range(max_with_offsets):
            tok_s, tok_e = token_char_offsets[tok_idx]
            if _span_overlap_chars(start, end, tok_s, tok_e) <= 0:
                continue
            if tok_idx in token_hits:
                continue
            token_hits.add(tok_idx)
            token_rewards[tok_idx] += float(token_penalty)

    if special_occurrences <= 0:
        return token_rewards, float(seq_reward), 0, 0

    adjusted_seq = float(seq_reward) + (float(seq_penalty_per_occurrence) * float(special_occurrences))
    return token_rewards, adjusted_seq, special_occurrences, len(token_hits)


def _sanitize_text_for_mqm_esa(target_text: str, *, special_tokens: list[str]) -> tuple[str, int]:
    return _sanitize_text_for_scoring_shared(target_text, special_tokens=special_tokens)


def _log_span_debug_for_rollout(
    *,
    rollout: Rollout,
    span_row: list[dict[str, Any]],
    token_rewards: list[float],
    raw_adv: list[float],
    adv_used: list[float] | None,
    seq_reward: float,
    overlap_policy: str,
    majority_threshold: float,
    max_tokens: int,
    only_nonzero: bool,
) -> None:
    adv_used_row = adv_used or []
    logger.info(
        "[span-debug] example_id=%s spans=%s tokens=%s seq_reward=%.4f completion=%r",
        rollout.example_id,
        len(span_row),
        len(rollout.token_char_offsets),
        float(seq_reward),
        _short_text(rollout.completion_text),
    )
    for span_idx, span in enumerate(span_row):
        try:
            start = int(span.get("start", 0))
            end = int(span.get("end", 0))
        except Exception:
            continue
        severity = str(span.get("severity", "")).upper()
        confidence = span.get("confidence", None)
        snippet = ""
        if end > start:
            snippet = rollout.completion_text[max(0, start) : max(0, end)]
        logger.info(
            "[span-debug] span[%s] severity=%s confidence=%s range=[%s,%s) text=%r",
            span_idx,
            severity,
            confidence,
            start,
            end,
            _short_text(snippet, limit=80),
        )

    printed = 0
    non_zero = 0
    for tok_idx, (tok_s, tok_e) in enumerate(rollout.token_char_offsets):
        tok_reward = token_rewards[tok_idx] if tok_idx < len(token_rewards) else 0.0
        tok_adv_raw = raw_adv[tok_idx] if tok_idx < len(raw_adv) else 0.0
        tok_adv_used = tok_adv_raw
        if tok_idx < len(adv_used_row):
            tok_adv_used = adv_used_row[tok_idx]
        if abs(tok_reward) > 0:
            non_zero += 1
        overlap_ids: list[int] = []
        for span_idx, span in enumerate(span_row):
            if _span_applies_to_token(
                span=span,
                tok_s=tok_s,
                tok_e=tok_e,
                overlap_policy=overlap_policy,
                majority_threshold=majority_threshold,
            ):
                overlap_ids.append(span_idx)

        if only_nonzero and abs(tok_reward) <= 0 and not overlap_ids:
            continue
        if printed >= max_tokens:
            break

        tok_text = ""
        if tok_e > tok_s:
            tok_text = rollout.completion_text[max(0, tok_s) : max(0, tok_e)]
        tok_id = int(rollout.completion_token_ids[tok_idx]) if tok_idx < len(rollout.completion_token_ids) else -1
        if adv_used_row:
            logger.info(
                "[span-debug] tok[%03d] id=%s range=[%d,%d) text=%r spans=%s tok_reward=%.4f raw_adv=%.4f adv_used=%.4f",
                tok_idx,
                tok_id,
                tok_s,
                tok_e,
                _short_text(tok_text, limit=60),
                overlap_ids,
                float(tok_reward),
                float(tok_adv_raw),
                float(tok_adv_used),
            )
        else:
            logger.info(
                "[span-debug] tok[%03d] id=%s range=[%d,%d) text=%r spans=%s tok_reward=%.4f raw_adv=%.4f",
                tok_idx,
                tok_id,
                tok_s,
                tok_e,
                _short_text(tok_text, limit=60),
                overlap_ids,
                float(tok_reward),
                float(tok_adv_raw),
            )
        printed += 1

    if len(rollout.token_char_offsets) > printed:
        logger.info(
            "[span-debug] token rows truncated: printed=%s total=%s (non_zero_token_rewards=%s)",
            printed,
            len(rollout.token_char_offsets),
            non_zero,
        )
    else:
        logger.info(
            "[span-debug] token rows complete: printed=%s total=%s (non_zero_token_rewards=%s)",
            printed,
            len(rollout.token_char_offsets),
            non_zero,
        )


def _is_distributed_initialized() -> bool:
    return bool(torch.distributed.is_available() and torch.distributed.is_initialized())


def _distributed_rank() -> int:
    if _is_distributed_initialized():
        return int(torch.distributed.get_rank())
    raw = os.environ.get("RANK")
    if raw and raw.isdigit():
        return int(raw)
    return 0


def _distributed_world_size() -> int:
    if _is_distributed_initialized():
        return int(torch.distributed.get_world_size())
    raw = os.environ.get("WORLD_SIZE")
    if raw and raw.isdigit():
        return int(raw)
    return 1


def _is_rank0() -> bool:
    return _distributed_rank() == 0


def _dist_barrier() -> None:
    if _is_distributed_initialized():
        torch.distributed.barrier()


def _configure_nccl_heartbeat_timeout(cfg: RLPostTrainConfig) -> None:
    if str(cfg.rl.backend).strip().lower() != "deepspeed":
        return
    if _distributed_world_size() <= 1:
        return
    # In this trainer, rank0 performs rollout/reward/eval while other ranks may
    # wait in a collective for several minutes. The default 480s watchdog can
    # abort these valid waits, so set a safer default unless the user already
    # configured it.
    key = "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"
    if key not in os.environ:
        os.environ[key] = "7200"
        logger.info("Set %s=%s (default for long rank0-only sections).", key, os.environ[key])


def _configure_cuda_allocator() -> None:
    if not torch.cuda.is_available():
        return
    key = "PYTORCH_CUDA_ALLOC_CONF"
    if key not in os.environ:
        os.environ[key] = "expandable_segments:True"
        logger.info("Set %s=%s (default to reduce CUDA allocator fragmentation).", key, os.environ[key])


def _set_rollout_sampling_seed(seed: int) -> None:
    # Keep stochastic generation aligned across ranks when using ZeRO-sharded models.
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _broadcast_object_list(payload: list[Any], src: int = 0) -> list[Any]:
    if _is_distributed_initialized():
        torch.distributed.broadcast_object_list(payload, src=src)
    return payload


def _gather_object_to_rank0(local_obj: Any) -> list[Any] | None:
    if not _is_distributed_initialized():
        return [local_obj]
    gathered: list[Any] | None = [None for _ in range(_distributed_world_size())] if _is_rank0() else None
    torch.distributed.gather_object(local_obj, object_gather_list=gathered, dst=0)
    return gathered


def _scatter_object_from_rank0(per_rank_objects: list[Any] | None, *, rank: int) -> Any:
    if not _is_distributed_initialized():
        if isinstance(per_rank_objects, list) and per_rank_objects:
            return per_rank_objects[0]
        return None

    world_size = _distributed_world_size()
    local_rank_raw = os.environ.get("LOCAL_RANK")
    if torch.cuda.is_available() and local_rank_raw and local_rank_raw.isdigit():
        torch.cuda.set_device(int(local_rank_raw))

    if hasattr(torch.distributed, "scatter_object_list"):
        try:
            recv: list[Any] = [None]
            if _is_rank0():
                payload = list(per_rank_objects or [])
                if len(payload) < world_size:
                    payload.extend([None for _ in range(world_size - len(payload))])
                elif len(payload) > world_size:
                    payload = payload[:world_size]
                torch.distributed.scatter_object_list(recv, scatter_object_input_list=payload, src=0)
            else:
                torch.distributed.scatter_object_list(recv, scatter_object_input_list=None, src=0)
            return recv[0]
        except Exception as exc:
            if _is_rank0():
                logger.warning(
                    "scatter_object_list failed; falling back to broadcast object distribution. error=%s",
                    exc,
                )

    shared: list[Any] = [per_rank_objects if _is_rank0() else None]
    torch.distributed.broadcast_object_list(shared, src=0)
    payload = shared[0]
    if isinstance(payload, list) and 0 <= rank < len(payload):
        return payload[rank]
    return None


def _local_rank_device(default_device: str) -> str:
    local_rank_text = os.environ.get("LOCAL_RANK")
    if local_rank_text and local_rank_text.isdigit() and torch.cuda.is_available():
        local_rank = int(local_rank_text)
        torch.cuda.set_device(local_rank)
        return f"cuda:{local_rank}"
    return default_device


def _build_deepspeed_config_dict(cfg: RLPostTrainConfig, world_size: int) -> dict[str, Any]:
    dtype_text = str(cfg.misc.dtype).strip().lower()
    use_bf16 = dtype_text in {"bf16", "bfloat16"}
    use_fp16 = dtype_text in {"fp16", "float16"}

    micro_batch = max(1, int(cfg.rl.batch_size))
    # GRPO update_policy already performs manual grad accumulation.
    ds_grad_accum = 1
    global_batch = micro_batch * ds_grad_accum * max(1, int(world_size))

    zero_stage = int(cfg.rl.deepspeed_zero_stage)
    zero_cfg: dict[str, Any] = {"stage": zero_stage}
    if cfg.rl.deepspeed_offload_optimizer:
        zero_cfg["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
    if cfg.rl.deepspeed_offload_param:
        zero_cfg["offload_param"] = {"device": "cpu", "pin_memory": True}

    ds_cfg: dict[str, Any] = {
        "train_micro_batch_size_per_gpu": micro_batch,
        "gradient_accumulation_steps": ds_grad_accum,
        "train_batch_size": global_batch,
        "zero_optimization": zero_cfg,
        "bf16": {"enabled": bool(use_bf16)},
        "fp16": {"enabled": bool(use_fp16 and not use_bf16)},
    }
    if cfg.rl.max_grad_norm > 0:
        ds_cfg["gradient_clipping"] = float(cfg.rl.max_grad_norm)
    ds_cfg["optimizer"] = {
        "type": "AdamW",
        "params": {
            "lr": float(cfg.rl.lr),
            "weight_decay": float(cfg.rl.weight_decay),
            "betas": [0.9, 0.999],
            "eps": float(cfg.rl.eps),
        },
    }
    return ds_cfg


def _load_deepspeed_config(cfg: RLPostTrainConfig, world_size: int) -> dict[str, Any]:
    if cfg.rl.deepspeed_config_path:
        path = Path(cfg.rl.deepspeed_config_path)
        if not path.exists():
            raise FileNotFoundError(f"rl.deepspeed_config_path not found: {path}")
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            payload = yaml.safe_load(text)
        else:
            payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid DeepSpeed config payload: {path}")
        return payload
    return _build_deepspeed_config_dict(cfg, world_size=world_size)


def _deepspeed_initialize(
    cfg: RLPostTrainConfig,
    policy_model: AutoModelForCausalLM,
) -> tuple[Any, Any]:
    if deepspeed is None:
        raise RuntimeError(
            "rl.backend=deepspeed but `deepspeed` is not installed. "
            "Install it and re-run."
        )

    world_size = _distributed_world_size()
    if world_size > 1 and not _is_distributed_initialized():
        deepspeed.init_distributed()

    ds_config = _load_deepspeed_config(cfg, world_size=world_size)
    engine, optimizer, _, _ = deepspeed.initialize(
        model=policy_model,
        model_parameters=[p for p in policy_model.parameters() if p.requires_grad],
        config=ds_config,
    )
    return engine, optimizer


def _configure_policy_train_memory(policy_model: AutoModelForCausalLM) -> None:
    cfg_obj = getattr(policy_model, "config", None)
    if cfg_obj is not None and getattr(cfg_obj, "use_cache", None):
        cfg_obj.use_cache = False
        logger.info("Disabled policy model KV cache for training (config.use_cache=False).")

    gc_enable = getattr(policy_model, "gradient_checkpointing_enable", None)
    if callable(gc_enable):
        enabled = False
        try:
            gc_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            enabled = True
        except TypeError:
            gc_enable()
            enabled = True
        except Exception as exc:
            logger.warning("Failed to enable gradient checkpointing: %s", exc)
        if enabled:
            logger.info("Enabled policy gradient checkpointing for training memory reduction.")


def _dtype_to_config_name(dtype: Any, fallback: str) -> str:
    if dtype is None:
        return fallback
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float32:
        return "float32"
    return fallback


def _looks_like_cuda_runtime_fault(text: str) -> bool:
    lowered = str(text).lower()
    markers = (
        "illegal memory access",
        "device-side assert",
        "cuda error",
        "acceleratorerror",
        "cublas",
        "cudnn",
        "driver shutting down",
    )
    return any(marker in lowered for marker in markers)


class ReferenceLogprobClient:
    def __init__(
        self,
        *,
        python_executable: str,
        timeout_sec: float,
        config_payload: dict[str, Any],
        env_overrides: dict[str, str] | None = None,
        remote_host: str | None = None,
        remote_workdir: str | None = None,
    ) -> None:
        self._timeout_sec = float(timeout_sec)
        self._remote_host = str(remote_host).strip() if remote_host else ""
        self._python_executable = python_executable
        self._config_payload = dict(config_payload)
        self._env_overrides = dict(env_overrides or {})
        self._remote_workdir = remote_workdir
        self._worker_script = Path(__file__).resolve().with_name("reference_worker.py")
        if not self._worker_script.exists():
            raise FileNotFoundError(f"reference worker script not found: {self._worker_script}")
        self._proc = None
        self._start_worker()

    def _start_worker(self) -> None:
        env = dict(os.environ)
        for key in ("LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            env.pop(key, None)
        # PYTHONHOME can break venv resolution and make installed packages invisible.
        env.pop("PYTHONHOME", None)
        if self._env_overrides and not self._remote_host:
            for key, value in self._env_overrides.items():
                env[str(key)] = str(value)

        cmd = build_worker_launch_command(
            python_executable=self._python_executable,
            worker_script=self._worker_script,
            worker_module="gemma27_rl.reference_worker",
            remote_host=self._remote_host or None,
            remote_workdir=self._remote_workdir,
            remote_env=self._env_overrides if self._remote_host else None,
        )
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                bufsize=1,
                env=env,
            )
        except Exception as exc:
            location = f" via ssh host={self._remote_host}" if self._remote_host else ""
            raise RuntimeError(f"Failed to start reference worker{location}: cmd={cmd}") from exc

        try:
            init_resp = self.request({"type": "init", "config": self._config_payload})
        except Exception:
            self.close()
            raise
        if not bool(init_resp.get("ok", False)):
            self.close()
            raise RuntimeError(f"reference worker init failed: {init_resp.get('error', 'unknown error')}")

    def _restart_worker(self, reason: str) -> None:
        logger.warning("Restarting reference worker after CUDA/runtime fault: %s", reason)
        self.close()
        self._start_worker()

    def _assert_alive(self) -> None:
        if self._proc.poll() is not None:
            raise RuntimeError(f"reference worker exited unexpectedly with code={self._proc.returncode}")

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._assert_alive()
        assert self._proc.stdin is not None
        assert self._proc.stdout is not None

        try:
            self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
        except Exception as exc:
            raise RuntimeError("Failed to send request to reference worker.") from exc

        ready, _, _ = select.select([self._proc.stdout], [], [], self._timeout_sec)
        if not ready:
            raise TimeoutError(f"reference worker timed out after {self._timeout_sec}s")

        line = self._proc.stdout.readline()
        if not line:
            self._assert_alive()
            raise RuntimeError("reference worker returned empty response")
        try:
            resp = json.loads(line)
        except Exception as exc:
            raise RuntimeError(f"Invalid JSON response from reference worker: {line[:200]!r}") from exc
        if not isinstance(resp, dict):
            raise RuntimeError(f"Unexpected reference worker response type: {type(resp)!r}")
        return resp

    def score_logprobs(self, prompt_ids: list[int], completion_ids: list[int]) -> list[float]:
        rows = self.score_logprobs_batch([(prompt_ids, completion_ids)])
        return rows[0] if rows else []

    def score_logprobs_batch(self, items: list[tuple[list[int], list[int]]]) -> list[list[float]]:
        if not items:
            return []
        payload_items = [
            {
                "prompt_ids": [int(v) for v in prompt_ids],
                "completion_ids": [int(v) for v in completion_ids],
            }
            for prompt_ids, completion_ids in items
        ]
        payload = {
            "type": "score_batch",
            "items": payload_items,
        }
        last_error = ""
        for attempt in range(2):
            try:
                resp = self.request(payload)
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                if attempt == 0 and _looks_like_cuda_runtime_fault(err):
                    self._restart_worker(err)
                    continue
                raise

            if not bool(resp.get("ok", False)):
                err = str(resp.get("error", "unknown error"))
                tb = str(resp.get("traceback") or "").strip()
                if tb:
                    err = f"{err}\nworker_traceback:\n{tb}"
                last_error = err
                if attempt == 0 and _looks_like_cuda_runtime_fault(err):
                    self._restart_worker(err)
                    continue
                raise RuntimeError(f"reference worker score_batch failed: {err}")

            rows_raw = resp.get("logprobs_rows", [])
            if not isinstance(rows_raw, list):
                raise RuntimeError("reference worker score_batch returned invalid logprobs_rows payload")
            rows: list[list[float]] = []
            for row_idx, row in enumerate(rows_raw):
                if not isinstance(row, list):
                    raise RuntimeError(
                        "reference worker score_batch returned invalid logprobs row "
                        f"at index={row_idx}: type={type(row).__name__}"
                    )
                rows.append([float(v) for v in row])
            if len(rows) != len(items):
                raise RuntimeError(
                    f"reference worker score_batch size mismatch: requested={len(items)} returned={len(rows)}"
                )
            return rows

        raise RuntimeError(f"reference worker score_batch failed after restart: {last_error or 'unknown error'}")

    def close(self) -> None:
        if getattr(self, "_proc", None) is None:
            return
        proc = self._proc
        if proc.poll() is None:
            try:
                self.request({"type": "close"})
            except Exception:
                pass
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self._proc = None  # type: ignore[assignment]

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass


def _unwrap_for_generation(model: Any) -> Any:
    module = getattr(model, "module", None)
    return module if module is not None else model


def _parse_cuda_index(device: str | None) -> int | None:
    if not device:
        return None
    text = str(device).strip().lower()
    if text == "cuda":
        return None
    if text.startswith("cuda:"):
        idx_text = text.split(":", 1)[1].strip()
        if idx_text.isdigit():
            return int(idx_text)
    return None


def _parse_cuda_visible_devices_env() -> list[int] | None:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    values: list[int] = []
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        if not token.isdigit():
            return None
        values.append(int(token))
    return values if values else None


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _effective_worker_host(component_host: str | None, fallback_host: str | None) -> str | None:
    return _normalize_optional_text(component_host) or _normalize_optional_text(fallback_host)


def _effective_worker_workdir(component_workdir: str | None, fallback_workdir: str | None) -> str | None:
    return _normalize_optional_text(component_workdir) or _normalize_optional_text(fallback_workdir)


def _get_rank_gpu_mapping_entry(policy_gpu_ids: list[int]) -> dict[str, Any]:
    local_rank_raw = os.environ.get("LOCAL_RANK")
    local_rank = int(local_rank_raw) if local_rank_raw and local_rank_raw.isdigit() else None
    visible = _parse_cuda_visible_devices_env()

    physical_gpu: int | None = None
    if local_rank is not None:
        if visible is not None and 0 <= local_rank < len(visible):
            physical_gpu = int(visible[local_rank])
        elif visible is None:
            physical_gpu = int(local_rank)
    elif len(policy_gpu_ids) == 1 and _distributed_world_size() == 1:
        physical_gpu = int(policy_gpu_ids[0])

    return {
        "rank": _distributed_rank(),
        "local_rank": local_rank,
        "physical_gpu": physical_gpu,
        "host": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _deepspeed_launch_hint(policy_gpu_ids: list[int]) -> str:
    csv = ",".join(str(i) for i in policy_gpu_ids)
    return (
        "deepspeed --include localhost:"
        f"{csv} <python_or_entrypoint> --config <config_path>"
    )


def _validate_deepspeed_partition_strict(cfg: RLPostTrainConfig) -> None:
    if cfg.rl.backend != "deepspeed":
        return

    policy_gpu_ids = _normalize_gpu_id_list(cfg.model.policy_gpu_ids)
    if not policy_gpu_ids:
        raise RuntimeError(
            "DeepSpeed strict mode requires model.policy_gpu_ids. "
            "Set explicit policy GPU ids and launch with matching --include. "
            f"Example: {_deepspeed_launch_hint([0, 1, 2, 3])}"
        )

    world_size = _distributed_world_size()
    local_world_size = len(policy_gpu_ids)
    if world_size < local_world_size or (world_size % local_world_size) != 0:
        raise RuntimeError(
            "DeepSpeed strict mode requires WORLD_SIZE to be a multiple of len(model.policy_gpu_ids). "
            f"world_size={world_size} policy_gpu_ids={policy_gpu_ids}. "
            f"Example: {_deepspeed_launch_hint(policy_gpu_ids)}"
        )

    reference_gpu_ids = _normalize_gpu_id_list(cfg.model.reference_gpu_ids)
    policy_set = set(policy_gpu_ids)
    reserved_map: dict[str, int] = {}
    reference_host = _effective_worker_host(cfg.model.reference_worker_host, cfg.misc.aux_worker_host)
    metricx_host = _effective_worker_host(cfg.reward.metricx.worker_host, cfg.misc.aux_worker_host)
    xcomet_host = _effective_worker_host(cfg.reward.xcomet.worker_host, cfg.misc.aux_worker_host)
    if reference_gpu_ids and reference_host is None and not _reference_uses_colocated_policy(cfg):
        reserved_map["reference"] = int(reference_gpu_ids[0])
    if cfg.reward.metricx.enabled and metricx_host is None:
        metricx_idx = _parse_cuda_index(cfg.reward.metricx.device)
        if metricx_idx is not None:
            reserved_map["metricx"] = metricx_idx
    if cfg.reward.xcomet.enabled and xcomet_host is None:
        xcomet_idx = _parse_cuda_index(cfg.reward.xcomet.device)
        if xcomet_idx is not None:
            reserved_map["xcomet"] = xcomet_idx

    overlap = sorted(policy_set & set(reserved_map.values()))
    if overlap:
        raise RuntimeError(
            "policy_gpu_ids must not overlap reserved reward/reference GPUs. "
            f"policy={sorted(policy_set)} reserved={reserved_map} overlap={overlap}"
        )

    local_entry = _get_rank_gpu_mapping_entry(policy_gpu_ids)
    gathered: list[Any] = [local_entry]
    if _is_distributed_initialized() and world_size > 1:
        # NCCL collectives used by all_gather_object require the current CUDA device
        # to match this process local rank.
        local_rank_raw = os.environ.get("LOCAL_RANK")
        if torch.cuda.is_available() and local_rank_raw and local_rank_raw.isdigit():
            torch.cuda.set_device(int(local_rank_raw))
        gathered = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(gathered, local_entry)

    entries: list[dict[str, Any]] = [e for e in gathered if isinstance(e, dict)]
    entries.sort(key=lambda x: int(x.get("rank", 0)))
    unresolved = [e for e in entries if not isinstance(e.get("physical_gpu"), int)]
    if unresolved:
        raise RuntimeError(
            "Could not resolve physical GPU id for one or more DeepSpeed ranks. "
            f"entries={entries}. Ensure LOCAL_RANK and CUDA_VISIBLE_DEVICES are set correctly. "
            f"Example: {_deepspeed_launch_hint(policy_gpu_ids)}"
        )

    expected_policy = sorted(policy_gpu_ids)
    by_host: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        host = str(entry.get("host") or "")
        by_host.setdefault(host, []).append(entry)

    for host, host_entries in sorted(by_host.items(), key=lambda kv: kv[0]):
        if len(host_entries) != local_world_size:
            raise RuntimeError(
                "DeepSpeed strict mode requires each policy host to run "
                "len(model.policy_gpu_ids) ranks. "
                f"host={host!r} host_ranks={len(host_entries)} expected={local_world_size} "
                f"entries={host_entries}"
            )
        host_policy = sorted({int(e["physical_gpu"]) for e in host_entries})
        if host_policy != expected_policy:
            raise RuntimeError(
                "DeepSpeed strict policy partition mismatch. "
                f"host={host!r} expected_policy={expected_policy} actual_policy={host_policy} "
                f"entries={host_entries}. "
                f"Launch with explicit include. Example: {_deepspeed_launch_hint(expected_policy)}"
            )

    if _is_rank0():
        logger.info(
            "DeepSpeed strict GPU mapping: policy=%s reserved=%s hosts=%s rank_mapping=%s",
            expected_policy,
            reserved_map,
            sorted(by_host.keys()),
            [{k: v for k, v in e.items() if k in {'rank', 'local_rank', 'physical_gpu'}} for e in entries],
        )


def _normalize_gpu_id_list(raw_ids: list[int] | None) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for idx in raw_ids or []:
        value = int(idx)
        if value < 0:
            raise ValueError(f"GPU index must be >= 0, got {value}")
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _pick_free_gpu(
    preferred: int | None,
    used: set[int],
    device_count: int,
) -> int:
    if preferred is not None and 0 <= preferred < device_count and preferred not in used:
        return preferred
    for cand in range(device_count):
        if cand not in used:
            return cand
    if preferred is not None and 0 <= preferred < device_count:
        return preferred
    return 0


def _assign_disjoint_gpu_devices(cfg: RLPostTrainConfig) -> None:
    if not torch.cuda.is_available():
        return

    device_count = int(torch.cuda.device_count())
    if device_count <= 0:
        return

    def _is_cuda_text(value: str | None) -> bool:
        return bool(value) and str(value).strip().lower().startswith("cuda")

    explicit_policy_ids = _normalize_gpu_id_list(cfg.model.policy_gpu_ids)
    explicit_reference_ids = _normalize_gpu_id_list(cfg.model.reference_gpu_ids)
    reference_host = _effective_worker_host(cfg.model.reference_worker_host, cfg.misc.aux_worker_host)
    metricx_host = _effective_worker_host(cfg.reward.metricx.worker_host, cfg.misc.aux_worker_host)
    xcomet_host = _effective_worker_host(cfg.reward.xcomet.worker_host, cfg.misc.aux_worker_host)
    local_reference_ids = (
        explicit_reference_ids if (reference_host is None and not _reference_uses_colocated_policy(cfg)) else []
    )

    if explicit_policy_ids or explicit_reference_ids:
        # In DeepSpeed launcher mode, CUDA_VISIBLE_DEVICES may expose only policy GPUs
        # while reference/reward workers are pinned with absolute physical GPU ids.
        # In that case, skip local visible-range checks here and let strict mapping
        # validation enforce policy rank placement.
        visible_ids = _parse_cuda_visible_devices_env()
        skip_local_range_check = bool(
            cfg.rl.backend == "deepspeed"
            and visible_ids is not None
            and len(visible_ids) < max(explicit_policy_ids + local_reference_ids + [0]) + 1
        )
        if not skip_local_range_check:
            for idx in explicit_policy_ids + local_reference_ids:
                if idx >= device_count:
                    raise ValueError(
                        f"Configured GPU index out of range: {idx} (cuda_count={device_count})"
                    )
        else:
            logger.info(
                "DeepSpeed explicit partition detected with restricted CUDA_VISIBLE_DEVICES=%s; "
                "treating policy/reference/reward GPU ids as physical indices.",
                os.environ.get("CUDA_VISIBLE_DEVICES"),
            )

        used: set[int] = set(explicit_policy_ids) | set(local_reference_ids)

        if _is_cuda_text(cfg.misc.device):
            if not explicit_policy_ids:
                explicit_policy_ids = [_pick_free_gpu(preferred=0, used=used, device_count=device_count)]
                used.update(explicit_policy_ids)
            cfg.misc.device = f"cuda:{explicit_policy_ids[0]}"
        cfg.model.policy_gpu_ids = explicit_policy_ids

        if explicit_reference_ids and not _reference_uses_colocated_policy(cfg):
            cfg.model.reference_device = f"cuda:{explicit_reference_ids[0]}"
        elif explicit_reference_ids:
            logger.info(
                "Ignoring model.reference_gpu_ids=%s because model.reference_runtime=colocate reuses the policy LoRA base.",
                explicit_reference_ids,
            )
        cfg.model.reference_gpu_ids = explicit_reference_ids

        # For explicit DeepSpeed partition, keep reward device ids exactly as configured
        # (physical GPU indices) and do not remap against local visible GPUs.
        if cfg.rl.backend != "deepspeed":
            if cfg.reward.metricx.enabled and metricx_host is None and _is_cuda_text(cfg.reward.metricx.device):
                metricx_idx = _pick_free_gpu(
                    preferred=_parse_cuda_index(cfg.reward.metricx.device),
                    used=used,
                    device_count=device_count,
                )
                used.add(metricx_idx)
                cfg.reward.metricx.device = f"cuda:{metricx_idx}"

            if cfg.reward.xcomet.enabled and xcomet_host is None and _is_cuda_text(cfg.reward.xcomet.device):
                xcomet_idx = _pick_free_gpu(
                    preferred=_parse_cuda_index(cfg.reward.xcomet.device),
                    used=used,
                    device_count=device_count,
                )
                used.add(xcomet_idx)
                cfg.reward.xcomet.device = f"cuda:{xcomet_idx}"

        logger.info(
            "Applied explicit GPU partition: policy_gpu_ids=%s reference_gpu_ids=%s metricx=%s xcomet=%s",
            cfg.model.policy_gpu_ids,
            cfg.model.reference_gpu_ids,
            cfg.reward.metricx.device,
            cfg.reward.xcomet.device,
        )
        return

    components: list[dict[str, Any]] = [
        {"name": "policy", "enabled": _is_cuda_text(cfg.misc.device), "raw": cfg.misc.device, "required": True},
        {
            "name": "metricx",
            "enabled": bool(
                cfg.reward.metricx.enabled
                and metricx_host is None
                and _is_cuda_text(cfg.reward.metricx.device)
            ),
            "raw": cfg.reward.metricx.device,
            "required": False,
        },
        {
            "name": "xcomet",
            "enabled": bool(
                cfg.reward.xcomet.enabled
                and xcomet_host is None
                and _is_cuda_text(cfg.reward.xcomet.device)
            ),
            "raw": cfg.reward.xcomet.device,
            "required": False,
        },
    ]

    used: set[int] = set()
    assigned: dict[str, int] = {}

    for comp in components:
        if not comp["enabled"]:
            continue
        raw = str(comp["raw"]).strip().lower()
        preferred = _parse_cuda_index(raw)
        if preferred is None and comp["name"] == "policy":
            preferred = 0

        idx: int | None = None
        if preferred is not None and 0 <= preferred < device_count and preferred not in used:
            idx = preferred
        else:
            for cand in range(device_count):
                if cand not in used:
                    idx = cand
                    break
            if idx is None:
                idx = preferred if preferred is not None else 0

        used.add(idx)
        assigned[comp["name"]] = idx

    if "policy" in assigned:
        cfg.misc.device = f"cuda:{assigned['policy']}"
    if "metricx" in assigned:
        cfg.reward.metricx.device = f"cuda:{assigned['metricx']}"
    if "xcomet" in assigned:
        cfg.reward.xcomet.device = f"cuda:{assigned['xcomet']}"

    if len({assigned.get("policy"), assigned.get("metricx"), assigned.get("xcomet")} - {None}) < len(assigned):
        logger.warning(
            "Could not assign fully disjoint GPUs for all models (cuda_count=%s, assigned=%s).",
            device_count,
            assigned,
        )
    else:
        logger.info("Assigned model devices: %s", assigned)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    m = sum(values) / len(values)
    var = sum((v - m) ** 2 for v in values) / len(values)
    return float(m), float(var**0.5)


def _apply_aux_worker_defaults(cfg: RLPostTrainConfig) -> None:
    aux_host = _normalize_optional_text(cfg.misc.aux_worker_host)
    aux_workdir = _normalize_optional_text(cfg.misc.aux_worker_remote_workdir)
    if aux_host is not None:
        if _normalize_optional_text(cfg.model.reference_worker_host) is None:
            cfg.model.reference_worker_host = aux_host
        if _normalize_optional_text(cfg.reward.metricx.worker_host) is None:
            cfg.reward.metricx.worker_host = aux_host
        if _normalize_optional_text(cfg.reward.xcomet.worker_host) is None:
            cfg.reward.xcomet.worker_host = aux_host
    if aux_workdir is not None:
        if _normalize_optional_text(cfg.model.reference_worker_remote_workdir) is None:
            cfg.model.reference_worker_remote_workdir = aux_workdir
        if _normalize_optional_text(cfg.reward.metricx.worker_remote_workdir) is None:
            cfg.reward.metricx.worker_remote_workdir = aux_workdir
        if _normalize_optional_text(cfg.reward.xcomet.worker_remote_workdir) is None:
            cfg.reward.xcomet.worker_remote_workdir = aux_workdir


def _flatten(rows: list[list[float]]) -> list[float]:
    out: list[float] = []
    for row in rows:
        out.extend(row)
    return out


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


class _AsyncJsonlWriter:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jsonl-writer")
        self._pending: deque[Any] = deque()

    def _reap_done(self) -> None:
        if not self._pending:
            return
        kept: deque[Any] = deque()
        while self._pending:
            fut = self._pending.popleft()
            if fut.done():
                fut.result()
            else:
                kept.append(fut)
        self._pending = kept

    def _submit(self, fn: Any, *args: Any, **kwargs: Any) -> None:
        self._reap_done()
        fut = self._executor.submit(fn, *args, **kwargs)
        self._pending.append(fut)

    def append_json(self, path: Path, payload: dict[str, Any]) -> None:
        self._submit(_append_jsonl, path, payload)

    def append_rollouts(
        self,
        *,
        path: Path,
        update_idx: int,
        rollouts: list[Rollout],
        advantages: list[list[float]],
        reward_stats: dict[str, float],
    ) -> None:
        self._submit(
            _append_rollout_jsonl,
            path,
            update_idx,
            rollouts,
            advantages,
            reward_stats,
        )

    def append_eval_rows(self, *, path: Path, update_idx: int, eval_rows: list[dict[str, Any]]) -> None:
        self._submit(_append_eval_output_jsonl, path, update_idx, eval_rows)

    def flush(self) -> None:
        while self._pending:
            fut = self._pending.popleft()
            fut.result()

    def close(self) -> None:
        self.flush()
        self._executor.shutdown(wait=True)


def _truncate_jsonl_by_update(path: Path, max_update: int) -> None:
    if not path.exists():
        return

    kept: list[str] = []
    dropped = 0
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                kept.append(line + "\n")
                continue
            if not isinstance(row, dict) or "update" not in row:
                kept.append(line + "\n")
                continue
            try:
                update_idx = int(row["update"])
            except (TypeError, ValueError):
                kept.append(line + "\n")
                continue
            if update_idx <= max_update:
                kept.append(line + "\n")
            else:
                dropped += 1

    if dropped > 0:
        with path.open("w", encoding="utf-8") as f:
            f.writelines(kept)
        logger.info("Truncated %s rows from %s for resume consistency.", dropped, path)


def _append_rollout_jsonl(
    path: Path,
    update_idx: int,
    rollouts: list[Rollout],
    advantages: list[list[float]],
    reward_stats: dict[str, float],
) -> None:
    with path.open("a", encoding="utf-8") as f:
        for ridx, rollout in enumerate(rollouts):
            adv_row = advantages[ridx] if ridx < len(advantages) else []
            payload = {
                "type": "rollout",
                "update": update_idx,
                "rollout_idx": ridx,
                "example_id": rollout.example_id,
                "src_text": rollout.src_text,
                "completion_text": rollout.completion_text,
                "prompt_instance_id": rollout.prompt_instance_id,
                "completion_raw_text": (
                    rollout.completion_raw_text
                    if rollout.completion_raw_text is not None
                    else rollout.completion_text
                ),
                "completion_clean_text": (
                    rollout.completion_clean_text
                    if rollout.completion_clean_text is not None
                    else rollout.completion_text
                ),
                "ref_text": rollout.ref_text,
                "completion_len": len(rollout.completion_token_ids),
                "adv_mean": float(sum(adv_row) / len(adv_row)) if adv_row else 0.0,
                "adv_sum": float(sum(adv_row)) if adv_row else 0.0,
                "old_logprob_mean": float(sum(rollout.old_logprobs) / len(rollout.old_logprobs))
                if rollout.old_logprobs
                else 0.0,
                "ref_logprob_mean": float(sum(rollout.ref_logprobs) / len(rollout.ref_logprobs))
                if rollout.ref_logprobs
                else None,
                "metricx_score_mean_batch": reward_stats.get("metricx_score_mean", 0.0),
                "xcomet_score_mean_batch": reward_stats.get("xcomet_score_mean", 0.0),
                "mqm_score_mean_batch": reward_stats.get("mqm_score_mean", 0.0),
                "esa_score_mean_batch": reward_stats.get("esa_score_mean", 0.0),
                "token_rewards_non_zero_ratio_batch": reward_stats.get("token_rewards_non_zero_ratio", 0.0),
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _append_eval_output_jsonl(path: Path, update_idx: int, eval_rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for ridx, row in enumerate(eval_rows):
            payload = {
                "type": "eval_output",
                "update": update_idx,
                "eval_row_idx": ridx,
                **row,
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _parse_checkpoint_update_idx(path: Path) -> int | None:
    name = path.name.strip()
    prefix = "checkpoint-"
    if not name.startswith(prefix):
        return None
    idx_text = name[len(prefix) :]
    if not idx_text.isdigit():
        return None
    return int(idx_text)


def _prune_old_checkpoints(output_dir: Path, keep_last_n: int) -> list[Path]:
    keep_n = max(0, int(keep_last_n))
    if keep_n <= 0:
        return []

    candidates: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        update_idx = _parse_checkpoint_update_idx(path)
        if update_idx is None:
            continue
        candidates.append((update_idx, path))

    if len(candidates) <= keep_n:
        return []

    # Keep newest N by update index and delete the rest.
    candidates.sort(key=lambda item: item[0], reverse=True)
    drop = candidates[keep_n:]
    removed: list[Path] = []
    for _, path in drop:
        try:
            shutil.rmtree(path)
            removed.append(path)
        except Exception as exc:
            logger.warning("Failed to prune old checkpoint %s: %s", path, exc)
    return removed


def _save_trainer_state(path: Path, payload: dict[str, Any]) -> None:
    state_path = path / "trainer_state.json"
    state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_trainer_state(path: Path) -> dict[str, Any] | None:
    state_path = path / "trainer_state.json"
    if not state_path.exists():
        return None
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read trainer_state.json from %s: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        logger.warning("Invalid trainer_state.json format at %s", path)
        return None
    return payload


def _restore_best_from_log(log_path: Path) -> tuple[float, int | None]:
    if not log_path.exists():
        return float("-inf"), None

    best_score = float("-inf")
    best_update: int | None = None
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            if row.get("type") != "eval":
                continue
            score_raw = row.get("model_select_score")
            update_raw = row.get("update")
            try:
                score = float(score_raw)
                update = int(update_raw)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(score):
                continue
            if score > best_score:
                best_score = score
                best_update = update
    return best_score, best_update


def _resolve_resume_checkpoint(
    cfg: RLPostTrainConfig,
    output_dir: Path,
) -> tuple[Path | None, int]:
    explicit = cfg.logging.resume_from_checkpoint
    if explicit:
        resume_path = Path(explicit)
        if not resume_path.exists():
            raise FileNotFoundError(f"logging.resume_from_checkpoint not found: {resume_path}")
        state = _load_trainer_state(resume_path) or {}
        update_idx = int(state.get("update_idx", _parse_checkpoint_update_idx(resume_path) or 0))
        return resume_path, update_idx

    if not cfg.logging.auto_resume:
        return None, 0

    candidates: list[tuple[int, Path]] = []
    resume_latest = output_dir / "resume_latest"
    if resume_latest.exists() and resume_latest.is_dir():
        state = _load_trainer_state(resume_latest) or {}
        update_raw = state.get("update_idx")
        try:
            update_idx = int(update_raw)
        except (TypeError, ValueError):
            update_idx = 0
        candidates.append((update_idx, resume_latest))

    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        update_idx = _parse_checkpoint_update_idx(path)
        if update_idx is None:
            continue
        candidates.append((update_idx, path))

    if not candidates:
        return None, 0

    update_idx, resume_path = max(candidates, key=lambda item: item[0])
    return resume_path, update_idx


def _resolve_model_dtype_and_attn(
    cfg: RLPostTrainConfig,
    device: str,
) -> tuple[torch.dtype | None, str | None]:
    dtype = resolve_torch_dtype(cfg.misc.dtype)
    attn_impl_raw = (cfg.model.attn_implementation or "").strip()
    attn_impl = attn_impl_raw.lower()
    is_cuda_device = str(device).strip().lower().startswith("cuda")

    if not is_cuda_device:
        if dtype in {torch.float16, torch.bfloat16}:
            dtype = torch.float32
        if attn_impl == "flash_attention_2":
            logger.warning(
                "flash_attention_2 requested on %s; falling back to sdpa for this model load.",
                device,
            )
            return dtype, "sdpa"
        if attn_impl and attn_impl != "auto":
            return dtype, attn_impl_raw
        return dtype, None

    if attn_impl == "flash_attention_2" and dtype not in {torch.float16, torch.bfloat16}:
        logger.warning(
            "flash_attention_2 requires fp16/bf16. Overriding model dtype from %s to bfloat16.",
            dtype,
        )
        dtype = torch.bfloat16

    if attn_impl and attn_impl != "auto":
        return dtype, attn_impl_raw
    return dtype, None


def _resolve_reference_attn_implementation(
    cfg: RLPostTrainConfig,
    requested_device: str,
    base_attn_impl: str | None,
) -> str | None:
    override_raw = cfg.model.reference_attn_implementation
    if override_raw is not None and str(override_raw).strip():
        return str(override_raw).strip()

    device_text = str(requested_device).strip().lower()
    if not device_text.startswith("cuda"):
        return base_attn_impl

    # Reference logprob scoring is latency-insensitive versus policy generation,
    # and FA2 kernel crashes are much harder to recover from than slower SDPA.
    base_text = str(base_attn_impl or "").strip().lower()
    if base_text in {"", "auto", "flash_attention_2"}:
        return "sdpa"
    return base_attn_impl


_ADAPTER_CONFIG_FILENAME = "adapter_config.json"


def _lora_enabled(cfg: RLPostTrainConfig) -> bool:
    return bool(getattr(getattr(cfg.model, "lora", None), "enabled", False))


def _reference_uses_colocated_policy(cfg: RLPostTrainConfig) -> bool:
    if not _reference_kl_enabled(cfg):
        return False
    return str(cfg.model.reference_runtime or "worker").strip().lower() == "colocate"


def _require_peft_for_lora() -> None:
    if LoraConfig is None or PeftModel is None or TaskType is None or get_peft_model is None:
        raise RuntimeError(
            "model.lora.enabled=true requires the `peft` package with a compatible transformers install."
        )


def _read_local_adapter_base_model_name_or_path(model_name_or_path: str) -> str | None:
    path = Path(str(model_name_or_path)).expanduser()
    cfg_path = path / _ADAPTER_CONFIG_FILENAME
    if not cfg_path.exists():
        return None
    try:
        payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to read LoRA adapter config: {cfg_path}") from exc
    base_source = str(payload.get("base_model_name_or_path") or "").strip()
    return base_source or None


def _checkpoint_has_model_artifacts(path: Path | None) -> bool:
    if path is None or (not path.is_dir()):
        return False
    return (path / "config.json").exists() or (path / _ADAPTER_CONFIG_FILENAME).exists()


def _build_lora_config(cfg: RLPostTrainConfig) -> Any:
    _require_peft_for_lora()
    return LoraConfig(
        r=int(cfg.model.lora.r),
        lora_alpha=int(cfg.model.lora.alpha),
        lora_dropout=float(cfg.model.lora.dropout),
        bias=str(cfg.model.lora.bias).strip().lower(),
        target_modules=[str(name).strip() for name in cfg.model.lora.target_modules],
        task_type=TaskType.CAUSAL_LM,
    )


def _log_trainable_parameter_summary(model: Any, *, tag: str) -> None:
    total = 0
    trainable = 0
    for param in model.parameters():
        count = int(param.numel())
        total += count
        if param.requires_grad:
            trainable += count
    pct = (100.0 * trainable / total) if total > 0 else 0.0
    logger.info("%s trainable parameters: %s / %s (%.4f%%)", tag, trainable, total, pct)


class _AdapterDisabledReferenceProxy(torch.nn.Module):
    def __init__(self, wrapped_model: Any) -> None:
        super().__init__()
        self.wrapped_model = wrapped_model

    def forward(self, *args, **kwargs):
        disable_adapter = getattr(self.wrapped_model, "disable_adapter", None)
        if not callable(disable_adapter):
            return self.wrapped_model(*args, **kwargs)
        with disable_adapter():
            return self.wrapped_model(*args, **kwargs)

    def get_input_embeddings(self):
        getter = getattr(self.wrapped_model, "get_input_embeddings", None)
        if callable(getter):
            return getter()
        raise AttributeError("wrapped model has no get_input_embeddings()")

    @property
    def config(self):
        return getattr(self.wrapped_model, "config", None)


def _load_causal_lm(
    model_name_or_path: str,
    kwargs: dict[str, Any],
    single_device: str,
    gpu_ids: list[int],
    component_name: str,
) -> AutoModelForCausalLM:
    explicit_gpu_ids = _normalize_gpu_id_list(gpu_ids)
    if len(explicit_gpu_ids) <= 1:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        model.to(single_device)
        return model
    raise RuntimeError(
        f"{component_name} model requested on multiple GPUs {explicit_gpu_ids}, "
        "but device_map=auto path is disabled. "
        "Use rl.backend=deepspeed for multi-GPU policy training, and keep reference model on a single GPU."
    )


def _register_qwen35_zero3_external_parameters(model: Any, cfg: RLPostTrainConfig) -> None:
    if str(cfg.rl.backend).strip().lower() != "deepspeed":
        return
    if int(cfg.rl.deepspeed_zero_stage) != 3:
        return
    if deepspeed is None:
        return
    zero = getattr(deepspeed, "zero", None)
    register_external_parameter = getattr(zero, "register_external_parameter", None)
    if not callable(register_external_parameter):
        return

    registered = 0
    for module in model.modules():
        conv1d = getattr(module, "conv1d", None)
        if type(module).__name__ != "Qwen3_5GatedDeltaNet":
            continue
        weight = getattr(conv1d, "weight", None)
        if isinstance(weight, torch.nn.Parameter):
            register_external_parameter(module, weight)
            registered += 1
        bias = getattr(conv1d, "bias", None)
        if isinstance(bias, torch.nn.Parameter):
            register_external_parameter(module, bias)
            registered += 1
    if registered:
        logger.info(
            "Registered %s external parameter(s) for Qwen3.5 gated-delta conv1d under DeepSpeed ZeRO-3.",
            registered,
        )


def _load_policy_model(
    cfg: RLPostTrainConfig,
    device: str,
    model_name_or_path: str | None = None,
) -> Any:
    dtype, attn_impl = _resolve_model_dtype_and_attn(cfg, device)
    if cfg.model.disable_policy_flash_attention:
        attn_text = str(attn_impl or "").strip().lower()
        if attn_text == "flash_attention_2":
            logger.warning(
                "Policy flash_attention_2 is disabled by model.disable_policy_flash_attention=true; "
                "falling back to sdpa for training stability."
            )
            attn_impl = "sdpa"

    kwargs: dict[str, Any] = {
        "trust_remote_code": cfg.model.trust_remote_code,
    }
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl

    source = model_name_or_path or cfg.model.policy_name_or_path
    policy_gpu_ids = [] if cfg.rl.backend == "deepspeed" else cfg.model.policy_gpu_ids
    adapter_base_source = _read_local_adapter_base_model_name_or_path(source)
    if adapter_base_source is not None and not _lora_enabled(cfg):
        raise RuntimeError(
            f"Policy source {source} is a LoRA adapter checkpoint, but model.lora.enabled=false."
        )

    base_source = adapter_base_source or source
    model = _load_causal_lm(
        model_name_or_path=base_source,
        kwargs=kwargs,
        single_device=device,
        gpu_ids=policy_gpu_ids,
        component_name="policy",
    )
    if adapter_base_source is not None:
        _require_peft_for_lora()
        try:
            model = PeftModel.from_pretrained(model, source, is_trainable=True)
            logger.info("Loaded policy LoRA adapter checkpoint from %s (base=%s).", source, adapter_base_source)
        except Exception as exc:
            source_path = Path(str(source)).expanduser()
            text = str(exc)
            can_fallback_to_deepspeed_resume = (
                str(cfg.rl.backend).strip().lower() == "deepspeed"
                and _is_deepspeed_checkpoint_dir(source_path)
                and _lora_enabled(cfg)
                and (("size mismatch" in text) or ("torch.Size([0])" in text))
            )
            if not can_fallback_to_deepspeed_resume:
                raise
            logger.warning(
                "LoRA adapter artifacts at %s appear invalid for resume (%s). "
                "Falling back to fresh adapter init from base model %s and relying on DeepSpeed shards.",
                source,
                text or type(exc).__name__,
                adapter_base_source,
            )
            model = get_peft_model(model, _build_lora_config(cfg))
    elif _lora_enabled(cfg):
        _require_peft_for_lora()
        model = get_peft_model(model, _build_lora_config(cfg))
        logger.info(
            "Enabled policy LoRA: r=%s alpha=%s dropout=%s targets=%s",
            cfg.model.lora.r,
            cfg.model.lora.alpha,
            cfg.model.lora.dropout,
            [str(name) for name in cfg.model.lora.target_modules],
        )

    _register_qwen35_zero3_external_parameters(model, cfg)

    if _lora_enabled(cfg):
        enable_input_require_grads = getattr(model, "enable_input_require_grads", None)
        if callable(enable_input_require_grads):
            try:
                enable_input_require_grads()
            except Exception as exc:
                logger.warning("Failed to enable input grads for LoRA training: %s", exc)
        _log_trainable_parameter_summary(model, tag="Policy")

    return model


def _reference_kl_enabled(cfg: RLPostTrainConfig) -> bool:
    return bool(cfg.model.use_reference_model and float(cfg.rl.kl_coef) > 0.0)


def _load_reference_model(cfg: RLPostTrainConfig, default_device: str) -> tuple[Any | None, str | None]:
    if not _reference_kl_enabled(cfg):
        return None, None

    ref_name = cfg.model.reference_name_or_path or cfg.model.policy_name_or_path
    reference_gpu_ids = _normalize_gpu_id_list(cfg.model.reference_gpu_ids)
    if reference_gpu_ids:
        if len(reference_gpu_ids) > 1:
            logger.warning(
                "reference_gpu_ids=%s requested, but multi-GPU reference loading is disabled. "
                "Using only cuda:%s",
                reference_gpu_ids,
                reference_gpu_ids[0],
            )
        ref_device = resolve_device(f"cuda:{reference_gpu_ids[0]}")
    else:
        ref_device = resolve_device(cfg.model.reference_device or default_device)

    dtype, attn_impl = _resolve_model_dtype_and_attn(cfg, ref_device)
    attn_impl = _resolve_reference_attn_implementation(cfg, ref_device, attn_impl)

    kwargs: dict[str, Any] = {
        "trust_remote_code": cfg.model.trust_remote_code,
    }
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl

    adapter_base_source = _read_local_adapter_base_model_name_or_path(ref_name)
    model = _load_causal_lm(
        model_name_or_path=adapter_base_source or ref_name,
        kwargs=kwargs,
        single_device=ref_device,
        gpu_ids=reference_gpu_ids[:1],
        component_name="reference",
    )
    if adapter_base_source is not None:
        _require_peft_for_lora()
        model = PeftModel.from_pretrained(model, ref_name, is_trainable=False)
        logger.info("Loaded reference LoRA adapter checkpoint from %s (base=%s).", ref_name, adapter_base_source)
    cfg_obj = getattr(model, "config", None)
    if cfg_obj is not None and getattr(cfg_obj, "use_cache", None):
        cfg_obj.use_cache = False
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, ref_device


def _create_colocated_reference_logprob_batch_fn(
    cfg: RLPostTrainConfig,
    policy_model: Any,
    *,
    device: str,
) -> tuple[Callable[[list[tuple[list[int], list[int]]]], list[list[float]]], str]:
    if not _lora_enabled(cfg):
        raise RuntimeError("model.reference_runtime=colocate requires model.lora.enabled=true.")
    disable_adapter = getattr(policy_model, "disable_adapter", None)
    if not callable(disable_adapter):
        raise RuntimeError(
            "model.reference_runtime=colocate requires a PEFT policy model with disable_adapter() support."
        )

    proxy = _AdapterDisabledReferenceProxy(policy_model)
    model_device = str(next(policy_model.parameters()).device)
    micro_batch = max(1, int(cfg.model.reference_logprob_micro_batch_size))

    def _score(items: list[tuple[list[int], list[int]]]) -> list[list[float]]:
        rows = compute_completion_logprobs_batch(
            model=proxy,
            items=items,
            device=device,
            micro_batch_size=micro_batch,
        )
        return [[float(v) for v in row.tolist()] for row in rows]

    logger.info(
        "Using colocated reference logprob path from the policy LoRA base (device=%s micro_batch=%s).",
        model_device,
        micro_batch,
    )
    return _score, model_device


def _score_reference_requests_with_batch_fn(
    *,
    requests: list[tuple[int, tuple[list[int], list[int]]]],
    ref_logprob_batch_fn: Callable[[list[tuple[list[int], list[int]]]], list[list[float]]],
    update_idx: int,
    source_label: str,
) -> tuple[dict[int, list[float]], int, int]:
    if not requests:
        return {}, 0, 1

    batch_chunk = _env_int("GEMMA27_RL_REF_FILL_CHUNK", default=16, minimum=1)
    responses_by_idx: dict[int, list[float]] = {}
    batch_calls = 0
    cursor = 0
    while cursor < len(requests):
        remaining = len(requests) - cursor
        chunk_size = min(batch_chunk, remaining)
        chunk = requests[cursor:cursor + chunk_size]
        try:
            chunk_rows = ref_logprob_batch_fn([pair for _, pair in chunk])
            batch_calls += 1
        except Exception as exc:
            if chunk_size > 1:
                batch_chunk = max(1, chunk_size // 2)
                logger.warning(
                    "%s failed at update=%s; reducing chunk size to %s and retrying. error=%s",
                    source_label,
                    update_idx,
                    batch_chunk,
                    exc,
                )
                continue
            bad_idx = chunk[0][0]
            logger.error(
                "%s failed for one rollout at update=%s idx=%s; skipping KL for this sample. error=%s",
                source_label,
                update_idx,
                bad_idx,
                exc,
            )
            cursor += 1
            continue

        if len(chunk_rows) != len(chunk):
            logger.warning(
                "%s returned mismatched size at update=%s requested=%s got=%s; retrying with smaller chunks.",
                source_label,
                update_idx,
                len(chunk),
                len(chunk_rows),
            )
            if chunk_size > 1:
                batch_chunk = max(1, chunk_size // 2)
                continue
            cursor += 1
            continue

        for (rollout_idx, _), row in zip(chunk, chunk_rows):
            responses_by_idx[rollout_idx] = [float(v) for v in row]
        cursor += chunk_size

    return responses_by_idx, batch_calls, batch_chunk


def _fill_missing_reference_logprobs_distributed_colocate(
    *,
    merged_rollouts: list[Rollout] | None,
    cfg: RLPostTrainConfig,
    update_idx: int,
    ref_logprob_batch_fn: Callable[[list[tuple[list[int], list[int]]]], list[list[float]]],
    rank: int,
) -> int:
    if (merged_rollouts is None and _is_rank0()) or (not _reference_kl_enabled(cfg)):
        return 0

    requests_by_rank: list[list[tuple[int, tuple[list[int], list[int]]]]] | None = None
    total_missing = 0
    if _is_rank0():
        assert merged_rollouts is not None
        missing_idx = [i for i, rollout in enumerate(merged_rollouts) if rollout.ref_logprobs is None]
        if missing_idx:
            for idx in list(missing_idx):
                if merged_rollouts[idx].completion_token_ids:
                    continue
                merged_rollouts[idx].ref_logprobs = []
            missing_idx = [i for i in missing_idx if merged_rollouts[i].ref_logprobs is None]
        total_missing = len(missing_idx)
        requests: list[tuple[int, tuple[list[int], list[int]]]] = [
            (
                idx,
                (
                    merged_rollouts[idx].prompt_input_ids,
                    merged_rollouts[idx].completion_token_ids,
                ),
            )
            for idx in missing_idx
        ]
        world_size = max(1, _distributed_world_size())
        requests_by_rank = [[] for _ in range(world_size)]
        for req_idx, item in enumerate(requests):
            requests_by_rank[req_idx % world_size].append(item)

    shared_meta: list[Any] = [total_missing if _is_rank0() else 0]
    _broadcast_object_list(shared_meta, src=0)
    total_missing = int(shared_meta[0] or 0)
    local_requests_raw = _scatter_object_from_rank0(requests_by_rank if _is_rank0() else None, rank=rank)
    local_requests = local_requests_raw if isinstance(local_requests_raw, list) else []

    local_rows, local_batch_calls, local_chunk = _score_reference_requests_with_batch_fn(
        requests=[
            (int(idx), (list(prompt_ids), list(completion_ids)))
            for idx, (prompt_ids, completion_ids) in local_requests
        ],
        ref_logprob_batch_fn=ref_logprob_batch_fn,
        update_idx=update_idx,
        source_label="Reference colocated score_batch",
    )
    gathered = _gather_object_to_rank0(
        {
            "rows": local_rows,
            "batch_calls": int(local_batch_calls),
            "chunk": int(local_chunk),
        }
    )

    if not _is_rank0():
        return 0

    assert merged_rollouts is not None
    merged_rows: dict[int, list[float]] = {}
    batch_calls_total = 0
    chunk_sizes: list[int] = []
    for payload in gathered or []:
        if not isinstance(payload, dict):
            continue
        batch_calls_total += int(payload.get("batch_calls", 0) or 0)
        chunk_raw = int(payload.get("chunk", 0) or 0)
        if chunk_raw > 0:
            chunk_sizes.append(chunk_raw)
        rows_payload = payload.get("rows")
        if not isinstance(rows_payload, dict):
            continue
        for idx_raw, row in rows_payload.items():
            try:
                rollout_idx = int(idx_raw)
            except Exception:
                continue
            if isinstance(row, list):
                merged_rows[rollout_idx] = [float(v) for v in row]

    filled = 0
    for idx, row in merged_rows.items():
        if 0 <= idx < len(merged_rollouts):
            merged_rollouts[idx].ref_logprobs = row
            filled += 1

    logger.info(
        "Filled colocated reference logprobs on rank0 for %s/%s gathered rollouts at update=%s "
        "(distributed batch_calls=%s chunk_candidates=%s).",
        filled,
        total_missing,
        update_idx,
        batch_calls_total,
        sorted(set(chunk_sizes)),
    )
    return filled


def _create_reference_logprob_client(
    cfg: RLPostTrainConfig,
    default_device: str,
) -> tuple[ReferenceLogprobClient | None, str | None]:
    if not _reference_kl_enabled(cfg):
        return None, None

    ref_name = cfg.model.reference_name_or_path or cfg.model.policy_name_or_path
    if cfg.model.reference_runtime == "cpu":
        requested_device = "cpu"
    else:
        reference_gpu_ids = _normalize_gpu_id_list(cfg.model.reference_gpu_ids)
        if reference_gpu_ids:
            requested_device = resolve_device(f"cuda:{reference_gpu_ids[0]}")
        else:
            requested_device = resolve_device(cfg.model.reference_device or default_device)

    dtype, attn_impl = _resolve_model_dtype_and_attn(cfg, requested_device)
    attn_impl = _resolve_reference_attn_implementation(cfg, requested_device, attn_impl)
    worker_device = requested_device
    worker_env_overrides: dict[str, str] | None = collect_huggingface_worker_env() or None
    remote_host = _effective_worker_host(cfg.model.reference_worker_host, cfg.misc.aux_worker_host)
    remote_workdir = _effective_worker_workdir(
        cfg.model.reference_worker_remote_workdir,
        cfg.misc.aux_worker_remote_workdir,
    )
    reference_gpu_idx = _parse_cuda_index(requested_device)
    if reference_gpu_idx is not None:
        worker_env_overrides = merge_env_overrides(
            worker_env_overrides,
            {"CUDA_VISIBLE_DEVICES": str(reference_gpu_idx)},
        )
        worker_device = "cuda:0"

    cfg_payload: dict[str, Any] = {
        "model_name_or_path": ref_name,
        "trust_remote_code": bool(cfg.model.trust_remote_code),
        "dtype": _dtype_to_config_name(dtype, str(cfg.misc.dtype)),
        "attn_implementation": attn_impl,
        "device": worker_device,
        "logprob_micro_batch_size": int(cfg.model.reference_logprob_micro_batch_size),
    }
    python_executable = cfg.model.reference_python_executable or sys.executable
    logger.info(
        "Reference worker init config: runtime=%s requested_device=%s worker_device=%s "
        "attn=%s dtype=%s micro_batch=%s python=%s host=%s",
        cfg.model.reference_runtime,
        requested_device,
        worker_device,
        cfg_payload.get("attn_implementation"),
        cfg_payload.get("dtype"),
        cfg_payload.get("logprob_micro_batch_size"),
        python_executable,
        remote_host or "local",
    )
    client = ReferenceLogprobClient(
        python_executable=python_executable,
        timeout_sec=float(cfg.model.reference_subprocess_timeout_sec),
        config_payload=cfg_payload,
        env_overrides=worker_env_overrides,
        remote_host=remote_host,
        remote_workdir=remote_workdir,
    )
    logger.info(
        "Reference worker started with python=%s host=%s requested_device=%s worker_device=%s model=%s",
        python_executable,
        remote_host or "local",
        requested_device,
        worker_device,
        ref_name,
    )
    return client, requested_device


def _sample_batch(examples: list, batch_size: int, rng: random.Random) -> list:
    if not examples:
        return []
    n = min(batch_size, len(examples))
    if n == len(examples):
        indices = list(range(len(examples)))
        rng.shuffle(indices)
        return [examples[i] for i in indices]
    indices = [rng.randrange(len(examples)) for _ in range(n)]
    return [examples[i] for i in indices]


def _resolve_group_ids_for_rollouts(
    rollouts: list[Rollout],
    *,
    num_samples_per_prompt: int,
) -> list[str]:
    prompt_instance_ids: list[str] = []
    all_present = True
    for rollout in rollouts:
        marker = str(rollout.prompt_instance_id or "").strip()
        prompt_instance_ids.append(marker)
        if not marker:
            all_present = False

    if all_present:
        return [f"prompt_instance:{marker}" for marker in prompt_instance_ids]

    # Backward-compatible fallback for legacy rollouts that do not carry
    # prompt_instance_id yet: infer prompt-instance groups by rollout ordering.
    group_size = max(1, int(num_samples_per_prompt))
    return [f"fallback_prompt_instance:{idx // group_size}" for idx in range(len(rollouts))]


def _validate_scorer_batch_lengths(
    *,
    scorer_name: str,
    requested: int,
    sequence_scores: Any,
    error_spans: Any | None = None,
) -> tuple[list[Any], list[Any] | None]:
    if not isinstance(sequence_scores, (list, tuple)):
        raise RuntimeError(
            f"{scorer_name} scorer returned non-list sequence_scores "
            f"(type={type(sequence_scores).__name__}, requested={requested})."
        )
    score_rows = list(sequence_scores)
    if len(score_rows) != int(requested):
        raise RuntimeError(
            f"{scorer_name} scorer returned mismatched sequence_scores length: "
            f"requested={requested} returned={len(score_rows)}"
        )

    span_rows: list[Any] | None = None
    if error_spans is not None:
        if not isinstance(error_spans, (list, tuple)):
            raise RuntimeError(
                f"{scorer_name} scorer returned non-list error_spans "
                f"(type={type(error_spans).__name__}, requested={requested})."
            )
        span_rows = list(error_spans)
        if len(span_rows) != int(requested):
            raise RuntimeError(
                f"{scorer_name} scorer returned mismatched error_spans length: "
                f"requested={requested} returned={len(span_rows)}"
            )

    return score_rows, span_rows


def _score_with_cache_metricx(
    samples: list[SampleForScoring],
    scorer: MetricXQEScorer,
    cache: dict[tuple[str, str, str], float],
    use_cache: bool,
) -> list[float]:
    out = [0.0 for _ in samples]
    uncached: list[SampleForScoring] = []
    uncached_idx: list[int] = []

    for idx, sample in enumerate(samples):
        key = (sample.src, sample.mt, (sample.ref or "") if scorer.cfg.use_reference else "")
        if use_cache and key in cache:
            out[idx] = cache[key]
        else:
            uncached.append(sample)
            uncached_idx.append(idx)

    if uncached:
        raw_scores = scorer.score_batch(uncached).sequence_scores
        scores, _ = _validate_scorer_batch_lengths(
            scorer_name="MetricX",
            requested=len(uncached),
            sequence_scores=raw_scores,
        )
        for idx, score, sample in zip(uncached_idx, scores, uncached):
            out[idx] = float(score)
            if use_cache:
                cache[(sample.src, sample.mt, (sample.ref or "") if scorer.cfg.use_reference else "")] = float(score)

    return out


def _score_with_cache_xcomet(
    samples: list[SampleForScoring],
    scorer: XCometXLScorer,
    cache: dict[tuple[str, str, str], tuple[float, list[dict[str, Any]]]],
    use_cache: bool,
) -> tuple[list[float], list[list[dict[str, Any]]]]:
    scores = [0.0 for _ in samples]
    spans = [[] for _ in samples]

    uncached: list[SampleForScoring] = []
    uncached_idx: list[int] = []
    for idx, sample in enumerate(samples):
        key = (sample.src, sample.mt, sample.ref or "")
        if use_cache and key in cache:
            scores[idx], spans[idx] = cache[key]
        else:
            uncached.append(sample)
            uncached_idx.append(idx)

    if uncached:
        out = scorer.score_batch(uncached)
        raw_span_rows = (out.metadata or {}).get("error_spans", [[] for _ in uncached])
        sequence_scores, span_rows = _validate_scorer_batch_lengths(
            scorer_name="xCOMET",
            requested=len(uncached),
            sequence_scores=out.sequence_scores,
            error_spans=raw_span_rows,
        )
        assert span_rows is not None
        for idx, score, span_row, sample in zip(uncached_idx, sequence_scores, span_rows, uncached):
            score_f = float(score)
            span_iter = span_row if isinstance(span_row, (list, tuple)) else []
            span_list = [s for s in span_iter if isinstance(s, dict)]
            scores[idx] = score_f
            spans[idx] = span_list
            if use_cache:
                cache[(sample.src, sample.mt, sample.ref or "")] = (score_f, span_list)

    return scores, spans


def _score_with_cache_mqm(
    samples: list[SampleForScoring],
    scorer: OpenAICompatibleMQMScorer,
    cache: dict[tuple[str, str, str], tuple[float, list[dict[str, Any]]]],
    use_cache: bool,
) -> tuple[list[float], list[list[dict[str, Any]]]]:
    scores = [0.0 for _ in samples]
    spans = [[] for _ in samples]
    uncached: list[SampleForScoring] = []
    uncached_idx: list[int] = []

    for idx, sample in enumerate(samples):
        key = (sample.src, sample.mt, (sample.ref or "") if scorer.cfg.use_reference else "")
        if use_cache and key in cache:
            scores[idx], spans[idx] = cache[key]
        else:
            uncached.append(sample)
            uncached_idx.append(idx)

    if uncached:
        out = scorer.score_batch(uncached)
        raw_span_rows = (out.metadata or {}).get("error_spans", [[] for _ in uncached])
        sequence_scores, span_rows = _validate_scorer_batch_lengths(
            scorer_name="MQM",
            requested=len(uncached),
            sequence_scores=out.sequence_scores,
            error_spans=raw_span_rows,
        )
        assert span_rows is not None
        for idx, score, span_row, sample in zip(uncached_idx, sequence_scores, span_rows, uncached):
            score_f = float(score)
            span_iter = span_row if isinstance(span_row, (list, tuple)) else []
            span_list = [s for s in span_iter if isinstance(s, dict)]
            scores[idx] = score_f
            spans[idx] = span_list
            if use_cache:
                cache[(sample.src, sample.mt, (sample.ref or "") if scorer.cfg.use_reference else "")] = (
                    score_f,
                    span_list,
                )

    return scores, spans


def _score_with_cache_esa(
    samples: list[SampleForScoring],
    scorer: OpenAICompatibleESAScorer,
    cache: dict[tuple[str, str, str], float],
    use_cache: bool,
) -> list[float]:
    out = [0.0 for _ in samples]
    uncached: list[SampleForScoring] = []
    uncached_idx: list[int] = []

    for idx, sample in enumerate(samples):
        key = (sample.src, sample.mt, (sample.ref or "") if scorer.cfg.use_reference else "")
        if use_cache and key in cache:
            out[idx] = float(cache[key])
        else:
            uncached.append(sample)
            uncached_idx.append(idx)

    if uncached:
        raw_scores = scorer.score_batch(uncached).sequence_scores
        scores, _ = _validate_scorer_batch_lengths(
            scorer_name="ESA",
            requested=len(uncached),
            sequence_scores=raw_scores,
        )
        for idx, score, sample in zip(uncached_idx, scores, uncached):
            out[idx] = float(score)
            if use_cache:
                cache[(sample.src, sample.mt, (sample.ref or "") if scorer.cfg.use_reference else "")] = float(score)

    return out


def _prepare_rewards_and_advantages(
    rollouts: list[Rollout],
    cfg: RLPostTrainConfig,
    metricx_scorer: MetricXQEScorer | None,
    xcomet_scorer: XCometXLScorer | None,
    mqm_scorer: OpenAICompatibleMQMScorer | None,
    esa_scorer: OpenAICompatibleESAScorer | None,
    metricx_cache: dict[tuple[str, str, str], float],
    xcomet_cache: dict[tuple[str, str, str], tuple[float, list[dict[str, Any]]]],
    mqm_cache: dict[tuple[str, str, str], tuple[float, list[dict[str, Any]]]],
    esa_cache: dict[tuple[str, str, str], float],
    tokenizer: Any | None = None,
) -> tuple[list[list[float]], dict[str, float], dict[str, float]]:
    global _ESA_ALL_ZERO_WARNED

    def _sanitize(values: list[float], fallback: float) -> tuple[list[float], int]:
        out: list[float] = []
        replaced = 0
        for value in values:
            if math.isfinite(value):
                out.append(float(value))
            else:
                out.append(float(fallback))
                replaced += 1
        return out, replaced

    samples: list[SampleForScoring] = []
    special_token_strings = _collect_tokenizer_special_token_strings(tokenizer)
    special_token_ids = _collect_tokenizer_special_token_ids(tokenizer)
    special_token_id_labels = _build_special_token_id_label_map(tokenizer, special_token_ids)
    special_token_penalty_strings = [
        tok for tok in special_token_strings if str(tok).strip().lower() not in {"<think>", "</think>"}
    ]
    exempt_final_special_ids, exempt_final_special_strings = _collect_exempt_final_end_of_turn_markers(
        tokenizer,
        special_token_strings=special_token_penalty_strings,
        special_token_ids=special_token_ids,
    )
    span_reward_texts: list[str] = []
    span_reward_samples: list[SampleForScoring] = []
    mqm_esa_samples: list[SampleForScoring] = []
    raw_completion_texts: list[str] = []
    clean_completion_texts: list[str] = []
    sanitized_target_rows = 0
    sanitized_marker_total = 0
    for rollout in rollouts:
        raw_mt = str(rollout.completion_raw_text if rollout.completion_raw_text is not None else rollout.completion_text or "")
        sanitized_mt, replacement_count = _sanitize_text_for_mqm_esa(
            raw_mt,
            special_tokens=special_token_strings,
        )
        clean_mt = str(rollout.completion_clean_text if rollout.completion_clean_text is not None else sanitized_mt)
        raw_completion_texts.append(raw_mt)
        clean_completion_texts.append(clean_mt)
        if replacement_count > 0:
            sanitized_target_rows += 1
            sanitized_marker_total += int(replacement_count)
        samples.append(SampleForScoring(src=rollout.src_text, mt=clean_mt, ref=rollout.ref_text))
        span_reward_texts.append(clean_mt)
        span_reward_samples.append(SampleForScoring(src=rollout.src_text, mt=clean_mt, ref=rollout.ref_text))
        mqm_esa_samples.append(SampleForScoring(src=rollout.src_text, mt=clean_mt, ref=rollout.ref_text))
    debug_span_loss = _env_flag("GEMMA27_RL_DEBUG_SPAN_LOSS", default=False)
    debug_span_max_rollouts = _env_int("GEMMA27_RL_DEBUG_SPAN_MAX_ROLLOUTS", default=1, minimum=1)
    debug_span_max_tokens = _env_int("GEMMA27_RL_DEBUG_SPAN_MAX_TOKENS", default=256, minimum=1)
    debug_span_only_nonzero = _env_flag("GEMMA27_RL_DEBUG_SPAN_ONLY_NONZERO", default=False)
    think_tag_token_penalty = -abs(
        _env_float("GEMMA27_RL_THINK_TAG_TOKEN_PENALTY", default=_DEFAULT_THINK_TAG_TOKEN_PENALTY)
    )
    think_tag_seq_penalty = -abs(
        _env_float("GEMMA27_RL_THINK_TAG_SEQ_PENALTY", default=_DEFAULT_THINK_TAG_SEQUENCE_PENALTY)
    )
    repeat_token_penalty = -abs(
        _env_float("GEMMA27_RL_REPEAT_TOKEN_PENALTY", default=_DEFAULT_REPEAT_TOKEN_PENALTY)
    )
    repeat_seq_penalty = -abs(
        _env_float("GEMMA27_RL_REPEAT_SEQ_PENALTY", default=_DEFAULT_REPEAT_SEQUENCE_PENALTY)
    )
    repeat_min_run = _env_int("GEMMA27_RL_REPEAT_MIN_RUN", default=2, minimum=2)
    repeat_max_pattern = _env_int("GEMMA27_RL_REPEAT_MAX_PATTERN", default=4, minimum=1)
    ngram_token_penalty = -abs(
        _env_float("GEMMA27_RL_NGRAM_TOKEN_PENALTY", default=_DEFAULT_NGRAM_TOKEN_PENALTY)
    )
    ngram_seq_penalty = -abs(
        _env_float("GEMMA27_RL_NGRAM_SEQ_PENALTY", default=_DEFAULT_NGRAM_SEQUENCE_PENALTY)
    )
    ngram_repeat_n = _env_int("GEMMA27_RL_NGRAM_REPEAT_N", default=3, minimum=2)
    ngram_min_occurrences = _env_int("GEMMA27_RL_NGRAM_REPEAT_MIN_OCCURS", default=2, minimum=2)
    special_token_penalty = -abs(
        _env_float("GEMMA27_RL_SPECIAL_TOKEN_PENALTY", default=_DEFAULT_SPECIAL_TOKEN_PENALTY)
    )
    special_seq_penalty = -abs(
        _env_float("GEMMA27_RL_SPECIAL_SEQ_PENALTY", default=_DEFAULT_SPECIAL_SEQUENCE_PENALTY)
    )
    debug_span_records: list[tuple[int, Rollout, list[dict[str, Any]], list[float], list[float], float]] = []

    metricx_enabled = cfg.reward.metricx.enabled and metricx_scorer is not None
    xcomet_enabled = cfg.reward.xcomet.enabled and xcomet_scorer is not None
    mqm_enabled = cfg.reward.mqm.enabled and mqm_scorer is not None
    esa_enabled = cfg.reward.esa.enabled and esa_scorer is not None

    metricx_scores = [cfg.reward.metricx.offset for _ in rollouts]
    xcomet_scores = [0.0 for _ in rollouts]
    mqm_scores = [0.0 for _ in rollouts]
    esa_scores = [0.0 for _ in rollouts]
    span_rows = [[] for _ in rollouts]
    mqm_span_rows = [[] for _ in rollouts]

    enabled_scorers = sum(int(flag) for flag in (metricx_enabled, xcomet_enabled, mqm_enabled, esa_enabled))
    if enabled_scorers > 1:
        logger.info(
            "reward scoring: running scorers in parallel for %s rollouts (metricx=%s xcomet=%s mqm=%s esa=%s)",
            len(rollouts),
            metricx_enabled,
            xcomet_enabled,
            mqm_enabled,
            esa_enabled,
        )
        with ThreadPoolExecutor(max_workers=enabled_scorers, thread_name_prefix="reward-scorer") as executor:
            futures: dict[str, Any] = {}
            if metricx_enabled:
                futures["metricx"] = executor.submit(
                    _score_with_cache_metricx,
                    samples=samples,
                    scorer=metricx_scorer,
                    cache=metricx_cache,
                    use_cache=cfg.reward.cache_enabled,
                )
            if xcomet_enabled:
                futures["xcomet"] = executor.submit(
                    _score_with_cache_xcomet,
                    samples=span_reward_samples,
                    scorer=xcomet_scorer,
                    cache=xcomet_cache,
                    use_cache=cfg.reward.cache_enabled,
                )
            if mqm_enabled:
                futures["mqm"] = executor.submit(
                    _score_with_cache_mqm,
                    samples=mqm_esa_samples,
                    scorer=mqm_scorer,
                    cache=mqm_cache,
                    use_cache=cfg.reward.cache_enabled and cfg.reward.mqm.enabled,
                )
            if esa_enabled:
                futures["esa"] = executor.submit(
                    _score_with_cache_esa,
                    samples=mqm_esa_samples,
                    scorer=esa_scorer,
                    cache=esa_cache,
                    use_cache=cfg.reward.cache_enabled and cfg.reward.esa.enabled,
                )
            if "metricx" in futures:
                metricx_scores = futures["metricx"].result()
            if "xcomet" in futures:
                xcomet_scores, span_rows = futures["xcomet"].result()
            if "mqm" in futures:
                mqm_scores, mqm_span_rows = futures["mqm"].result()
            if "esa" in futures:
                esa_scores = futures["esa"].result()
    else:
        if metricx_enabled:
            metricx_scores = _score_with_cache_metricx(
                samples=samples,
                scorer=metricx_scorer,
                cache=metricx_cache,
                use_cache=cfg.reward.cache_enabled,
            )
        if xcomet_enabled:
            xcomet_scores, span_rows = _score_with_cache_xcomet(
                samples=span_reward_samples,
                scorer=xcomet_scorer,
                cache=xcomet_cache,
                use_cache=cfg.reward.cache_enabled,
            )
        if mqm_enabled:
            mqm_scores, mqm_span_rows = _score_with_cache_mqm(
                samples=mqm_esa_samples,
                scorer=mqm_scorer,
                cache=mqm_cache,
                use_cache=cfg.reward.cache_enabled and cfg.reward.mqm.enabled,
            )
        if esa_enabled:
            esa_scores = _score_with_cache_esa(
                samples=mqm_esa_samples,
                scorer=esa_scorer,
                cache=esa_cache,
                use_cache=cfg.reward.cache_enabled and cfg.reward.esa.enabled,
            )

    metricx_scores, metricx_replaced = _sanitize(metricx_scores, fallback=cfg.reward.metricx.offset)
    if metricx_replaced > 0:
        msg = (
            f"MetricX produced {metricx_replaced} non-finite scores "
            f"(model={cfg.reward.metricx.model_name}, device={cfg.reward.metricx.device}, "
            f"dtype={cfg.reward.metricx.dtype})."
        )
        if cfg.reward.metricx.overflow_policy == "skip":
            logger.warning("%s Replacing with fallback offset %.4f due to overflow_policy=skip.", msg, cfg.reward.metricx.offset)
        else:
            raise RuntimeError(
                f"{msg} Training aborted to avoid silent fallback-to-{cfg.reward.metricx.offset:.4f}. "
                "Check MetricX model load/inference and dtype/device settings."
            )

    metricx_rewards = [metricx_score_to_reward(v, offset=cfg.reward.metricx.offset) for v in metricx_scores]

    xcomet_scores, xcomet_replaced = _sanitize(xcomet_scores, fallback=0.0)
    if xcomet_replaced > 0:
        logger.warning(
            "xCOMET produced %s non-finite scores; replaced with fallback 0.0.",
            xcomet_replaced,
        )

    mqm_scores, mqm_replaced = _sanitize(mqm_scores, fallback=0.0)
    if mqm_replaced > 0:
        logger.warning(
            "MQM scorer produced %s non-finite scores; replaced with fallback 0.0.",
            mqm_replaced,
        )
    esa_scores, esa_replaced = _sanitize(esa_scores, fallback=0.0)
    if esa_replaced > 0:
        logger.warning(
            "ESA scorer produced %s non-finite scores; replaced with fallback 0.0.",
            esa_replaced,
        )
    if (
        esa_enabled
        and esa_scores
        and (not _ESA_ALL_ZERO_WARNED)
        and all(abs(float(v)) <= 1e-12 for v in esa_scores)
    ):
        _ESA_ALL_ZERO_WARNED = True
        logger.warning(
            "ESA scores are all 0.0 for this batch (n=%s). "
            "Check reward.esa.score_min/score_max (recommended 0..100), "
            "reward.esa.error_policy (zero can hide API failures), and ESA API logs "
            "(GEMMA27_RL_LOG_ESA_IO=1).",
            len(esa_scores),
        )
    span_rows = [
        [
            *(span_rows[idx] if idx < len(span_rows) else []),
            *(mqm_span_rows[idx] if idx < len(mqm_span_rows) else []),
        ]
        for idx in range(len(rollouts))
    ]

    seq_rewards = build_sequence_rewards(
        metricx_scores=metricx_scores,
        xcomet_scores=xcomet_scores,
        metricx_offset=cfg.reward.metricx.offset,
        w_metricx=cfg.reward.w_metricx,
        w_xcomet_seq=cfg.reward.w_xcomet_seq,
        xcomet_seq_scale=cfg.reward.xcomet_seq_scale,
        mqm_scores=mqm_scores,
        w_mqm_seq=cfg.reward.w_mqm_seq,
        mqm_seq_scale=cfg.reward.mqm_seq_scale,
        esa_scores=esa_scores,
        w_esa_seq=cfg.reward.w_esa_seq,
        esa_seq_scale=cfg.reward.esa_seq_scale,
    )

    token_reward_rows: list[list[float]] = []
    raw_adv_rows: list[list[float]] = []
    severity_counts: dict[str, list[float]] = {"MINOR": [], "MAJOR": [], "CRITICAL": []}
    think_tag_counts: list[float] = []
    think_tag_token_hits: list[float] = []
    think_tag_penalties: list[float] = []
    repeat_token_counts: list[float] = []
    repeat_run_counts: list[float] = []
    repeat_penalties: list[float] = []
    ngram_repeat_token_hits: list[float] = []
    ngram_repeat_occurrences: list[float] = []
    ngram_repeat_penalties: list[float] = []
    special_token_occurrence_counts: list[float] = []
    special_token_hit_counts: list[float] = []
    special_token_penalties: list[float] = []
    special_token_id_occurrence_totals: list[float] = []
    special_token_text_occurrence_totals: list[float] = []
    span_special_masked_counts: list[float] = []
    group_scalar_rewards: list[float] = []
    special_token_id_occurrence_counts: dict[int, int] = {}
    special_token_text_occurrence_counts: dict[str, int] = {}

    for row_idx, (rollout, span_row, seq_reward) in enumerate(zip(rollouts, span_rows, seq_rewards)):
        raw_completion_text = (
            raw_completion_texts[row_idx]
            if row_idx < len(raw_completion_texts)
            else str(rollout.completion_raw_text if rollout.completion_raw_text is not None else rollout.completion_text or "")
        )
        span_reward_text = (
            span_reward_texts[row_idx]
            if row_idx < len(span_reward_texts)
            else str(rollout.completion_clean_text if rollout.completion_clean_text is not None else raw_completion_text)
        )
        token_rewards = spans_to_token_rewards(
            mt_text=span_reward_text,
            token_char_offsets=rollout.token_char_offsets,
            error_spans=span_row,
            severity_weights=cfg.reward.severity_weights,
            overlap_policy=cfg.reward.overlap_policy,
            majority_threshold=cfg.reward.majority_threshold,
            use_confidence=cfg.reward.use_confidence,
            combine_policy=cfg.reward.span_combine_policy,
        )
        span_special_masked = _zero_token_rewards_on_special_token_ids(
            token_rewards=token_rewards,
            completion_token_ids=rollout.completion_token_ids,
            special_token_ids=special_token_ids,
        )
        token_reward_sum_before = float(sum(token_rewards))
        seq_reward_before = float(seq_reward)
        token_rewards, seq_reward, forbidden_tag_count, forbidden_token_hits = _apply_forbidden_think_tag_penalty(
            completion_text=raw_completion_text,
            token_char_offsets=rollout.token_char_offsets,
            token_rewards=token_rewards,
            seq_reward=float(seq_reward),
            token_penalty=think_tag_token_penalty,
            seq_penalty_per_match=think_tag_seq_penalty,
        )
        think_penalty_delta = (float(sum(token_rewards)) - token_reward_sum_before) + (float(seq_reward) - seq_reward_before)
        special_sum_before = float(sum(token_rewards))
        special_seq_before = float(seq_reward)
        rollout_special_id_hits: dict[int, int] = {}
        rollout_special_text_hits: dict[str, int] = {}
        token_rewards, seq_reward, special_occurrences, special_token_hits = _apply_special_token_penalty(
            completion_text=raw_completion_text,
            completion_token_ids=rollout.completion_token_ids,
            penalty_token_ids=rollout.raw_completion_token_ids,
            token_char_offsets=rollout.token_char_offsets,
            token_rewards=token_rewards,
            seq_reward=float(seq_reward),
            special_token_ids=special_token_ids,
            special_token_strings=special_token_penalty_strings,
            token_penalty=special_token_penalty,
            seq_penalty_per_occurrence=special_seq_penalty,
            exempt_final_token_ids=exempt_final_special_ids,
            exempt_final_token_strings=exempt_final_special_strings,
            id_hit_counter=rollout_special_id_hits,
            text_hit_counter=rollout_special_text_hits,
        )
        special_penalty_delta = (float(sum(token_rewards)) - special_sum_before) + (float(seq_reward) - special_seq_before)
        special_id_occurrences = float(sum(int(v) for v in rollout_special_id_hits.values()))
        special_text_occurrences = float(sum(int(v) for v in rollout_special_text_hits.values()))
        if special_occurrences > 0 and (rollout_special_id_hits or rollout_special_text_hits):
            for tok_id, count in rollout_special_id_hits.items():
                special_token_id_occurrence_counts[tok_id] = int(special_token_id_occurrence_counts.get(tok_id, 0)) + int(count)
            for token_text, count in rollout_special_text_hits.items():
                special_token_text_occurrence_counts[token_text] = (
                    int(special_token_text_occurrence_counts.get(token_text, 0)) + int(count)
                )
            logger.info(
                "special token penalty detail: example_id=%s id_hits=[%s] text_hits=[%s]",
                rollout.example_id,
                _format_top_special_id_counts(
                    rollout_special_id_hits,
                    id_label_map=special_token_id_labels,
                    limit=8,
                ),
                _format_top_special_text_counts(rollout_special_text_hits, limit=8),
            )
        repeat_sum_before = float(sum(token_rewards))
        repeat_seq_before = float(seq_reward)
        token_rewards, seq_reward, repeat_token_count, repeat_run_count = _apply_repeated_token_penalty(
            completion_token_ids=rollout.completion_token_ids,
            token_rewards=token_rewards,
            seq_reward=float(seq_reward),
            token_penalty=repeat_token_penalty,
            seq_penalty_per_repeat=repeat_seq_penalty,
            min_repeat_run_length=repeat_min_run,
            max_repeat_pattern_length=repeat_max_pattern,
        )
        repeat_penalty_delta = (float(sum(token_rewards)) - repeat_sum_before) + (float(seq_reward) - repeat_seq_before)
        ngram_sum_before = float(sum(token_rewards))
        ngram_seq_before = float(seq_reward)
        token_rewards, seq_reward, ngram_token_hit_count, ngram_repeat_count = _apply_ngram_repeat_penalty(
            completion_token_ids=rollout.completion_token_ids,
            token_rewards=token_rewards,
            seq_reward=float(seq_reward),
            token_penalty=ngram_token_penalty,
            seq_penalty_per_repeat=ngram_seq_penalty,
            ngram_size=ngram_repeat_n,
            min_occurrences=ngram_min_occurrences,
        )
        ngram_penalty_delta = (float(sum(token_rewards)) - ngram_sum_before) + (float(seq_reward) - ngram_seq_before)
        group_scalar_rewards.append(float(seq_reward))
        seq_row = broadcast_sequence_reward(seq_reward, token_count=len(token_rewards))
        raw_adv = combine_advantages(seq_row, token_rewards)

        token_reward_rows.append(token_rewards)
        raw_adv_rows.append(raw_adv)
        think_tag_counts.append(float(forbidden_tag_count))
        think_tag_token_hits.append(float(forbidden_token_hits))
        think_tag_penalties.append(float(think_penalty_delta))
        repeat_token_counts.append(float(repeat_token_count))
        repeat_run_counts.append(float(repeat_run_count))
        repeat_penalties.append(float(repeat_penalty_delta))
        ngram_repeat_token_hits.append(float(ngram_token_hit_count))
        ngram_repeat_occurrences.append(float(ngram_repeat_count))
        ngram_repeat_penalties.append(float(ngram_penalty_delta))
        special_token_occurrence_counts.append(float(special_occurrences))
        special_token_hit_counts.append(float(special_token_hits))
        special_token_penalties.append(float(special_penalty_delta))
        special_token_id_occurrence_totals.append(float(special_id_occurrences))
        special_token_text_occurrence_totals.append(float(special_text_occurrences))
        span_special_masked_counts.append(float(span_special_masked))
        if debug_span_loss and len(debug_span_records) < debug_span_max_rollouts:
            debug_span_records.append(
                (
                    len(raw_adv_rows) - 1,
                    rollout,
                    span_row,
                    list(token_rewards),
                    list(raw_adv),
                    float(seq_reward),
                )
            )

        span_counter = {"MINOR": 0, "MAJOR": 0, "CRITICAL": 0}
        for span in span_row:
            sev = str(span.get("severity", "")).upper()
            if sev in span_counter:
                span_counter[sev] += 1
        for key in severity_counts:
            severity_counts[key].append(float(span_counter[key]))

    total_forbidden_tags = int(sum(think_tag_counts))
    if total_forbidden_tags > 0:
        logger.info(
            "forbidden think tag penalty applied: tags=%s token_hits=%s token_penalty=%.1f seq_penalty=%.1f",
            total_forbidden_tags,
            int(sum(think_tag_token_hits)),
            float(think_tag_token_penalty),
            float(think_tag_seq_penalty),
        )
    total_repeat_tokens = int(sum(repeat_token_counts))
    if total_repeat_tokens > 0:
        logger.info(
            "repeat token penalty applied: repeat_tokens=%s repeat_runs=%s min_run=%s max_pattern=%s token_penalty=%.2f seq_penalty=%.2f",
            total_repeat_tokens,
            int(sum(repeat_run_counts)),
            int(repeat_min_run),
            int(repeat_max_pattern),
            float(repeat_token_penalty),
            float(repeat_seq_penalty),
        )
    total_ngram_occurrences = int(sum(ngram_repeat_occurrences))
    if total_ngram_occurrences > 0:
        logger.info(
            "n-gram repeat penalty applied: n=%s min_occurs=%s repeats=%s token_hits=%s token_penalty=%.2f seq_penalty=%.2f",
            int(ngram_repeat_n),
            int(ngram_min_occurrences),
            total_ngram_occurrences,
            int(sum(ngram_repeat_token_hits)),
            float(ngram_token_penalty),
            float(ngram_seq_penalty),
        )
    total_special_occurrences = int(sum(special_token_occurrence_counts))
    if total_special_occurrences > 0:
        logger.info(
            "special token penalty applied: occurrences=%s id_occurrences=%s text_occurrences=%s token_hits=%s token_penalty=%.1f seq_penalty=%.1f ids=%s strings=%s top_id_hits=[%s] top_text_hits=[%s]",
            total_special_occurrences,
            int(sum(special_token_id_occurrence_totals)),
            int(sum(special_token_text_occurrence_totals)),
            int(sum(special_token_hit_counts)),
            float(special_token_penalty),
            float(special_seq_penalty),
            len(special_token_ids),
            len(special_token_penalty_strings),
            _format_top_special_id_counts(
                special_token_id_occurrence_counts,
                id_label_map=special_token_id_labels,
                limit=12,
            ),
            _format_top_special_text_counts(special_token_text_occurrence_counts, limit=12),
        )
    if sanitized_target_rows > 0:
        logger.info(
            "MQM/ESA target sanitize applied: rows=%s/%s marker_replacements=%s tokenizer_special_tokens=%s",
            int(sanitized_target_rows),
            len(rollouts),
            int(sanitized_marker_total),
            len(special_token_strings),
        )
    total_span_special_masked = int(sum(span_special_masked_counts))
    if total_span_special_masked > 0:
        logger.info(
            "span reward special-token mask applied: masked_token_rewards=%s",
            total_span_special_masked,
        )

    if cfg.rl.group_normalize:
        group_ids = _resolve_group_ids_for_rollouts(
            rollouts,
            num_samples_per_prompt=int(getattr(cfg.generation, "num_samples_per_prompt", 1) or 1),
        )
        raw_adv_rows, _ = apply_group_relative_advantage(
            raw_advantages=raw_adv_rows,
            group_ids=group_ids,
            rollout_scalars=group_scalar_rewards,
            coef=cfg.rl.group_advantage_coef,
            eps=cfg.rl.eps,
        )

    if cfg.rl.normalize_advantage:
        norm_adv_rows, norm_stats = normalize_advantages(raw_adv_rows, eps=cfg.rl.eps)
    else:
        norm_adv_rows = raw_adv_rows
        flat_raw = _flatten(raw_adv_rows)
        raw_m, raw_s = _mean_std(flat_raw)
        norm_stats = {
            "raw_mean": raw_m,
            "raw_std": raw_s,
            "norm_mean": raw_m,
            "norm_std": raw_s,
        }

    if debug_span_loss and debug_span_records:
        for row_idx, rollout, span_row, token_rewards, raw_adv_pre, seq_reward in debug_span_records:
            adv_used = norm_adv_rows[row_idx] if row_idx < len(norm_adv_rows) else []
            _log_span_debug_for_rollout(
                rollout=rollout,
                span_row=span_row,
                token_rewards=token_rewards,
                raw_adv=raw_adv_pre,
                adv_used=adv_used,
                seq_reward=seq_reward,
                overlap_policy=cfg.reward.overlap_policy,
                majority_threshold=cfg.reward.majority_threshold,
                max_tokens=debug_span_max_tokens,
                only_nonzero=debug_span_only_nonzero,
            )

    flat_token_rewards = _flatten(token_reward_rows)
    token_reward_m, token_reward_s = _mean_std(flat_token_rewards)
    non_zero_token_ratio = (
        sum(1 for v in flat_token_rewards if abs(v) > 0) / max(1, len(flat_token_rewards))
        if flat_token_rewards
        else 0.0
    )

    metricx_m, metricx_s = _mean_std(metricx_scores)
    metricx_r_m, metricx_r_s = _mean_std(metricx_rewards)
    xcomet_m, xcomet_s = _mean_std(xcomet_scores)
    mqm_m, mqm_s = _mean_std(mqm_scores)
    esa_m, esa_s = _mean_std(esa_scores)

    reward_stats = {
        "metricx_score_mean": metricx_m,
        "metricx_score_std": metricx_s,
        "metricx_reward_mean": metricx_r_m,
        "metricx_reward_std": metricx_r_s,
        "xcomet_score_mean": xcomet_m,
        "xcomet_score_std": xcomet_s,
        "mqm_score_mean": mqm_m,
        "mqm_score_std": mqm_s,
        "esa_score_mean": esa_m,
        "esa_score_std": esa_s,
        "token_rewards_mean": token_reward_m,
        "token_rewards_std": token_reward_s,
        "token_rewards_non_zero_ratio": float(non_zero_token_ratio),
        "span_minor_mean": float(mean(severity_counts["MINOR"]) if severity_counts["MINOR"] else 0.0),
        "span_major_mean": float(mean(severity_counts["MAJOR"]) if severity_counts["MAJOR"] else 0.0),
        "span_critical_mean": float(mean(severity_counts["CRITICAL"]) if severity_counts["CRITICAL"] else 0.0),
        "forbidden_think_tag_count_mean": float(mean(think_tag_counts) if think_tag_counts else 0.0),
        "forbidden_think_tag_count_total": float(sum(think_tag_counts)),
        "forbidden_think_tag_token_hits_mean": float(mean(think_tag_token_hits) if think_tag_token_hits else 0.0),
        "forbidden_think_penalty_mean": float(mean(think_tag_penalties) if think_tag_penalties else 0.0),
        "forbidden_think_penalty_total": float(sum(think_tag_penalties)),
        "repeat_token_count_mean": float(mean(repeat_token_counts) if repeat_token_counts else 0.0),
        "repeat_token_count_total": float(sum(repeat_token_counts)),
        "repeat_run_count_mean": float(mean(repeat_run_counts) if repeat_run_counts else 0.0),
        "repeat_penalty_mean": float(mean(repeat_penalties) if repeat_penalties else 0.0),
        "repeat_penalty_total": float(sum(repeat_penalties)),
        "ngram_repeat_token_hit_mean": float(mean(ngram_repeat_token_hits) if ngram_repeat_token_hits else 0.0),
        "ngram_repeat_token_hit_total": float(sum(ngram_repeat_token_hits)),
        "ngram_repeat_occurrence_mean": float(mean(ngram_repeat_occurrences) if ngram_repeat_occurrences else 0.0),
        "ngram_repeat_occurrence_total": float(sum(ngram_repeat_occurrences)),
        "ngram_repeat_penalty_mean": float(mean(ngram_repeat_penalties) if ngram_repeat_penalties else 0.0),
        "ngram_repeat_penalty_total": float(sum(ngram_repeat_penalties)),
        "special_token_occurrence_count_mean": float(
            mean(special_token_occurrence_counts) if special_token_occurrence_counts else 0.0
        ),
        "special_token_occurrence_count_total": float(sum(special_token_occurrence_counts)),
        "special_token_hit_count_mean": float(mean(special_token_hit_counts) if special_token_hit_counts else 0.0),
        "special_token_penalty_mean": float(mean(special_token_penalties) if special_token_penalties else 0.0),
        "special_token_penalty_total": float(sum(special_token_penalties)),
        "special_token_id_occurrence_count_mean": float(
            mean(special_token_id_occurrence_totals) if special_token_id_occurrence_totals else 0.0
        ),
        "special_token_id_occurrence_count_total": float(sum(special_token_id_occurrence_totals)),
        "special_token_text_occurrence_count_mean": float(
            mean(special_token_text_occurrence_totals) if special_token_text_occurrence_totals else 0.0
        ),
        "special_token_text_occurrence_count_total": float(sum(special_token_text_occurrence_totals)),
        "span_special_masked_token_count_mean": float(
            mean(span_special_masked_counts) if span_special_masked_counts else 0.0
        ),
        "span_special_masked_token_count_total": float(sum(span_special_masked_counts)),
        "judge_sanitized_target_count": float(sanitized_target_rows),
        "judge_sanitized_marker_total": float(sanitized_marker_total),
    }

    return norm_adv_rows, reward_stats, norm_stats


def _fill_missing_reference_logprobs(
    *,
    merged_rollouts: list[Rollout],
    cfg: RLPostTrainConfig,
    update_idx: int,
    ref_logprob_batch_fn: Callable[[list[tuple[list[int], list[int]]]], list[list[float]]] | None,
    ref_logprob_client: ReferenceLogprobClient | None,
    ref_model: Any | None,
    ref_device: str | None,
    device: str,
) -> int:
    if (not merged_rollouts) or (not _reference_kl_enabled(cfg)):
        return 0

    missing_idx = [i for i, rollout in enumerate(merged_rollouts) if rollout.ref_logprobs is None]
    if missing_idx:
        empty_completion_missing = 0
        for idx in list(missing_idx):
            if merged_rollouts[idx].completion_token_ids:
                continue
            merged_rollouts[idx].ref_logprobs = []
            empty_completion_missing += 1
        if empty_completion_missing > 0:
            missing_idx = [i for i in missing_idx if merged_rollouts[i].ref_logprobs is None]
            logger.info(
                "Skipping reference logprob fill for %s empty-completion rollouts at update=%s.",
                empty_completion_missing,
                update_idx,
            )
    if not missing_idx:
        return 0

    if ref_logprob_batch_fn is not None or ref_logprob_client is not None:
        requests: list[tuple[int, tuple[list[int], list[int]]]] = [
            (
                i,
                (
                    merged_rollouts[i].prompt_input_ids,
                    merged_rollouts[i].completion_token_ids,
                ),
            )
            for i in missing_idx
        ]
        source_label = "Reference colocated score_batch" if ref_logprob_batch_fn is not None else "Reference worker score_batch"
        responses_by_idx, batch_calls, batch_chunk = _score_reference_requests_with_batch_fn(
            requests=requests,
            ref_logprob_batch_fn=(
                ref_logprob_batch_fn
                if ref_logprob_batch_fn is not None
                else (lambda items: ref_logprob_client.score_logprobs_batch(items))  # type: ignore[union-attr]
            ),
            update_idx=update_idx,
            source_label=source_label,
        )

        filled = 0
        for idx in missing_idx:
            row = responses_by_idx.get(idx)
            if row is None:
                continue
            merged_rollouts[idx].ref_logprobs = row
            filled += 1

        logger.info(
            "Filled reference logprobs on rank0 for %s/%s gathered rollouts at update=%s "
            "(source=%s batch_calls=%s chunk=%s).",
            filled,
            len(missing_idx),
            update_idx,
            "colocate" if ref_logprob_batch_fn is not None else "worker",
            batch_calls,
            batch_chunk,
        )
        return filled

    if ref_model is not None:
        for idx in missing_idx:
            rollout = merged_rollouts[idx]
            rollout.ref_logprobs = compute_completion_logprobs(
                ref_model,
                rollout.prompt_input_ids,
                rollout.completion_token_ids,
                device=ref_device or device,
            ).tolist()
        logger.info(
            "Filled reference logprobs on rank0 for %s gathered rollouts at update=%s.",
            len(missing_idx),
            update_idx,
        )
        return len(missing_idx)

    return 0


def _save_checkpoint_to_dir(
    ckpt_dir: Path,
    model: AutoModelForCausalLM,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    trainer_state: dict[str, Any] | None = None,
) -> Path:
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    if trainer_state:
        _save_trainer_state(ckpt_dir, trainer_state)
    return ckpt_dir


def _build_zero3_peft_state_dict(model: Any) -> dict[str, Any] | None:
    if PeftModel is None or get_peft_model_state_dict is None:
        return None
    if not isinstance(model, PeftModel):
        return None
    if deepspeed is None or _distributed_world_size() <= 1:
        return None
    zero = getattr(deepspeed, "zero", None)
    gathered_parameters = getattr(zero, "GatheredParameters", None)
    if gathered_parameters is None:
        return None

    params_to_gather = [param for param in model.parameters() if param.requires_grad]
    if not params_to_gather:
        return None

    with gathered_parameters(params_to_gather, modifier_rank=0):
        if not _is_rank0():
            return None
        state_dict = get_peft_model_state_dict(model, state_dict=model.state_dict())
        out: dict[str, Any] = {}
        for key, value in state_dict.items():
            if torch.is_tensor(value):
                out[key] = value.detach().cpu().clone()
            else:
                out[key] = value
        return out


def _save_pretrained_model(model: Any, output_dir: Path) -> None:
    zero3_state_dict = _build_zero3_peft_state_dict(model)
    _dist_barrier()
    if _is_rank0():
        if zero3_state_dict is None:
            model.save_pretrained(output_dir)
        else:
            model.save_pretrained(output_dir, state_dict=zero3_state_dict)
    _dist_barrier()


def _save_deepspeed_checkpoint_to_dir(
    ckpt_dir: Path,
    engine: Any,
    tokenizer,
    hf_model: AutoModelForCausalLM | None = None,
    trainer_state: dict[str, Any] | None = None,
) -> Path:
    if _is_rank0():
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    _dist_barrier()
    engine.save_checkpoint(str(ckpt_dir), tag="state")
    _dist_barrier()
    if hf_model is not None:
        _save_pretrained_model(hf_model, ckpt_dir)
    if _is_rank0():
        tokenizer.save_pretrained(ckpt_dir)
        if trainer_state:
            _save_trainer_state(ckpt_dir, trainer_state)
    _dist_barrier()
    return ckpt_dir


def _save_checkpoint(
    output_dir: Path,
    update_idx: int,
    model: AutoModelForCausalLM,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    trainer_state: dict[str, Any] | None = None,
) -> Path:
    ckpt_dir = output_dir / f"checkpoint-{update_idx}"
    return _save_checkpoint_to_dir(
        ckpt_dir=ckpt_dir,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        trainer_state=trainer_state,
    )


def _save_deepspeed_checkpoint(
    output_dir: Path,
    update_idx: int,
    engine: Any,
    tokenizer,
    hf_model: AutoModelForCausalLM | None = None,
    trainer_state: dict[str, Any] | None = None,
) -> Path:
    ckpt_dir = output_dir / f"checkpoint-{update_idx}"
    return _save_deepspeed_checkpoint_to_dir(
        ckpt_dir=ckpt_dir,
        engine=engine,
        tokenizer=tokenizer,
        hf_model=hf_model,
        trainer_state=trainer_state,
    )


def _save_resume_checkpoint(
    output_dir: Path,
    update_idx: int,
    model: AutoModelForCausalLM,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    trainer_state: dict[str, Any] | None = None,
) -> Path:
    ckpt_dir = output_dir / "resume_latest"
    state = dict(trainer_state or {})
    state["update_idx"] = int(update_idx)
    return _save_checkpoint_to_dir(
        ckpt_dir=ckpt_dir,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        trainer_state=state,
    )


def _save_deepspeed_resume_checkpoint(
    output_dir: Path,
    update_idx: int,
    engine: Any,
    tokenizer,
    hf_model: AutoModelForCausalLM | None = None,
    trainer_state: dict[str, Any] | None = None,
) -> Path:
    ckpt_dir = output_dir / "resume_latest"
    state = dict(trainer_state or {})
    state["update_idx"] = int(update_idx)
    return _save_deepspeed_checkpoint_to_dir(
        ckpt_dir=ckpt_dir,
        engine=engine,
        tokenizer=tokenizer,
        hf_model=hf_model,
        trainer_state=state,
    )


def _is_deepspeed_checkpoint_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    if (path / "latest").exists():
        return True
    for pattern in ("**/*_model_states.pt", "**/*_optim_states.pt"):
        if next(path.glob(pattern), None) is not None:
            return True
    return False


def _compute_eval_selection_score(report: dict[str, Any], cfg: RLPostTrainConfig) -> float:
    metricx_term = float(report.get("metricx_reward_mean", 0.0)) * float(cfg.reward.w_metricx)
    xcomet_term = (
        float(report.get("xcomet_score_mean", 0.0))
        * float(cfg.reward.w_xcomet_seq)
        * float(cfg.reward.xcomet_seq_scale)
    )
    mqm_term = (
        float(report.get("mqm_score_mean", 0.0))
        * float(cfg.reward.w_mqm_seq)
        * float(cfg.reward.mqm_seq_scale)
    )
    esa_term = (
        float(report.get("esa_score_mean", 0.0))
        * float(cfg.reward.w_esa_seq)
        * float(cfg.reward.esa_seq_scale)
    )
    return float(metricx_term + xcomet_term + mqm_term + esa_term)


def _should_enable_xcomet_runtime(cfg: RLPostTrainConfig) -> bool:
    if not bool(cfg.reward.xcomet.enabled):
        return False
    effective_weight = float(cfg.reward.w_xcomet_seq) * float(cfg.reward.xcomet_seq_scale)
    if abs(effective_weight) <= 0.0:
        logger.info(
            "xCOMET scorer disabled at runtime because effective xcomet weight is zero "
            "(w_xcomet_seq=%.6f xcomet_seq_scale=%.6f).",
            float(cfg.reward.w_xcomet_seq),
            float(cfg.reward.xcomet_seq_scale),
        )
        return False
    return True


def run_metric_only_eval(cfg: RLPostTrainConfig) -> dict[str, Any]:
    set_seed(cfg.misc.seed)
    hf_token = resolve_huggingface_token(
        explicit_token=cfg.misc.huggingface_token,
        token_env_name=cfg.misc.huggingface_token_env,
    )
    configure_huggingface_cache(cfg.misc.huggingface_cache_dir, token=hf_token)
    _apply_aux_worker_defaults(cfg)
    _assign_disjoint_gpu_devices(cfg)
    device = resolve_device(cfg.misc.device)

    tokenizer_name = cfg.model.tokenizer_name_or_path or cfg.model.policy_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=cfg.model.use_fast_tokenizer)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_policy_model(cfg, device=device)

    eval_limit = cfg.eval.eval_limit if cfg.eval.eval_limit is not None else cfg.data.eval_limit
    eval_examples = load_examples(cfg.data, split="eval", limit=eval_limit)
    if eval_limit is not None and len(eval_examples) > int(eval_limit):
        eval_examples = eval_examples[: int(eval_limit)]
    logger.info(
        "Prepared eval examples (metric-only): requested_limit=%s loaded=%s eval_file=%s hf_eval_split=%s",
        eval_limit,
        len(eval_examples),
        cfg.data.eval_file,
        cfg.data.hf_eval_split,
    )
    if (
        cfg.data.hf_dataset_name
        and not cfg.data.eval_file
        and (cfg.data.hf_eval_split or cfg.data.hf_train_split) == cfg.data.hf_train_split
    ):
        logger.warning(
            "Eval is currently read from the same HF split as train (%s). "
            "For SFT eval-set selection, set data.eval_file (recommended) or data.hf_eval_split to a distinct split.",
            cfg.data.hf_train_split,
        )

    if not (cfg.reward.mqm.source_lang or "").strip():
        cfg.reward.mqm.source_lang = cfg.data.default_src_lang
    if not (cfg.reward.mqm.target_lang or "").strip():
        cfg.reward.mqm.target_lang = cfg.data.default_tgt_lang
    if not (cfg.reward.esa.source_lang or "").strip():
        cfg.reward.esa.source_lang = cfg.data.default_src_lang
    if not (cfg.reward.esa.target_lang or "").strip():
        cfg.reward.esa.target_lang = cfg.data.default_tgt_lang

    metricx_scorer = MetricXQEScorer(cfg.reward.metricx) if cfg.reward.metricx.enabled else None
    xcomet_runtime_enabled = _should_enable_xcomet_runtime(cfg)
    xcomet_scorer = XCometXLScorer(cfg.reward.xcomet) if xcomet_runtime_enabled else None
    mqm_scorer = OpenAICompatibleMQMScorer(cfg.reward.mqm) if cfg.reward.mqm.enabled else None
    esa_scorer = OpenAICompatibleESAScorer(cfg.reward.esa) if cfg.reward.esa.enabled else None

    report = evaluate_on_dataset(
        examples=eval_examples,
        policy_model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        device=device,
        metricx_scorer=metricx_scorer,
        xcomet_scorer=xcomet_scorer,
        mqm_scorer=mqm_scorer,
        esa_scorer=esa_scorer,
        show_progress=True,
    )
    logger.info("Eval report: %s", report)
    return report


def run_toy_rl(cfg: RLPostTrainConfig) -> dict[str, Any]:
    set_seed(cfg.misc.seed)
    _configure_nccl_heartbeat_timeout(cfg)
    _configure_cuda_allocator()
    hf_token = resolve_huggingface_token(
        explicit_token=cfg.misc.huggingface_token,
        token_env_name=cfg.misc.huggingface_token_env,
    )
    configure_huggingface_cache(cfg.misc.huggingface_cache_dir, token=hf_token)
    _apply_aux_worker_defaults(cfg)
    _assign_disjoint_gpu_devices(cfg)
    use_deepspeed = cfg.rl.backend == "deepspeed"
    if use_deepspeed and _distributed_world_size() > 1 and not _is_distributed_initialized():
        if deepspeed is None:
            raise RuntimeError(
                "rl.backend=deepspeed but deepspeed is not installed. Install it first."
            )
        deepspeed.init_distributed()

    base_device = resolve_device(cfg.misc.device)
    if use_deepspeed:
        base_device = _local_rank_device(base_device)
        _validate_deepspeed_partition_strict(cfg)
    device = base_device
    rank = _distributed_rank()
    rank0 = _is_rank0()
    world_size = _distributed_world_size()

    output_dir = Path(cfg.logging.output_dir)
    if rank0:
        output_dir.mkdir(parents=True, exist_ok=True)
        dump_config(cfg, output_dir / "resolved_config.yaml")
    _dist_barrier()

    log_path = output_dir / cfg.logging.jsonl_name
    rollout_log_path = output_dir / cfg.logging.rollout_jsonl_name
    eval_output_path = output_dir / cfg.logging.eval_output_jsonl_name

    resume_ckpt, resume_update_idx = _resolve_resume_checkpoint(cfg, output_dir)
    resume_state = _load_trainer_state(resume_ckpt) if resume_ckpt is not None else None
    if resume_state and "update_idx" in resume_state:
        try:
            resume_update_idx = int(resume_state["update_idx"])
        except (TypeError, ValueError):
            pass
    is_resuming = resume_ckpt is not None
    start_update = resume_update_idx + 1 if is_resuming else 1

    if rank0:
        if log_path.exists() and not is_resuming:
            log_path.unlink()
        if cfg.logging.save_rollouts and rollout_log_path.exists() and not is_resuming:
            rollout_log_path.unlink()
        if cfg.logging.save_eval_outputs and eval_output_path.exists() and not is_resuming:
            eval_output_path.unlink()
        if is_resuming:
            _truncate_jsonl_by_update(log_path, resume_update_idx)
            if cfg.logging.save_rollouts:
                _truncate_jsonl_by_update(rollout_log_path, resume_update_idx)
            if cfg.logging.save_eval_outputs:
                _truncate_jsonl_by_update(eval_output_path, resume_update_idx)
    _dist_barrier()

    tokenizer_name = cfg.model.tokenizer_name_or_path or cfg.model.policy_name_or_path
    tokenizer_source = (
        str(resume_ckpt)
        if resume_ckpt is not None and (resume_ckpt / "tokenizer_config.json").exists()
        else tokenizer_name
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=cfg.model.use_fast_tokenizer)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    resume_has_hf_artifacts = _checkpoint_has_model_artifacts(resume_ckpt)
    if use_deepspeed:
        # Prefer resume dir as model init when it includes HF artifacts (e.g., `best`).
        # Optimizer/lr states are still restored through DeepSpeed shards when present.
        policy_source = str(resume_ckpt) if resume_has_hf_artifacts else cfg.model.policy_name_or_path
    else:
        policy_source = str(resume_ckpt) if resume_ckpt is not None else cfg.model.policy_name_or_path
    policy_model = _load_policy_model(cfg, device=device, model_name_or_path=policy_source)
    _configure_policy_train_memory(policy_model)
    train_model: Any = policy_model
    optimizer: torch.optim.Optimizer | None = None
    if use_deepspeed:
        train_model, _ = _deepspeed_initialize(cfg, policy_model)
    policy_runtime_mode = str(cfg.model.policy_runtime_mode or "colocate").strip().lower()
    if policy_runtime_mode == "separate":
        if use_deepspeed:
            raise ValueError(
                "model.policy_runtime_mode=separate is not supported with rl.backend=deepspeed. "
                "Use model.policy_runtime_mode=colocate."
            )
        # Native-only: keep a dedicated rollout/eval policy copy detached from optimizer/backward.
        policy_eval_model = _load_policy_model(cfg, device=device, model_name_or_path=policy_source)
    else:
        policy_eval_model = _unwrap_for_generation(train_model)

    if rank0:
        configured_attn = str(cfg.model.attn_implementation or "auto").strip() or "auto"
        logger.info(
            "Policy attention uses model.attn_implementation=%s (disable_policy_flash_attention=%s).",
            configured_attn,
            bool(cfg.model.disable_policy_flash_attention),
        )
        logger.info(
            "Policy runtime mode=%s (colocate means single policy instance is shared for rollout/eval/update).",
            policy_runtime_mode,
        )

    ref_model: AutoModelForCausalLM | None = None
    ref_device: str | None = None
    ref_logprob_client: ReferenceLogprobClient | None = None
    ref_logprob_batch_fn: Callable[[list[tuple[list[int], list[int]]]], list[list[float]]] | None = None
    reference_kl_enabled = _reference_kl_enabled(cfg)
    reference_runtime = str(cfg.model.reference_runtime or "worker").strip().lower()
    if rank0 and (float(cfg.rl.kl_coef) > 0.0) and (not reference_kl_enabled):
        logger.info(
            "Reference model is disabled (model.use_reference_model=false); KL-to-reference term will be skipped."
        )
    if reference_kl_enabled and reference_runtime == "colocate":
        ref_logprob_batch_fn, ref_device = _create_colocated_reference_logprob_batch_fn(
            cfg,
            policy_eval_model,
            device=device,
        )
    elif (not use_deepspeed) or rank0:
        if reference_kl_enabled:
            if reference_runtime == "worker":
                ref_logprob_client, ref_device = _create_reference_logprob_client(cfg, default_device=device)
            elif reference_runtime == "cpu":
                ref_model, ref_device = _load_reference_model(cfg, default_device="cpu")
            else:
                ref_model, ref_device = _load_reference_model(cfg, default_device=device)

    train_examples = load_examples(cfg.data, split="train", limit=cfg.data.limit)
    if not train_examples:
        raise ValueError(
            "No train examples loaded. "
            f"Check data fields and filters: "
            f"src_text_field={cfg.data.src_text_field!r}, "
            f"id_field={cfg.data.id_field!r}, "
            f"skip_bad_source={cfg.data.skip_bad_source}, "
            f"is_bad_source_field={cfg.data.is_bad_source_field!r}, "
            f"limit={cfg.data.limit}, "
            f"hf_dataset_name={cfg.data.hf_dataset_name!r}, "
            f"hf_train_split={cfg.data.hf_train_split!r}, "
            f"train_file={cfg.data.train_file!r}."
        )

    eval_limit = cfg.eval.eval_limit if cfg.eval.eval_limit is not None else cfg.data.eval_limit
    eval_examples = load_examples(cfg.data, split="eval", limit=eval_limit)
    if eval_limit is not None and len(eval_examples) > int(eval_limit):
        eval_examples = eval_examples[: int(eval_limit)]
    logger.info(
        "Prepared eval examples: requested_limit=%s loaded=%s eval_file=%s hf_eval_split=%s",
        eval_limit,
        len(eval_examples),
        cfg.data.eval_file,
        cfg.data.hf_eval_split,
    )
    if (
        cfg.data.hf_dataset_name
        and not cfg.data.eval_file
        and (cfg.data.hf_eval_split or cfg.data.hf_train_split) == cfg.data.hf_train_split
    ):
        logger.warning(
            "Eval is currently read from the same HF split as train (%s). "
            "For SFT eval-set selection, set data.eval_file (recommended) or data.hf_eval_split to a distinct split.",
            cfg.data.hf_train_split,
        )

    if not (cfg.reward.mqm.source_lang or "").strip():
        cfg.reward.mqm.source_lang = cfg.data.default_src_lang
    if not (cfg.reward.mqm.target_lang or "").strip():
        cfg.reward.mqm.target_lang = cfg.data.default_tgt_lang
    if not (cfg.reward.esa.source_lang or "").strip():
        cfg.reward.esa.source_lang = cfg.data.default_src_lang
    if not (cfg.reward.esa.target_lang or "").strip():
        cfg.reward.esa.target_lang = cfg.data.default_tgt_lang

    metricx_scorer = (
        MetricXQEScorer(cfg.reward.metricx) if cfg.reward.metricx.enabled and ((not use_deepspeed) or rank0) else None
    )
    xcomet_runtime_enabled = _should_enable_xcomet_runtime(cfg)
    xcomet_scorer = (
        XCometXLScorer(cfg.reward.xcomet) if xcomet_runtime_enabled and ((not use_deepspeed) or rank0) else None
    )
    mqm_scorer = (
        OpenAICompatibleMQMScorer(cfg.reward.mqm)
        if cfg.reward.mqm.enabled and ((not use_deepspeed) or rank0)
        else None
    )
    esa_scorer = (
        OpenAICompatibleESAScorer(cfg.reward.esa)
        if cfg.reward.esa.enabled and ((not use_deepspeed) or rank0)
        else None
    )
    if rank0:
        effective_esa_weight = float(cfg.reward.w_esa_seq) * float(cfg.reward.esa_seq_scale)
        logger.info(
            "reward config: esa_enabled=%s esa_runtime=%s w_esa_seq=%.6f esa_seq_scale=%.6f "
            "effective_esa_weight=%.6f score_range=[%.3f, %.3f] scale_to_unit_interval=%s error_policy=%s",
            bool(cfg.reward.esa.enabled),
            bool(esa_scorer is not None),
            float(cfg.reward.w_esa_seq),
            float(cfg.reward.esa_seq_scale),
            effective_esa_weight,
            float(cfg.reward.esa.score_min),
            float(cfg.reward.esa.score_max),
            bool(cfg.reward.esa.scale_to_unit_interval),
            str(cfg.reward.esa.error_policy),
        )
        if cfg.reward.esa.enabled and abs(effective_esa_weight) <= 0.0:
            logger.warning(
                "ESA is enabled but effective sequence weight is 0.0 "
                "(w_esa_seq * esa_seq_scale == 0). ESA score will not affect loss."
            )
        if cfg.reward.esa.enabled and float(cfg.reward.esa.score_max) <= 0.0:
            logger.warning(
                "reward.esa.score_max <= 0.0. GEMBA-ESA usually returns 0..100, "
                "so outputs may clip to 0.0. Recommended range is score_min=0, score_max=100."
            )
        if cfg.reward.esa.enabled and str(cfg.reward.esa.error_policy).strip().lower() == "zero":
            logger.warning(
                "reward.esa.error_policy=zero: ESA API/parsing failures are converted to 0.0."
            )

    if not use_deepspeed:
        optimizer = torch.optim.AdamW(
            [p for p in policy_model.parameters() if p.requires_grad],
            lr=cfg.rl.lr,
            weight_decay=cfg.rl.weight_decay,
        )
        if is_resuming:
            optimizer_path = resume_ckpt / "optimizer.pt"
            if optimizer_path.exists():
                optimizer_state = torch.load(optimizer_path, map_location="cpu")
                optimizer.load_state_dict(optimizer_state)
            else:
                logger.warning("Resume checkpoint has no optimizer.pt: %s", resume_ckpt)
    elif is_resuming and resume_ckpt is not None:
        if _is_deepspeed_checkpoint_dir(resume_ckpt):
            ds_resume_failed = False
            try:
                load_path, _ = train_model.load_checkpoint(
                    str(resume_ckpt),
                    tag="state",
                    load_optimizer_states=True,
                    load_lr_scheduler_states=False,
                )
                if load_path is None and rank0:
                    logger.warning("DeepSpeed resume checkpoint could not be loaded from %s", resume_ckpt)
                    ds_resume_failed = True
            except AssertionError as exc:
                # DeepSpeed can assert on empty ckpt shard list when shard files are
                # missing for current rank/world-size. In that case, keep model
                # weights already loaded from `policy_source` and reset optimizer.
                text = str(exc)
                if ("ckpt_list" in text) or ("len(self.ckpt_list)" in text):
                    ds_resume_failed = True
                    if rank0:
                        logger.warning(
                            "DeepSpeed optimizer-state resume failed from %s due to shard mismatch (%s). "
                            "Continuing with model weights from %s and resetting optimizer state.",
                            resume_ckpt,
                            text or type(exc).__name__,
                            policy_source,
                        )
                else:
                    raise RuntimeError(f"Failed to load DeepSpeed checkpoint from {resume_ckpt}") from exc
            except Exception as exc:
                raise RuntimeError(f"Failed to load DeepSpeed checkpoint from {resume_ckpt}") from exc
            if ds_resume_failed and rank0:
                logger.warning(
                    "DeepSpeed resume fallback active: start_update=%s (from trainer_state), "
                    "but optimizer/momentum states were not restored.",
                    start_update,
                )
        elif rank0:
            logger.warning(
                "Resume checkpoint has no DeepSpeed shard states: %s. "
                "Continuing with model weights only from %s (optimizer state reset).",
                resume_ckpt,
                policy_source,
            )

    artifacts: dict[str, Any] = {
        "output_dir": str(output_dir),
        "checkpoints": [],
    }
    if is_resuming and resume_ckpt is not None:
        artifacts["resumed_from"] = str(resume_ckpt)
        artifacts["resume_update"] = resume_update_idx

    async_json_writer: _AsyncJsonlWriter | None = _AsyncJsonlWriter() if ((not use_deepspeed) or rank0) else None

    def _log_json_row(payload: dict[str, Any]) -> None:
        if async_json_writer is not None:
            async_json_writer.append_json(log_path, payload)
        else:
            _append_jsonl(log_path, payload)

    def _log_rollout_rows(
        *,
        update_idx: int,
        rollouts: list[Rollout],
        advantages: list[list[float]],
        reward_stats: dict[str, float],
    ) -> None:
        if async_json_writer is not None:
            async_json_writer.append_rollouts(
                path=rollout_log_path,
                update_idx=update_idx,
                rollouts=rollouts,
                advantages=advantages,
                reward_stats=reward_stats,
            )
        else:
            _append_rollout_jsonl(
                path=rollout_log_path,
                update_idx=update_idx,
                rollouts=rollouts,
                advantages=advantages,
                reward_stats=reward_stats,
            )

    def _log_eval_rows(*, update_idx: int, eval_rows: list[dict[str, Any]]) -> None:
        if async_json_writer is not None:
            async_json_writer.append_eval_rows(
                path=eval_output_path,
                update_idx=update_idx,
                eval_rows=eval_rows,
            )
        else:
            _append_eval_output_jsonl(eval_output_path, update_idx=update_idx, eval_rows=eval_rows)

    best_dir = output_dir / "best"
    best_eval_score = float("-inf")
    best_eval_update: int | None = None

    def _sync_and_save_best_checkpoint(candidate_score: float | None, *, update_idx: int) -> None:
        nonlocal best_eval_score, best_eval_update

        is_new_best_local = False
        best_score_local: float | None = None
        if rank0 and candidate_score is not None:
            try:
                parsed_score = float(candidate_score)
            except (TypeError, ValueError):
                parsed_score = float("-inf")
            if math.isfinite(parsed_score) and parsed_score > best_eval_score:
                is_new_best_local = True
                best_score_local = parsed_score

        shared: list[Any] = [
            {
                "is_new_best": bool(is_new_best_local),
                "update_idx": int(update_idx),
                "score": best_score_local,
            }
            if rank0
            else None
        ]
        _broadcast_object_list(shared, src=0)
        payload = shared[0] if shared and isinstance(shared[0], dict) else {}
        should_save = bool(payload.get("is_new_best", False))
        if not should_save:
            return

        resolved_update_idx = int(payload.get("update_idx", update_idx))
        score_raw = payload.get("score")
        try:
            resolved_score = float(score_raw)
        except (TypeError, ValueError):
            resolved_score = float("-inf")
        if not math.isfinite(resolved_score):
            return

        trainer_state_payload = {
            "update_idx": int(resolved_update_idx),
            "best_eval_update": int(resolved_update_idx),
            "best_eval_score": float(resolved_score),
        }

        if use_deepspeed:
            _save_deepspeed_checkpoint_to_dir(
                ckpt_dir=best_dir,
                engine=train_model,
                tokenizer=tokenizer,
                hf_model=policy_eval_model,
                trainer_state=trainer_state_payload,
            )
        elif rank0:
            assert optimizer is not None
            _save_checkpoint_to_dir(
                ckpt_dir=best_dir,
                model=policy_eval_model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                trainer_state=trainer_state_payload,
            )

        if rank0:
            best_eval_score = resolved_score
            best_eval_update = resolved_update_idx
            logger.info("new best eval at update=%s score=%.6f", resolved_update_idx, resolved_score)

    if is_resuming:
        if resume_state:
            score_raw = resume_state.get("best_eval_score")
            update_raw = resume_state.get("best_eval_update")
            try:
                score = float(score_raw)
            except (TypeError, ValueError):
                score = float("-inf")
            try:
                update = int(update_raw) if update_raw is not None else None
            except (TypeError, ValueError):
                update = None
            if math.isfinite(score):
                best_eval_score = score
                best_eval_update = update

        if log_path.exists():
            log_best_score, log_best_update = _restore_best_from_log(log_path)
            if log_best_update is not None and (
                best_eval_update is None or log_best_score > best_eval_score
            ):
                best_eval_score = log_best_score
                best_eval_update = log_best_update

        logger.info(
            "Resuming training from %s (resume_update=%s, start_update=%s, best_eval_update=%s, best_eval_score=%s)",
            resume_ckpt,
            resume_update_idx,
            start_update,
            best_eval_update,
            best_eval_score if math.isfinite(best_eval_score) else None,
        )

    if cfg.logging.save_only_best and cfg.logging.save_every_n_updates <= 0:
        logger.warning(
            "logging.save_only_best=true with logging.save_every_n_updates<=0: "
            "resume checkpoints will not be written during training."
        )
    if rank0 and cfg.logging.keep_last_n_checkpoints > 0:
        logger.info(
            "Periodic checkpoint retention enabled: keep_last_n_checkpoints=%s (best checkpoint is managed separately).",
            int(cfg.logging.keep_last_n_checkpoints),
        )

    distributed_eval_shard = bool(use_deepspeed and world_size > 1 and cfg.eval.distributed_shard)
    if use_deepspeed and world_size > 1 and (not distributed_eval_shard) and rank0:
        logger.info(
            "Eval distributed sharding is disabled; keeping all ranks in eval generation to avoid ZeRO/NCCL deadlock."
        )

    def _run_eval_once(*, collect_outputs: bool, show_progress: bool) -> dict[str, Any]:
        return evaluate_on_dataset(
            examples=eval_examples,
            policy_model=policy_eval_model,
            tokenizer=tokenizer,
            cfg=cfg,
            device=device,
            metricx_scorer=metricx_scorer,
            xcomet_scorer=xcomet_scorer,
            mqm_scorer=mqm_scorer,
            esa_scorer=esa_scorer,
            collect_outputs=collect_outputs,
            show_progress=show_progress,
            distributed_eval_shard=distributed_eval_shard,
            distributed_rank=rank,
            distributed_world_size=world_size,
        )

    if cfg.eval.run_before_train and eval_examples and start_update <= 1:
        if (not use_deepspeed) or rank0:
            logger.info(
                "starting eval (run_before_train): examples=%s metricx=%s xcomet=%s mqm=%s esa=%s",
                len(eval_examples),
                bool(metricx_scorer is not None and cfg.reward.metricx.enabled),
                bool(xcomet_scorer is not None and xcomet_runtime_enabled),
                bool(mqm_scorer is not None and cfg.reward.mqm.enabled),
                bool(esa_scorer is not None and cfg.reward.esa.enabled),
            )
        report = _run_eval_once(
            collect_outputs=bool(cfg.logging.save_eval_outputs and ((not use_deepspeed) or rank0)),
            show_progress=bool((not use_deepspeed) or rank0),
        )
        if (not use_deepspeed) or rank0:
            eval_select_score = _compute_eval_selection_score(report, cfg)
            report["model_select_score"] = eval_select_score
            eval_rows = report.pop("eval_rows", [])
            _log_json_row({"type": "eval", "update": 0, **report})
            if cfg.logging.save_eval_outputs:
                _log_eval_rows(update_idx=0, eval_rows=eval_rows)
            logger.info(
                "finished eval (run_before_train): update=0 model_select_score=%.6f metricx=%.4f xcomet=%.4f mqm=%.4f esa=%.4f",
                float(eval_select_score),
                float(report.get("metricx_score_mean", 0.0)),
                float(report.get("xcomet_score_mean", 0.0)),
                float(report.get("mqm_score_mean", 0.0)),
                float(report.get("esa_score_mean", 0.0)),
            )
        _sync_and_save_best_checkpoint(eval_select_score if rank0 else None, update_idx=0)
        _dist_barrier()
    elif cfg.eval.run_before_train and eval_examples and start_update > 1:
        logger.info(
            "Skipping run_before_train eval because training is resumed from update=%s.",
            start_update - 1,
        )

    metricx_cache: dict[tuple[str, str, str], float] = {}
    xcomet_cache: dict[tuple[str, str, str], tuple[float, list[dict[str, Any]]]] = {}
    mqm_cache: dict[tuple[str, str, str], tuple[float, list[dict[str, Any]]]] = {}
    esa_cache: dict[tuple[str, str, str], float] = {}
    rng = random.Random(cfg.misc.seed)
    train_indices = list(range(len(train_examples)))
    rng.shuffle(train_indices)
    train_cursor = 0
    per_rank_batch_size = max(1, int(cfg.rl.batch_size))
    effective_batch_size = per_rank_batch_size * (world_size if (use_deepspeed and world_size > 1) else 1)
    updates_per_epoch = math.ceil(len(train_examples) / max(1, effective_batch_size))
    if (not use_deepspeed) or rank0:
        logger.info(
            "train_examples=%s per_rank_batch_size=%s effective_batch_size=%s updates_per_epoch=%s configured_updates=%s",
            len(train_examples),
            per_rank_batch_size,
            effective_batch_size,
            updates_per_epoch,
            cfg.rl.updates,
        )

    if start_update > cfg.rl.updates:
        logger.info(
            "Nothing to train: start_update=%s exceeds configured updates=%s.",
            start_update,
            cfg.rl.updates,
        )

    for update_idx in range(start_update, cfg.rl.updates + 1):
        rollouts: list[Rollout] = []
        advantages: list[list[float]] = []
        reward_stats: dict[str, float] = {}
        adv_stats: dict[str, float] = {}
        log_rollouts: list[Rollout] = rollouts
        log_advantages: list[list[float]] = advantages
        if use_deepspeed and world_size > 1:
            per_rank_batches: list[list[int]] = []
            if rank0:
                global_batch_size = per_rank_batch_size * world_size
                global_indices: list[int] = []
                while len(global_indices) < global_batch_size:
                    if train_cursor >= len(train_indices):
                        rng.shuffle(train_indices)
                        train_cursor = 0
                    remaining = len(train_indices) - train_cursor
                    take = min(global_batch_size - len(global_indices), remaining)
                    if take <= 0:
                        break
                    global_indices.extend(train_indices[train_cursor:train_cursor + take])
                    train_cursor += take
                per_rank_batches = [
                    global_indices[i * per_rank_batch_size : (i + 1) * per_rank_batch_size]
                    for i in range(world_size)
                ]

            shared_batch: list[Any] = [per_rank_batches]
            _broadcast_object_list(shared_batch, src=0)
            per_rank_batches = shared_batch[0] or []
            local_indices = (
                [int(i) for i in per_rank_batches[rank]]
                if rank < len(per_rank_batches)
                else []
            )
            if not local_indices:
                logger.warning("Rank %s has empty rollout shard at update=%s; skipping step.", rank, update_idx)
            local_examples = [train_examples[i] for i in local_indices]
            local_prompt_instance_ids = [
                f"u{update_idx}:r{rank}:p{pos}:idx{int(local_indices[pos])}"
                for pos in range(len(local_examples))
            ]

            _set_rollout_sampling_seed(cfg.misc.seed + update_idx + (rank * 1009))
            local_rollouts = generate_rollouts(
                examples=local_examples,
                policy_model=policy_eval_model,
                tokenizer=tokenizer,
                gen_cfg=cfg.generation,
                device=device,
                ref_model=None,
                ref_device=None,
                # In distributed mode we fill reference logprobs later in rank0 batch
                # (`_fill_missing_reference_logprobs`). Avoid per-sample worker calls here,
                # which are fragile under long-running CUDA workers.
                ref_logprob_fn=None,
                prompt_template=cfg.prompt.template,
                show_progress=bool(rank0),
                progress_desc=f"rollout u{update_idx}",
                prompt_instance_ids=local_prompt_instance_ids,
            )
            gathered_rollouts = _gather_object_to_rank0(local_rollouts)
            per_rank_payload: list[Any] | None = None
            shared_stats: list[Any] = [reward_stats if rank0 else {}, adv_stats if rank0 else {}]
            per_rank_rollouts: list[list[Rollout]] = []
            merged_rollouts: list[Rollout] = []
            merged_advantages: list[list[float]] = []
            if rank0:
                for shard_idx in range(world_size):
                    shard_rollouts: list[Rollout] = []
                    if gathered_rollouts is not None and shard_idx < len(gathered_rollouts):
                        raw_shard = gathered_rollouts[shard_idx]
                        if isinstance(raw_shard, list):
                            shard_rollouts = [r for r in raw_shard if isinstance(r, Rollout)]
                    per_rank_rollouts.append(shard_rollouts)

                for shard_rollouts in per_rank_rollouts:
                    merged_rollouts.extend(shard_rollouts)

                if merged_rollouts:
                    if reference_kl_enabled and ref_logprob_batch_fn is None:
                        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="rank0-step") as step_executor:
                            reward_future = step_executor.submit(
                                _prepare_rewards_and_advantages,
                                rollouts=merged_rollouts,
                                cfg=cfg,
                                metricx_scorer=metricx_scorer,
                                xcomet_scorer=xcomet_scorer,
                                mqm_scorer=mqm_scorer,
                                esa_scorer=esa_scorer,
                                metricx_cache=metricx_cache,
                                xcomet_cache=xcomet_cache,
                                mqm_cache=mqm_cache,
                                esa_cache=esa_cache,
                                tokenizer=tokenizer,
                            )
                            ref_fill_future = step_executor.submit(
                                _fill_missing_reference_logprobs,
                                merged_rollouts=merged_rollouts,
                                cfg=cfg,
                                update_idx=update_idx,
                                ref_logprob_batch_fn=ref_logprob_batch_fn,
                                ref_logprob_client=ref_logprob_client,
                                ref_model=ref_model,
                                ref_device=ref_device,
                                device=device,
                            )
                            merged_advantages, reward_stats, adv_stats = reward_future.result()
                            ref_fill_future.result()
                    else:
                        merged_advantages, reward_stats, adv_stats = _prepare_rewards_and_advantages(
                            rollouts=merged_rollouts,
                            cfg=cfg,
                            metricx_scorer=metricx_scorer,
                            xcomet_scorer=xcomet_scorer,
                            mqm_scorer=mqm_scorer,
                            esa_scorer=esa_scorer,
                            metricx_cache=metricx_cache,
                            xcomet_cache=xcomet_cache,
                            mqm_cache=mqm_cache,
                            esa_cache=esa_cache,
                            tokenizer=tokenizer,
                        )
                else:
                    logger.warning("No rollouts generated at update=%s; skipping step.", update_idx)

            if reference_kl_enabled and ref_logprob_batch_fn is not None:
                _ = _fill_missing_reference_logprobs_distributed_colocate(
                    merged_rollouts=merged_rollouts if rank0 else None,
                    cfg=cfg,
                    update_idx=update_idx,
                    ref_logprob_batch_fn=ref_logprob_batch_fn,
                    rank=rank,
                )

            if rank0:
                shard_sizes = [len(shard) for shard in per_rank_rollouts]
                can_shard_update = (
                    bool(merged_rollouts)
                    and bool(shard_sizes)
                    and min(shard_sizes) > 0
                    and len(set(shard_sizes)) == 1
                    and len(merged_advantages) == len(merged_rollouts)
                )
                if can_shard_update:
                    logger.info(
                        "Using per-rank policy update shards at update=%s shard_size=%s merged_rollouts=%s.",
                        update_idx,
                        shard_sizes[0] if shard_sizes else 0,
                        len(merged_rollouts),
                    )
                    per_rank_advantages: list[list[list[float]]] = []
                    cursor = 0
                    for shard_rollouts in per_rank_rollouts:
                        take = len(shard_rollouts)
                        per_rank_advantages.append(merged_advantages[cursor:cursor + take])
                        cursor += take
                    per_rank_payload = [
                        {"rollouts": per_rank_rollouts[i], "advantages": per_rank_advantages[i]}
                        for i in range(world_size)
                    ]
                else:
                    if merged_rollouts and len(set(shard_sizes)) > 1:
                        logger.warning(
                            "Uneven rollout shard sizes at update=%s; falling back to replicated policy update this step. shard_sizes=%s",
                            update_idx,
                            shard_sizes,
                        )
                    per_rank_payload = [
                        {"rollouts": merged_rollouts, "advantages": merged_advantages}
                        for _ in range(world_size)
                    ]

                shared_stats = [reward_stats, adv_stats]
                log_rollouts = merged_rollouts
                log_advantages = merged_advantages

            local_payload = _scatter_object_from_rank0(per_rank_payload if rank0 else None, rank=rank)
            if isinstance(local_payload, dict):
                rollouts = list(local_payload.get("rollouts") or [])
                advantages = list(local_payload.get("advantages") or [])
            else:
                rollouts = []
                advantages = []

            _broadcast_object_list(shared_stats, src=0)
            reward_stats = shared_stats[0] or {}
            adv_stats = shared_stats[1] or {}
            if not rank0:
                log_rollouts = rollouts
                log_advantages = advantages
        elif (not use_deepspeed) or rank0:
            if train_cursor >= len(train_indices):
                rng.shuffle(train_indices)
                train_cursor = 0
            batch_end = min(train_cursor + max(1, cfg.rl.batch_size), len(train_indices))
            batch_indices = train_indices[train_cursor:batch_end]
            train_cursor = batch_end
            batch_examples = [train_examples[i] for i in batch_indices]
            prompt_instance_ids = [
                f"u{update_idx}:r{rank}:p{pos}:idx{int(batch_indices[pos])}"
                for pos in range(len(batch_examples))
            ]
            rollouts = generate_rollouts(
                examples=batch_examples,
                policy_model=policy_eval_model,
                tokenizer=tokenizer,
                gen_cfg=cfg.generation,
                device=device,
                # Keep rollout generation free of reference model calls; fill reference
                # logprobs in one batched step right after rewards.
                ref_model=None,
                ref_device=None,
                ref_logprob_fn=None,
                prompt_template=cfg.prompt.template,
                show_progress=True,
                progress_desc=f"rollout u{update_idx}",
                prompt_instance_ids=prompt_instance_ids,
            )
            if rollouts:
                advantages, reward_stats, adv_stats = _prepare_rewards_and_advantages(
                    rollouts=rollouts,
                    cfg=cfg,
                    metricx_scorer=metricx_scorer,
                    xcomet_scorer=xcomet_scorer,
                    mqm_scorer=mqm_scorer,
                    esa_scorer=esa_scorer,
                    metricx_cache=metricx_cache,
                    xcomet_cache=xcomet_cache,
                    mqm_cache=mqm_cache,
                    esa_cache=esa_cache,
                    tokenizer=tokenizer,
                )
                if reference_kl_enabled:
                    _ = _fill_missing_reference_logprobs(
                        merged_rollouts=rollouts,
                        cfg=cfg,
                        update_idx=update_idx,
                        ref_logprob_batch_fn=ref_logprob_batch_fn,
                        ref_logprob_client=ref_logprob_client,
                        ref_model=ref_model,
                        ref_device=ref_device,
                        device=device,
                    )
            else:
                logger.warning("No rollouts generated at update=%s; skipping step.", update_idx)
            log_rollouts = rollouts
            log_advantages = advantages

        if not rollouts:
            _dist_barrier()
            continue

        step_stats = []
        for _ in range(max(1, cfg.rl.ppo_epochs)):
            step_stats.append(
                update_policy(
                    rollouts=rollouts,
                    advantages=advantages,
                    policy_model=train_model,
                    optimizer=optimizer,
                    rl_cfg=cfg.rl,
                    device=device,
                    tokenizer=tokenizer,
                )
            )
        train_stats = step_stats[-1]

        completion_lens = [len(r.completion_token_ids) for r in log_rollouts]
        payload = {
            "type": "train",
            "update": update_idx,
            "rollout_avg_completion_len": float(mean(completion_lens) if completion_lens else 0.0),
            "adv_raw_mean": adv_stats["raw_mean"],
            "adv_raw_std": adv_stats["raw_std"],
            "adv_norm_mean": adv_stats["norm_mean"],
            "adv_norm_std": adv_stats["norm_std"],
            "policy_loss": train_stats.policy_loss,
            "approx_kl": train_stats.approx_kl,
            "clip_fraction": train_stats.clip_fraction,
            "entropy": train_stats.entropy,
            "kl_to_reference": train_stats.kl_to_reference,
            "token_count": train_stats.token_count,
            **reward_stats,
        }
        if (not use_deepspeed) or rank0:
            _log_json_row(payload)
            if cfg.logging.save_rollouts:
                _log_rollout_rows(
                    update_idx=update_idx,
                    rollouts=log_rollouts,
                    advantages=log_advantages,
                    reward_stats=reward_stats,
                )

            logger.info(
                "update=%s loss=%.6f len=%.2f metricx=%.4f±%.4f xcomet=%.4f±%.4f mqm=%.4f±%.4f esa=%.4f±%.4f token_nonzero=%.4f A(raw)=%.4f/%.4f A(norm)=%.4f/%.4f",
                update_idx,
                train_stats.policy_loss,
                payload["rollout_avg_completion_len"],
                payload["metricx_score_mean"],
                payload["metricx_score_std"],
                payload["xcomet_score_mean"],
                payload["xcomet_score_std"],
                payload["mqm_score_mean"],
                payload["mqm_score_std"],
                payload["esa_score_mean"],
                payload["esa_score_std"],
                payload["token_rewards_non_zero_ratio"],
                payload["adv_raw_mean"],
                payload["adv_raw_std"],
                payload["adv_norm_mean"],
                payload["adv_norm_std"],
            )

        trainer_state_payload = {
            "update_idx": int(update_idx),
            "best_eval_update": int(best_eval_update) if best_eval_update is not None else None,
            "best_eval_score": float(best_eval_score) if math.isfinite(best_eval_score) else None,
        }

        if cfg.logging.save_every_n_updates > 0 and update_idx % cfg.logging.save_every_n_updates == 0:
            keep_recent_n = int(cfg.logging.keep_last_n_checkpoints)
            save_periodic_checkpoint = bool(keep_recent_n > 0 or (not cfg.logging.save_only_best))
            if use_deepspeed:
                hf_checkpoint_model = policy_eval_model if _lora_enabled(cfg) else None
                if save_periodic_checkpoint:
                    ckpt = _save_deepspeed_checkpoint(
                        output_dir=output_dir,
                        update_idx=update_idx,
                        engine=train_model,
                        tokenizer=tokenizer,
                        hf_model=hf_checkpoint_model,
                        trainer_state=trainer_state_payload,
                    )
                    if rank0:
                        artifacts["checkpoints"].append(str(ckpt))
                        if cfg.logging.save_only_best:
                            artifacts["resume_checkpoint"] = str(ckpt)
                        if keep_recent_n > 0:
                            removed = _prune_old_checkpoints(output_dir, keep_recent_n)
                            if removed:
                                removed_set = {str(path) for path in removed}
                                artifacts["checkpoints"] = [
                                    path for path in artifacts["checkpoints"] if path not in removed_set
                                ]
                                logger.info(
                                    "Pruned %s old checkpoints; keeping latest %s periodic checkpoints.",
                                    len(removed),
                                    keep_recent_n,
                                )
                else:
                    resume_path = _save_deepspeed_resume_checkpoint(
                        output_dir=output_dir,
                        update_idx=update_idx,
                        engine=train_model,
                        tokenizer=tokenizer,
                        hf_model=hf_checkpoint_model,
                        trainer_state=trainer_state_payload,
                    )
                    if rank0:
                        artifacts["resume_checkpoint"] = str(resume_path)
                if keep_recent_n > 0:
                    _dist_barrier()
            elif rank0:
                if save_periodic_checkpoint:
                    assert optimizer is not None
                    ckpt = _save_checkpoint(
                        output_dir=output_dir,
                        update_idx=update_idx,
                        model=policy_eval_model,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        trainer_state=trainer_state_payload,
                    )
                    artifacts["checkpoints"].append(str(ckpt))
                    if cfg.logging.save_only_best:
                        artifacts["resume_checkpoint"] = str(ckpt)
                    if keep_recent_n > 0:
                        removed = _prune_old_checkpoints(output_dir, keep_recent_n)
                        if removed:
                            removed_set = {str(path) for path in removed}
                            artifacts["checkpoints"] = [
                                path for path in artifacts["checkpoints"] if path not in removed_set
                            ]
                            logger.info(
                                "Pruned %s old checkpoints; keeping latest %s periodic checkpoints.",
                                len(removed),
                                keep_recent_n,
                            )
                else:
                    assert optimizer is not None
                    resume_path = _save_resume_checkpoint(
                        output_dir=output_dir,
                        update_idx=update_idx,
                        model=policy_eval_model,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        trainer_state=trainer_state_payload,
                    )
                    artifacts["resume_checkpoint"] = str(resume_path)

        if (
            cfg.eval.eval_every_n_updates > 0
            and eval_examples
            and update_idx % cfg.eval.eval_every_n_updates == 0
        ):
            if (not use_deepspeed) or rank0:
                logger.info(
                    "starting eval: update=%s examples=%s metricx=%s xcomet=%s mqm=%s esa=%s",
                    update_idx,
                    len(eval_examples),
                    bool(metricx_scorer is not None and cfg.reward.metricx.enabled),
                    bool(xcomet_scorer is not None and xcomet_runtime_enabled),
                    bool(mqm_scorer is not None and cfg.reward.mqm.enabled),
                    bool(esa_scorer is not None and cfg.reward.esa.enabled),
                )
            report = _run_eval_once(
                collect_outputs=bool(cfg.logging.save_eval_outputs and ((not use_deepspeed) or rank0)),
                show_progress=bool((not use_deepspeed) or rank0),
            )
            if (not use_deepspeed) or rank0:
                eval_select_score = _compute_eval_selection_score(report, cfg)
                report["model_select_score"] = eval_select_score
                eval_rows = report.pop("eval_rows", [])
                _log_json_row({"type": "eval", "update": update_idx, **report})
                if cfg.logging.save_eval_outputs:
                    _log_eval_rows(update_idx=update_idx, eval_rows=eval_rows)
                logger.info(
                    "finished eval: update=%s model_select_score=%.6f metricx=%.4f xcomet=%.4f mqm=%.4f esa=%.4f",
                    update_idx,
                    float(eval_select_score),
                    float(report.get("metricx_score_mean", 0.0)),
                    float(report.get("xcomet_score_mean", 0.0)),
                    float(report.get("mqm_score_mean", 0.0)),
                    float(report.get("esa_score_mean", 0.0)),
                )
            _sync_and_save_best_checkpoint(eval_select_score if rank0 else None, update_idx=update_idx)

        # Early-stop guard for divergence in toy runs.
        if not math.isfinite(train_stats.policy_loss):
            raise RuntimeError(f"Non-finite loss at update {update_idx}")
        _dist_barrier()

    if async_json_writer is not None:
        async_json_writer.flush()

    final_dir = output_dir / "final"
    if rank0 and final_dir.exists():
        shutil.rmtree(final_dir)
    _dist_barrier()
    if best_eval_update is not None and best_dir.exists():
        if rank0:
            shutil.copytree(best_dir, final_dir)
    else:
        if rank0:
            final_dir.mkdir(parents=True, exist_ok=True)
        _dist_barrier()
        if use_deepspeed:
            _save_pretrained_model(policy_eval_model, final_dir)
            if rank0:
                tokenizer.save_pretrained(final_dir)
        elif rank0:
            policy_eval_model.save_pretrained(final_dir)
            tokenizer.save_pretrained(final_dir)
    if (not use_deepspeed) or rank0:
        artifacts["final_model_dir"] = str(final_dir)
        artifacts["best_model_dir"] = str(best_dir) if best_eval_update is not None and best_dir.exists() else None
        artifacts["best_eval_update"] = best_eval_update
        artifacts["best_eval_score"] = best_eval_score if best_eval_update is not None else None
        artifacts["log_path"] = str(log_path)
        if cfg.logging.save_rollouts:
            artifacts["rollout_log_path"] = str(rollout_log_path)
        if cfg.logging.save_eval_outputs:
            artifacts["eval_output_path"] = str(eval_output_path)
    _dist_barrier()

    if ref_logprob_client is not None:
        ref_logprob_client.close()
    if async_json_writer is not None:
        async_json_writer.close()

    if use_deepspeed and (not rank0):
        return {
            "output_dir": str(output_dir),
            "worker_rank": _distributed_rank(),
            "status": "ok",
        }

    return artifacts
