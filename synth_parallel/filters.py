from __future__ import annotations

from dataclasses import dataclass

from .config import FiltersConfig
from .prompts import build_judge_messages, parse_judge_json
from .teacher import TeacherClient
from .utils import jaccard_overlap


@dataclass
class FilterDecision:
    passed: bool
    reason_code: str
    notes: str = ""


def apply_rule_based_filters(source_text: str, target_text: str, cfg: FiltersConfig) -> FilterDecision:
    cleaned = target_text.strip()
    if not cleaned:
        return FilterDecision(False, "empty")

    if len(cleaned) < cfg.min_chars:
        return FilterDecision(False, "too_short")

    if len(cleaned) > cfg.max_chars:
        return FilterDecision(False, "too_long")

    lowered = cleaned.lower()
    for token in cfg.blocked_substrings:
        if token.lower() in lowered:
            return FilterDecision(False, "blocked_substring", notes=token)

    ratio = len(cleaned) / max(1, len(source_text.strip()))
    if ratio < cfg.length_ratio_min:
        return FilterDecision(False, "ratio_too_small", notes=f"ratio={ratio:.3f}")
    if ratio > cfg.length_ratio_max:
        return FilterDecision(False, "ratio_too_large", notes=f"ratio={ratio:.3f}")

    overlap = jaccard_overlap(source_text, cleaned)
    if overlap > cfg.max_copy_overlap:
        return FilterDecision(False, "copy_overlap", notes=f"overlap={overlap:.3f}")

    return FilterDecision(True, "pass")


def apply_llm_judge_filter(
    teacher: TeacherClient,
    source_lang: str,
    target_lang: str,
    source_text: str,
    target_text: str,
    cfg: FiltersConfig,
) -> FilterDecision:
    messages = build_judge_messages(source_lang, target_lang, source_text, target_text)

    try:
        content = teacher.complete(
            messages=messages,
            temperature=cfg.llm_judge.temperature,
            top_p=1.0,
            max_tokens=cfg.llm_judge.max_tokens,
            model=cfg.llm_judge.model,
        )
    except Exception as exc:  # pylint: disable=broad-except
        if cfg.llm_judge.fail_policy == "permissive":
            return FilterDecision(True, "judge_failed_permissive", notes=str(exc))
        return FilterDecision(False, "judge_failed", notes=str(exc))

    parsed = parse_judge_json(content)
    if parsed is None:
        if cfg.llm_judge.fail_policy == "permissive":
            return FilterDecision(True, "judge_parse_failed_permissive")
        return FilterDecision(False, "judge_parse_failed")

    passed = bool(parsed.get("pass", False))
    reason = str(parsed.get("reason_code", "judge_reject" if not passed else "pass"))
    notes = str(parsed.get("notes", ""))
    return FilterDecision(passed=passed, reason_code=reason, notes=notes)
