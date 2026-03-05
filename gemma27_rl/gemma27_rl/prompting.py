from __future__ import annotations

import re
from typing import Any

from .rl_types import Example


DEFAULT_TRANSLATION_PROMPT_TEMPLATE = (
    "You are a professional {source_lang} ({src_lang_code}) to {target_lang} ({tgt_lang_code}) "
    "translator. Your goal is to accurately convey the meaning and nuances of the original "
    "{source_lang} text while adhering to {target_lang} grammar, vocabulary, and cultural "
    "sensitivities. Produce only the {target_lang} translation, without any additional "
    "explanations or commentary. Please translate the following {source_lang} text into "
    "{target_lang}:\\n\\n{text}"
)


_THINK_BLOCK_PATTERN = re.compile(r"<\s*think\s*>.*?<\s*/\s*think\s*>", flags=re.IGNORECASE | re.DOTALL)
_THINK_TAG_PATTERN = re.compile(r"<\s*/?\s*think\s*>", flags=re.IGNORECASE)
_PIPE_SPECIAL_PATTERN = re.compile(r"<\|[^>\n]{1,128}\|>")


def collect_tokenizer_special_token_strings(tokenizer: Any | None) -> list[str]:
    if tokenizer is None:
        return []
    tokens: set[str] = set()

    all_special = getattr(tokenizer, "all_special_tokens", None)
    if isinstance(all_special, (list, tuple, set)):
        for tok in all_special:
            if isinstance(tok, str) and tok:
                tokens.add(tok)

    additional_special = getattr(tokenizer, "additional_special_tokens", None)
    if isinstance(additional_special, (list, tuple, set)):
        for tok in additional_special:
            if isinstance(tok, str) and tok:
                tokens.add(tok)

    special_map = getattr(tokenizer, "special_tokens_map", None)
    if isinstance(special_map, dict):
        for value in special_map.values():
            if isinstance(value, str) and value:
                tokens.add(value)
            elif isinstance(value, (list, tuple, set)):
                for tok in value:
                    if isinstance(tok, str) and tok:
                        tokens.add(tok)

    return sorted(tokens, key=len, reverse=True)


def _replace_matches_with_spaces(text: str, pattern: re.Pattern[str]) -> tuple[str, int]:
    match_count = 0

    def _repl(match: re.Match[str]) -> str:
        nonlocal match_count
        match_count += 1
        return " " * (match.end() - match.start())

    return pattern.sub(_repl, text), match_count


def sanitize_text_for_scoring(target_text: str, *, special_tokens: list[str] | None = None) -> tuple[str, int]:
    sanitized = str(target_text or "")
    replacement_count = 0
    special_tokens = list(special_tokens or [])

    sanitized, replaced = _replace_matches_with_spaces(sanitized, _THINK_BLOCK_PATTERN)
    replacement_count += replaced

    sanitized, replaced = _replace_matches_with_spaces(sanitized, _THINK_TAG_PATTERN)
    replacement_count += replaced

    for tok in special_tokens:
        token_text = str(tok or "")
        if not token_text:
            continue
        if token_text.lower() in {"<think>", "</think>"}:
            continue
        occurrences = sanitized.count(token_text)
        if occurrences <= 0:
            continue
        sanitized = sanitized.replace(token_text, " " * len(token_text))
        replacement_count += occurrences

    sanitized, replaced = _replace_matches_with_spaces(sanitized, _PIPE_SPECIAL_PATTERN)
    replacement_count += replaced
    return sanitized, replacement_count


def format_translation_prompt(example: Example, template: str = DEFAULT_TRANSLATION_PROMPT_TEMPLATE) -> str:
    src_code = (example.src_lang_code or example.src_lang or "").strip()
    tgt_code = (example.tgt_lang_code or example.tgt_lang or "").strip()
    return template.format(
        source_lang=example.src_lang,
        src_lang_code=src_code,
        target_lang=example.tgt_lang,
        tgt_lang_code=tgt_code,
        text=example.src_text,
    )


def postprocess_translation(raw_text: str) -> str:
    return raw_text.strip()
