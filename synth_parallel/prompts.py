from __future__ import annotations

import json
import re


def build_translation_messages(
    source_lang: str,
    target_lang: str,
    src_lang_code: str,
    tgt_lang_code: str,
    text: str,
) -> list[dict[str, str]]:
    system = "You are a professional translator."
    user = (
        f"Translate the following text from {source_lang} ({src_lang_code}) "
        f"to {target_lang} ({tgt_lang_code}).\n"
        "Preserve meaning and nuance. Use natural grammar in the target language.\n"
        "Output only the translation with no commentary.\n\n"
        f"Text:\n{text}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_judge_messages(
    source_lang: str,
    target_lang: str,
    source_text: str,
    candidate_translation: str,
) -> list[dict[str, str]]:
    system = "You are a strict translation quality gate. Return JSON only."
    user = (
        "Given a translation candidate, decide whether it should be accepted.\n"
        "Return exactly one JSON object: "
        '{"pass": true|false, "reason_code": "...", "notes": "..."}.\n'
        "Criteria: target-language correctness, no extra commentary, no severe meaning loss.\n\n"
        f"Source language: {source_lang}\n"
        f"Target language: {target_lang}\n"
        f"Source text:\n{source_text}\n\n"
        f"Candidate translation:\n{candidate_translation}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_judge_json(text: str) -> dict[str, str | bool] | None:
    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and "pass" in payload:
            return payload
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
        if isinstance(payload, dict) and "pass" in payload:
            return payload
    except json.JSONDecodeError:
        return None
    return None
