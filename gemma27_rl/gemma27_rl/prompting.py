from __future__ import annotations

from .types import Example


DEFAULT_TRANSLATION_PROMPT_TEMPLATE = (
    "You are a professional {source_lang} ({src_lang_code}) to {target_lang} ({tgt_lang_code}) "
    "translator. Your goal is to accurately convey the meaning and nuances of the original "
    "{source_lang} text while adhering to {target_lang} grammar, vocabulary, and cultural "
    "sensitivities. Produce only the {target_lang} translation, without any additional "
    "explanations or commentary. Please translate the following {source_lang} text into "
    "{target_lang}:\\n\\n{text}"
)

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
