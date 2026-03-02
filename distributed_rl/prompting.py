"""Translation prompt template for chat-based LLMs.

In GRPO for machine translation, the "prompt" is a user-turn message that asks
the model to translate text.  The model's "completion" (response) is the
translation.  The reward model then scores the quality of that translation.

The template uses Python str.format() placeholders:
  {source_lang}  — full language name, e.g. "English"
  {target_lang}  — full language name, e.g. "Korean"
  {src_lang_code} — ISO code, e.g. "en"
  {tgt_lang_code} — ISO code, e.g. "ko"
  {text}          — the actual source text to translate

This prompt is wrapped into a chat message list for TRL:
  [{"role": "user", "content": "<formatted prompt>"}]

TRL's GRPOTrainer then applies the model's chat template (e.g. <|im_start|>
for Qwen, <start_of_turn> for Gemma) to convert this into the actual
token sequence the model sees.

Ref: Original template — qwen3.5-35b-a3b/qwen35_moe_rl/prompting.py
"""

from __future__ import annotations


DEFAULT_TRANSLATION_PROMPT_TEMPLATE = (
    "You are a professional {source_lang} ({src_lang_code}) to {target_lang} ({tgt_lang_code}) "
    "translator. Your goal is to accurately convey the meaning and nuances of the original "
    "{source_lang} text while adhering to {target_lang} grammar, vocabulary, and cultural "
    "sensitivities. Produce only the {target_lang} translation, without any additional "
    "explanations or commentary. Please translate the following {source_lang} text into "
    "{target_lang}:\\n\\n{text}"
)


def format_translation_prompt(
    *,
    src_text: str,
    src_lang: str,
    tgt_lang: str,
    src_lang_code: str = "",
    tgt_lang_code: str = "",
    template: str = DEFAULT_TRANSLATION_PROMPT_TEMPLATE,
) -> str:
    """Fill in the translation prompt template with concrete values.

    Args:
        src_text: The source text to translate (goes into ``{text}``).
        src_lang: Full source language name (e.g. ``"English"``).
        tgt_lang: Full target language name (e.g. ``"Korean"``).
        src_lang_code: ISO code for source language (e.g. ``"en"``).
                       Falls back to ``src_lang`` if empty.
        tgt_lang_code: ISO code for target language (e.g. ``"ko"``).
                       Falls back to ``tgt_lang`` if empty.
        template: The prompt template string with ``{...}`` placeholders.

    Returns:
        The fully formatted prompt string ready to be wrapped in a chat message.
    """
    return template.format(
        source_lang=src_lang,
        src_lang_code=src_lang_code or src_lang,
        target_lang=tgt_lang,
        tgt_lang_code=tgt_lang_code or tgt_lang,
        text=src_text,
    )
