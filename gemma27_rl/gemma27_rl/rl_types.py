from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Example:
    example_id: str
    src_text: str
    src_lang: str
    tgt_lang: str
    src_lang_code: str | None = None
    tgt_lang_code: str | None = None
    ref_text: str | None = None


@dataclass
class Rollout:
    example_id: str
    prompt_text: str
    prompt_input_ids: list[int]
    completion_text: str
    completion_token_ids: list[int]
    old_logprobs: list[float]
    ref_logprobs: list[float] | None
    token_char_offsets: list[tuple[int, int]]
    src_text: str
    src_lang: str | None = None
    tgt_lang: str | None = None
    src_lang_code: str | None = None
    tgt_lang_code: str | None = None
    ref_text: str | None = None
    raw_completion_token_ids: list[int] | None = None
    completion_raw_text: str | None = None
    completion_clean_text: str | None = None
    prompt_instance_id: str | None = None


@dataclass
class SampleForScoring:
    src: str
    mt: str
    ref: str | None = None
    source_lang: str | None = None
    target_lang: str | None = None


@dataclass
class RewardOutput:
    sequence_scores: list[float]
    metadata: Any = None


@dataclass
class TrainStats:
    policy_loss: float
    approx_kl: float
    clip_fraction: float
    entropy: float
    kl_to_reference: float
    token_count: int
    extra: dict[str, float] = field(default_factory=dict)
