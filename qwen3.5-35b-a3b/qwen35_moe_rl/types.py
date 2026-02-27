from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - optional during lightweight tests
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


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
    ref_text: str | None = None
    # Precomputed sequence-level logprob statistics for GSPO.
    old_logprob_mean: float | None = None
    ref_logprob_mean: float | None = None


@dataclass
class SampleForScoring:
    src: str
    mt: str
    ref: str | None = None


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
    # MoE monitoring fields.
    aux_loss: float = 0.0
    expert_utilization_entropy: float = 0.0
    max_expert_load: float = 0.0
    min_expert_load: float = 0.0


@dataclass
class DistributedContext:
    """Encapsulates distributed training state.

    When ``deepspeed_engine`` is set, backward / step / zero_grad are delegated
    to the DeepSpeed engine.  Otherwise (``None`` or default construction) the
    context is a no-op wrapper that preserves the original non-distributed code
    paths exactly.
    """

    backend: str = "none"  # none | deepspeed | fsdp | ddp
    deepspeed_engine: Any = None

    def backward(self, loss: Any) -> None:
        if self.deepspeed_engine is not None:
            self.deepspeed_engine.backward(loss)
        else:
            loss.backward()

    def step(self, optimizer: Any, model: Any, max_grad_norm: float) -> None:
        if self.deepspeed_engine is not None:
            self.deepspeed_engine.step()
        else:
            if max_grad_norm > 0 and torch is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

    def zero_grad(self, optimizer: Any) -> None:
        if self.deepspeed_engine is not None:
            self.deepspeed_engine.zero_grad()
        else:
            optimizer.zero_grad(set_to_none=True)
