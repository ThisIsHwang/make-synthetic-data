"""Abstract reward model interface and TRL reward function wrapper.

=== Reward Models in GRPO ===

In GRPO, the reward model is the "environment" that tells the agent (LLM)
how good its action (translation) was.  The training loop is:

  1. Policy generates translations (completions)
  2. Reward model scores each translation → scalar reward
  3. GRPO computes advantages from group-relative rewards
  4. Policy is updated to maximize expected reward

This module defines the interface that all reward models must implement,
and the wrapper that converts them into TRL-compatible callable functions.

=== Reward Direction and Offset ===

Different metrics have different reward directions:

  MetricX:  raw score in [0, 25], LOWER = better
            reward = offset - metricx_score  (with offset=5.0)
            So a perfect translation (score=0) → reward=5.0
            A terrible translation (score=25) → reward=-20.0

  XComet:   raw score in [0, 1], HIGHER = better
            reward = weight * xcomet_score  (no offset needed)
            So a perfect translation (score=1.0) → reward=1.0

When using both rewards together, TRL sums them:
  total_reward = metricx_reward + xcomet_reward

The weights (w_metricx, w_xcomet) control the relative importance of each.

Ref: Original interface — qwen3.5-35b-a3b/qwen35_moe_rl/types.py (SampleForScoring, RewardOutput)
Ref: TRL reward function API — https://huggingface.co/docs/trl/grpo_trainer#using-custom-reward-functions
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class SampleForScoring:
    """A single translation sample to be scored by a reward model.

    This is the universal input format for all reward models.
    It contains the minimal information needed to evaluate translation quality:
      - src: the original source text (what was supposed to be translated)
      - mt:  the machine translation (the model's completion / output)
      - ref: the human reference translation (optional, for ref-based metrics)

    In QE (Quality Estimation) mode, ``ref`` is None — the reward model
    judges quality using only the source and translation.
    """
    src: str       # Source text (e.g. "Hello world")
    mt: str        # Machine translation / model completion (e.g. "안녕하세요")
    ref: str | None = None  # Reference translation (optional)


@dataclass
class RewardOutput:
    """Output from a reward model's ``score()`` method.

    sequence_scores: One scalar reward per input sample.
                     Length must match the input samples list.
                     The scale and direction depend on the specific model
                     (MetricX: lower=better, XComet: higher=better).
                     The offset/weight transformation is applied by
                     ``make_trl_reward_func()``, not here.

    metadata: Optional extra data (e.g. error spans from XComet).
              Not used by the training loop, but useful for analysis.
    """
    sequence_scores: list[float]
    metadata: Any = None


class RewardModel(ABC):
    """Abstract base class for all reward models.

    To add a new reward model, subclass this and implement ``score()``.
    The model can load weights in ``__init__()`` and release them in ``close()``.

    GPU placement and subprocess isolation are the responsibility of each
    implementation.  See MetricXReward and XCometReward for examples of
    subprocess-based GPU isolation.
    """

    @abstractmethod
    def score(self, samples: list[SampleForScoring]) -> RewardOutput:
        """Score a batch of (source, translation, optional reference) samples.

        Args:
            samples: List of SampleForScoring objects to evaluate.

        Returns:
            RewardOutput with one score per sample.
        """
        ...

    def close(self) -> None:
        """Release resources (GPU memory, subprocesses).

        Called when training ends or when the reward model is no longer needed.
        Override this if your model holds external resources.
        """
        pass


# ---------------------------------------------------------------------------
# TRL reward function wrapper
# ---------------------------------------------------------------------------

def make_trl_reward_func(
    reward_model: RewardModel,
    *,
    src_column: str = "src_text",
    ref_column: str = "ref_text",
    weight: float = 1.0,
    offset: float = 0.0,
    clip_value: float = 0.0,
) -> Callable[..., list[float]]:
    """Wrap a RewardModel as a TRL-compatible reward function.

    === How TRL Calls Reward Functions ===

    TRL's GRPOTrainer calls each reward function like this:

        scores = reward_func(
            completions=["translation1", "translation2", ...],
            prompts=[...],          # the original prompts
            src_text=["source1", "source2", ...],  # from our dataset
            ref_text=["ref1", "ref2", ...],        # from our dataset
        )

    All columns from the dataset are passed as keyword arguments.
    This is why we include ``src_text`` and ``ref_text`` in our dataset —
    the reward functions need them to evaluate translation quality.

    === Offset and Weight ===

    The returned reward is:
      - If offset != 0: ``weight * (offset - raw_score)``
        Used for MetricX where lower score = better quality.
        Example: MetricX score=0.5, offset=5.0, weight=1.0 → reward=4.5

      - If offset == 0: ``weight * raw_score``
        Used for XComet where higher score = better quality.
        Example: XComet score=0.85, weight=1.0 → reward=0.85

    Args:
        reward_model: The reward model to wrap.
        src_column: Name of the dataset column containing source text.
        ref_column: Name of the dataset column containing reference text.
        weight: Multiplier for the reward (controls relative importance
                when using multiple reward models).
        offset: Subtraction offset for "lower=better" metrics.
                Set to 0 for "higher=better" metrics.
        clip_value: If > 0, clip transformed rewards to [-clip_value, +clip_value].
                Prevents extreme outliers from destabilizing advantage
                normalization.  Set to 0 to disable (default).
                Ref: MiniRL (arxiv 2512.01374)

    Returns:
        A callable with signature:
        ``(completions: list[str], **kwargs) -> list[float]``
    """

    def _reward_func(completions: list[str], **kwargs: Any) -> list[float]:
        # Extract source texts from kwargs.
        # These come from the dataset columns we set up in data.py.
        # If the column is missing (shouldn't happen), fall back to empty strings.
        sources: list[str] = kwargs.get(src_column, [""] * len(completions))
        references: list[str | None] = kwargs.get(ref_column, [None] * len(completions))

        # Build SampleForScoring objects from the TRL-provided data.
        # Each sample pairs a source text with the model's translation.
        samples = [
            SampleForScoring(src=str(src), mt=str(mt), ref=(str(ref) if ref else None))
            for src, mt, ref in zip(sources, completions, references)
        ]

        # Call the actual reward model to score all samples.
        output = reward_model.score(samples)

        # Apply offset and weight transformation.
        # For MetricX (offset=5.0): reward = weight * (5.0 - metricx_score)
        # For XComet (offset=0.0):  reward = weight * xcomet_score
        if offset != 0.0:
            rewards = [weight * (offset - s) for s in output.sequence_scores]
        else:
            rewards = [weight * s for s in output.sequence_scores]

        # Clip extreme rewards to prevent advantage destabilization.
        # MiniRL (arxiv 2512.01374) recommends bounding reward signals.
        if clip_value > 0.0:
            rewards = [max(-clip_value, min(clip_value, r)) for r in rewards]

        return rewards

    return _reward_func
