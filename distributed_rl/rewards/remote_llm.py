"""Remote LLM reward model via OpenAI-compatible API.

=== What This Module Does ===

RemoteLLMReward calls a remote LLM served via an OpenAI-compatible API
(vLLM, TGI, etc.) to score translation quality.  Two modes:

  1. JUDGE MODE (LLM-as-a-Judge):
     Sends a chat completion prompt asking the LLM to evaluate the
     translation and rate it on a numeric scale.  The score is extracted
     from the text response via regex.

  2. SCORING MODE (custom scoring endpoint):
     The model directly returns a numeric score — the entire response
     content is parsed as a float.  Useful for fine-tuned scoring models
     served via an OpenAI-compatible endpoint.

=== Concurrency ===

The OpenAI SDK's chat.completions.create() is per-request (no native
batching).  For a batch of N samples, we fire N concurrent requests via
ThreadPoolExecutor.  This is I/O-bound work (waiting for HTTP responses),
so threads achieve true concurrency (GIL is released during socket I/O).

=== Network Latency and NCCL Timeout ===

This reward model runs only on rank 0.  During scoring, ranks 1-N are
idle waiting for TRL's internal broadcast.  The NCCL timeout (default:
60 min in DistributedConfig) must be large enough to cover the slowest
batch of API calls.

=== Score Output ===

Both modes output normalized scores in [0, 1] where HIGHER = BETTER.
This matches XComet's convention, so ``make_trl_reward_func()`` should
use ``offset=0.0`` (no inversion needed).

Ref: OpenAI Python SDK — https://github.com/openai/openai-python
"""

from __future__ import annotations

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from ..config import RemoteLLMRewardConfig
from .base import RewardModel, RewardOutput, SampleForScoring

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default prompt templates for judge mode
# ---------------------------------------------------------------------------

DEFAULT_JUDGE_SYSTEM_MESSAGE = (
    "You are an expert translation quality evaluator. "
    "You assess translations on a scale from 0 to 10, where "
    "0 is completely wrong and 10 is a perfect translation."
)

DEFAULT_JUDGE_PROMPT_TEMPLATE = """\
Evaluate the quality of the following translation.

Source text: {src}
Translation: {mt}
Reference translation: {ref}

Consider the following criteria:
- Accuracy: Does the translation convey the same meaning as the source?
- Fluency: Is the translation natural and grammatically correct?
- Adequacy: Is all important information preserved?

Provide a brief assessment (2-3 sentences), then give your final score.
You MUST end your response with exactly: Score: X
where X is a number from 0 to 10 (decimals allowed, e.g. Score: 7.5)."""


# ---------------------------------------------------------------------------
# RemoteLLMReward
# ---------------------------------------------------------------------------

class RemoteLLMReward(RewardModel):
    """Remote LLM translation quality scorer via OpenAI-compatible API.

    Args:
        cfg: RemoteLLMRewardConfig with API endpoint, mode, prompt, etc.
        predict_fn: Optional custom function for testing (bypasses API).
                    Signature: (samples: list[SampleForScoring]) -> list[float]
    """

    def __init__(
        self,
        cfg: RemoteLLMRewardConfig,
        predict_fn: Callable[..., list[float]] | None = None,
    ) -> None:
        self.cfg = cfg
        self._predict_fn = predict_fn
        self._client: Any = None
        self._executor: ThreadPoolExecutor | None = None
        self._score_pattern: re.Pattern[str] | None = None

        if predict_fn is not None or not cfg.enabled:
            return

        if OpenAI is None:
            raise RuntimeError(
                "RemoteLLMReward requires the 'openai' package. "
                "Install with: pip install openai"
            )

        # Resolve API key: direct value → env var → "EMPTY" (vLLM accepts any).
        api_key = cfg.api_key
        if not api_key and cfg.api_key_env:
            api_key = os.environ.get(cfg.api_key_env)
        if not api_key:
            api_key = "EMPTY"

        self._client = OpenAI(
            base_url=cfg.base_url,
            api_key=api_key,
            timeout=cfg.request_timeout_sec,
            max_retries=cfg.max_retries,
        )

        self._executor = ThreadPoolExecutor(
            max_workers=cfg.max_concurrent_requests,
        )

        if cfg.mode == "judge":
            self._score_pattern = re.compile(cfg.score_regex)

        self._system_message = cfg.judge_system_message or DEFAULT_JUDGE_SYSTEM_MESSAGE
        self._prompt_template = cfg.judge_prompt_template or DEFAULT_JUDGE_PROMPT_TEMPLATE

        logger.info(
            "RemoteLLMReward initialized: mode=%s base_url=%s model=%s "
            "max_concurrent=%d",
            cfg.mode, cfg.base_url, cfg.model, cfg.max_concurrent_requests,
        )

    # ------------------------------------------------------------------
    # RewardModel interface
    # ------------------------------------------------------------------

    def score(self, samples: list[SampleForScoring]) -> RewardOutput:
        """Score a batch of translations via the remote LLM API.

        Fires one API request per sample, concurrently via ThreadPoolExecutor.

        Returns:
            RewardOutput with normalized scores in [0, 1], higher = better.
        """
        if not samples:
            return RewardOutput(sequence_scores=[])

        # Testing mode: bypass API calls.
        if self._predict_fn is not None:
            scores = self._predict_fn(samples)
            return RewardOutput(sequence_scores=[float(s) for s in scores])

        if self._client is None:
            raise RuntimeError("RemoteLLMReward is not initialized.")

        # Fire concurrent requests.
        futures = {}
        for idx, sample in enumerate(samples):
            future = self._executor.submit(self._score_single, sample)
            futures[future] = idx

        # Collect results in original order.
        results: list[float] = [self.cfg.fallback_score] * len(samples)
        metadata_entries: list[dict[str, Any]] = [{}] * len(samples)

        for future in as_completed(futures):
            idx = futures[future]
            try:
                score_val, meta = future.result()
                results[idx] = score_val
                metadata_entries[idx] = meta
            except Exception as exc:
                logger.warning(
                    "RemoteLLMReward: request %d/%d failed: %s. "
                    "Using fallback score %.2f.",
                    idx + 1, len(samples), exc, self.cfg.fallback_score,
                )
                results[idx] = self.cfg.fallback_score

        return RewardOutput(
            sequence_scores=results,
            metadata={"responses": metadata_entries},
        )

    def close(self) -> None:
        """Release thread pool resources."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        self._client = None

    # ------------------------------------------------------------------
    # Internal scoring methods
    # ------------------------------------------------------------------

    def _score_single(
        self, sample: SampleForScoring,
    ) -> tuple[float, dict[str, Any]]:
        """Score a single sample.  Called from ThreadPoolExecutor."""
        if self.cfg.mode == "judge":
            return self._score_judge(sample)
        else:
            return self._score_direct(sample)

    def _score_judge(
        self, sample: SampleForScoring,
    ) -> tuple[float, dict[str, Any]]:
        """LLM-as-a-Judge mode: send prompt, parse score from response."""
        ref_text = sample.ref if sample.ref else "N/A"
        user_content = self._prompt_template.format(
            src=sample.src, mt=sample.mt, ref=ref_text,
        )

        response = self._client.chat.completions.create(
            model=self.cfg.model,
            messages=[
                {"role": "system", "content": self._system_message},
                {"role": "user", "content": user_content},
            ],
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )

        text = response.choices[0].message.content or ""
        raw_score = self._parse_score(text)
        normalized = self._normalize_score(raw_score)

        return normalized, {"raw_text": text, "raw_score": raw_score}

    def _score_direct(
        self, sample: SampleForScoring,
    ) -> tuple[float, dict[str, Any]]:
        """Scoring mode: model returns numeric score directly."""
        user_content = f"Source: {sample.src}\nTranslation: {sample.mt}"
        if sample.ref:
            user_content += f"\nReference: {sample.ref}"

        response = self._client.chat.completions.create(
            model=self.cfg.model,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.0,
            max_tokens=32,
        )

        text = (response.choices[0].message.content or "").strip()
        try:
            raw_score = float(text)
        except ValueError:
            # Try to extract any number from the response.
            numbers = re.findall(r"[\d]+(?:\.[\d]+)?", text)
            if numbers:
                raw_score = float(numbers[0])
            else:
                logger.warning(
                    "RemoteLLMReward scoring mode: could not parse score "
                    "from response: %r. Using fallback.",
                    text[:100],
                )
                return self.cfg.fallback_score, {
                    "raw_text": text, "parse_error": True,
                }

        normalized = self._normalize_score(raw_score)
        return normalized, {"raw_text": text, "raw_score": raw_score}

    # ------------------------------------------------------------------
    # Score parsing and normalization
    # ------------------------------------------------------------------

    def _parse_score(self, text: str) -> float:
        """Extract a numeric score from judge LLM response text.

        Uses the configured regex.  Falls back to progressively more
        lenient patterns if the primary regex fails.

        Returns:
            The raw (un-normalized) score.

        Raises:
            ValueError: If no score can be extracted at all.
        """
        # Primary: use the configured regex.
        if self._score_pattern is not None:
            match = self._score_pattern.search(text)
            if match:
                return float(match.group(1))

        # Fallback 1: look for "X/10" pattern.
        match = re.search(r"([\d]+(?:\.[\d]+)?)\s*/\s*10", text)
        if match:
            return float(match.group(1))

        # Fallback 2: last standalone number in plausible range.
        numbers = re.findall(r"\b([\d]+(?:\.[\d]+)?)\b", text)
        if numbers:
            candidate = float(numbers[-1])
            if self.cfg.score_min <= candidate <= self.cfg.score_max:
                return candidate

        raise ValueError(f"Could not parse score from: {text[:200]!r}")

    def _normalize_score(self, raw_score: float) -> float:
        """Normalize raw score to [0, 1] via linear mapping, clamped."""
        score_range = self.cfg.score_max - self.cfg.score_min
        if score_range <= 0:
            return max(0.0, min(1.0, raw_score))
        normalized = (raw_score - self.cfg.score_min) / score_range
        return max(0.0, min(1.0, normalized))
