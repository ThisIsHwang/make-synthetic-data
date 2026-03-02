"""MetricX reward model with subprocess GPU isolation.

=== What This Module Does ===

MetricXReward wraps the MetricX quality estimation model (an MT5-based
regression model) into our RewardModel interface.  It supports two modes:

  1. SUBPROCESS MODE (production):
     When ``python_executable`` is set in the config, the model runs in a
     separate Python process on a dedicated GPU (e.g. GPU 6).  Communication
     happens via JSON over stdin/stdout (see subprocess_client.py).

     This is the mode used during actual training because:
     - The training process can only see GPUs 0-5 (CUDA_VISIBLE_DEVICES)
     - The MetricX subprocess can see ALL GPUs (env var removed)
     - MetricX may need a different Python environment (dependency conflicts)

  2. IN-PROCESS MODE (testing/debugging):
     When ``python_executable`` is None, the model loads directly in the
     current process.  Used for unit testing and single-GPU debugging.

=== MetricX Input Formats ===

QE mode (use_reference=False):
  "source: Hello world candidate: 안녕하세요"

Reference mode (use_reference=True):
  "source: Hello world candidate: 안녕하세요 reference: 안녕하세요"

=== Score → Reward Conversion ===

MetricX outputs scores where LOWER = BETTER (0=perfect, 25=worst).
The conversion to RL reward (higher=better) is done in make_trl_reward_func()
with the offset parameter:
  reward = weight * (offset - metricx_score)

This module just returns the raw MetricX scores.

Ref: Adapted from qwen3.5-35b-a3b/qwen35_moe_rl/rewards.py:140-543
     (MetricXQEScorer class)
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..config import MetricXConfig
from .base import RewardModel, RewardOutput, SampleForScoring
from .subprocess_client import ScorerSubprocessClient

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover
    AutoTokenizer = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MetricX input formatting helpers
# ---------------------------------------------------------------------------

def metricx_qe_input(src: str, mt: str) -> str:
    """Format input for MetricX Quality Estimation (no reference).

    This is the standard input format when we don't have a reference translation.
    MetricX was trained to predict quality scores from this exact string format.
    """
    return f"source: {src} candidate: {mt}"


def metricx_ref_input(src: str, mt: str, ref: str) -> str:
    """Format input for MetricX with reference translation.

    More accurate than QE mode because the model can compare the candidate
    against a known-good reference, but requires reference translations.
    """
    return f"source: {src} candidate: {mt} reference: {ref}"


def metricx_score_to_reward(metricx_score: float, offset: float = 5.0) -> float:
    """Convert MetricX score (lower=better) to reward (higher=better).

    Example:
      metricx_score=0.5, offset=5.0 → reward=4.5 (good translation)
      metricx_score=3.0, offset=5.0 → reward=2.0 (mediocre translation)
      metricx_score=10.0, offset=5.0 → reward=-5.0 (bad translation)

    Note: This function is provided as a standalone utility.
    In the actual training pipeline, the conversion is handled by
    make_trl_reward_func(offset=5.0) in base.py.
    """
    return float(offset) - float(metricx_score)


# ---------------------------------------------------------------------------
# MetricX Reward Model
# ---------------------------------------------------------------------------

class MetricXReward(RewardModel):
    """MetricX translation quality scorer.

    Operates in one of two modes:

    1. **Subprocess** (when ``python_executable`` is set):
       Launches a separate process with ``CUDA_VISIBLE_DEVICES`` cleared so it
       can place the model on a dedicated GPU (e.g. GPU 6).

       Flow:
         MetricXReward.score(samples)
           → ScorerSubprocessClient.request({"type": "score", "samples": [...]})
             → scorer_worker.py receives the request
               → scorer_worker._init_metricx() loaded MetricXReward in-process
               → MetricXReward._predict_scores() runs MT5 inference on GPU 6
             ← {"ok": true, "scores": [0.5, 1.2, ...]}
           ← RewardOutput(sequence_scores=[0.5, 1.2, ...])

    2. **In-process** (when ``python_executable`` is None):
       Loads the model directly in the current process.
       Used when this class is instantiated inside scorer_worker.py
       (the subprocess itself) or for direct testing.

    Args:
        cfg: MetricXConfig with model name, GPU ID, batch size, etc.
        predict_fn: Optional custom prediction function (for testing only).
    """

    def __init__(self, cfg: MetricXConfig, predict_fn: Callable[..., list[float]] | None = None) -> None:
        self.cfg = cfg
        self._predict_fn = predict_fn
        self._worker: ScorerSubprocessClient | None = None
        self._model: Any = None
        self._tokenizer: Any = None
        self._dtype: Any = None
        self._device: str = f"cuda:{cfg.gpu_id}"

        # If a custom prediction function is provided (testing), or if the
        # model is disabled, skip initialization entirely.
        if predict_fn is not None or not cfg.enabled:
            return

        # --- Mode 1: Subprocess ---
        # Spawn a separate Python process for GPU-isolated inference.
        if cfg.python_executable:
            self._worker = ScorerSubprocessClient(
                backend="metricx",
                python_executable=cfg.python_executable,
                timeout_sec=cfg.subprocess_timeout_sec,
                config_payload={
                    "model_name": cfg.model_name,
                    "tokenizer_name": cfg.tokenizer_name,
                    "use_reference": bool(cfg.use_reference),
                    "batch_size": int(cfg.batch_size),
                    "gpu_id": int(cfg.gpu_id),
                    "dtype": cfg.dtype,
                    "max_input_length": int(cfg.max_input_length),
                    "overflow_policy": cfg.overflow_policy,
                },
            )
            logger.info("MetricX scorer → subprocess (python=%s gpu=%d)", cfg.python_executable, cfg.gpu_id)
            return

        # --- Mode 2: In-process ---
        # Load the model directly (used inside scorer_worker.py or for testing).
        if torch is None or AutoTokenizer is None:
            raise RuntimeError("MetricX in-process mode requires torch + transformers.")

        from .metricx_model import MT5ForRegression

        self._dtype = _resolve_dtype(cfg.dtype)
        # MetricX uses the MT5-XL tokenizer regardless of model size.
        self._tokenizer = AutoTokenizer.from_pretrained(
            cfg.tokenizer_name or "google/mt5-xl", use_fast=False
        )

        # Load the MT5ForRegression model from a HuggingFace checkpoint or local path.
        model_source = self._resolve_model_source(cfg.model_name)
        kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
        if self._dtype is not None:
            kwargs["dtype"] = self._dtype
        try:
            self._model = MT5ForRegression.from_pretrained(model_source, **kwargs)
        except TypeError:
            # Some older transformers versions don't accept "dtype" directly.
            kwargs.pop("low_cpu_mem_usage", None)
            if "dtype" in kwargs:
                kwargs["torch_dtype"] = kwargs.pop("dtype")
            self._model = MT5ForRegression.from_pretrained(model_source, **kwargs)

        # Place on the dedicated GPU (e.g. cuda:6).
        self._model.to(self._device)
        self._model.eval()  # Set to inference mode (no dropout, etc.)
        logger.info("MetricX loaded in-process (device=%s dtype=%s)", self._device, self._dtype)

    def score(self, samples: list[SampleForScoring]) -> RewardOutput:
        """Score a batch of translations.

        Dispatches to the appropriate mode (subprocess, predict_fn, or in-process)
        based on how the model was initialized.

        Args:
            samples: List of (source, translation, optional reference) tuples.

        Returns:
            RewardOutput with raw MetricX scores (lower=better, [0, 25] range).
        """
        if not samples:
            return RewardOutput(sequence_scores=[])

        # --- Subprocess mode ---
        # Send samples to the worker process and get scores back.
        if self._worker is not None:
            rows = [{"src": s.src, "mt": s.mt, "ref": s.ref} for s in samples]
            resp = self._worker.request({"type": "score", "samples": rows})
            if not resp.get("ok", False):
                raise RuntimeError(f"MetricX subprocess error: {resp.get('error', 'unknown')}")
            return RewardOutput(
                sequence_scores=[float(v) for v in resp["scores"]],
                metadata=resp.get("metadata"),
            )

        # --- Predict function mode (testing) ---
        inputs = self._build_inputs(samples)
        if self._predict_fn is not None:
            scores = [float(v) for v in self._predict_fn(inputs)]
            return RewardOutput(sequence_scores=scores)

        # --- In-process mode ---
        # Run inference in batches to avoid OOM on large sample sets.
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("MetricXReward is not initialized.")

        all_scores: list[float] = []
        for i in range(0, len(inputs), self.cfg.batch_size):
            batch = inputs[i: i + self.cfg.batch_size]
            all_scores.extend(self._predict_scores(batch))
        return RewardOutput(sequence_scores=all_scores)

    def _build_inputs(self, samples: list[SampleForScoring]) -> list[str]:
        """Format samples into MetricX input strings.

        Chooses between QE format ("source: X candidate: Y") and reference
        format ("source: X candidate: Y reference: Z") based on config.
        """
        inputs: list[str] = []
        for s in samples:
            ref = (s.ref or "").strip()
            if self.cfg.use_reference and ref:
                inputs.append(metricx_ref_input(s.src, s.mt, ref))
            else:
                inputs.append(metricx_qe_input(s.src, s.mt))
        return inputs

    def _predict_scores(self, batch_inputs: list[str]) -> list[float]:
        """Run MT5ForRegression inference on a batch of formatted inputs.

        Steps:
          1. Tokenize the input strings (with padding and truncation)
          2. Move tensors to the GPU
          3. Run forward pass (no gradient computation — inference only)
          4. Extract the prediction values
          5. Check for NaN/Inf (numerical instability warning)
          6. Return as a list of Python floats
        """
        tokenized = self._tokenizer(
            batch_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_input_length,
        )
        input_ids = tokenized["input_ids"].to(self._device)
        attention_mask = tokenized["attention_mask"].to(self._device)

        # No gradient computation needed for reward scoring (saves memory).
        with torch.no_grad():
            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
        preds = outputs.predictions
        # Check for numerical issues — NaN or Inf predictions indicate a problem
        # with the model or input (e.g. extremely long input that causes overflow).
        if not torch.isfinite(preds).all():
            bad = int((~torch.isfinite(preds)).sum().item())
            logger.warning("MetricX produced %d non-finite predictions", bad)
        return [float(v) for v in preds.detach().float().cpu().tolist()]

    @staticmethod
    def _resolve_model_source(model_name: str) -> str:
        """Resolve model name to a local path or HuggingFace download.

        If the model_name is a local path that exists, use it directly.
        Otherwise, download from HuggingFace Hub using snapshot_download.
        Falls back to the raw name (letting from_pretrained handle it).
        """
        text = model_name.strip()
        path = Path(text).expanduser()
        if path.exists():
            return str(path)
        try:
            from huggingface_hub import snapshot_download

            cache_dir = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
            local_path = snapshot_download(repo_id=text, cache_dir=cache_dir, max_workers=1)
            return str(local_path)
        except Exception:
            return text

    def close(self) -> None:
        """Release GPU memory and subprocess resources."""
        if self._worker is not None:
            self._worker.close()
            self._worker = None
        self._model = None
        self._tokenizer = None


def _resolve_dtype(dtype_str: str) -> Any:
    """Convert a dtype string to a PyTorch dtype object.

    Supports common aliases like "bf16" → torch.bfloat16.
    """
    if torch is None:
        return None
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(dtype_str.lower(), torch.bfloat16)
