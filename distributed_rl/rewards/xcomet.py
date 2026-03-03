"""XComet reward model with subprocess GPU isolation.

=== What is XComet? ===

XComet (XCOMET-XL by Unbabel) is a learned translation quality metric.
Unlike MetricX (which uses MT5), XComet is based on XLM-RoBERTa with
additional estimation heads fine-tuned on human quality judgments.

Key difference from MetricX:
  - XComet outputs scores in [0, 1] where HIGHER = BETTER
  - No score inversion needed for RL (reward = weight * xcomet_score)
  - Can optionally output error spans (which tokens are wrong)

=== Subprocess Architecture ===

Like MetricXReward, XCometReward supports two modes:

  1. SUBPROCESS mode: Runs in a separate Python process on a dedicated GPU
     (e.g. GPU 7).  Uses the ``unbabel-comet`` package which may have
     different dependency requirements than the training environment.

  2. IN-PROCESS mode: Loads directly.  Used inside scorer_worker.py
     or for testing.

=== XComet API ===

XComet uses the ``unbabel-comet`` library:
  from comet import download_model, load_from_checkpoint
  model = load_from_checkpoint(download_model("Unbabel/XCOMET-XL"))
  model.predict([{"src": "hello", "mt": "안녕"}])

The input is a list of dicts with "src", "mt", and optionally "ref" keys.
The output contains a "scores" field with one score per sample.

Ref: Adapted from qwen3.5-35b-a3b/qwen35_moe_rl/rewards.py:547-738
     (XCometXLScorer class)
Ref: XComet paper — https://arxiv.org/abs/2310.10482
Ref: unbabel-comet — https://github.com/Unbabel/COMET
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from ..config import XCometConfig
from .base import RewardModel, RewardOutput, SampleForScoring
from .subprocess_client import ScorerSubprocessClient

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class XCometReward(RewardModel):
    """XComet translation quality scorer.

    Returns sequence-level scores in [0, 1] where higher = better.

    Initialization modes:
      1. Subprocess mode (python_executable set): Spawns a separate process
         on a dedicated GPU.  Communicates via JSON-RPC.
      2. In-process mode (python_executable=None): Loads the COMET model
         directly using ``unbabel-comet``.

    Args:
        cfg: XCometConfig with model name, GPU ID, batch size, etc.
        predict_fn: Optional custom prediction function (for testing only).
    """

    def __init__(self, cfg: XCometConfig, predict_fn: Callable[..., Any] | None = None) -> None:
        self.cfg = cfg
        self._predict_fn = predict_fn
        self._worker: ScorerSubprocessClient | None = None
        self._model: Any = None
        self._device: str = f"cuda:{cfg.gpu_id}"

        if predict_fn is not None or not cfg.enabled:
            return

        # --- Mode 1: Subprocess ---
        if cfg.python_executable:
            self._worker = ScorerSubprocessClient(
                backend="xcomet",
                python_executable=cfg.python_executable,
                timeout_sec=cfg.subprocess_timeout_sec,
                config_payload={
                    "model_name": cfg.model_name,
                    "batch_size": int(cfg.batch_size),
                    "gpu_id": int(cfg.gpu_id),
                    "use_reference": bool(cfg.use_reference),
                },
            )
            logger.info("XComet scorer → subprocess (python=%s gpu=%d)", cfg.python_executable, cfg.gpu_id)
            return

        # --- Mode 2: In-process ---
        # Requires the ``unbabel-comet`` package (pip install unbabel-comet).
        import sys
        logger.warning(
            "[XComet diag] executable=%s  prefix=%s  path=%s",
            sys.executable, sys.prefix, sys.path,
        )
        # Diagnose pkg_resources / setuptools availability
        try:
            import setuptools
            logger.warning("[XComet diag] setuptools=%s loc=%s", setuptools.__version__, setuptools.__file__)
        except Exception as e:
            logger.warning("[XComet diag] setuptools import FAILED: %s", e)
        try:
            import pkg_resources
            logger.warning("[XComet diag] pkg_resources OK loc=%s", pkg_resources.__file__)
        except Exception as e:
            logger.warning("[XComet diag] pkg_resources import FAILED: %s", e)
        try:
            import pytorch_lightning
            logger.warning("[XComet diag] pytorch_lightning=%s loc=%s", pytorch_lightning.__version__, pytorch_lightning.__file__)
        except Exception as e:
            logger.warning("[XComet diag] pytorch_lightning import FAILED: %s", e)
        try:
            from comet import download_model, load_from_checkpoint
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "XCometReward requires unbabel-comet>=2.2.0."
            ) from exc

        # download_model() downloads the model weights from HuggingFace Hub
        # and returns a local path to the checkpoint.
        model_path = download_model(cfg.model_name)
        # load_from_checkpoint() loads the model and returns a CometModel instance.
        self._model = load_from_checkpoint(model_path)
        if self._device.startswith("cuda") and torch is not None:
            self._model.to(torch.device(self._device))
        self._model.eval()
        logger.info("XComet loaded in-process (device=%s)", self._device)

    def score(self, samples: list[SampleForScoring]) -> RewardOutput:
        """Score a batch of translations.

        Args:
            samples: List of (source, translation, optional reference) tuples.

        Returns:
            RewardOutput with XComet scores (higher=better, [0, 1] range).
            Metadata includes ``error_spans`` if available.
        """
        if not samples:
            return RewardOutput(sequence_scores=[], metadata={"error_spans": []})

        # --- Subprocess mode ---
        if self._worker is not None:
            rows = [{"src": s.src, "mt": s.mt, "ref": s.ref} for s in samples]
            # Note: XComet uses "payload" key (not "samples") for historical compatibility.
            resp = self._worker.request({"type": "score", "payload": rows})
            if not resp.get("ok", False):
                raise RuntimeError(f"XComet subprocess error: {resp.get('error', 'unknown')}")
            return RewardOutput(
                sequence_scores=[float(v) for v in resp["scores"]],
                metadata={"error_spans": resp.get("error_spans", [[] for _ in samples])},
            )

        # --- Predict function mode (testing) ---
        payload = self._build_payload(samples)
        if self._predict_fn is not None:
            result = self._predict_fn(payload)
            scores = _extract_scores(result, expected=len(samples))
            return RewardOutput(sequence_scores=scores)

        # --- In-process mode ---
        if self._model is None:
            raise RuntimeError("XCometReward is not initialized.")

        # Process in batches to control GPU memory usage.
        all_scores: list[float] = []
        for i in range(0, len(payload), self.cfg.batch_size):
            batch = payload[i: i + self.cfg.batch_size]
            # prepare_for_inference() tokenizes and batches the inputs
            # in the format COMET expects.
            batch_inputs = self._model.prepare_for_inference(batch)
            # Move all tensors to the GPU.
            batch_inputs = _move_to_device(batch_inputs, self._device)
            with torch.no_grad():
                # predict_step() runs the model and returns scores.
                pred = self._model.predict_step(batch_inputs)
            # Extract the "scores" field from the prediction output.
            scores_tensor = pred.get("scores") if isinstance(pred, dict) else getattr(pred, "scores", None)
            if scores_tensor is None:
                raise ValueError("XComet predict_step returned no scores.")
            batch_scores = torch.as_tensor(scores_tensor).detach().float().cpu().tolist()
            all_scores.extend(float(v) for v in batch_scores)

        return RewardOutput(sequence_scores=all_scores)

    def _build_payload(self, samples: list[SampleForScoring]) -> list[dict[str, str]]:
        """Build the input payload in COMET's expected format.

        COMET expects a list of dicts with keys "src", "mt", and optionally "ref":
          [{"src": "Hello", "mt": "안녕", "ref": "안녕하세요"}, ...]

        The "ref" key is only included if use_reference=True and a reference
        translation is available.
        """
        payload: list[dict[str, str]] = []
        for s in samples:
            record: dict[str, str] = {"src": s.src, "mt": s.mt}
            if self.cfg.use_reference and s.ref:
                record["ref"] = s.ref
            payload.append(record)
        return payload

    def close(self) -> None:
        """Release GPU memory and subprocess resources."""
        if self._worker is not None:
            self._worker.close()
            self._worker = None
        self._model = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_scores(result: Any, expected: int) -> list[float]:
    """Extract scores from COMET's prediction output.

    COMET's predict_step() can return scores in different formats depending
    on the model version:
      - An object with a ``.scores`` attribute
      - A dict with a ``"scores"`` key
      - A plain list/tuple of floats

    This function handles all three cases.
    """
    if hasattr(result, "scores"):
        return [float(v) for v in list(result.scores)][:expected]
    if isinstance(result, dict) and "scores" in result:
        return [float(v) for v in list(result["scores"])][:expected]
    if isinstance(result, (list, tuple)):
        return [float(v) for v in result][:expected]
    raise ValueError("Unsupported XComet prediction output format.")


def _move_to_device(obj: Any, device: str) -> Any:
    """Recursively move tensors in a nested structure to the specified device.

    COMET's prepare_for_inference() returns a nested dict/list of tensors.
    We need all tensors on the same GPU for inference.

    Example:
      {"input_ids": tensor_on_cpu, "attention_mask": tensor_on_cpu}
      → {"input_ids": tensor_on_gpu, "attention_mask": tensor_on_gpu}
    """
    if torch is None:
        return obj
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_to_device(v, device) for v in obj)
    return obj
