"""Subprocess worker for reward model inference.

=== Purpose ===

This script runs in a SEPARATE Python process from the training loop.
It is launched by ``ScorerSubprocessClient`` (from the training process)
and handles the "server" side of the JSON-RPC protocol.

The worker:
  1. Receives an "init" message → loads the reward model onto a dedicated GPU
  2. Receives "score" messages → runs inference and returns scores
  3. Receives a "close" message → exits cleanly

=== Why a Separate Process? ===

  1. GPU isolation: The training process sees GPUs 0-5 (via CUDA_VISIBLE_DEVICES).
     This subprocess sees ALL GPUs (env var removed by the client) and can
     place the reward model on GPU 6 or 7.

  2. Dependency isolation: MetricX uses a different Python environment
     (``python_executable`` in config) that may have incompatible packages.
     Running it in a separate process with a separate venv avoids conflicts.

  3. Memory isolation: The reward model's GPU memory is completely separate
     from the training process.  No risk of OOM interference.

=== Protocol ===

Communication is via JSON lines on stdin/stdout:

  → {"type": "init", "config": {"model_name": "...", "gpu_id": 6, ...}}
  ← {"ok": true}

  → {"type": "score", "samples": [{"src": "Hello", "mt": "안녕", "ref": null}, ...]}
  ← {"ok": true, "scores": [0.5, 1.2, ...]}

  → {"type": "close"}
  ← {"ok": true}
  (Process exits)

Errors are returned as:
  ← {"ok": false, "error": "traceback..."}

=== Extensibility ===

To add a new reward model backend:
  1. Write an ``_init_mymodel(cfg_payload)`` function that returns a RewardModel
  2. Add ``"mymodel": _init_mymodel`` to the ``BACKEND_INIT`` dict
  3. The rest of the protocol (score, close) works automatically because
     all reward models implement the same ``RewardModel.score()`` interface

Ref: Generalized from qwen3.5-35b-a3b/qwen35_moe_rl/scorer_worker.py
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any


def _ensure_import_path() -> None:
    """Add distributed_rl package root to sys.path for standalone execution.

    This worker is executed as a standalone script by ScorerSubprocessClient:
      /path/to/python scorer_worker.py --backend metricx

    It needs to import from ``distributed_rl``, so we add the package's
    grandparent directory (the project root) to sys.path.
    """
    pkg_root = Path(__file__).resolve().parents[2]
    pkg_root_text = str(pkg_root)
    if pkg_root_text not in sys.path:
        sys.path.insert(0, pkg_root_text)


# Must run before any distributed_rl imports.
_ensure_import_path()


def _reply(payload: dict[str, Any]) -> None:
    """Send a JSON response back to the client via stdout.

    Important: stdout is the ONLY communication channel back to the client.
    Any print() or logging to stdout would corrupt the protocol.
    Worker logging should go to stderr (which is inherited from the parent).
    """
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _to_samples(rows: list[dict[str, Any]]) -> list[Any]:
    """Convert raw JSON dicts to SampleForScoring objects.

    The client sends samples as plain dicts:
      {"src": "Hello", "mt": "안녕", "ref": null}

    We convert them to the typed SampleForScoring dataclass that all
    reward models expect.
    """
    from distributed_rl.rewards.base import SampleForScoring

    return [
        SampleForScoring(
            src=str(row.get("src", "")),
            mt=str(row.get("mt", "")),
            ref=(None if row.get("ref") is None else str(row["ref"])),
        )
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Backend initializers — one per reward model type
# ---------------------------------------------------------------------------

def _init_metricx(cfg_payload: dict[str, Any]) -> Any:
    """Initialize MetricX reward model from config payload.

    Sets ``python_executable=None`` to prevent the MetricXReward from
    spawning ANOTHER subprocess (we're already in the subprocess!).
    This forces in-process mode: the model loads directly in this process.
    """
    from distributed_rl.rewards.metricx import MetricXReward
    from distributed_rl.config import MetricXConfig

    cfg_payload = dict(cfg_payload)
    cfg_payload["python_executable"] = None  # Force in-process mode
    cfg = MetricXConfig(**cfg_payload)
    return MetricXReward(cfg)


def _init_xcomet(cfg_payload: dict[str, Any]) -> Any:
    """Initialize XComet reward model from config payload.

    Same pattern as MetricX: force in-process mode to avoid recursive subprocess.
    """
    from distributed_rl.rewards.xcomet import XCometReward
    from distributed_rl.config import XCometConfig

    cfg_payload = dict(cfg_payload)
    cfg_payload["python_executable"] = None  # Force in-process mode
    cfg = XCometConfig(**cfg_payload)
    return XCometReward(cfg)


# Registry of supported backends.
# To add a new reward model, just add an entry here.
BACKEND_INIT = {
    "metricx": _init_metricx,
    "xcomet": _init_xcomet,
}


# ---------------------------------------------------------------------------
# Main loop — reads JSON from stdin, dispatches to the appropriate handler
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reward scorer worker")
    parser.add_argument("--backend", required=True, choices=list(BACKEND_INIT.keys()))
    args = parser.parse_args(argv)

    scorer: Any = None  # Will be set by "init" message

    # Read JSON messages from stdin, one per line.
    # The client (ScorerSubprocessClient) writes to our stdin.
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if not isinstance(req, dict):
                raise ValueError("request must be a JSON object")

            req_type = str(req.get("type", "")).strip().lower()

            # --- CLOSE: graceful shutdown ---
            if req_type == "close":
                _reply({"ok": True})
                return 0

            # --- INIT: load the reward model ---
            # The client sends the model config (model_name, gpu_id, etc.)
            # and we use the appropriate backend initializer to load it.
            if req_type == "init":
                cfg_payload = req.get("config") or {}
                if not isinstance(cfg_payload, dict):
                    raise ValueError("init.config must be an object")
                init_fn = BACKEND_INIT[args.backend]
                scorer = init_fn(cfg_payload)
                _reply({"ok": True})
                continue

            # All subsequent requests require an initialized scorer.
            if scorer is None:
                raise RuntimeError("Worker not initialized. Send init first.")

            # --- SCORE: run reward model inference ---
            # The client sends a list of samples (src, mt, ref).
            # We run inference and return the scores.
            if req_type == "score":
                sample_rows = req.get("samples") or req.get("payload") or []
                if not isinstance(sample_rows, list):
                    raise ValueError("score.samples must be a list")
                # Convert raw dicts to SampleForScoring objects.
                samples = _to_samples(sample_rows)
                # Call the reward model's score() method.
                output = scorer.score(samples)
                # Build response with scores.
                resp: dict[str, Any] = {
                    "ok": True,
                    "scores": output.sequence_scores,
                }
                # Include metadata if available (e.g. XComet error spans).
                if output.metadata is not None:
                    resp["metadata"] = output.metadata
                    if isinstance(output.metadata, dict) and "error_spans" in output.metadata:
                        resp["error_spans"] = output.metadata["error_spans"]
                _reply(resp)
                continue

            raise ValueError(f"unsupported request type: {req_type}")

        except Exception:
            # Any error is caught and sent back as a JSON error response.
            # The worker keeps running — it doesn't crash on a single bad request.
            _reply({
                "ok": False,
                "error": traceback.format_exc(limit=4),
            })

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
