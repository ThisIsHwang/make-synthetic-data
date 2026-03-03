"""Generic subprocess client for reward model GPU isolation.

=== Why Subprocess Isolation? ===

The training process runs under ``torchrun`` with:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --nproc_per_node=6 ...

This means the training process can ONLY see GPUs 0-5.  But the reward
models need GPUs 6 and 7.  We can't just do ``torch.device("cuda:6")``
from inside the training process — GPU 6 doesn't exist in its CUDA context.

Solution: Spawn a SEPARATE Python process with CUDA_VISIBLE_DEVICES removed.
This subprocess can see ALL 8 GPUs and place the reward model on GPU 6 or 7.
Communication happens via JSON messages over stdin/stdout.

  Training process (GPU 0-5)     Subprocess (can see GPU 0-7)
  ┌────────────────────────┐     ┌──────────────────────────┐
  │  rank 0 training       │     │  MetricX model on GPU 6  │
  │                        │────→│  (reads JSON from stdin)  │
  │  Sends {"type":"score",│     │  Runs inference           │
  │    "samples": [...]}   │←────│  Returns {"scores": [...]}│
  └────────────────────────┘     └──────────────────────────┘
          stdin/stdout pipe (JSON lines)

=== JSON-RPC Protocol ===

The protocol uses JSON lines (one JSON object per line) for simplicity:

  Client → Worker:  {"type": "init",  "config": {...}}
  Worker → Client:  {"ok": true}

  Client → Worker:  {"type": "score", "samples": [{"src": ..., "mt": ...}, ...]}
  Worker → Client:  {"ok": true, "scores": [0.5, 1.2, ...]}

  Client → Worker:  {"type": "close"}
  Worker → Client:  {"ok": true}
  (Worker exits)

=== Timeout Handling ===

Reward model inference can be slow (especially for large batches with long texts).
We use ``select.select()`` to wait for the subprocess's stdout with a timeout.
If the subprocess doesn't respond within ``timeout_sec``, we raise TimeoutError
rather than hanging forever.

Ref: Extracted and generalized from qwen3.5-35b-a3b/qwen35_moe_rl/rewards.py:31-137
     (_ScorerSubprocessClient class)
"""

from __future__ import annotations

import json
import logging
import os
import select
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ScorerSubprocessClient:
    """JSON-RPC-like client that talks to ``scorer_worker.py``.

    Lifecycle:
      1. ``__init__`` spawns the worker subprocess and sends an "init" message
         with the model configuration (model name, GPU ID, etc.)
      2. ``request()`` sends a JSON message and waits for a JSON response
      3. ``close()`` sends a "close" message and terminates the subprocess

    The subprocess runs ``scorer_worker.py`` which loads the reward model
    and processes scoring requests in a loop.
    """

    def __init__(
        self,
        *,
        backend: str,             # "metricx" or "xcomet"
        python_executable: str,   # Path to Python in the reward model's venv
        timeout_sec: float,       # Max seconds to wait for a response
        config_payload: dict[str, Any],  # Model config sent to the worker
    ) -> None:
        self._backend = backend
        self._timeout_sec = float(timeout_sec)

        # The worker script lives in the same directory as this file.
        worker_script = Path(__file__).resolve().with_name("scorer_worker.py")
        if not worker_script.exists():
            raise FileNotFoundError(f"scorer worker script not found: {worker_script}")

        cmd = [python_executable, str(worker_script), "--backend", backend]

        # --- CRITICAL: Clean subprocess environment ---
        # 1. Remove CUDA_VISIBLE_DEVICES so the subprocess can see ALL GPUs.
        #    The parent (torchrun) restricts to GPUs 0-5, but the reward model
        #    needs GPU 6 or 7.
        # 2. Remove VIRTUAL_ENV / PYTHONHOME so the subprocess uses its own
        #    venv's site-packages (not the parent training venv's).
        env = os.environ.copy()
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.pop("VIRTUAL_ENV", None)
        env.pop("PYTHONHOME", None)

        # Spawn the subprocess with stdin/stdout pipes for JSON communication.
        # stderr is inherited (goes to the same terminal/log as the parent).
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,   # We write JSON to the worker
                stdout=subprocess.PIPE,  # Worker writes JSON responses back
                stderr=None,             # Inherit stderr for error logs
                text=True,               # Use text mode (not binary)
                bufsize=1,               # Line-buffered (important for JSON lines)
                env=env,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to start {backend} scorer worker: cmd={cmd}"
            ) from exc

        # Send the init message — worker loads the model and responds.
        # This can take minutes for large models (downloading + GPU loading).
        try:
            init_resp = self.request({"type": "init", "config": config_payload})
        except Exception:
            self.close()
            raise
        if not bool(init_resp.get("ok", False)):
            self.close()
            raise RuntimeError(
                f"{backend} scorer worker init failed: {init_resp.get('error', 'unknown error')}"
            )

    def _assert_alive(self) -> None:
        """Check that the subprocess hasn't crashed."""
        if self._proc.poll() is not None:
            raise RuntimeError(
                f"{self._backend} scorer worker exited unexpectedly "
                f"with code={self._proc.returncode}"
            )

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON request and wait for a JSON response.

        This is the core communication method.  Each call:
          1. Writes one line of JSON to the subprocess's stdin
          2. Waits for one line of JSON from the subprocess's stdout
          3. Returns the parsed JSON response

        Uses ``select.select()`` for timeout — if the worker doesn't respond
        within ``timeout_sec``, raises TimeoutError instead of hanging.
        """
        self._assert_alive()
        assert self._proc.stdin is not None
        assert self._proc.stdout is not None

        # --- Send request ---
        try:
            self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()  # Ensure the data is actually sent
        except Exception as exc:
            raise RuntimeError(
                f"Failed to send request to {self._backend} scorer worker."
            ) from exc

        # --- Wait for response with timeout ---
        # select.select() returns when stdout has data ready to read,
        # or when the timeout expires.  This prevents indefinite blocking
        # if the worker crashes or hangs during inference.
        ready, _, _ = select.select([self._proc.stdout], [], [], self._timeout_sec)
        if not ready:
            raise TimeoutError(
                f"{self._backend} scorer worker timed out after {self._timeout_sec}s"
            )

        # --- Read response ---
        try:
            line = self._proc.stdout.readline()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read response from {self._backend} scorer worker."
            ) from exc

        if not line:
            # Empty read means the subprocess closed its stdout (crashed).
            self._assert_alive()
            raise RuntimeError(
                f"{self._backend} scorer worker returned empty response."
            )

        # --- Parse JSON ---
        try:
            resp = json.loads(line)
        except Exception as exc:
            raise RuntimeError(
                f"Invalid JSON from {self._backend} scorer worker: {line[:200]!r}"
            ) from exc

        if not isinstance(resp, dict):
            raise RuntimeError(
                f"Unexpected response type from {self._backend} worker: {type(resp)!r}"
            )
        return resp

    def close(self) -> None:
        """Gracefully shut down the subprocess.

        Sends a "close" message first (so the worker can release GPU memory),
        then terminates the process.  Falls back to kill() if terminate() hangs.
        """
        if getattr(self, "_proc", None) is None:
            return
        proc = self._proc
        if proc.poll() is None:  # Still running
            # Try graceful shutdown first.
            try:
                self.request({"type": "close"})
            except Exception:
                pass
            # Then terminate (SIGTERM).
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                # If terminate doesn't work, force kill (SIGKILL).
                try:
                    proc.kill()
                except Exception:
                    pass
        self._proc = None  # type: ignore[assignment]

    def __del__(self) -> None:  # pragma: no cover
        """Ensure subprocess is cleaned up on garbage collection."""
        try:
            self.close()
        except Exception:
            pass
