from __future__ import annotations

import json
import logging
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

from .caches import SQLiteKVCache
from .config import MetricXConfig
from .stats import StatsCollector
from .utils import jaccard_overlap, stable_hash


class MetricXScorer:
    def __init__(
        self,
        cfg: MetricXConfig,
        cache_db_path: str,
        stats: StatsCollector,
        logger: logging.Logger | None = None,
    ):
        self.cfg = cfg
        self.cache = SQLiteKVCache(cache_db_path, table_name="metricx_cache")
        self.stats = stats
        self.logger = logger or logging.getLogger(__name__)
        self._worker_proc: subprocess.Popen[str] | None = None
        self._worker_stdout_q: queue.Queue[str] | None = None
        self._worker_stdout_thread: threading.Thread | None = None
        self._worker_stderr_thread: threading.Thread | None = None
        self._worker_lock = threading.Lock()
        if self.cfg.batch_size != 1:
            self.logger.warning(
                "metricx.batch_size=%s is overridden to 1 (fixed policy).",
                self.cfg.batch_size,
            )
            self.cfg.batch_size = 1

    def _cache_key(self, source: str, hypothesis: str) -> str:
        return stable_hash({"source": source, "hypothesis": hypothesis, "checkpoint": self.cfg.checkpoint})

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        results: list[float | None] = [None] * len(pairs)
        uncached_indices: list[int] = []
        uncached_pairs: list[tuple[str, str]] = []

        for i, (source, hyp) in enumerate(pairs):
            key = self._cache_key(source, hyp)
            cached = self.cache.get(key)
            if cached is not None:
                self.stats.inc("metricx.cache_hit")
                results[i] = float(cached["score"])
            else:
                self.stats.inc("metricx.cache_miss")
                uncached_indices.append(i)
                uncached_pairs.append((source, hyp))

        if uncached_pairs:
            with self.stats.time_block("metricx.score_batch"):
                if self.cfg.backend == "metricx24_cli":
                    if self.cfg.persistent_worker:
                        scores = self._score_metricx24_persistent(uncached_pairs)
                    else:
                        scores = self._score_metricx24_cli(uncached_pairs)
                else:
                    scores = [self._heuristic_score(src, hyp) for src, hyp in uncached_pairs]
                    self.logger.warning(
                        "metricx backend '%s' is using heuristic fallback; scores are not MetricX.",
                        self.cfg.backend,
                    )
            cache_batch: list[tuple[str, dict[str, float | str]]] = []
            for idx, score, pair in zip(uncached_indices, scores, uncached_pairs):
                results[idx] = score
                key = self._cache_key(pair[0], pair[1])
                cache_batch.append((key, {"score": float(score), "backend": self.cfg.backend}))
            self.cache.set_many(cache_batch)

        return [float(x) if x is not None else 25.0 for x in results]

    @staticmethod
    def _reader_thread(stream, out_q: queue.Queue[str] | None = None, logger: logging.Logger | None = None) -> None:
        try:
            for line in iter(stream.readline, ""):
                if out_q is not None:
                    out_q.put(line)
                elif logger is not None:
                    text = line.strip()
                    if text:
                        logger.info("MetricX worker stderr | %s", text)
        finally:
            try:
                stream.close()
            except Exception:  # pylint: disable=broad-except
                pass

    def _read_worker_json(self, timeout_s: int) -> dict:
        if self._worker_stdout_q is None:
            raise RuntimeError("MetricX worker stdout queue is not initialized.")
        deadline = time.time() + timeout_s
        while True:
            remaining = max(0.0, deadline - time.time())
            if remaining == 0.0:
                raise TimeoutError(f"Timed out waiting MetricX worker response ({timeout_s}s).")
            try:
                line = self._worker_stdout_q.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError(f"Timed out waiting MetricX worker response ({timeout_s}s).") from exc

            payload = line.strip()
            if not payload:
                continue
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"MetricX worker emitted invalid JSON: {payload[:200]}") from exc
            if not isinstance(parsed, dict):
                raise RuntimeError(f"MetricX worker response must be object, got: {type(parsed)}")
            return parsed

    def _ensure_metricx_worker(self) -> None:
        with self._worker_lock:
            if self._worker_proc is not None and self._worker_proc.poll() is None:
                return

            python_bin = self.cfg.python_bin.strip() if self.cfg.python_bin else ""
            if not python_bin:
                python_bin = sys.executable

            repo_dir = self.cfg.repo_dir.strip() if getattr(self.cfg, "repo_dir", "") else ""
            cwd = repo_dir or None
            if repo_dir and not Path(repo_dir).exists():
                raise RuntimeError(f"metricx.repo_dir does not exist: {repo_dir}")

            self._validate_metricx_runtime(python_bin, cwd)

            worker_script = Path(__file__).with_name("metricx_worker.py").resolve()
            env = self._build_metricx_env()
            if repo_dir:
                existing_py_path = env.get("PYTHONPATH", "")
                env["PYTHONPATH"] = (
                    repo_dir
                    if not existing_py_path
                    else f"{repo_dir}{os.pathsep}{existing_py_path}"
                )
            cmd = [
                python_bin,
                "-u",
                str(worker_script),
                "--tokenizer",
                self.cfg.tokenizer,
                "--model_name_or_path",
                self.cfg.checkpoint,
                "--max_input_length",
                str(self.cfg.max_input_length),
                "--device",
                self.cfg.device,
                "--qe",
            ]

            self.logger.info("Starting persistent MetricX worker: %s", " ".join(cmd))
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
                cwd=cwd,
            )
            if proc.stdin is None or proc.stdout is None or proc.stderr is None:
                raise RuntimeError("Failed to create MetricX worker stdio pipes.")

            stdout_q: queue.Queue[str] = queue.Queue()
            stdout_thread = threading.Thread(
                target=self._reader_thread,
                args=(proc.stdout, stdout_q, None),
                daemon=True,
            )
            stderr_thread = threading.Thread(
                target=self._reader_thread,
                args=(proc.stderr, None, self.logger),
                daemon=True,
            )
            stdout_thread.start()
            stderr_thread.start()

            self._worker_proc = proc
            self._worker_stdout_q = stdout_q
            self._worker_stdout_thread = stdout_thread
            self._worker_stderr_thread = stderr_thread

            init_msg = self._read_worker_json(timeout_s=self.cfg.worker_start_timeout_s)
            if init_msg.get("event") == "ready" and init_msg.get("ok", True):
                self.logger.info("MetricX worker ready device=%s", init_msg.get("device"))
                return

            error = init_msg.get("error") or init_msg
            if isinstance(error, str) and "No module named 'metricx24'" in error:
                error = (
                    "MetricX worker cannot import metricx24. "
                    f"Check metricx.repo_dir={repo_dir} and that the repo exists; "
                    "worker PYTHONPATH now includes metricx.repo_dir."
                )
            self._stop_metricx_worker()
            raise RuntimeError(f"MetricX worker failed to initialize: {error}")

    def _score_metricx24_persistent(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []

        self._ensure_metricx_worker()
        if self._worker_proc is None or self._worker_proc.stdin is None:
            raise RuntimeError("MetricX worker is not available.")

        req_id = uuid.uuid4().hex
        payload = {
            "id": req_id,
            "pairs": [{"source": source, "hypothesis": hyp} for source, hyp in pairs],
        }
        started = time.perf_counter()
        self.logger.info("MetricX worker request start pairs=%s", len(pairs))
        try:
            self._worker_proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._worker_proc.stdin.flush()
        except Exception as exc:  # pylint: disable=broad-except
            self._stop_metricx_worker()
            raise RuntimeError(f"Failed to send request to MetricX worker: {exc}") from exc

        # Worker emits one response per request in order.
        response = self._read_worker_json(timeout_s=self.cfg.worker_response_timeout_s)
        if response.get("id") != req_id:
            self._stop_metricx_worker()
            raise RuntimeError(
                "MetricX worker response id mismatch. "
                f"expected={req_id}, got={response.get('id')}"
            )
        if not response.get("ok", False):
            error = response.get("error", "unknown worker error")
            tb = response.get("traceback", "")
            raise RuntimeError(f"MetricX worker failed: {error}\n{tb}")

        scores = response.get("scores")
        if not isinstance(scores, list):
            raise RuntimeError(f"MetricX worker response missing scores list: {response}")
        if len(scores) != len(pairs):
            raise RuntimeError(
                f"MetricX worker score size mismatch: expected={len(pairs)} got={len(scores)}"
            )
        self.logger.info(
            "MetricX worker request done pairs=%s elapsed=%.2fs",
            len(pairs),
            time.perf_counter() - started,
        )
        self.stats.inc("metricx.success", value=len(scores))
        return [float(x) for x in scores]

    def _stop_metricx_worker(self) -> None:
        with self._worker_lock:
            proc = self._worker_proc
            if proc is None:
                return
            try:
                if proc.poll() is None and proc.stdin is not None:
                    proc.stdin.write(json.dumps({"event": "shutdown"}) + "\n")
                    proc.stdin.flush()
            except Exception:  # pylint: disable=broad-except
                pass

            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)

            self._worker_proc = None
            self._worker_stdout_q = None
            self._worker_stdout_thread = None
            self._worker_stderr_thread = None

    def _score_metricx24_cli(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []

        with tempfile.TemporaryDirectory(prefix="metricx-") as tmp_dir:
            tmp = Path(tmp_dir)
            in_path = tmp / "input.jsonl"
            out_path = tmp / "output.jsonl"

            with in_path.open("w", encoding="utf-8") as fp:
                for source, hyp in pairs:
                    row = {
                        "source": source,
                        "hypothesis": hyp,
                        "reference": "",
                    }
                    fp.write(json.dumps(row, ensure_ascii=False) + "\n")

            self._run_metricx_command(in_path, out_path)

            scores: list[float] = []
            with out_path.open("r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    score = (
                        row.get("metricx_score")
                        or row.get("predicted_score")
                        or row.get("score")
                        or row.get("prediction")
                    )
                    if score is None:
                        raise RuntimeError(f"MetricX output missing score field: {row}")
                    scores.append(float(score))

            if len(scores) != len(pairs):
                raise RuntimeError(
                    f"MetricX output size mismatch: expected {len(pairs)} rows, got {len(scores)}"
                )
            self.stats.inc("metricx.success", value=len(scores))
            return scores

    def _run_metricx_command(self, in_path: Path, out_path: Path) -> None:
        python_bin = self.cfg.python_bin.strip() if self.cfg.python_bin else ""
        if not python_bin:
            python_bin = sys.executable

        repo_dir = self.cfg.repo_dir.strip() if getattr(self.cfg, "repo_dir", "") else ""
        cwd = repo_dir or None
        if repo_dir and not Path(repo_dir).exists():
            raise RuntimeError(f"metricx.repo_dir does not exist: {repo_dir}")

        self._validate_metricx_runtime(python_bin, cwd)

        # Align with google-research/metricx usage:
        # python -m metricx24.predict --tokenizer ... --model_name_or_path ... --max_input_length ...
        #   --batch_size ... --input_file ... --output_file ... --qe
        variants = [
            [
                python_bin,
                "-m",
                self.cfg.module,
                "--tokenizer",
                self.cfg.tokenizer,
                "--model_name_or_path",
                self.cfg.checkpoint,
                "--max_input_length",
                str(self.cfg.max_input_length),
                "--batch_size",
                str(self.cfg.batch_size),
                "--input_file",
                str(in_path),
                "--output_file",
                str(out_path),
                "--qe",
            ]
        ]

        env = self._build_metricx_env()
        last_error = ""
        with in_path.open("r", encoding="utf-8") as fp:
            pair_count = sum(1 for _ in fp)
        for cmd in variants:
            start = time.perf_counter()
            self.logger.info(
                "MetricX command start pairs=%s batch_size=%s device=%s",
                pair_count,
                self.cfg.batch_size,
                self.cfg.device,
            )
            self.logger.info("MetricX command: %s", " ".join(cmd))
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=cwd,
            )
            heartbeat_s = 30
            while proc.poll() is None:
                try:
                    proc.wait(timeout=heartbeat_s)
                except subprocess.TimeoutExpired:
                    self.logger.info(
                        "MetricX still running elapsed=%.1fs pairs=%s",
                        time.perf_counter() - start,
                        pair_count,
                    )

            stdout, stderr = proc.communicate()
            if proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
                self.logger.info("MetricX command succeeded: %s", " ".join(cmd))
                return
            last_error = (stderr or stdout or "").strip()
            if last_error:
                self.logger.error("MetricX command failed rc=%s output=%s", proc.returncode, last_error[:1200])

        self.stats.inc("metricx.error")
        if "loading a dataset cached in a localfilesystem is not supported" in last_error.lower():
            raise RuntimeError(
                "MetricX runtime hit datasets/fsspec incompatibility "
                "(NotImplementedError: LocalFileSystem). "
                "In the metricx env, pin fsspec to <2023.10.0 and retry: "
                f"{python_bin} -m pip install 'fsspec<2023.10.0'"
            )
        if "unable to avoid copy while creating an array as requested" in last_error.lower():
            raise RuntimeError(
                "MetricX runtime hit NumPy 2.x compatibility issue. "
                "In the metricx env, pin numpy to <2 and retry: "
                f"{python_bin} -m pip install 'numpy<2'"
            )
        if "pyextensiontype" in last_error.lower():
            raise RuntimeError(
                "MetricX runtime has incompatible pyarrow version for datasets==2.13.1. "
                f"Install pyarrow<21 in metricx env: {python_bin} -m pip install 'pyarrow<21'"
            )
        if "use_auth_token" in last_error and "hf_hub_download" in last_error:
            raise RuntimeError(
                "MetricX runtime has incompatible huggingface stack. "
                "This usually means datasets is too old (e.g. 1.x). "
                "Reinstall metricx env with: "
                f"{python_bin} -m pip install -r {Path(repo_dir) / 'requirements.txt' if repo_dir else 'requirements.txt'}"
            )
        raise RuntimeError(
            "Failed to run MetricX (google-research/metricx). "
            "Verify metricx.repo_dir (git clone) and that requirements are installed in metricx.python_bin env. "
            f"Last error: {last_error}"
        )

    def _validate_metricx_runtime(self, python_bin: str, cwd: str | None) -> None:
        check_cmd = [
            python_bin,
            "-c",
            (
                "import datasets,sys;"
                "print(datasets.__version__)"
            ),
        ]
        proc = subprocess.run(check_cmd, capture_output=True, text=True, cwd=cwd)
        if proc.returncode != 0:
            stderr = (proc.stderr or proc.stdout or "").strip()
            if "PyExtensionType" in stderr or "pyextensiontype" in stderr.lower():
                raise RuntimeError(
                    "metricx.python_bin hits datasets/pyarrow incompatibility "
                    "(missing pyarrow.PyExtensionType). "
                    f"Install pyarrow<21 in that env: {python_bin} -m pip install 'pyarrow<21'"
                )
            raise RuntimeError(
                "metricx.python_bin cannot import datasets. "
                "Install metricx requirements in that exact interpreter. "
                f"python_bin={python_bin}, error={stderr}"
            )
        version = (proc.stdout or "").strip().splitlines()[-1]
        match = re.match(r"^(\\d+)\\.(\\d+)\\.(\\d+)", version)
        if match and int(match.group(1)) < 2:
            raise RuntimeError(
                "metricx.python_bin has datasets<2 installed "
                f"(detected {version}). MetricX repo expects datasets==2.13.1."
            )

    def _build_metricx_env(self) -> dict[str, str]:
        env = dict(os.environ)
        device = (self.cfg.device or "").strip().lower()
        if device.startswith("cuda:"):
            gpu_id = device.split(":", maxsplit=1)[1]
            if "CUDA_VISIBLE_DEVICES" in env:
                self.logger.info(
                    "Preserving existing CUDA_VISIBLE_DEVICES=%s for MetricX worker (metricx.device=%s).",
                    env.get("CUDA_VISIBLE_DEVICES", ""),
                    self.cfg.device,
                )
            elif gpu_id:
                env["CUDA_VISIBLE_DEVICES"] = gpu_id
        elif device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        return env

    @staticmethod
    def _heuristic_score(source: str, hypothesis: str) -> float:
        if not hypothesis.strip():
            return 25.0
        ratio = len(hypothesis) / max(1, len(source))
        ratio_penalty = abs(1.0 - ratio) * 5.0
        overlap_penalty = jaccard_overlap(source, hypothesis) * 10.0
        score = 5.0 + ratio_penalty + overlap_penalty
        return max(0.0, min(25.0, score))

    def close(self) -> None:
        self._stop_metricx_worker()
        self.cache.close()
