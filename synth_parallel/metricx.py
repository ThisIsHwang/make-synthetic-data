from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
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
                    scores = self._score_metricx24_cli(uncached_pairs)
                else:
                    scores = [self._heuristic_score(src, hyp) for src, hyp in uncached_pairs]
                    self.logger.warning(
                        "metricx backend '%s' is using heuristic fallback; scores are not MetricX.",
                        self.cfg.backend,
                    )
            for idx, score, pair in zip(uncached_indices, scores, uncached_pairs):
                results[idx] = score
                key = self._cache_key(pair[0], pair[1])
                self.cache.set(key, {"score": score, "backend": self.cfg.backend})

        return [float(x) if x is not None else 25.0 for x in results]

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
        for cmd in variants:
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd)
            if proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
                self.logger.info("MetricX command succeeded: %s", " ".join(cmd))
                return
            last_error = (proc.stderr or proc.stdout or "").strip()

        self.stats.inc("metricx.error")
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
            if gpu_id:
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
        self.cache.close()
