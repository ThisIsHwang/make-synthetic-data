from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback
from typing import Any


def _ensure_repo_import_path() -> None:
    # scorer_worker.py lives at gemma27_rl/gemma27_rl/scorer_worker.py
    # Add gemma27_rl/ to sys.path so `import gemma27_rl.*` works without install.
    repo_pkg_root = Path(__file__).resolve().parents[1]
    repo_pkg_root_text = str(repo_pkg_root)
    if repo_pkg_root_text not in sys.path:
        sys.path.insert(0, repo_pkg_root_text)


_ensure_repo_import_path()

from gemma27_rl.config import MetricXConfig, XCometConfig  # noqa: E402
from gemma27_rl.rewards import MetricXQEScorer, XCometXLScorer  # noqa: E402
from gemma27_rl.types import SampleForScoring  # noqa: E402


def _reply(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _to_samples(rows: list[dict[str, Any]]) -> list[SampleForScoring]:
    out: list[SampleForScoring] = []
    for row in rows:
        out.append(
            SampleForScoring(
                src=str(row.get("src", "")),
                mt=str(row.get("mt", "")),
                ref=(None if row.get("ref") is None else str(row.get("ref"))),
            )
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="External scorer worker")
    parser.add_argument("--backend", required=True, choices=["metricx", "xcomet"])
    args = parser.parse_args(argv)

    scorer: Any = None

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if not isinstance(req, dict):
                raise ValueError("request must be a JSON object")

            req_type = str(req.get("type", "")).strip().lower()
            if req_type == "close":
                _reply({"ok": True})
                return 0

            if req_type == "init":
                cfg_payload = req.get("config") or {}
                if not isinstance(cfg_payload, dict):
                    raise ValueError("init.config must be an object")

                if args.backend == "metricx":
                    cfg_payload["python_executable"] = None
                    cfg = MetricXConfig(**cfg_payload)
                    scorer = MetricXQEScorer(cfg=cfg)
                else:
                    cfg_payload["python_executable"] = None
                    cfg = XCometConfig(**cfg_payload)
                    scorer = XCometXLScorer(cfg=cfg)
                _reply({"ok": True})
                continue

            if scorer is None:
                raise RuntimeError("worker is not initialized. send init first.")

            if req_type == "score":
                if args.backend == "metricx":
                    sample_rows = req.get("samples") or []
                    if not isinstance(sample_rows, list):
                        raise ValueError("score.samples must be a list")
                    samples = _to_samples(sample_rows)
                    out = scorer.score_batch(samples)
                    _reply({"ok": True, "scores": out.sequence_scores, "metadata": out.metadata})
                else:
                    payload_rows = req.get("payload") or []
                    if not isinstance(payload_rows, list):
                        raise ValueError("score.payload must be a list")
                    samples = _to_samples(payload_rows)
                    out = scorer.score_batch(samples)
                    metadata = out.metadata or {}
                    error_spans = metadata.get("error_spans", [[] for _ in out.sequence_scores])
                    _reply({"ok": True, "scores": out.sequence_scores, "error_spans": error_spans})
                continue

            raise ValueError(f"unsupported request type: {req_type}")
        except Exception as exc:
            _reply(
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=4),
                }
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

