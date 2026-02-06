from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .io_utils import ensure_parent


class StatsCollector:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.counters: Counter[str] = Counter()
        self.timings: dict[str, list[float]] = defaultdict(list)
        self.stage_started_at: dict[str, float] = {}

    def inc(self, key: str, value: int = 1) -> None:
        self.counters[key] += value

    @contextmanager
    def time_block(self, key: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[key].append(time.perf_counter() - start)

    def stage_start(self, stage: str) -> None:
        self.stage_started_at[stage] = time.time()

    def stage_end(self, stage: str) -> None:
        start = self.stage_started_at.get(stage)
        if start is not None:
            self.timings[f"stage.{stage}.duration_s"].append(time.time() - start)

    def to_dict(self) -> dict:
        return {
            "counters": dict(self.counters),
            "timings": {
                key: {
                    "count": len(values),
                    "total_s": sum(values),
                    "avg_s": (sum(values) / len(values)) if values else 0.0,
                    "max_s": max(values) if values else 0.0,
                }
                for key, values in self.timings.items()
            },
            "updated_at": time.time(),
        }

    def flush(self) -> None:
        ensure_parent(self.path)
        self.path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
