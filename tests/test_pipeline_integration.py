from pathlib import Path

import synth_parallel.pipeline as pipeline_mod
from synth_parallel.config import PipelineConfig
from synth_parallel.io_utils import read_jsonl
from synth_parallel.pipeline import PipelineRunner


class FakeTeacher:
    def run_tasks(self, tasks, worker_fn, max_workers=None):
        out = []
        for task in tasks:
            if task.item_id.endswith("::greedy"):
                text = "greedy translation"
            elif task.item_id.endswith("::sample"):
                text = "sample translation"
            else:
                idx = int(task.item_id.split("::", maxsplit=1)[1])
                text = f"candidate_{idx}"
            out.append(worker_fn(task, text))
        return out

    def complete(self, *args, **kwargs):
        return '{"pass": true, "reason_code": "pass", "notes": "ok"}'

    def close(self):
        return None


class FakeMetricX:
    def score_batch(self, pairs):
        scores = []
        for _src, hyp in pairs:
            if hyp.startswith("greedy"):
                scores.append(10.0)
            elif hyp.startswith("sample"):
                scores.append(5.0)
            elif hyp.startswith("candidate_"):
                idx = int(hyp.split("_", maxsplit=1)[1])
                scores.append(float(idx))
            else:
                scores.append(12.0)
        return scores

    def close(self):
        return None


def test_pipeline_all_stages(tmp_path, monkeypatch):
    fake_ds = [
        {"id": "d1", "text": ["line one source text", "line two source text"]},
        {"id": "d2", "text": "document two. another sentence."},
        {"id": "d3", "text": "document three only line"},
    ]

    monkeypatch.setattr(pipeline_mod, "load_dataset", lambda *args, **kwargs: fake_ds)

    cfg = PipelineConfig()
    cfg.run.out_dir = str(tmp_path / "run")
    cfg.data.sample_pool_size = 6
    cfg.data.target_examples_total = 2
    cfg.data.streaming = False
    cfg.data.sentence_ratio = 1.0
    cfg.data.blob_ratio = 0.0
    cfg.final_generation.blob.enabled = False
    cfg.final_generation.num_candidates = 4
    cfg.final_generation.store_top_k = 2
    cfg.filters.llm_judge.enabled = False
    cfg.teacher.max_concurrency = 4

    runner = PipelineRunner(cfg=cfg, overwrite=True)
    runner._teacher = FakeTeacher()
    runner._metricx = FakeMetricX()

    try:
        runner.run("all")
    finally:
        runner.close()

    out_path = Path(cfg.run.out_dir) / "final_dataset.jsonl"
    assert out_path.exists()

    rows = list(read_jsonl(out_path))
    assert len(rows) == cfg.data.target_examples_total
    assert all("target_text" in row for row in rows)
    assert all(row["metricx_qe_score_best"] == 0.0 for row in rows)
