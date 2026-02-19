from synth_parallel.config import MetricXConfig
from synth_parallel.metricx import MetricXScorer
from synth_parallel.stats import StatsCollector


def _build_scorer(tmp_path, device: str) -> MetricXScorer:
    cfg = MetricXConfig(device=device)
    return MetricXScorer(
        cfg=cfg,
        cache_db_path=str(tmp_path / "metricx_cache.sqlite"),
        stats=StatsCollector(tmp_path / "stats.json"),
    )


def test_metricx_env_preserves_existing_cuda_visible_devices(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    scorer = _build_scorer(tmp_path, device="cuda:0")
    try:
        env = scorer._build_metricx_env()
        assert env.get("CUDA_VISIBLE_DEVICES") == "1"
    finally:
        scorer.close()


def test_metricx_env_sets_cuda_visible_devices_when_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    scorer = _build_scorer(tmp_path, device="cuda:3")
    try:
        env = scorer._build_metricx_env()
        assert env.get("CUDA_VISIBLE_DEVICES") == "3"
    finally:
        scorer.close()


def test_metricx_env_clears_cuda_visible_devices_for_cpu(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    scorer = _build_scorer(tmp_path, device="cpu")
    try:
        env = scorer._build_metricx_env()
        assert env.get("CUDA_VISIBLE_DEVICES") == ""
    finally:
        scorer.close()
