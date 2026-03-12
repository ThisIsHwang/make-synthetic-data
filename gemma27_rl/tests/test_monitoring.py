from __future__ import annotations

import sys
from types import ModuleType

import pytest

pytest.importorskip("torch")

from gemma27_rl.config import RLPostTrainConfig
from gemma27_rl.trainer import _ExperimentTracker, _flatten_monitor_metrics


def test_flatten_monitor_metrics_flattens_nested_numeric_values() -> None:
    metrics = _flatten_monitor_metrics(
        prefix="eval",
        payload={
            "type": "eval",
            "update": 3,
            "metricx_score_mean": 1.5,
            "severity_counts": {"MAJOR": 2, "label": "ignored"},
            "direction_metrics": {
                "en->ko": {"metricx_reward_mean": 4.0},
                "ko->en": {"metricx_reward_mean": 1.0},
            },
            "eval_rows": [{"example_id": "ex-1"}],
            "flag": True,
        },
    )

    assert metrics == {
        "eval/metricx_score_mean": 1.5,
        "eval/severity_counts/MAJOR": 2.0,
        "eval/direction_metrics/en->ko/metricx_reward_mean": 4.0,
        "eval/direction_metrics/ko->en/metricx_reward_mean": 1.0,
    }


def test_experiment_tracker_logs_to_tensorboard_and_wandb(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    tb_events: list[tuple[str, float, int]] = []
    tb_text: list[tuple[str, str, int]] = []
    wandb_logs: list[tuple[int, dict[str, float]]] = []
    wandb_finish_calls: list[int] = []
    wandb_init_calls: list[dict[str, object]] = []

    class _FakeSummaryWriter:
        def __init__(self, log_dir: str) -> None:
            self.log_dir = log_dir

        def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
            tb_events.append((tag, float(scalar_value), int(global_step)))

        def add_text(self, tag: str, text_string: str, global_step: int = 0) -> None:
            tb_text.append((tag, str(text_string), int(global_step)))

        def flush(self) -> None:
            return

        def close(self) -> None:
            return

    class _FakeWandbRun:
        def __init__(self) -> None:
            self.id = "wandb-run-123"
            self.url = "https://wandb.example/runs/wandb-run-123"

        def log(self, metrics: dict[str, float], step: int) -> None:
            wandb_logs.append((int(step), dict(metrics)))

        def finish(self, exit_code: int = 0) -> None:
            wandb_finish_calls.append(int(exit_code))

    fake_tb_module = ModuleType("torch.utils.tensorboard")
    fake_tb_module.SummaryWriter = _FakeSummaryWriter  # type: ignore[attr-defined]

    fake_wandb_module = ModuleType("wandb")

    def _fake_wandb_init(**kwargs):  # type: ignore[no-untyped-def]
        wandb_init_calls.append(dict(kwargs))
        return _FakeWandbRun()

    fake_wandb_module.init = _fake_wandb_init  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", fake_tb_module)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb_module)

    cfg = RLPostTrainConfig()
    cfg.logging.tensorboard_enabled = True
    cfg.logging.wandb_enabled = True
    cfg.logging.wandb_project = "monitor-test"
    cfg.logging.wandb_mode = "offline"
    cfg.logging.wandb_tags = ["rl", "monitoring"]

    tracker = _ExperimentTracker(cfg=cfg, output_dir=tmp_path)
    tracker.log_metrics(
        prefix="train",
        payload={
            "type": "train",
            "update": 5,
            "policy_loss": 1.25,
            "adv_raw_std": 0.5,
            "nested": {"mqm_score_mean": -2.0},
        },
        step=5,
    )
    tracker.close()

    assert tb_text and tb_text[0][0] == "config/resolved"
    assert ("train/policy_loss", 1.25, 5) in tb_events
    assert ("train/adv_raw_std", 0.5, 5) in tb_events
    assert ("train/nested/mqm_score_mean", -2.0, 5) in tb_events
    assert wandb_init_calls and wandb_init_calls[0]["project"] == "monitor-test"
    assert wandb_logs == [
        (
            5,
            {
                "train/policy_loss": 1.25,
                "train/adv_raw_std": 0.5,
                "train/nested/mqm_score_mean": -2.0,
            },
        )
    ]
    assert wandb_finish_calls == [0]
    assert tracker.wandb_run_id == "wandb-run-123"
    assert tracker.wandb_url == "https://wandb.example/runs/wandb-run-123"
    assert (tmp_path / "wandb_run_id.txt").read_text(encoding="utf-8").strip() == "wandb-run-123"
