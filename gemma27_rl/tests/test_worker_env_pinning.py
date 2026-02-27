from __future__ import annotations

import io

import pytest

pytest.importorskip("torch")

from gemma27_rl.config import MetricXConfig
from gemma27_rl.rewards import MetricXQEScorer, _ScorerSubprocessClient
from gemma27_rl.trainer import ReferenceLogprobClient


class _FakeProc:
    def __init__(self, env: dict[str, str] | None = None) -> None:
        self._env = env or {}
        self.stdin = io.StringIO()
        self.stdout = io.StringIO()
        self.returncode = 0

    def poll(self) -> int:
        return 0

    def terminate(self) -> None:
        return None

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        return None


def test_scorer_worker_env_strips_dist_vars(monkeypatch) -> None:
    captured: dict[str, dict[str, str]] = {}

    def _fake_popen(*args, **kwargs):
        captured["env"] = dict(kwargs.get("env") or {})
        return _FakeProc(env=kwargs.get("env"))

    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    monkeypatch.setattr("gemma27_rl.rewards.subprocess.Popen", _fake_popen)
    monkeypatch.setattr(_ScorerSubprocessClient, "request", lambda self, payload: {"ok": True})

    client = _ScorerSubprocessClient(
        backend="metricx",
        python_executable="python",
        timeout_sec=1.0,
        config_payload={"device": "cuda:0"},
        env_overrides={"CUDA_VISIBLE_DEVICES": "7"},
    )
    client.close()

    env = captured["env"]
    assert env.get("CUDA_VISIBLE_DEVICES") == "7"
    assert "LOCAL_RANK" not in env
    assert "RANK" not in env
    assert "WORLD_SIZE" not in env
    assert "MASTER_ADDR" not in env
    assert "MASTER_PORT" not in env


def test_metricx_worker_device_is_pinned_to_cuda0(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def close(self) -> None:
            return None

    monkeypatch.setattr("gemma27_rl.rewards._ScorerSubprocessClient", _FakeClient)
    scorer = MetricXQEScorer(
        MetricXConfig(
            enabled=True,
            python_executable="python",
            device="cuda:7",
        )
    )
    del scorer

    assert captured["env_overrides"] == {"CUDA_VISIBLE_DEVICES": "7"}
    cfg_payload = captured["config_payload"]
    assert isinstance(cfg_payload, dict)
    assert cfg_payload["device"] == "cuda:0"


def test_reference_worker_env_strips_dist_vars(monkeypatch) -> None:
    captured: dict[str, dict[str, str]] = {}

    def _fake_popen(*args, **kwargs):
        captured["env"] = dict(kwargs.get("env") or {})
        return _FakeProc(env=kwargs.get("env"))

    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29501")
    monkeypatch.setattr("gemma27_rl.trainer.subprocess.Popen", _fake_popen)
    monkeypatch.setattr(ReferenceLogprobClient, "request", lambda self, payload: {"ok": True})

    client = ReferenceLogprobClient(
        python_executable="python",
        timeout_sec=1.0,
        config_payload={"model_name_or_path": "dummy", "device": "cpu"},
        env_overrides={"CUDA_VISIBLE_DEVICES": "6"},
    )
    client.close()

    env = captured["env"]
    assert env.get("CUDA_VISIBLE_DEVICES") == "6"
    assert "LOCAL_RANK" not in env
    assert "RANK" not in env
    assert "WORLD_SIZE" not in env
    assert "MASTER_ADDR" not in env
    assert "MASTER_PORT" not in env
