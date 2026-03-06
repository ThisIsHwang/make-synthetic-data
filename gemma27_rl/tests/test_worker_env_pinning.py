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
    monkeypatch.setenv("PYTHONHOME", "/tmp/scorer-home")
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
    assert "PYTHONHOME" not in env


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

    env_overrides = captured["env_overrides"]
    assert isinstance(env_overrides, dict)
    assert env_overrides.get("CUDA_VISIBLE_DEVICES") == "7"
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
    monkeypatch.setenv("PYTHONHOME", "/tmp/ref-home")
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
    assert "PYTHONHOME" not in env


def test_scorer_worker_remote_launch_uses_ssh(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_popen(*args, **kwargs):
        captured["cmd"] = list(args[0])
        captured["env"] = dict(kwargs.get("env") or {})
        return _FakeProc(env=kwargs.get("env"))

    monkeypatch.setattr("gemma27_rl.rewards.subprocess.Popen", _fake_popen)
    monkeypatch.setattr(_ScorerSubprocessClient, "request", lambda self, payload: {"ok": True})

    client = _ScorerSubprocessClient(
        backend="metricx",
        python_executable="python",
        timeout_sec=1.0,
        config_payload={"device": "cuda:0"},
        env_overrides={"CUDA_VISIBLE_DEVICES": "7"},
        remote_host="aux-node-1",
        remote_workdir="/shared/gemma27_rl",
    )
    client.close()

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[0] == "ssh"
    assert cmd[1] == "aux-node-1"
    remote_cmd = str(cmd[2])
    assert "CUDA_VISIBLE_DEVICES=7" in remote_cmd
    assert "cd /shared/gemma27_rl" in remote_cmd
    assert "scorer_worker.py" in remote_cmd
    assert "--backend metricx" in remote_cmd
    env = captured["env"]
    assert isinstance(env, dict)
    assert "CUDA_VISIBLE_DEVICES" not in env


def test_reference_worker_remote_launch_uses_ssh(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_popen(*args, **kwargs):
        captured["cmd"] = list(args[0])
        captured["env"] = dict(kwargs.get("env") or {})
        return _FakeProc(env=kwargs.get("env"))

    monkeypatch.setattr("gemma27_rl.trainer.subprocess.Popen", _fake_popen)
    monkeypatch.setattr(ReferenceLogprobClient, "request", lambda self, payload: {"ok": True})

    client = ReferenceLogprobClient(
        python_executable="python",
        timeout_sec=1.0,
        config_payload={"model_name_or_path": "dummy", "device": "cpu"},
        env_overrides={"CUDA_VISIBLE_DEVICES": "6"},
        remote_host="aux-node-1",
        remote_workdir="/shared/gemma27_rl",
    )
    client.close()

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[0] == "ssh"
    assert cmd[1] == "aux-node-1"
    remote_cmd = str(cmd[2])
    assert "CUDA_VISIBLE_DEVICES=6" in remote_cmd
    assert "cd /shared/gemma27_rl" in remote_cmd
    assert "reference_worker.py" in remote_cmd
    env = captured["env"]
    assert isinstance(env, dict)
    assert "CUDA_VISIBLE_DEVICES" not in env


def test_reference_worker_rejects_invalid_logprobs_row_payload() -> None:
    client = object.__new__(ReferenceLogprobClient)
    client.request = lambda payload: {"ok": True, "logprobs_rows": [123]}  # type: ignore[attr-defined]

    with pytest.raises(RuntimeError, match="invalid logprobs row"):
        client.score_logprobs_batch([([1, 2], [3])])
