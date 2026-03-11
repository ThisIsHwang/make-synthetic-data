from __future__ import annotations

import datetime

import pytest

from gemma27_rl import eval as eval_mod
from gemma27_rl.config import RLPostTrainConfig


def _reset_eval_object_group_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eval_mod, "_EVAL_OBJECT_GROUP", None)
    monkeypatch.setattr(eval_mod, "_EVAL_OBJECT_GROUP_WORLD_SIZE", -1)
    monkeypatch.setattr(eval_mod, "_EVAL_OBJECT_GROUP_TIMEOUT_SEC", -1.0)


def test_eval_object_group_uses_configured_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    if eval_mod.torch is None:
        pytest.skip("torch is not installed")

    cfg = RLPostTrainConfig()
    cfg.rl.backend = "deepspeed"
    cfg.misc.distributed_timeout_sec = 5400
    captured: dict[str, object] = {}
    fake_group = object()

    _reset_eval_object_group_cache(monkeypatch)
    monkeypatch.delenv("TORCH_DISTRIBUTED_TIMEOUT_SEC", raising=False)
    monkeypatch.setattr(eval_mod.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(eval_mod.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(eval_mod.torch.distributed, "get_backend", lambda: "nccl")
    monkeypatch.setattr(
        eval_mod.torch.distributed,
        "new_group",
        lambda **kwargs: captured.update(kwargs) or fake_group,
    )

    group = eval_mod._get_eval_object_collective_group(cfg, rank=0, world_size=8)

    assert group is fake_group
    assert captured["backend"] == "gloo"
    assert captured["timeout"] == datetime.timedelta(seconds=5400)


def test_eval_object_group_uses_deepspeed_default_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    if eval_mod.torch is None:
        pytest.skip("torch is not installed")

    cfg = RLPostTrainConfig()
    cfg.rl.backend = "deepspeed"
    cfg.misc.distributed_timeout_sec = None
    captured: dict[str, object] = {}
    fake_group = object()

    _reset_eval_object_group_cache(monkeypatch)
    monkeypatch.delenv("TORCH_DISTRIBUTED_TIMEOUT_SEC", raising=False)
    monkeypatch.setattr(eval_mod.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(eval_mod.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(eval_mod.torch.distributed, "get_backend", lambda: "nccl")
    monkeypatch.setattr(
        eval_mod.torch.distributed,
        "new_group",
        lambda **kwargs: captured.update(kwargs) or fake_group,
    )

    group = eval_mod._get_eval_object_collective_group(cfg, rank=0, world_size=8)

    assert group is fake_group
    assert captured["timeout"] == datetime.timedelta(seconds=7200)
