from __future__ import annotations

from qwen35_moe_rl.utils import (
    all_reduce_scalar,
    get_rank,
    get_world_size,
    is_main_process,
    resolve_huggingface_token,
)


def test_resolve_hf_token_from_explicit_and_env(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "env-token")
    assert resolve_huggingface_token("explicit-token", "HF_TOKEN") == "explicit-token"


def test_resolve_hf_token_from_env_name(monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("MY_HF_TOKEN", "custom-token")
    assert resolve_huggingface_token(None, "MY_HF_TOKEN") == "custom-token"


def test_is_main_process_not_initialized() -> None:
    assert is_main_process() is True


def test_get_rank_not_initialized() -> None:
    assert get_rank() == 0


def test_get_world_size_not_initialized() -> None:
    assert get_world_size() == 1


def test_all_reduce_scalar_not_distributed() -> None:
    assert all_reduce_scalar(3.14, op="mean") == 3.14
    assert all_reduce_scalar(2.71, op="sum") == 2.71
