from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch

from gemma27_rl.config import GenerationConfig, VLLMConfig
from gemma27_rl.rl_types import Example
import gemma27_rl.vllm_rollout as vllm_mod


class _FakeProc:
    def __init__(self) -> None:
        self.returncode = None

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = 0
        return 0

    def kill(self) -> None:
        self.returncode = 0


class _TokenizerStub:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.chat_template = None
        self.special_tokens_map = {"pad_token": "<pad>", "eos_token": "</s>"}

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):  # type: ignore[no-untyped-def]
        del skip_special_tokens, clean_up_tokenization_spaces
        return "|".join(str(int(tok)) for tok in list(token_ids))

    def __call__(
        self,
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    ):  # type: ignore[no-untyped-def]
        del add_special_tokens, return_attention_mask, return_token_type_ids
        if str(text) == "fallback":
            return {"input_ids": [901, 902]}
        return {"input_ids": []}


class _PolicyModel:
    generation_config = type("GenConfig", (), {"eos_token_id": 2})()

    def eval(self) -> None:
        return None


class _ClientStub:
    def __init__(self, rows) -> None:  # type: ignore[no-untyped-def]
        self.rows = rows
        self._warned_missing_token_ids = False

    def generate_choices(self, **kwargs):  # type: ignore[no-untyped-def]
        return self.rows


def _example() -> Example:
    return Example(
        example_id="ex-0",
        src_text="hello",
        src_lang="English",
        tgt_lang="Korean",
        src_lang_code="en",
        tgt_lang_code="ko",
    )


def test_local_vllm_client_start_strips_dist_vars(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_popen(*args, **kwargs):  # type: ignore[no-untyped-def]
        captured["cmd"] = list(args[0])
        captured["env"] = dict(kwargs.get("env") or {})
        return _FakeProc()

    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    monkeypatch.setenv("PYTHONHOME", "/tmp/fake-home")
    monkeypatch.setattr(vllm_mod, "_detect_vllm_log_request_flag_style", lambda python_executable: "enable")
    monkeypatch.setattr(vllm_mod.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(vllm_mod.LocalVLLMRolloutClient, "_wait_until_ready", lambda self: None)

    cfg = VLLMConfig(
        enabled=True,
        gpu_ids=[6, 7],
        tensor_parallel_size=2,
        adapter_root_dir=str(tmp_path / "adapters"),
        python_executable="python",
    )
    client = vllm_mod.LocalVLLMRolloutClient(
        cfg=cfg,
        base_model_name_or_path="google/gemma-3-27b-it",
        tokenizer_name_or_path="google/gemma-3-27b-it",
        lora_rank=64,
        trust_remote_code=False,
        dtype="bfloat16",
        log_path=tmp_path / "vllm.log",
        owns_server=True,
    )
    client.start()
    client.close()

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[:3] == ["python", "-m", "vllm.entrypoints.openai.api_server"]
    assert "--enable-lora" in cmd
    assert "--max-lora-rank" in cmd
    assert "--no-enable-log-requests" in cmd
    env = captured["env"]
    assert isinstance(env, dict)
    assert env.get("CUDA_VISIBLE_DEVICES") == "6,7"
    assert env.get("VLLM_ALLOW_RUNTIME_LORA_UPDATING") == "True"
    assert "LOCAL_RANK" not in env
    assert "RANK" not in env
    assert "WORLD_SIZE" not in env
    assert "MASTER_ADDR" not in env
    assert "MASTER_PORT" not in env
    assert "PYTHONHOME" not in env


def test_generate_rollouts_vllm_uses_returned_token_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vllm_mod, "_collect_end_of_turn_token_ids", lambda tokenizer: [])
    monkeypatch.setattr(
        vllm_mod,
        "_resolve_eos_token_ids",
        lambda tokenizer_eos, model_eos, extra_token_ids=None: [2],
    )
    monkeypatch.setattr(vllm_mod, "_encode_prompt_rows", lambda **kwargs: [[11, 12]])
    monkeypatch.setattr(vllm_mod, "_get_model_vocab_size", lambda model: 4096)
    monkeypatch.setattr(vllm_mod, "_validate_item_token_ids", lambda **kwargs: None)
    monkeypatch.setattr(vllm_mod, "_validate_token_ids_in_vocab", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        vllm_mod,
        "compute_token_char_offsets",
        lambda **kwargs: [(0, len(str(kwargs["completion_text"])))],
    )
    monkeypatch.setattr(
        vllm_mod,
        "_compute_logprobs_batch_with_backoff",
        lambda **kwargs: [torch.tensor([-0.1, -0.2], dtype=torch.float32)],
    )

    rollouts = vllm_mod.generate_rollouts_vllm(
        examples=[_example()],
        policy_model=_PolicyModel(),
        tokenizer=_TokenizerStub(),
        gen_cfg=GenerationConfig(max_new_tokens=8, num_samples_per_prompt=1),
        device="cpu",
        vllm_rollout_client=_ClientStub([[vllm_mod._VLLMChoice(text="unused", token_ids=[101, 102])]]),
    )

    assert len(rollouts) == 1
    rollout = rollouts[0]
    assert rollout.prompt_input_ids == [11, 12]
    assert rollout.completion_token_ids == [101, 102]
    assert rollout.completion_text == "101|102"
    assert rollout.old_logprobs == [-0.10000000149011612, -0.20000000298023224]


def test_generate_rollouts_vllm_retokenizes_when_response_omits_token_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vllm_mod, "_collect_end_of_turn_token_ids", lambda tokenizer: [])
    monkeypatch.setattr(
        vllm_mod,
        "_resolve_eos_token_ids",
        lambda tokenizer_eos, model_eos, extra_token_ids=None: [2],
    )
    monkeypatch.setattr(vllm_mod, "_encode_prompt_rows", lambda **kwargs: [[11, 12]])
    monkeypatch.setattr(vllm_mod, "_get_model_vocab_size", lambda model: 4096)
    monkeypatch.setattr(vllm_mod, "_validate_item_token_ids", lambda **kwargs: None)
    monkeypatch.setattr(vllm_mod, "_validate_token_ids_in_vocab", lambda *args, **kwargs: None)
    monkeypatch.setattr(vllm_mod, "compute_token_char_offsets", lambda **kwargs: [])
    monkeypatch.setattr(
        vllm_mod,
        "_compute_logprobs_batch_with_backoff",
        lambda **kwargs: [torch.tensor([-0.3, -0.4], dtype=torch.float32)],
    )

    client = _ClientStub([[vllm_mod._VLLMChoice(text="fallback", token_ids=[])]])
    rollouts = vllm_mod.generate_rollouts_vllm(
        examples=[_example()],
        policy_model=_PolicyModel(),
        tokenizer=_TokenizerStub(),
        gen_cfg=GenerationConfig(max_new_tokens=8, num_samples_per_prompt=1),
        device="cpu",
        vllm_rollout_client=client,
    )

    assert len(rollouts) == 1
    assert rollouts[0].completion_token_ids == [901, 902]
    assert client._warned_missing_token_ids is True


def test_detect_vllm_log_request_flag_style_prefers_new_enable_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Completed:
        stdout = "usage: ... [--enable-log-requests | --no-enable-log-requests]"
        stderr = ""

    monkeypatch.setattr(vllm_mod.subprocess, "run", lambda *args, **kwargs: _Completed())
    monkeypatch.setattr(vllm_mod, "_VLLM_LOG_REQUEST_FLAG_STYLE_CACHE", {})

    assert vllm_mod._detect_vllm_log_request_flag_style("python") == "enable"


def test_detect_vllm_log_request_flag_style_falls_back_to_old_disable_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Completed:
        stdout = "usage: ... [--disable-log-requests]"
        stderr = ""

    monkeypatch.setattr(vllm_mod.subprocess, "run", lambda *args, **kwargs: _Completed())
    monkeypatch.setattr(vllm_mod, "_VLLM_LOG_REQUEST_FLAG_STYLE_CACHE", {})

    assert vllm_mod._detect_vllm_log_request_flag_style("python") == "disable"
