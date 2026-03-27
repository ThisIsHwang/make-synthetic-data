from __future__ import annotations

import json
import os
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
        self.calls: list[dict[str, object]] = []

    def generate_choices(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(dict(kwargs))
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
    monkeypatch.setattr(vllm_mod, "_detect_vllm_custom_all_reduce_flag_style", lambda python_executable: "toggle")
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
    assert "--no-disable-custom-all-reduce" in cmd
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


@pytest.mark.parametrize("adapter_root_dir", ["", "relative/adapters"])
def test_local_vllm_client_adapter_dir_rejects_invalid_root_dir(
    adapter_root_dir: str,
    tmp_path: Path,
) -> None:
    cfg = VLLMConfig(
        enabled=True,
        gpu_ids=[6],
        tensor_parallel_size=1,
        adapter_root_dir=adapter_root_dir,
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
        owns_server=False,
    )

    with pytest.raises(ValueError, match="non-empty absolute path"):
        _ = client.adapter_dir


def test_local_vllm_client_request_json_keeps_proxy_env_intact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"data": []}).encode("utf-8")

    class _FakeOpener:
        def open(self, req, timeout=None):
            captured["timeout"] = timeout
            captured["url"] = req.full_url
            return _FakeResponse()

    def _fake_build_opener(*handlers):  # type: ignore[no-untyped-def]
        captured["handlers"] = handlers
        return _FakeOpener()

    monkeypatch.setenv("ALL_PROXY", "socks5://proxy.local:1080")
    monkeypatch.setattr(vllm_mod.urllib_request, "build_opener", _fake_build_opener)

    cfg = VLLMConfig(
        enabled=True,
        gpu_ids=[6],
        tensor_parallel_size=1,
        adapter_root_dir=str(tmp_path / "adapters"),
        python_executable="python",
        request_timeout_sec=17.0,
    )
    client = vllm_mod.LocalVLLMRolloutClient(
        cfg=cfg,
        base_model_name_or_path="google/gemma-3-27b-it",
        tokenizer_name_or_path="google/gemma-3-27b-it",
        lora_rank=64,
        trust_remote_code=False,
        dtype="bfloat16",
        log_path=tmp_path / "vllm.log",
        owns_server=False,
    )

    payload = client._request_json("/v1/models", method="GET")

    assert payload == {"data": []}
    assert captured["timeout"] == 17.0
    assert captured["url"] == "http://127.0.0.1:8000/v1/models"
    handlers = captured["handlers"]
    assert isinstance(handlers, tuple)
    assert len(handlers) == 1
    assert isinstance(handlers[0], vllm_mod.urllib_request.ProxyHandler)
    assert os.environ["ALL_PROXY"] == "socks5://proxy.local:1080"


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

    client = _ClientStub([[vllm_mod._VLLMChoice(text="unused", token_ids=[101, 102])]])
    rollouts = vllm_mod.generate_rollouts_vllm(
        examples=[_example()],
        policy_model=_PolicyModel(),
        tokenizer=_TokenizerStub(),
        gen_cfg=GenerationConfig(max_new_tokens=8, num_samples_per_prompt=1),
        device="cpu",
        vllm_rollout_client=client,
    )

    assert len(rollouts) == 1
    rollout = rollouts[0]
    assert len(client.calls) == 1
    assert client.calls[0]["prompt_token_id_rows"] == [[11, 12]]
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


def test_local_vllm_client_generate_one_prompt_submits_prompt_token_ids(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_request_json(path, *, method="POST", payload=None):  # type: ignore[no-untyped-def]
        captured["path"] = path
        captured["method"] = method
        captured["payload"] = payload
        return {
            "choices": [
                {
                    "text": "unused",
                    "token_ids": [101, 102],
                    "prompt_token_ids": [11, 12],
                }
            ]
        }

    client = vllm_mod.LocalVLLMRolloutClient(
        cfg=VLLMConfig(
            enabled=True,
            gpu_ids=[6],
            tensor_parallel_size=1,
            adapter_root_dir=str(tmp_path / "adapters"),
            python_executable="python",
        ),
        base_model_name_or_path="google/gemma-3-27b-it",
        tokenizer_name_or_path="google/gemma-3-27b-it",
        lora_rank=64,
        trust_remote_code=False,
        dtype="bfloat16",
        log_path=tmp_path / "vllm.log",
        owns_server=False,
    )
    monkeypatch.setattr(client, "_request_json", _fake_request_json)

    choices = client._generate_one_prompt(
        prompt_token_ids=[11, 12],
        gen_cfg=GenerationConfig(max_new_tokens=8, num_samples_per_prompt=1),
        stop_token_ids=[2],
    )

    assert len(choices) == 1
    assert captured["path"] == "/v1/completions"
    assert captured["method"] == "POST"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["prompt"] == [11, 12]
    assert payload["return_token_ids"] is True


def test_local_vllm_client_generate_one_prompt_rejects_mismatched_echoed_prompt_token_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _fake_request_json(path, *, method="POST", payload=None):  # type: ignore[no-untyped-def]
        return {
            "choices": [
                {
                    "text": "unused",
                    "token_ids": [101, 102],
                    "prompt_token_ids": [11, 99],
                }
            ]
        }

    client = vllm_mod.LocalVLLMRolloutClient(
        cfg=VLLMConfig(
            enabled=True,
            gpu_ids=[6],
            tensor_parallel_size=1,
            adapter_root_dir=str(tmp_path / "adapters"),
            python_executable="python",
        ),
        base_model_name_or_path="google/gemma-3-27b-it",
        tokenizer_name_or_path="google/gemma-3-27b-it",
        lora_rank=64,
        trust_remote_code=False,
        dtype="bfloat16",
        log_path=tmp_path / "vllm.log",
        owns_server=False,
    )
    monkeypatch.setattr(client, "_request_json", _fake_request_json)

    with pytest.raises(RuntimeError, match="prompt token ids mismatch"):
        client._generate_one_prompt(
            prompt_token_ids=[11, 12],
            gen_cfg=GenerationConfig(max_new_tokens=8, num_samples_per_prompt=1),
            stop_token_ids=[2],
        )


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


def test_detect_vllm_custom_all_reduce_flag_style_prefers_toggle_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Completed:
        stdout = "usage: ... [--disable-custom-all-reduce | --no-disable-custom-all-reduce]"
        stderr = ""

    monkeypatch.setattr(vllm_mod.subprocess, "run", lambda *args, **kwargs: _Completed())
    monkeypatch.setattr(vllm_mod, "_VLLM_CUSTOM_ALL_REDUCE_FLAG_STYLE_CACHE", {})

    assert vllm_mod._detect_vllm_custom_all_reduce_flag_style("python") == "toggle"


def test_start_retries_with_disable_custom_all_reduce_when_startup_log_matches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_cmds: list[list[str]] = []

    def _fake_popen(*args, **kwargs):  # type: ignore[no-untyped-def]
        captured_cmds.append(list(args[0]))
        return _FakeProc()

    wait_calls = {"count": 0}

    def _fake_wait(self):  # type: ignore[no-untyped-def]
        wait_calls["count"] += 1
        if wait_calls["count"] == 1:
            self._log_path.write_text(
                "Failed: Cuda error /workspace/csrc/custom_all_reduce.cuh:455 'invalid argument'\n",
                encoding="utf-8",
            )
            raise RuntimeError("startup failed")
        return None

    monkeypatch.setattr(vllm_mod, "_detect_vllm_log_request_flag_style", lambda python_executable: "enable")
    monkeypatch.setattr(vllm_mod, "_detect_vllm_custom_all_reduce_flag_style", lambda python_executable: "toggle")
    monkeypatch.setattr(vllm_mod.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(vllm_mod.LocalVLLMRolloutClient, "_wait_until_ready", _fake_wait)

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

    assert len(captured_cmds) == 2
    assert "--no-disable-custom-all-reduce" in captured_cmds[0]
    assert "--disable-custom-all-reduce" in captured_cmds[1]


def test_start_ignores_stale_custom_all_reduce_log_from_previous_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_cmds: list[list[str]] = []

    def _fake_popen(*args, **kwargs):  # type: ignore[no-untyped-def]
        captured_cmds.append(list(args[0]))
        return _FakeProc()

    def _fake_wait(self):  # type: ignore[no-untyped-def]
        self._log_path.write_text("fatal: unrelated startup failure\n", encoding="utf-8")
        raise RuntimeError("startup failed")

    log_path = tmp_path / "vllm.log"
    log_path.write_text(
        "stale: custom_all_reduce.cuh invalid argument from previous run\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(vllm_mod, "_detect_vllm_log_request_flag_style", lambda python_executable: "enable")
    monkeypatch.setattr(vllm_mod, "_detect_vllm_custom_all_reduce_flag_style", lambda python_executable: "toggle")
    monkeypatch.setattr(vllm_mod.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(vllm_mod.LocalVLLMRolloutClient, "_wait_until_ready", _fake_wait)

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
        log_path=log_path,
        owns_server=True,
    )

    with pytest.raises(RuntimeError, match="startup failed"):
        client.start()

    assert len(captured_cmds) == 1
    assert "--no-disable-custom-all-reduce" in captured_cmds[0]
    assert "--disable-custom-all-reduce" not in captured_cmds[0]
    assert log_path.read_text(encoding="utf-8") == "fatal: unrelated startup failure\n"


def test_start_prefers_disable_custom_all_reduce_when_configured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_cmds: list[list[str]] = []

    def _fake_popen(*args, **kwargs):  # type: ignore[no-untyped-def]
        captured_cmds.append(list(args[0]))
        return _FakeProc()

    monkeypatch.setattr(vllm_mod, "_detect_vllm_log_request_flag_style", lambda python_executable: "enable")
    monkeypatch.setattr(vllm_mod, "_detect_vllm_custom_all_reduce_flag_style", lambda python_executable: "toggle")
    monkeypatch.setattr(vllm_mod.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(vllm_mod.LocalVLLMRolloutClient, "_wait_until_ready", lambda self: None)

    cfg = VLLMConfig(
        enabled=True,
        gpu_ids=[6, 7],
        tensor_parallel_size=2,
        adapter_root_dir=str(tmp_path / "adapters"),
        python_executable="python",
        disable_custom_all_reduce=True,
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

    assert len(captured_cmds) == 1
    assert "--disable-custom-all-reduce" in captured_cmds[0]
    assert "--no-disable-custom-all-reduce" not in captured_cmds[0]
