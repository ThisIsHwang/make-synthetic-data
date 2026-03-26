from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Callable
from urllib import error as urllib_error
from urllib import request as urllib_request

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None  # type: ignore[assignment]

try:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase
except Exception:  # pragma: no cover - optional during lightweight tests
    PreTrainedModel = Any  # type: ignore[assignment,misc]
    PreTrainedTokenizerBase = Any  # type: ignore[assignment,misc]

from .config import GenerationConfig, VLLMConfig
from .prompting import (
    DEFAULT_TRANSLATION_PROMPT_TEMPLATE,
    collect_tokenizer_special_token_strings,
    format_translation_prompt,
    sanitize_text_for_scoring,
)
from .rl_types import Example, Rollout
from .rollout import (
    TokenDecodeConfig,
    _build_generation_chat_kwargs,
    _collect_end_of_turn_token_ids,
    _compute_logprobs_batch_with_backoff,
    _encode_prompt_rows,
    _env_flag,
    _env_int,
    _get_model_vocab_size,
    _is_rank0_process,
    _resolve_eos_token_ids,
    _safe_convert_ids_to_tokens,
    _safe_decode_ids_with_specials,
    _truncate_for_log,
    _trim_completion_ids,
    _validate_item_token_ids,
    _validate_token_ids_in_vocab,
    compute_token_char_offsets,
)
from .utils import collect_huggingface_worker_env, merge_env_overrides


logger = logging.getLogger(__name__)
_VLLM_LOG_REQUEST_FLAG_STYLE_CACHE: dict[str, str] = {}
_VLLM_CUSTOM_ALL_REDUCE_FLAG_STYLE_CACHE: dict[str, str] = {}


@dataclass
class _VLLMChoice:
    text: str
    token_ids: list[int]


def _temporarily_unset_proxy_env() -> Callable[[], None]:
    keys = (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    )
    backup: dict[str, str] = {}
    for key in keys:
        if key in os.environ:
            backup[key] = os.environ.pop(key)

    def _restore() -> None:
        for key, value in backup.items():
            os.environ[key] = value

    return _restore


def _extract_text_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "").strip().lower() != "text":
            continue
        text = item.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def _detect_vllm_log_request_flag_style(python_executable: str) -> str:
    cached = _VLLM_LOG_REQUEST_FLAG_STYLE_CACHE.get(str(python_executable))
    if cached is not None:
        return cached

    cmd = [str(python_executable), "-m", "vllm.entrypoints.openai.api_server", "--help"]
    style = "enable"
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30.0,
            check=False,
        )
        help_text = f"{proc.stdout or ''}\n{proc.stderr or ''}"
        if "--enable-log-requests" in help_text or "--no-enable-log-requests" in help_text:
            style = "enable"
        elif "--disable-log-requests" in help_text:
            style = "disable"
    except Exception as exc:
        logger.warning(
            "Failed to inspect vLLM api_server --help for log-request flags; defaulting to new-style flags: %s",
            exc,
        )

    _VLLM_LOG_REQUEST_FLAG_STYLE_CACHE[str(python_executable)] = style
    return style


def _detect_vllm_custom_all_reduce_flag_style(python_executable: str) -> str:
    cached = _VLLM_CUSTOM_ALL_REDUCE_FLAG_STYLE_CACHE.get(str(python_executable))
    if cached is not None:
        return cached

    cmd = [str(python_executable), "-m", "vllm.entrypoints.openai.api_server", "--help"]
    style = "disable"
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30.0,
            check=False,
        )
        help_text = f"{proc.stdout or ''}\n{proc.stderr or ''}"
        if "--no-disable-custom-all-reduce" in help_text:
            style = "toggle"
        elif "--disable-custom-all-reduce" in help_text:
            style = "disable"
    except Exception as exc:
        logger.warning(
            "Failed to inspect vLLM api_server --help for custom-all-reduce flags; defaulting to disable-only flags: %s",
            exc,
        )

    _VLLM_CUSTOM_ALL_REDUCE_FLAG_STYLE_CACHE[str(python_executable)] = style
    return style


def _looks_like_custom_all_reduce_startup_failure(log_text: str) -> bool:
    text = str(log_text or "").lower()
    return ("custom_all_reduce" in text or "custom all reduce" in text) and "invalid argument" in text


class LocalVLLMRolloutClient:
    def __init__(
        self,
        *,
        cfg: VLLMConfig,
        base_model_name_or_path: str,
        tokenizer_name_or_path: str | None,
        lora_rank: int,
        trust_remote_code: bool,
        dtype: str | None,
        log_path: Path,
        owns_server: bool,
    ) -> None:
        self._cfg = cfg
        self._base_model_name_or_path = str(base_model_name_or_path)
        self._tokenizer_name_or_path = str(tokenizer_name_or_path).strip() if tokenizer_name_or_path else None
        self._lora_rank = max(1, int(lora_rank))
        self._trust_remote_code = bool(trust_remote_code)
        self._dtype = str(dtype).strip().lower() if dtype else None
        self._log_path = Path(log_path)
        self._owns_server = bool(owns_server)
        self._proc: subprocess.Popen[str] | None = None
        self._log_handle: Any | None = None
        self._adapter_loaded = False
        self._warned_missing_token_ids = False
        self._disable_custom_all_reduce_active = False

    @property
    def adapter_dir(self) -> Path:
        return Path(self._cfg.adapter_root_dir or "").resolve() / "current"

    @property
    def server_base_url(self) -> str:
        return f"http://{self._cfg.host}:{int(self._cfg.port)}"

    @property
    def adapter_name(self) -> str:
        return str(self._cfg.adapter_name or "policy-lora").strip() or "policy-lora"

    def start(self) -> None:
        if not self._owns_server:
            return
        if self._proc is not None and self._proc.poll() is None:
            return

        worker_env = dict(os.environ)
        for key in ("LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            worker_env.pop(key, None)
        worker_env.pop("PYTHONHOME", None)
        hf_env = collect_huggingface_worker_env()
        env_overrides = merge_env_overrides(
            hf_env,
            {
                "CUDA_VISIBLE_DEVICES": ",".join(str(int(idx)) for idx in self._cfg.gpu_ids),
                "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
            },
        )
        if env_overrides:
            worker_env.update(env_overrides)

        python_executable = str(self._cfg.python_executable or "python").strip() or "python"
        self._disable_custom_all_reduce_active = False
        try:
            self._launch_server_process(
                python_executable=python_executable,
                worker_env=worker_env,
                disable_custom_all_reduce=False,
            )
            self._wait_until_ready()
            return
        except RuntimeError as exc:
            failure_text = self._read_recent_log_text()
            if _looks_like_custom_all_reduce_startup_failure(failure_text):
                logger.warning(
                    "vLLM startup hit custom_all_reduce failure; retrying with custom all-reduce disabled. log=%s",
                    self._log_path,
                )
                self.close()
                self._launch_server_process(
                    python_executable=python_executable,
                    worker_env=worker_env,
                    disable_custom_all_reduce=True,
                )
                self._wait_until_ready()
                return
            self.close()
            raise exc

    def close(self) -> None:
        if self._proc is not None:
            proc = self._proc
            self._proc = None
            try:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=20.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            self._adapter_loaded = False
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            finally:
                self._log_handle = None

    def _wait_until_ready(self) -> None:
        deadline = time.time() + float(self._cfg.startup_timeout_sec)
        last_error: str | None = None
        while time.time() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    "local vLLM server exited during startup "
                    f"(code={self._proc.returncode}). See log: {self._log_path}"
                )
            try:
                response = self._request_json("/v1/models", method="GET")
                if isinstance(response, dict):
                    return
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(2.0)
        raise RuntimeError(
            "Timed out waiting for local vLLM server startup "
            f"(timeout={self._cfg.startup_timeout_sec}s, last_error={last_error}, log={self._log_path})"
        )

    def _build_server_command(self, *, python_executable: str, disable_custom_all_reduce: bool) -> list[str]:
        log_request_flag_style = _detect_vllm_log_request_flag_style(python_executable)
        custom_all_reduce_flag_style = _detect_vllm_custom_all_reduce_flag_style(python_executable)
        cmd = [
            python_executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self._base_model_name_or_path,
            "--host",
            str(self._cfg.host),
            "--port",
            str(int(self._cfg.port)),
            "--served-model-name",
            str(self._cfg.served_model_name or "policy-base"),
            "--enable-lora",
            "--max-lora-rank",
            str(self._lora_rank),
            "--tensor-parallel-size",
            str(int(self._cfg.tensor_parallel_size or 1)),
            "--gpu-memory-utilization",
            str(float(self._cfg.gpu_memory_utilization)),
        ]
        if self._tokenizer_name_or_path:
            cmd.extend(["--tokenizer", self._tokenizer_name_or_path])
        if self._trust_remote_code:
            cmd.append("--trust-remote-code")
        if self._dtype:
            cmd.extend(["--dtype", self._dtype])
        if self._cfg.max_model_len is not None:
            cmd.extend(["--max-model-len", str(int(self._cfg.max_model_len))])
        if self._cfg.max_num_seqs is not None:
            cmd.extend(["--max-num-seqs", str(int(self._cfg.max_num_seqs))])
        if self._cfg.max_num_batched_tokens is not None:
            cmd.extend(["--max-num-batched-tokens", str(int(self._cfg.max_num_batched_tokens))])
        if log_request_flag_style == "disable":
            if bool(self._cfg.disable_log_requests):
                cmd.append("--disable-log-requests")
        else:
            cmd.append("--no-enable-log-requests" if bool(self._cfg.disable_log_requests) else "--enable-log-requests")
        if custom_all_reduce_flag_style == "toggle":
            cmd.append(
                "--disable-custom-all-reduce"
                if disable_custom_all_reduce
                else "--no-disable-custom-all-reduce"
            )
        elif disable_custom_all_reduce:
            cmd.append("--disable-custom-all-reduce")
        if bool(self._cfg.enforce_eager):
            cmd.append("--enforce-eager")
        return cmd

    def _launch_server_process(
        self,
        *,
        python_executable: str,
        worker_env: dict[str, str],
        disable_custom_all_reduce: bool,
    ) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self._log_path.open("a", encoding="utf-8")
        cmd = self._build_server_command(
            python_executable=python_executable,
            disable_custom_all_reduce=disable_custom_all_reduce,
        )
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=worker_env,
            )
        except Exception as exc:
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None
            raise RuntimeError(f"Failed to start local vLLM server: cmd={cmd}") from exc
        self._disable_custom_all_reduce_active = bool(disable_custom_all_reduce)
        logger.info(
            "Started local vLLM server pid=%s host=%s port=%s gpus=%s disable_custom_all_reduce=%s log=%s",
            getattr(self._proc, "pid", None),
            self._cfg.host,
            int(self._cfg.port),
            self._cfg.gpu_ids,
            disable_custom_all_reduce,
            self._log_path,
        )

    def _read_recent_log_text(self, max_chars: int = 20000) -> str:
        try:
            if self._log_handle is not None:
                self._log_handle.flush()
        except Exception:
            pass
        try:
            text = self._log_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""
        if len(text) <= max_chars:
            return text
        return text[-max_chars:]

    def _request_json(self, path: str, *, method: str = "POST", payload: dict[str, Any] | None = None) -> Any:
        url = f"{self.server_base_url}{path}"
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        restore_proxy_env = _temporarily_unset_proxy_env()
        try:
            opener = urllib_request.build_opener(urllib_request.ProxyHandler({}))
            with opener.open(req, timeout=float(self._cfg.request_timeout_sec)) as resp:
                body = resp.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"vLLM request failed status={exc.code} path={path} body={body}") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"vLLM request failed path={path}: {exc}") from exc
        finally:
            restore_proxy_env()
        try:
            return json.loads(body)
        except Exception:
            return body

    def unload_adapter(self) -> None:
        if not self._owns_server or not self._adapter_loaded:
            return
        try:
            _ = self._request_json(
                "/v1/unload_lora_adapter",
                method="POST",
                payload={"lora_name": self.adapter_name},
            )
        except Exception as exc:
            logger.warning("Failed to unload vLLM LoRA adapter %s: %s", self.adapter_name, exc)
        self._adapter_loaded = False

    def load_adapter(self, adapter_path: Path) -> None:
        if not self._owns_server:
            return
        response = self._request_json(
            "/v1/load_lora_adapter",
            method="POST",
            payload={"lora_name": self.adapter_name, "lora_path": str(adapter_path)},
        )
        logger.info("Loaded vLLM LoRA adapter %s from %s response=%s", self.adapter_name, adapter_path, response)
        self._adapter_loaded = True

    def _generate_one_prompt(
        self,
        *,
        rendered_prompt_text: str,
        gen_cfg: GenerationConfig,
        stop_token_ids: list[int],
    ) -> list[_VLLMChoice]:
        do_sample = bool(gen_cfg.do_sample and gen_cfg.temperature > 0)
        payload: dict[str, Any] = {
            "model": self.adapter_name,
            "prompt": rendered_prompt_text,
            "max_tokens": int(gen_cfg.max_new_tokens),
            "n": max(1, int(gen_cfg.num_samples_per_prompt)),
            "stream": False,
            "temperature": float(gen_cfg.temperature) if do_sample else 0.0,
            "top_p": float(gen_cfg.top_p) if do_sample else 1.0,
            "top_k": int(gen_cfg.top_k) if do_sample else -1,
            "repetition_penalty": float(gen_cfg.repetition_penalty),
            "return_token_ids": True,
            "stop_token_ids": [int(tok) for tok in stop_token_ids],
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
        }
        response = self._request_json("/v1/completions", method="POST", payload=payload)
        if not isinstance(response, dict):
            raise RuntimeError(f"Unexpected vLLM response type: {type(response).__name__}")
        choices = response.get("choices")
        if not isinstance(choices, list):
            raise RuntimeError("vLLM response is missing choices.")

        parsed: list[_VLLMChoice] = []
        for raw_choice in choices:
            if not isinstance(raw_choice, dict):
                raise RuntimeError(f"vLLM response contains invalid choice: {raw_choice!r}")
            token_ids_raw = raw_choice.get("token_ids")
            token_ids: list[int] = []
            if isinstance(token_ids_raw, list):
                token_ids = [int(tok) for tok in token_ids_raw]
            text = raw_choice.get("text")
            if not isinstance(text, str):
                text = _extract_text_content(raw_choice.get("message", {}).get("content"))
            parsed.append(_VLLMChoice(text=str(text or ""), token_ids=token_ids))

        requested = max(1, int(gen_cfg.num_samples_per_prompt))
        if len(parsed) != requested:
            raise RuntimeError(
                "vLLM returned a mismatched number of choices: "
                f"requested={requested} returned={len(parsed)}"
            )
        return parsed

    def generate_choices(
        self,
        *,
        rendered_prompt_texts: list[str],
        gen_cfg: GenerationConfig,
        stop_token_ids: list[int],
        show_progress: bool = False,
        progress_desc: str | None = None,
    ) -> list[list[_VLLMChoice]]:
        if not rendered_prompt_texts:
            return []
        max_workers = max(1, min(len(rendered_prompt_texts), int(self._cfg.max_num_seqs or 8), 8))
        results: list[list[_VLLMChoice] | None] = [None] * len(rendered_prompt_texts)

        iterable = list(enumerate(rendered_prompt_texts))
        bar = None
        if show_progress and tqdm is not None:
            bar = tqdm(
                total=len(rendered_prompt_texts),
                desc=progress_desc or "vllm rollout",
                leave=False,
                mininterval=2.0,
            )

        try:
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="vllm-rollout") as executor:
                futures = {
                    executor.submit(
                        self._generate_one_prompt,
                        rendered_prompt_text=prompt_text,
                        gen_cfg=gen_cfg,
                        stop_token_ids=stop_token_ids,
                    ): idx
                    for idx, prompt_text in iterable
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    results[idx] = future.result()
                    if bar is not None:
                        bar.update(1)
        finally:
            if bar is not None:
                bar.close()

        out: list[list[_VLLMChoice]] = []
        for idx, row in enumerate(results):
            if row is None:
                raise RuntimeError(f"Missing vLLM rollout result at index={idx}")
            out.append(row)
        return out


def generate_rollouts_vllm(
    *,
    examples: list[Example],
    policy_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    gen_cfg: GenerationConfig,
    device: str,
    vllm_rollout_client: LocalVLLMRolloutClient,
    ref_model: PreTrainedModel | None = None,
    ref_device: str | None = None,
    ref_logprob_fn: Callable[[list[int], list[int]], list[float]] | None = None,
    prompt_template: str | None = None,
    show_progress: bool = False,
    progress_desc: str | None = None,
    compute_old_logprobs: bool = True,
    compute_token_offsets: bool = True,
    include_prompt_input_ids: bool = True,
    prompt_instance_ids: list[str] | None = None,
) -> list[Rollout]:
    if not examples:
        return []
    if prompt_instance_ids is not None and len(prompt_instance_ids) != len(examples):
        raise ValueError(
            "prompt_instance_ids and examples length mismatch: "
            f"{len(prompt_instance_ids)} != {len(examples)}"
        )

    pad_token_id = tokenizer.pad_token_id
    model_eos = getattr(getattr(policy_model, "generation_config", None), "eos_token_id", None)
    eot_token_ids = _collect_end_of_turn_token_ids(tokenizer)
    eos_token_ids = _resolve_eos_token_ids(tokenizer.eos_token_id, model_eos, extra_token_ids=eot_token_ids)
    if pad_token_id is None and eos_token_ids:
        pad_token_id = eos_token_ids[0]

    generation_chat_kwargs = _build_generation_chat_kwargs(gen_cfg)
    thinking_enabled = bool(generation_chat_kwargs.get("enable_thinking"))
    if eot_token_ids:
        logger.info("generate_rollouts_vllm: end-of-turn stop token ids enabled: %s", eot_token_ids)
    if thinking_enabled and eot_token_ids:
        logger.warning(
            "generate_rollouts_vllm: chat_template_kwargs.enable_thinking=true while end-of-turn stop token ids are active."
        )

    policy_model.eval()
    if ref_model is not None:
        ref_model.eval()

    decode_cfg = TokenDecodeConfig()
    policy_vocab_size = _get_model_vocab_size(policy_model)
    ref_dev = ref_device or device
    special_token_strings = collect_tokenizer_special_token_strings(tokenizer)
    raw_io_log_enabled = _env_flag("GEMMA27_RL_LOG_RAW_IO", default=False)
    raw_io_log_all_ranks = _env_flag("GEMMA27_RL_LOG_RAW_IO_ALL_RANKS", default=False)
    raw_io_max_chars = _env_int("GEMMA27_RL_LOG_RAW_IO_MAX_CHARS", default=20000, minimum=256)
    raw_io_max_rows = _env_int("GEMMA27_RL_LOG_RAW_IO_MAX_ROWS", default=0, minimum=0)
    should_log_raw_io = raw_io_log_enabled and (raw_io_log_all_ranks or _is_rank0_process())
    raw_phase = str(progress_desc or "rollout")

    prompt_texts: list[str] = [
        format_translation_prompt(
            ex,
            template=prompt_template or DEFAULT_TRANSLATION_PROMPT_TEMPLATE,
        )
        for ex in examples
    ]
    prompt_id_rows = _encode_prompt_rows(
        tokenizer=tokenizer,
        prompt_texts=prompt_texts,
        gen_cfg=gen_cfg,
        pad_token_id=pad_token_id,
    )
    rendered_prompt_texts = [
        tokenizer.decode(
            prompt_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for prompt_ids in prompt_id_rows
    ]

    if should_log_raw_io:
        for ex_idx, ex in enumerate(examples):
            if raw_io_max_rows > 0 and ex_idx >= raw_io_max_rows:
                break
            prompt_ids = prompt_id_rows[ex_idx] if ex_idx < len(prompt_id_rows) else []
            prompt_tokens = _safe_convert_ids_to_tokens(tokenizer, prompt_ids)
            prompt_decoded_with_specials = _safe_decode_ids_with_specials(tokenizer, prompt_ids)
            rendered_prompt = rendered_prompt_texts[ex_idx] if ex_idx < len(rendered_prompt_texts) else ""
            logger.info(
                "[raw-io][%s][input] ex_idx=%s example_id=%s prompt=%r rendered_prompt=%r prompt_ids=%s prompt_tokens=%s "
                "prompt_decoded_with_specials=%r",
                raw_phase,
                ex_idx,
                ex.example_id,
                _truncate_for_log(prompt_texts[ex_idx], raw_io_max_chars),
                _truncate_for_log(rendered_prompt, raw_io_max_chars),
                _truncate_for_log(json.dumps(prompt_ids, ensure_ascii=False), raw_io_max_chars),
                _truncate_for_log(json.dumps(prompt_tokens, ensure_ascii=False), raw_io_max_chars),
                _truncate_for_log(prompt_decoded_with_specials, raw_io_max_chars),
            )

    if policy_vocab_size is not None:
        _validate_item_token_ids(
            items=[(row, []) for row in prompt_id_rows],
            vocab_size=policy_vocab_size,
            tag="generate_rollouts_vllm.prompt_ids",
        )

    completion_rows = vllm_rollout_client.generate_choices(
        rendered_prompt_texts=rendered_prompt_texts,
        gen_cfg=gen_cfg,
        stop_token_ids=eos_token_ids,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )

    empty_completion_count = 0
    pending_policy_logprob_items: list[tuple[int, list[int], list[int]]] = []
    pending_ref_model_logprob_items: list[tuple[int, list[int], list[int]]] = []
    rollouts: list[Rollout] = []

    for ex_idx, choices in enumerate(completion_rows):
        if ex_idx >= len(examples):
            break
        ex = examples[ex_idx]
        prompt_text = prompt_texts[ex_idx]
        prompt_ids = prompt_id_rows[ex_idx]
        prompt_instance_id = (
            str(prompt_instance_ids[ex_idx])
            if prompt_instance_ids is not None and ex_idx < len(prompt_instance_ids)
            else str(ex_idx)
        )

        for sample_idx, choice in enumerate(choices):
            completion_untrimmed_ids = [int(v) for v in choice.token_ids]
            if not completion_untrimmed_ids and choice.text:
                if not vllm_rollout_client._warned_missing_token_ids:
                    logger.warning(
                        "vLLM response omitted token_ids; falling back to local retokenization of completion text."
                    )
                    vllm_rollout_client._warned_missing_token_ids = True
                tokenized = tokenizer(
                    choice.text,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )
                token_ids = tokenized.get("input_ids", []) if isinstance(tokenized, dict) else []
                completion_untrimmed_ids = [int(v) for v in list(token_ids)]

            completion_raw_ids = _trim_completion_ids(
                list(completion_untrimmed_ids),
                eos_token_ids=eos_token_ids,
                pad_token_id=pad_token_id,
            )
            completion_text = tokenizer.decode(
                completion_raw_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            completion_ids = [int(x) for x in completion_raw_ids]
            if not completion_ids:
                empty_completion_count += 1
            completion_raw_text = str(completion_text or "")
            completion_clean_text, _ = sanitize_text_for_scoring(
                completion_raw_text,
                special_tokens=special_token_strings,
            )

            if should_log_raw_io:
                output_row_idx = len(rollouts)
                if raw_io_max_rows <= 0 or output_row_idx < raw_io_max_rows:
                    completion_untrimmed_tokens = _safe_convert_ids_to_tokens(tokenizer, completion_untrimmed_ids)
                    completion_untrimmed_decoded = _safe_decode_ids_with_specials(tokenizer, completion_untrimmed_ids)
                    completion_raw_tokens = _safe_convert_ids_to_tokens(tokenizer, completion_raw_ids)
                    completion_raw_decoded = _safe_decode_ids_with_specials(tokenizer, completion_raw_ids)
                    completion_tokens = _safe_convert_ids_to_tokens(tokenizer, completion_ids)
                    completion_decoded = _safe_decode_ids_with_specials(tokenizer, completion_ids)
                    logger.info(
                        "[raw-io][%s][output] row_idx=%s ex_idx=%s sample_idx=%s example_id=%s "
                        "completion_untrimmed_ids=%s completion_untrimmed_tokens=%s completion_untrimmed_decoded_with_specials=%r "
                        "completion_raw_ids=%s completion_raw_tokens=%s completion_raw_decoded_with_specials=%r "
                        "completion_ids=%s completion_tokens=%s completion_decoded_with_specials=%r "
                        "completion_raw_text=%r completion_clean_text=%r",
                        raw_phase,
                        output_row_idx,
                        ex_idx,
                        sample_idx,
                        ex.example_id,
                        _truncate_for_log(json.dumps(completion_untrimmed_ids, ensure_ascii=False), raw_io_max_chars),
                        _truncate_for_log(json.dumps(completion_untrimmed_tokens, ensure_ascii=False), raw_io_max_chars),
                        _truncate_for_log(completion_untrimmed_decoded, raw_io_max_chars),
                        _truncate_for_log(json.dumps(completion_raw_ids, ensure_ascii=False), raw_io_max_chars),
                        _truncate_for_log(json.dumps(completion_raw_tokens, ensure_ascii=False), raw_io_max_chars),
                        _truncate_for_log(completion_raw_decoded, raw_io_max_chars),
                        _truncate_for_log(json.dumps(completion_ids, ensure_ascii=False), raw_io_max_chars),
                        _truncate_for_log(json.dumps(completion_tokens, ensure_ascii=False), raw_io_max_chars),
                        _truncate_for_log(completion_decoded, raw_io_max_chars),
                        _truncate_for_log(completion_raw_text, raw_io_max_chars),
                        _truncate_for_log(completion_clean_text, raw_io_max_chars),
                    )

            if policy_vocab_size is not None:
                _validate_token_ids_in_vocab(
                    completion_ids,
                    vocab_size=policy_vocab_size,
                    context=f"generate_rollouts_vllm.completion_ids(example_id={ex.example_id})",
                )

            old_lp: list[float] = []
            rollout_idx = len(rollouts)
            if compute_old_logprobs:
                pending_policy_logprob_items.append((rollout_idx, list(prompt_ids), list(completion_ids)))
            ref_lp = None
            if compute_old_logprobs and ref_logprob_fn is not None:
                ref_lp = [float(v) for v in ref_logprob_fn(prompt_ids, completion_ids)]
            elif compute_old_logprobs and ref_model is not None:
                pending_ref_model_logprob_items.append((rollout_idx, list(prompt_ids), list(completion_ids)))

            offsets: list[tuple[int, int]] = []
            if compute_token_offsets:
                offsets = compute_token_char_offsets(
                    tokenizer=tokenizer,
                    completion_token_ids=completion_ids,
                    decode_cfg=decode_cfg,
                    completion_text=completion_text,
                )

            rollouts.append(
                Rollout(
                    example_id=ex.example_id,
                    prompt_text=prompt_text,
                    prompt_input_ids=(prompt_ids if include_prompt_input_ids else []),
                    completion_text=completion_text,
                    completion_token_ids=completion_ids,
                    old_logprobs=old_lp,
                    ref_logprobs=ref_lp,
                    token_char_offsets=offsets,
                    src_text=ex.src_text,
                    src_lang=ex.src_lang,
                    tgt_lang=ex.tgt_lang,
                    src_lang_code=ex.src_lang_code,
                    tgt_lang_code=ex.tgt_lang_code,
                    ref_text=ex.ref_text,
                    raw_completion_token_ids=list(completion_raw_ids),
                    completion_raw_text=completion_raw_text,
                    completion_clean_text=completion_clean_text,
                    prompt_instance_id=prompt_instance_id,
                )
            )

    if compute_old_logprobs and pending_policy_logprob_items:
        policy_rows = _compute_logprobs_batch_with_backoff(
            model=policy_model,
            items=[(prompt_ids, completion_ids) for _, prompt_ids, completion_ids in pending_policy_logprob_items],
            device=device,
            tag="policy_old_logprobs_vllm",
        )
        if len(policy_rows) != len(pending_policy_logprob_items):
            raise RuntimeError(
                "policy old_logprobs batch size mismatch: "
                f"requested={len(pending_policy_logprob_items)} returned={len(policy_rows)}"
            )
        for (rollout_idx, _, _), row in zip(pending_policy_logprob_items, policy_rows):
            if rollout_idx < len(rollouts):
                rollouts[rollout_idx].old_logprobs = [float(v) for v in row.tolist()]

    if compute_old_logprobs and pending_ref_model_logprob_items and ref_model is not None:
        ref_rows = _compute_logprobs_batch_with_backoff(
            model=ref_model,
            items=[(prompt_ids, completion_ids) for _, prompt_ids, completion_ids in pending_ref_model_logprob_items],
            device=ref_dev,
            tag="reference_model_logprobs_vllm",
        )
        if len(ref_rows) != len(pending_ref_model_logprob_items):
            raise RuntimeError(
                "reference logprobs batch size mismatch: "
                f"requested={len(pending_ref_model_logprob_items)} returned={len(ref_rows)}"
            )
        for (rollout_idx, _, _), row in zip(pending_ref_model_logprob_items, ref_rows):
            if rollout_idx < len(rollouts):
                rollouts[rollout_idx].ref_logprobs = [float(v) for v in row.tolist()]

    if empty_completion_count > 0:
        logger.info(
            "generate_rollouts_vllm: observed %s empty completions after eos/pad trimming.",
            empty_completion_count,
        )

    return rollouts
