from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
import json
import logging
import math
import os
from pathlib import Path
import re
import select
import subprocess
import threading
import time
from textwrap import dedent
from typing import Any, Callable
import unicodedata
from urllib import error as urllib_error
from urllib import request as urllib_request

_TORCH_IMPORT_ERROR: Exception | None = None
try:
    import torch
except Exception as exc:  # pragma: no cover - optional during lightweight tests
    _TORCH_IMPORT_ERROR = exc
    torch = None  # type: ignore[assignment]

_TRANSFORMERS_IMPORT_ERROR: Exception | None = None
try:
    from transformers import AutoTokenizer
except Exception as exc:  # pragma: no cover - optional during lightweight tests
    _TRANSFORMERS_IMPORT_ERROR = exc
    AutoTokenizer = None  # type: ignore[assignment]

from .config import ESAConfig, GroupRankConfig, MQMConfig, MetricXConfig, XCometConfig
from .rl_types import GroupRankSample, RewardOutput, SampleForScoring
from .utils import (
    build_worker_launch_command,
    collect_huggingface_worker_env,
    merge_env_overrides,
    resolve_device,
    resolve_torch_dtype,
)


logger = logging.getLogger(__name__)


def _capture_exception(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - simple wrapper
        return exc


def _run_jobs_with_bounded_concurrency(
    *,
    executor: ThreadPoolExecutor,
    jobs: list[tuple[Any, ...]],
    worker_fn: Callable[..., Any],
    max_in_flight: int,
) -> list[Any]:
    if not jobs:
        return []

    limit = max(1, min(int(max_in_flight), len(jobs)))
    results: list[Any] = [None for _ in jobs]
    in_flight: dict[Any, int] = {}
    next_job_idx = 0

    def _submit(job_idx: int) -> None:
        future = executor.submit(worker_fn, *jobs[job_idx])
        in_flight[future] = int(job_idx)

    while next_job_idx < len(jobs) and len(in_flight) < limit:
        _submit(next_job_idx)
        next_job_idx += 1

    while in_flight:
        done, _ = wait(tuple(in_flight.keys()), return_when=FIRST_COMPLETED)
        for future in done:
            job_idx = in_flight.pop(future)
            results[job_idx] = future.result()
            if next_job_idx < len(jobs):
                _submit(next_job_idx)
                next_job_idx += 1

    return results


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return max(minimum, int(default))
    try:
        value = int(raw.strip())
    except Exception:
        return max(minimum, int(default))
    return max(minimum, value)


def _truncate_for_log(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"...[truncated {len(text) - max_chars} chars]"


_PARSE_FAILURE_LOG_LOCK = threading.Lock()


def _append_jsonl_record(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    with _PARSE_FAILURE_LOG_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)


def _record_scorer_parse_failure(
    *,
    log_path: Path | None,
    scorer_name: str,
    model_name: str,
    sample: SampleForScoring,
    enable_thinking: bool,
    stage: str,
    error: str,
    details: dict[str, Any] | None = None,
) -> None:
    if log_path is None:
        return
    payload: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "pid": int(os.getpid()),
        "scorer": scorer_name,
        "model_name": model_name,
        "stage": stage,
        "enable_thinking": bool(enable_thinking),
        "error": str(error),
        "source_lang": sample.source_lang,
        "target_lang": sample.target_lang,
        "src": sample.src,
        "mt": sample.mt,
        "ref": sample.ref,
    }
    if details:
        for key, value in details.items():
            if value is None:
                continue
            payload[str(key)] = value
    try:
        _append_jsonl_record(log_path, payload)
    except Exception as exc:
        logger.warning("Failed to append %s parse failure record to %s: %s", scorer_name, log_path, exc)


def _extract_openai_message_content_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value

    if not isinstance(value, list):
        return None

    parts: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type", "") or "").strip().lower()
        if item_type != "text":
            continue
        text = item.get("text")
        if isinstance(text, str):
            parts.append(text)

    if not parts:
        return None
    return "\n".join(parts)


def _extract_openai_response_text(
    *,
    parsed: dict[str, Any],
    scorer_name: str,
    log_io: bool,
    log_max_chars: int,
) -> str:
    choices = parsed.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"{scorer_name} API response has no choices.")

    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError(
            f"{scorer_name} API response format is unsupported; expected choices[0].message.content."
        )
    message = first.get("message")
    if not isinstance(message, dict):
        raise RuntimeError(
            f"{scorer_name} API response format is unsupported; expected choices[0].message.content."
        )

    content = _extract_openai_message_content_text(message.get("content"))
    if content is None:
        raise RuntimeError(
            f"{scorer_name} API response format is unsupported; expected choices[0].message.content."
        )

    if log_io:
        logger.info(
            "[%s-io] parsed_content=%s",
            scorer_name.lower(),
            _truncate_for_log(content, log_max_chars),
        )
    return content


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


class _ScorerSubprocessClient:
    def __init__(
        self,
        *,
        backend: str,
        python_executable: str,
        timeout_sec: float,
        config_payload: dict[str, Any],
        env_overrides: dict[str, str] | None = None,
        remote_host: str | None = None,
        remote_workdir: str | None = None,
    ) -> None:
        self._backend = backend
        self._timeout_sec = float(timeout_sec)
        self._remote_host = str(remote_host).strip() if remote_host else ""
        worker_script = Path(__file__).resolve().with_name("scorer_worker.py")
        if not worker_script.exists():
            raise FileNotFoundError(f"scorer worker script not found: {worker_script}")

        cmd = build_worker_launch_command(
            python_executable=python_executable,
            worker_script=worker_script,
            worker_module="gemma27_rl.scorer_worker",
            worker_args=["--backend", backend],
            remote_host=self._remote_host or None,
            remote_workdir=remote_workdir,
            remote_env=env_overrides if self._remote_host else None,
        )
        worker_env = dict(os.environ)
        for key in ("LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            worker_env.pop(key, None)
        # PYTHONHOME can break venv resolution and make installed packages invisible.
        worker_env.pop("PYTHONHOME", None)
        if env_overrides and not self._remote_host:
            for key, value in env_overrides.items():
                worker_env[str(key)] = str(value)

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                bufsize=1,
                env=worker_env,
            )
        except Exception as exc:
            location = f" via ssh host={self._remote_host}" if self._remote_host else ""
            raise RuntimeError(f"Failed to start {backend} scorer worker{location}: cmd={cmd}") from exc

        logger.info(
            "%s scorer worker process started (pid=%s, host=%s). waiting for init...",
            backend,
            getattr(self._proc, "pid", None),
            self._remote_host or "local",
        )
        try:
            init_resp = self.request({"type": "init", "config": config_payload})
        except Exception:
            self.close()
            raise
        if not bool(init_resp.get("ok", False)):
            self.close()
            err = init_resp.get("error", "unknown error")
            worker_tb = init_resp.get("traceback")
            worker_runtime = init_resp.get("runtime")
            tb_text = f"\nworker_traceback:\n{worker_tb}" if worker_tb else ""
            runtime_text = f"\nworker_runtime:\n{worker_runtime}" if worker_runtime else ""
            raise RuntimeError(
                f"{backend} scorer worker init failed: {err}{tb_text}{runtime_text}"
            )

    def _assert_alive(self) -> None:
        if self._proc.poll() is not None:
            raise RuntimeError(f"{self._backend} scorer worker exited unexpectedly with code={self._proc.returncode}")

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._assert_alive()
        assert self._proc.stdin is not None
        assert self._proc.stdout is not None
        request_type = str(payload.get("type", "request")).strip() or "request"

        try:
            self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
        except Exception as exc:
            raise RuntimeError(f"Failed to send request to {self._backend} scorer worker.") from exc

        started = time.monotonic()
        next_wait_log_sec = 30.0
        while True:
            elapsed = time.monotonic() - started
            remaining = self._timeout_sec - elapsed
            if remaining <= 0:
                raise TimeoutError(
                    f"{self._backend} scorer worker timed out after {self._timeout_sec}s while waiting for {request_type}"
                )
            wait_slice = min(2.0, remaining)
            ready, _, _ = select.select([self._proc.stdout], [], [], wait_slice)
            if ready:
                break
            if elapsed >= next_wait_log_sec:
                logger.info(
                    "%s scorer worker still waiting for %s response: elapsed=%.1fs timeout=%.1fs host=%s",
                    self._backend,
                    request_type,
                    elapsed,
                    self._timeout_sec,
                    self._remote_host or "local",
                )
                next_wait_log_sec += 30.0

        try:
            line = self._proc.stdout.readline()
        except Exception as exc:
            raise RuntimeError(f"Failed to read response from {self._backend} scorer worker.") from exc

        if not line:
            self._assert_alive()
            raise RuntimeError(f"{self._backend} scorer worker returned empty response.")

        try:
            resp = json.loads(line)
        except Exception as exc:
            raise RuntimeError(
                f"Invalid JSON response from {self._backend} scorer worker: {line[:200]!r}"
            ) from exc

        if not isinstance(resp, dict):
            raise RuntimeError(f"Unexpected response type from {self._backend} scorer worker: {type(resp)!r}")
        return resp

    def close(self) -> None:
        if getattr(self, "_proc", None) is None:
            return
        proc = self._proc
        if proc.poll() is None:
            try:
                self.request({"type": "close"})
            except Exception:
                pass
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self._proc = None  # type: ignore[assignment]

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass


def _parse_cuda_device_index(device: str | None) -> int | None:
    if device is None:
        return None
    text = str(device).strip().lower()
    if text.startswith("cuda:"):
        suffix = text.split(":", 1)[1].strip()
        if suffix.isdigit():
            return int(suffix)
    return None


# NOTE: GEMBA-MQM prompts and few-shots below are copied from:
# /home/seungyoonee/initial_translation/evalmt/metrics/gemba_mqm_metric.py
GEMBA_SYSTEM_PROMPT = (
    "You are an annotator for the quality of machine translation. "
    "Your task is to identify errors and assess the quality of the translation."
)

GEMBA_USER_TASK_PROMPT = (
    "Based on the source segment and machine translation surrounded with triple backticks, "
    "identify error types in the translation and classify them. The categories of errors are: "
    "accuracy (addition, mistranslation, omission, untranslated text), fluency (character encoding, "
    "grammar, inconsistency, punctuation, register, spelling), style (awkward), terminology "
    "(inappropriate for context, inconsistent use), non-translation, other, or no-error.\n"
    "Each error is classified as one of three categories: critical, major, and minor. "
    "Critical errors inhibit comprehension of the text. Major errors disrupt the flow, but what "
    "the text is trying to say is still understandable. Minor errors are technically errors, "
    "but do not disrupt the flow or hinder comprehension."
)

GEMBA_MQM_REPAIR_SYSTEM_PROMPT = (
    "You normalize machine translation MQM annotations into a strict JSON format."
)

GEMBA_MQM_REPAIR_PROMPT_TEMPLATE = (
    "Rewrite the evaluator output below into the exact MQM JSON format.\n\n"
    "Return only valid JSON with this schema:\n"
    '{{\n'
    '  "errors": [\n'
    '    {{\n'
    '      "severity": "critical",\n'
    '      "type": "accuracy/mistranslation",\n'
    '      "target_span": "exact target text or null",\n'
    '      "source_span": "exact source text or null",\n'
    '      "confidence": 0.92\n'
    '    }}\n'
    '  ]\n'
    '}}\n\n'
    "Rules:\n"
    '- severity must be one of "critical", "major", or "minor".\n'
    '- type must be category/subcategory in lowercase.\n'
    "- Copy target_span exactly from the translation when possible.\n"
    "- For omissions or anchorless errors, use target_span=null and source_span when possible.\n"
    "- confidence must be a number between 0 and 1.\n"
    '- If there are no errors, return {{"errors": []}}.\n'
    "- Do not include explanations or any text outside the JSON object.\n\n"
    "Source:\n"
    "```{source_seg}```\n"
    "Translation:\n"
    "```{target_seg}```\n"
    "Evaluator output:\n"
    "```{raw_output}```"
)

GEMBA_FEWSHOT_USER_1 = dedent(
    """\
    English source:
    ```I do apologise about this, we must gain permission from the account holder to discuss an order with another person, I apologise if this was done previously, however, I would not be able to discuss this with yourself without the account holders permission.```
    German translation:
    ```Ich entschuldige mich dafür, wir müssen die Erlaubnis einholen, um eine Bestellung mit einer anderen Person zu besprechen. Ich entschuldige mich, falls dies zuvor geschehen wäre, aber ohne die Erlaubnis des Kontoinhabers wäre ich nicht in der Lage, dies mit dir involvement.```

    Based on the source segment and machine translation surrounded with triple backticks, identify error types in the translation and classify them. The categories of errors are: accuracy (addition, mistranslation, omission, untranslated text), fluency (character encoding, grammar, inconsistency, punctuation, register, spelling), style (awkward), terminology (inappropriate for context, inconsistent use), non-translation, other, or no-error.
    Each error is classified as one of three categories: critical, major, and minor. Critical errors inhibit comprehension of the text. Major errors disrupt the flow, but what the text is trying to say is still understandable. Minor errors are technically errors, but do not disrupt the flow or hinder comprehension.
    """
).strip()

GEMBA_FEWSHOT_ASSISTANT_1 = dedent(
    """\
    {
      "errors": [
        {"severity": "major", "type": "accuracy/mistranslation", "target_span": "involvement", "source_span": null, "confidence": 0.98},
        {"severity": "major", "type": "accuracy/omission", "target_span": "the account holder", "source_span": null, "confidence": 0.95},
        {"severity": "minor", "type": "fluency/grammar", "target_span": "wäre", "source_span": null, "confidence": 0.93},
        {"severity": "minor", "type": "fluency/register", "target_span": "dir", "source_span": null, "confidence": 0.91}
      ]
    }
    """
).strip()

GEMBA_FEWSHOT_USER_2 = dedent(
    """\
    English source:
    ```Talks have resumed in Vienna to try to revive the nuclear pact, with both sides trying to gauge the prospects of success after the latest exchanges in the stop-start negotiations.```
    Czech translation:
    ```Ve Vídni se ve Vídni obnovily rozhovory o oživení jaderného paktu, přičemž obě partaje se snaží posoudit vyhlídky na úspěch po posledních výměnách v jednáních.```

    Based on the source segment and machine translation surrounded with triple backticks, identify error types in the translation and classify them. The categories of errors are: accuracy (addition, mistranslation, omission, untranslated text), fluency (character encoding, grammar, inconsistency, punctuation, register, spelling), style (awkward), terminology (inappropriate for context, inconsistent use), non-translation, other, or no-error.
    Each error is classified as one of three categories: critical, major, and minor. Critical errors inhibit comprehension of the text. Major errors disrupt the flow, but what the text is trying to say is still understandable. Minor errors are technically errors, but do not disrupt the flow or hinder comprehension.
    """
).strip()

GEMBA_FEWSHOT_ASSISTANT_2 = dedent(
    """\
    {
      "errors": [
        {"severity": "major", "type": "accuracy/addition", "target_span": "ve Vídni", "source_span": null, "confidence": 0.97},
        {"severity": "major", "type": "accuracy/omission", "target_span": null, "source_span": "the stop-start", "confidence": 0.94},
        {"severity": "minor", "type": "terminology/inappropriate for context", "target_span": "partaje", "source_span": null, "confidence": 0.9}
      ]
    }
    """
).strip()

GEMBA_FEWSHOT_USER_3 = dedent(
    """\
    Chinese source:
    ```大众点评乌鲁木齐家居卖场频道为您提供高铁居然之家地址，电话，营业时间等最新商户信息，找装修公司，就上大众点评```
    English translation:
    ```Urumqi Home Furnishing Store Channel provides you with the latest business information such as the address, telephone number, business hours, etc., of high-speed rail, and find a decoration company, and go to the reviews.```

    Based on the source segment and machine translation surrounded with triple backticks, identify error types in the translation and classify them. The categories of errors are: accuracy (addition, mistranslation, omission, untranslated text), fluency (character encoding, grammar, inconsistency, punctuation, register, spelling), style (awkward), terminology (inappropriate for context, inconsistent use), non-translation, other, or no-error.
    Each error is classified as one of three categories: critical, major, and minor. Critical errors inhibit comprehension of the text. Major errors disrupt the flow, but what the text is trying to say is still understandable. Minor errors are technically errors, but do not disrupt the flow or hinder comprehension.
    """
).strip()

GEMBA_FEWSHOT_ASSISTANT_3 = dedent(
    """\
    {
      "errors": [
        {"severity": "critical", "type": "accuracy/addition", "target_span": "of high-speed rail", "source_span": null, "confidence": 0.98},
        {"severity": "major", "type": "accuracy/mistranslation", "target_span": "go to the reviews", "source_span": null, "confidence": 0.96},
        {"severity": "minor", "type": "style/awkward", "target_span": "etc.,", "source_span": null, "confidence": 0.88}
      ]
    }
    """
).strip()


_GEMBA_ERROR_LINE_PATTERN = re.compile(
    r"^((?:accuracy|fluency|style|terminology|non-translation|other)"
    r"(?:\s*/\s*[^:]+?)?)\s*(?:-|:|–|—)\s*(.+)$",
    flags=re.IGNORECASE,
)
_MQM_PARSE_ATTEMPTS_WITHOUT_THINKING = 1
_MQM_PARSE_ATTEMPTS_WITH_THINKING = 1
_ESA_SCORE_ATTEMPTS_WITHOUT_THINKING = 1
_ESA_SCORE_ATTEMPTS_WITH_THINKING = 10


class GembaParseError(ValueError):
    pass


def _configured_enable_thinking(chat_template_kwargs: dict[str, Any] | None) -> bool:
    return bool((chat_template_kwargs or {}).get("enable_thinking"))


def _mqm_parse_phase_specs(chat_template_kwargs: dict[str, Any] | None) -> tuple[tuple[bool, int], ...]:
    if _configured_enable_thinking(chat_template_kwargs):
        return ((True, _MQM_PARSE_ATTEMPTS_WITH_THINKING),)
    return (
        (False, _MQM_PARSE_ATTEMPTS_WITHOUT_THINKING),
        (True, _MQM_PARSE_ATTEMPTS_WITH_THINKING),
    )


def _esa_score_phase_specs(chat_template_kwargs: dict[str, Any] | None) -> tuple[tuple[bool, int], ...]:
    if _configured_enable_thinking(chat_template_kwargs):
        return ((True, _ESA_SCORE_ATTEMPTS_WITH_THINKING),)
    return (
        (False, _ESA_SCORE_ATTEMPTS_WITHOUT_THINKING),
        (True, _ESA_SCORE_ATTEMPTS_WITH_THINKING),
    )


def _override_enable_thinking(
    chat_template_kwargs: dict[str, Any] | None,
    *,
    enable_thinking: bool,
) -> dict[str, Any]:
    out = dict(chat_template_kwargs or {})
    out["enable_thinking"] = bool(enable_thinking)
    return out


def _resolve_sample_lang_pair(
    sample: SampleForScoring,
    *,
    default_source_lang: str,
    default_target_lang: str,
) -> tuple[str, str]:
    source_lang = str(sample.source_lang or default_source_lang or "").strip() or str(default_source_lang)
    target_lang = str(sample.target_lang or default_target_lang or "").strip() or str(default_target_lang)
    return source_lang, target_lang


def _gemba_json_output_instructions(*, allowed_levels: tuple[str, ...]) -> str:
    level_text = ", ".join(f'"{level}"' for level in allowed_levels)
    return (
        "\n\nReturn only valid JSON with this schema:\n"
        '{\n'
        '  "errors": [\n'
        '    {\n'
        '      "severity": "major",\n'
        '      "type": "accuracy/mistranslation",\n'
        '      "target_span": "exact target text or null",\n'
        '      "source_span": "exact source text or null",\n'
        '      "confidence": 0.92\n'
        '    }\n'
        '  ]\n'
        '}\n'
        "Rules:\n"
        f"- severity must be lowercase and one of: {level_text}.\n"
        "- type must be category/subcategory in lowercase.\n"
        "- Use target_span when you can anchor the error to exact translated text.\n"
        "- For omissions or anchorless errors, use target_span=null and source_span when possible.\n"
        "- confidence must be a number between 0 and 1.\n"
        '- If there are no errors, return {"errors": []}.\n'
        "- Do not include explanations or text outside the JSON object."
    )


def _normalize_gemba_response_line(raw_line: str) -> str:
    line = str(raw_line).strip()
    if not line:
        return ""
    line = re.sub(r"^\s*(?:[-*+]|(?:\d+[\.\)]))\s*", "", line)
    line = line.replace("**", "").replace("__", "").replace("`", "").strip()
    return line


def _is_gemba_no_error_line(line: str) -> bool:
    text = str(line).strip().lower()
    return text in {"no-error", "no error"}


def _looks_like_gemba_level_header(line: str, *, allowed_levels: tuple[str, ...]) -> bool:
    line_l = str(line).strip().lower()
    if not line_l:
        return False
    for allowed in allowed_levels:
        header = f"{allowed}:"
        if line_l == header or line_l.startswith(header):
            return True
    return False


def _strip_matching_quotes(text: str) -> str:
    out = str(text or "").strip()
    if not out:
        return ""
    pairs = (('"', '"'), ("“", "”"), ("'", "'"), ("`", "`"))
    changed = True
    while changed:
        changed = False
        for open_quote, close_quote in pairs:
            if out.startswith(open_quote) and out.endswith(close_quote) and len(out) > (len(open_quote) + len(close_quote)):
                out = out[len(open_quote): len(out) - len(close_quote)].strip()
                changed = True
                break
    return out


def _extract_gemba_quoted_text(line: str) -> str | None:
    candidates: list[str] = []
    text = str(line or "")
    for open_quote, close_quote in (('"', '"'), ("“", "”"), ("'", "'"), ("`", "`")):
        start = text.find(open_quote)
        end = text.rfind(close_quote)
        if start < 0 or end <= start:
            continue
        value = text[start + len(open_quote):end].strip()
        if not value:
            continue
        if value.lower() in {"no-error", "no error"}:
            continue
        candidates.append(value)
    if not candidates:
        return None
    return max(candidates, key=len)


def _normalize_gemba_error_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _split_gemba_error_line(line: str) -> tuple[str, str] | None:
    match = _GEMBA_ERROR_LINE_PATTERN.match(str(line).strip())
    if match is None:
        return None
    error_type = _normalize_gemba_error_type(match.group(1))
    detail = str(match.group(2)).strip()
    if not error_type or not detail:
        return None
    return error_type, detail


def _extract_gemba_error_type(line: str) -> str | None:
    parsed = _split_gemba_error_line(line)
    if parsed is None:
        return None
    return parsed[0]


def _normalize_optional_gemba_span(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _normalize_gemba_confidence(value: Any) -> float:
    if value is None:
        return 1.0
    try:
        parsed = float(value)
    except Exception as exc:
        raise GembaParseError(f"GEMBA confidence must be numeric, got {value!r}.") from exc
    if not math.isfinite(parsed):
        raise GembaParseError("GEMBA confidence must be finite.")
    return min(1.0, max(0.0, parsed))


def _strip_json_code_fence(text: str) -> str:
    stripped = str(text or "").strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _extract_balanced_json_object(text: str) -> str | None:
    raw = str(text or "")
    for start in [idx for idx, ch in enumerate(raw) if ch == "{"]:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(raw)):
            ch = raw[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return raw[start : idx + 1]
    return None


def _repair_unescaped_json_inner_quotes(text: str | None) -> str:
    raw = str(text or "")
    if not raw:
        return raw

    out: list[str] = []
    in_string = False
    escape = False
    length = len(raw)

    for idx, ch in enumerate(raw):
        if escape:
            out.append(ch)
            escape = False
            continue

        if ch == "\\" and in_string:
            out.append(ch)
            escape = True
            continue

        if ch != '"':
            out.append(ch)
            continue

        if not in_string:
            in_string = True
            out.append(ch)
            continue

        next_idx = idx + 1
        while next_idx < length and raw[next_idx] in {" ", "\t", "\r", "\n"}:
            next_idx += 1
        next_char = raw[next_idx] if next_idx < length else ""
        if next_char in {",", "}", "]", ":"} or next_char == "":
            in_string = False
            out.append(ch)
            continue

        out.append('\\"')

    return "".join(out)


_LENIENT_GEMBA_JSON_FIELD_PATTERN = re.compile(
    r'"(severity|type|target_span|source_span|confidence)"\s*:',
    flags=re.IGNORECASE,
)


def _decode_lenient_json_string(text: str) -> str:
    raw = str(text or "")
    if not raw:
        return ""

    out: list[str] = []
    idx = 0
    length = len(raw)
    escape_map = {
        '"': '"',
        "\\": "\\",
        "/": "/",
        "b": "\b",
        "f": "\f",
        "n": "\n",
        "r": "\r",
        "t": "\t",
    }
    while idx < length:
        ch = raw[idx]
        if ch != "\\":
            out.append(ch)
            idx += 1
            continue
        if idx + 1 >= length:
            out.append("\\")
            idx += 1
            continue
        nxt = raw[idx + 1]
        if nxt == "u" and idx + 5 < length:
            hex_part = raw[idx + 2 : idx + 6]
            try:
                out.append(chr(int(hex_part, 16)))
                idx += 6
                continue
            except Exception:
                pass
        out.append(escape_map.get(nxt, nxt))
        idx += 2
    return "".join(out)


def _extract_lenient_gemba_errors_array_body(text: str | None) -> str | None:
    raw = str(text or "")
    if not raw:
        return None
    match = re.search(r'"errors"\s*:\s*\[', raw, flags=re.IGNORECASE)
    if match is None:
        return None

    start_idx = match.end() - 1
    depth = 0
    in_string = False
    escape = False
    for idx in range(start_idx, len(raw)):
        ch = raw[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return raw[start_idx + 1 : idx]
    if depth > 0:
        return raw[start_idx + 1 :]
    return None


def _extract_lenient_json_object_fragments(text: str | None) -> list[str]:
    raw = str(text or "")
    if not raw:
        return []

    fragments: list[str] = []
    depth = 0
    start_idx: int | None = None
    in_string = False
    escape = False
    for idx, ch in enumerate(raw):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start_idx = idx
            depth += 1
        elif ch == "}":
            if depth <= 0:
                continue
            depth -= 1
            if depth == 0 and start_idx is not None:
                fragments.append(raw[start_idx : idx + 1])
                start_idx = None
    if depth > 0 and start_idx is not None:
        fragments.append(raw[start_idx:])
    return fragments


def _parse_lenient_gemba_json_value(raw_value: str, *, field_name: str) -> Any:
    text = str(raw_value or "").strip().rstrip(",").strip()
    if field_name in {"target_span", "source_span"}:
        if not text or text.lower() == "null":
            return None
    if field_name == "confidence":
        match = re.search(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)
        if match is None:
            raise GembaParseError(f"GEMBA confidence must be numeric, got {raw_value!r}.")
        return float(match.group(0))

    if text.startswith('"'):
        if len(text) >= 2 and text.endswith('"'):
            text = text[1:-1]
        else:
            text = text[1:]
    elif text.lower() == "null":
        return None

    return _decode_lenient_json_string(text)


def _parse_lenient_gemba_error_object(fragment: str) -> dict[str, Any] | None:
    matches = list(_LENIENT_GEMBA_JSON_FIELD_PATTERN.finditer(str(fragment or "")))
    if not matches:
        return None

    parsed: dict[str, Any] = {}
    for idx, match in enumerate(matches):
        field_name = str(match.group(1)).strip().lower()
        value_start = match.end()
        value_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(fragment)
        raw_value = str(fragment[value_start:value_end]).strip()
        if raw_value.endswith("}"):
            raw_value = raw_value[:-1].rstrip()
        parsed[field_name] = _parse_lenient_gemba_json_value(raw_value, field_name=field_name)

    if "severity" not in parsed or "type" not in parsed:
        return None
    if "target_span" not in parsed:
        parsed["target_span"] = None
    if "source_span" not in parsed:
        parsed["source_span"] = None
    if "confidence" not in parsed:
        parsed["confidence"] = 1.0
    return parsed


def _parse_lenient_gemba_json_errors(
    model_output: str | None,
    *,
    allowed_levels: tuple[str, ...],
    scorer_name: str,
) -> list[dict[str, Any]] | None:
    errors_body = _extract_lenient_gemba_errors_array_body(model_output)
    if errors_body is None:
        return None
    if not str(errors_body).strip():
        return []

    fragments = _extract_lenient_json_object_fragments(errors_body)
    if not fragments:
        return []

    structured_errors: list[dict[str, Any]] = []
    for fragment in fragments:
        parsed_item = _parse_lenient_gemba_error_object(fragment)
        if parsed_item is None:
            return None
        structured_errors.append(
            _normalize_gemba_structured_error(
                parsed_item,
                allowed_levels=allowed_levels,
                scorer_name=scorer_name,
            )
        )
    return structured_errors


def _try_parse_json_object(text: str | None) -> dict[str, Any] | None:
    candidates: list[str] = []
    stripped = str(text or "").strip()
    if stripped:
        candidates.append(stripped)
        repaired = _repair_unescaped_json_inner_quotes(stripped)
        if repaired and repaired not in candidates:
            candidates.append(repaired)
        fenced = _strip_json_code_fence(stripped)
        if fenced and fenced not in candidates:
            candidates.append(fenced)
        repaired_fenced = _repair_unescaped_json_inner_quotes(fenced)
        if repaired_fenced and repaired_fenced not in candidates:
            candidates.append(repaired_fenced)
        balanced = _extract_balanced_json_object(fenced)
        if balanced and balanced not in candidates:
            candidates.append(balanced)
        repaired_balanced = _extract_balanced_json_object(repaired_fenced)
        if repaired_balanced and repaired_balanced not in candidates:
            candidates.append(repaired_balanced)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _normalize_gemba_structured_error(
    item: Any,
    *,
    allowed_levels: tuple[str, ...],
    scorer_name: str,
) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise GembaParseError(f"{scorer_name} JSON error item must be an object.")
    severity = str(item.get("severity", "")).strip().lower()
    if severity not in allowed_levels:
        raise GembaParseError(
            f"{scorer_name} JSON error severity must be one of {allowed_levels}, got {severity!r}."
        )
    error_type = _normalize_gemba_error_type(item.get("type"))
    if not error_type or error_type == "no-error":
        raise GembaParseError(f"{scorer_name} JSON error type must be non-empty.")
    target_span = _normalize_optional_gemba_span(item.get("target_span"))
    source_span = _normalize_optional_gemba_span(item.get("source_span"))
    confidence = _normalize_gemba_confidence(item.get("confidence", 1.0))
    return {
        "severity": severity,
        "type": error_type,
        "target_span": target_span,
        "source_span": source_span,
        "confidence": confidence,
    }


def _parse_gemba_json_errors(
    model_output: str | None,
    *,
    allowed_levels: tuple[str, ...],
    scorer_name: str,
) -> list[dict[str, Any]] | None:
    payload = _try_parse_json_object(model_output)
    if isinstance(payload, dict) and "errors" in payload:
        errors_value = payload.get("errors")
        if not isinstance(errors_value, list):
            raise GembaParseError(f"{scorer_name} JSON errors field must be a list.")
        return [
            _normalize_gemba_structured_error(item, allowed_levels=allowed_levels, scorer_name=scorer_name)
            for item in errors_value
        ]

    lenient_errors = _parse_lenient_gemba_json_errors(
        model_output,
        allowed_levels=allowed_levels,
        scorer_name=scorer_name,
    )
    if lenient_errors is not None:
        return lenient_errors

    if payload is None:
        return None
    if "errors" not in payload:
        raise GembaParseError(f"{scorer_name} JSON output must contain an errors field.")
    raise GembaParseError(f"{scorer_name} JSON errors field must be a list.")


def _has_unbalanced_gemba_quotes(text: str) -> bool:
    raw = str(text or "")
    if raw.count('"') % 2 != 0:
        return True
    if raw.count("`") % 2 != 0:
        return True
    if raw.count("“") != raw.count("”"):
        return True
    return False


def _coalesce_gemba_response_lines(
    model_output: str | None,
    *,
    allowed_levels: tuple[str, ...],
) -> list[str]:
    entries: list[str] = []
    pending: str | None = None

    for raw_line in str(model_output or "").splitlines():
        line = _normalize_gemba_response_line(raw_line)
        if not line:
            continue
        if pending is None:
            pending = line
            continue

        line_starts_new_item = (
            _looks_like_gemba_level_header(line, allowed_levels=allowed_levels)
            or (_GEMBA_ERROR_LINE_PATTERN.match(line) is not None)
            or _is_gemba_no_error_line(line)
        )
        if _has_unbalanced_gemba_quotes(pending) or (not line_starts_new_item):
            joiner = "" if pending.endswith(("-", ":", "–", "—", "/", "(", "[", "{", "“", '"', "'", "`")) else " "
            pending = (pending + joiner + line).strip()
            continue

        entries.append(pending)
        pending = line

    if pending is not None:
        entries.append(pending)
    return entries


def _is_structured_gemba_error_line(line: str) -> bool:
    return _split_gemba_error_line(line) is not None


def _parse_gemba_error_output(
    model_output: str | None,
    *,
    allowed_levels: tuple[str, ...],
    scorer_name: str,
) -> dict[str, list[str]]:
    text = str(model_output or "")
    if not text.strip():
        raise GembaParseError(f"{scorer_name} response is empty.")

    errors: dict[str, list[str]] = {level: [] for level in allowed_levels}
    level: str | None = None
    saw_structured_error = False
    saw_explicit_no_error = False
    invalid_lines: list[str] = []

    for line in _coalesce_gemba_response_lines(text, allowed_levels=allowed_levels):
        line_l = line.lower()

        matched_header = False
        for allowed in allowed_levels:
            header = f"{allowed}:"
            if line_l == header:
                level = allowed
                line = ""
                line_l = ""
                matched_header = True
                break
            if line_l.startswith(header):
                level = allowed
                line = line[len(header) :].strip()
                line_l = line.lower()
                matched_header = True
                break
        if matched_header and not line:
            continue

        if _is_gemba_no_error_line(line):
            saw_explicit_no_error = True
            continue
        if level is None:
            invalid_lines.append(line)
            continue
        if not _is_structured_gemba_error_line(line):
            invalid_lines.append(line)
            continue

        normalized_level = "critical" if "non-translation" in line_l and "critical" in errors else level
        errors[normalized_level].append(line)
        saw_structured_error = True

    if invalid_lines:
        joined = "; ".join(invalid_lines[:3])
        raise GembaParseError(f"{scorer_name} response has unparseable lines: {joined}")
    if not saw_structured_error and not saw_explicit_no_error:
        raise GembaParseError(f"{scorer_name} response contained neither structured errors nor explicit no-error.")
    return errors


def _legacy_gemba_error_to_structured(severity: str, line: str) -> dict[str, Any]:
    parsed = _split_gemba_error_line(line)
    if parsed is None:
        raise GembaParseError(f"GEMBA response has unparseable line: {line}")
    error_type, detail = parsed
    target_span = _normalize_optional_gemba_span(_strip_matching_quotes(_extract_gemba_quoted_text(detail) or detail))
    if target_span is not None and target_span.lower() in {"no-error", "no error"}:
        target_span = None
    if target_span == detail:
        detail_candidate = _strip_matching_quotes(detail)
        target_span = detail_candidate or None
    return {
        "severity": str(severity).strip().lower(),
        "type": error_type,
        "target_span": target_span,
        "source_span": None,
        "confidence": 1.0,
        "label": line,
    }


def _legacy_gemba_errors_to_structured(
    model_output: str | None,
    *,
    allowed_levels: tuple[str, ...],
    scorer_name: str,
) -> list[dict[str, Any]]:
    parsed = _parse_gemba_error_output(
        model_output,
        allowed_levels=allowed_levels,
        scorer_name=scorer_name,
    )
    out: list[dict[str, Any]] = []
    for level in allowed_levels:
        for line in parsed.get(level, []):
            out.append(_legacy_gemba_error_to_structured(level, line))
    return out


def _structured_gemba_error_label(error: dict[str, Any]) -> str:
    error_type = str(error.get("type", "")).strip()
    target_span = _normalize_optional_gemba_span(error.get("target_span"))
    source_span = _normalize_optional_gemba_span(error.get("source_span"))
    if target_span:
        return f'{error_type} - "{target_span}"'
    if source_span:
        return f'{error_type} - source: "{source_span}"'
    return error_type


def _structured_gemba_errors_to_legacy_dict(
    errors: list[dict[str, Any]],
    *,
    allowed_levels: tuple[str, ...],
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {level: [] for level in allowed_levels}
    for error in errors:
        severity = str(error.get("severity", "")).strip().lower()
        if severity in out:
            label = str(error.get("label") or _structured_gemba_error_label(error)).strip()
            out[severity].append(label)
    return out


def _format_gemba_structured_errors(errors: list[dict[str, Any]]) -> str:
    normalized = [
        {
            "severity": str(error.get("severity", "")).strip().lower(),
            "type": _normalize_gemba_error_type(error.get("type")),
            "target_span": _normalize_optional_gemba_span(error.get("target_span")),
            "source_span": _normalize_optional_gemba_span(error.get("source_span")),
            "confidence": _normalize_gemba_confidence(error.get("confidence", 1.0)),
        }
        for error in errors
    ]
    return json.dumps({"errors": normalized}, ensure_ascii=False, indent=2)


def gemba_mqm_parse_structured_errors(model_output: str) -> list[dict[str, Any]]:
    structured = _parse_gemba_json_errors(
        model_output,
        allowed_levels=("critical", "major", "minor"),
        scorer_name="MQM",
    )
    if structured is not None:
        return structured
    return _legacy_gemba_errors_to_structured(
        model_output,
        allowed_levels=("critical", "major", "minor"),
        scorer_name="MQM",
    )


def gemba_mqm_parse_errors(model_output: str) -> dict[str, list[str]]:
    return _structured_gemba_errors_to_legacy_dict(
        gemba_mqm_parse_structured_errors(model_output),
        allowed_levels=("critical", "major", "minor"),
    )


def gemba_mqm_score(model_output: str | None) -> int | None:
    if model_output is None:
        return None
    errors = gemba_mqm_parse_structured_errors(model_output)

    penalty = 0
    count = 0
    for lvl in ["critical", "major", "minor"]:
        for error in errors:
            if str(error.get("severity", "")).strip().lower() != lvl:
                continue
            if count >= 5:
                break
            penalty += 25 if lvl == "critical" else 5 if lvl == "major" else 1
            count += 1
    if penalty > 25:
        penalty = 25
    return -penalty


def _extract_mqm_quoted_text(line: str) -> str | None:
    return _extract_gemba_quoted_text(line)


def _extract_mqm_error_detail(line: str) -> str | None:
    parsed = _split_gemba_error_line(line)
    if parsed is None:
        return None
    return parsed[1] or None


def _normalize_mqm_error_text_candidate(text: str) -> str:
    value = re.sub(r"\s+", " ", str(text or "").strip())
    if not value:
        return ""
    value = re.sub(r"\s+([)\]\}])", r"\1", value)
    value = re.sub(r"([(\[\{])\s+", r"\1", value)
    return value.strip()


def _mqm_error_text_candidates(line: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def _add(candidate: str | None) -> None:
        if candidate is None:
            return
        normalized = _normalize_mqm_error_text_candidate(candidate)
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        out.append(normalized)

    quoted = _extract_mqm_quoted_text(line)
    detail = _extract_mqm_error_detail(line)
    for candidate in (quoted, detail):
        _add(candidate)
        stripped = _strip_matching_quotes(candidate or "")
        _add(stripped)
        if stripped and "(" in stripped and ")" in stripped:
            before_paren = stripped.split("(", 1)[0].strip()
            inner = stripped.split("(", 1)[1].rsplit(")", 1)[0].strip()
            _add(before_paren)
            _add(inner)
            if before_paren and inner:
                _add(f"{before_paren}({inner})")
                _add(f"{before_paren} ({inner})")
    return out


def _build_whitespace_flexible_pattern(text: str) -> str | None:
    parts = [re.escape(part) for part in re.split(r"\s+", str(text or "").strip()) if part]
    if len(parts) <= 1:
        return None
    return r"\s*".join(parts)


def _find_text_span(
    text: str,
    needle: str,
    used_spans: list[tuple[int, int]],
) -> tuple[int, int] | None:
    if not text or not needle:
        return None

    candidates: list[tuple[int, int]] = []

    start = 0
    while True:
        idx = text.find(needle, start)
        if idx < 0:
            break
        candidates.append((idx, idx + len(needle)))
        start = idx + 1

    if not candidates:
        text_l = text.lower()
        needle_l = needle.lower()
        start = 0
        while True:
            idx = text_l.find(needle_l, start)
            if idx < 0:
                break
            candidates.append((idx, idx + len(needle)))
            start = idx + 1

    if not candidates:
        flexible_pattern = _build_whitespace_flexible_pattern(needle)
        if flexible_pattern is not None:
            for flags in (0, re.IGNORECASE):
                try:
                    for match in re.finditer(flexible_pattern, text, flags=flags):
                        candidates.append((int(match.start()), int(match.end())))
                except re.error:
                    candidates = []
                if candidates:
                    break

    if not candidates:
        return None

    def _overlap(span: tuple[int, int], other: tuple[int, int]) -> bool:
        return span[0] < other[1] and other[0] < span[1]

    for span in candidates:
        if all(not _overlap(span, used) for used in used_spans):
            return span
    return candidates[0]


def _extract_mqm_error_detail_text(error: dict[str, Any], label: str) -> str:
    target_span = _normalize_optional_gemba_span(error.get("target_span"))
    if target_span:
        return target_span
    source_span = _normalize_optional_gemba_span(error.get("source_span"))
    if source_span:
        return source_span
    parsed = _split_gemba_error_line(label)
    if parsed is not None:
        _, detail = parsed
        quoted = _extract_gemba_quoted_text(detail)
        if quoted:
            return quoted
        detail = re.sub(r"(?i)^source:\s*", "", detail).strip()
        detail = _strip_matching_quotes(detail)
        if detail:
            return detail
    return label


def gemba_mqm_extract_error_annotations(
    model_output: str | None,
    mt_text: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if model_output is None:
        return [], []

    parsed = gemba_mqm_parse_structured_errors(model_output)
    anchored: list[dict[str, Any]] = []
    unanchored: list[dict[str, Any]] = []
    used_spans: list[tuple[int, int]] = []

    for severity in ("critical", "major", "minor"):
        for error in parsed:
            if str(error.get("severity", "")).strip().lower() != severity:
                continue
            target_span = _normalize_optional_gemba_span(error.get("target_span"))
            label = str(error.get("label") or _structured_gemba_error_label(error)).strip()
            error_type = (
                _normalize_gemba_error_type(error.get("type"))
                or _extract_gemba_error_type(label)
                or None
            )
            span = None
            if target_span:
                span = _find_text_span(mt_text, target_span, used_spans)
            if span is None:
                for candidate in _mqm_error_text_candidates(label):
                    span = _find_text_span(mt_text, candidate, used_spans)
                    if span is not None:
                        break
            if span is None:
                unanchored.append(
                    {
                        "severity": severity.upper(),
                        "source": "mqm",
                        "label": label,
                        "error_type": error_type,
                        "detail_text": _extract_mqm_error_detail_text(error, label),
                    }
                )
                continue
            start, end = span
            used_spans.append(span)
            anchored.append(
                {
                    "text": mt_text[start:end],
                    "start": int(start),
                    "end": int(end),
                    "severity": severity.upper(),
                    "confidence": _normalize_gemba_confidence(error.get("confidence", 1.0)),
                    "source": "mqm",
                    "label": label,
                    "type": _normalize_gemba_error_type(error.get("type")),
                    "error_type": error_type,
                    "target_span": target_span,
                    "source_span": _normalize_optional_gemba_span(error.get("source_span")),
                }
            )

    return anchored, unanchored


def gemba_mqm_extract_error_spans(model_output: str | None, mt_text: str) -> list[dict[str, Any]]:
    spans, _ = gemba_mqm_extract_error_annotations(model_output, mt_text)
    return spans


_GEMBA_PROMPT_DIRECTION_ANY: tuple[str, str] = ("*", "*")
_VALID_GEMBA_PROMPT_PACKS = frozenset({"generic", "ko_en_enterprise_v1"})


@dataclass(frozen=True)
class _GembaPromptPack:
    mqm_system_prompt: str
    mqm_task_prompt: str
    mqm_fewshot_turns: dict[tuple[str, str], list[dict[str, str]]]
    esa_system_prompt: str
    esa_task_prompt: str
    esa_fewshot_turns: dict[tuple[str, str], list[dict[str, str]]]


def _normalize_gemba_prompt_lang(value: str | None) -> str:
    text = str(value or "").strip().lower()
    if text in {"ko", "korean"}:
        return "korean"
    if text in {"en", "english"}:
        return "english"
    return text


def _gemba_prompt_direction_key(source_lang: str, target_lang: str) -> tuple[str, str]:
    return (
        _normalize_gemba_prompt_lang(source_lang),
        _normalize_gemba_prompt_lang(target_lang),
    )


def _copy_gemba_turns(turns: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{"role": str(turn["role"]), "content": str(turn["content"])} for turn in turns]


def _select_gemba_fewshot_turns(
    turns_by_direction: dict[tuple[str, str], list[dict[str, str]]],
    *,
    source_lang: str,
    target_lang: str,
) -> list[dict[str, str]]:
    out = _copy_gemba_turns(turns_by_direction.get(_GEMBA_PROMPT_DIRECTION_ANY, []))
    direction_key = _gemba_prompt_direction_key(source_lang, target_lang)
    if direction_key != _GEMBA_PROMPT_DIRECTION_ANY:
        out.extend(_copy_gemba_turns(turns_by_direction.get(direction_key, [])))
    return out


def _append_gemba_guidance(base_prompt: str, guidance: str) -> str:
    extra = str(guidance or "").strip()
    if not extra:
        return base_prompt
    return f"{base_prompt}\n\n{extra}"


def _gemba_mqm_fewshot_user_message(
    *,
    source_lang: str,
    target_lang: str,
    source_seg: str,
    target_seg: str,
    task_prompt: str,
) -> str:
    return (
        f"{source_lang} source:\n"
        f"```{source_seg}```\n"
        f"{target_lang} translation:\n"
        f"```{target_seg}```\n\n"
        f"{task_prompt}"
    )


def _gemba_eval_user_message(
    *,
    source_lang: str,
    target_lang: str,
    source_seg: str,
    target_seg: str,
    task_prompt: str = GEMBA_USER_TASK_PROMPT,
) -> str:
    return (
        f"{source_lang} source:\n"
        f"```{source_seg}```\n"
        f"{target_lang} translation:\n"
        f"```{target_seg}```\n\n"
        f"{task_prompt}"
        f"{_gemba_json_output_instructions(allowed_levels=('critical', 'major', 'minor'))}"
    )


GEMBA_ESA_SYSTEM_PROMPT = (
    "Your task is to identify machine translation errors and assess the quality of the translation."
)

GEMBA_ESA_USER_TASK_PROMPT = (
    "Based on the source segment and machine translation surrounded with triple backticks, identify "
    "error types in the translation and classify them. The categories of errors are: accuracy "
    "(addition, mistranslation, omission, untranslated text), fluency (character encoding, grammar, "
    "inconsistency, punctuation, register, spelling), style (awkward), terminology (inappropriate for "
    "context, inconsistent use), non-translation, other, or no-error.\n"
    "Each error is classified as one of two categories: major or minor. Major errors disrupt the flow "
    "and make the understandability of text difficult or impossible. Minor errors are errors that do "
    "not disrupt the flow significantly and what the text is trying to say is still understandable."
)

GEMBA_ESA_REPAIR_SYSTEM_PROMPT = (
    "You normalize machine translation ESA annotations into a strict JSON format."
)

GEMBA_ESA_REPAIR_PROMPT_TEMPLATE = (
    "Rewrite the evaluator output below into the exact ESA JSON format.\n\n"
    "Return only valid JSON with this schema:\n"
    '{{\n'
    '  "errors": [\n'
    '    {{\n'
    '      "severity": "major",\n'
    '      "type": "accuracy/mistranslation",\n'
    '      "target_span": "exact target text or null",\n'
    '      "source_span": "exact source text or null",\n'
    '      "confidence": 0.92\n'
    '    }}\n'
    '  ]\n'
    '}}\n\n'
    "Rules:\n"
    '- severity must be one of "major" or "minor".\n'
    '- type must be category/subcategory in lowercase.\n'
    "- Copy target_span exactly from the translation when possible.\n"
    "- For omissions or anchorless errors, use target_span=null and source_span when possible.\n"
    "- confidence must be a number between 0 and 1.\n"
    '- If there are no errors, return {{"errors": []}}.\n'
    "- Do not include explanations or any text outside the JSON object.\n\n"
    "Source:\n"
    "```{source_seg}```\n"
    "Translation:\n"
    "```{target_seg}```\n"
    "Evaluator output:\n"
    "```{raw_output}```"
)

GEMBA_ESA_FEWSHOT_ASSISTANT_1 = dedent(
    """\
    {
      "errors": [
        {"severity": "major", "type": "accuracy/mistranslation", "target_span": "involvement", "source_span": null, "confidence": 0.98},
        {"severity": "major", "type": "accuracy/omission", "target_span": "the account holder", "source_span": null, "confidence": 0.95},
        {"severity": "minor", "type": "fluency/grammar", "target_span": "wäre", "source_span": null, "confidence": 0.93},
        {"severity": "minor", "type": "fluency/register", "target_span": "dir", "source_span": null, "confidence": 0.91}
      ]
    }
    """
).strip()

GEMBA_ESA_FEWSHOT_ASSISTANT_2 = dedent(
    """\
    {
      "errors": [
        {"severity": "major", "type": "accuracy/addition", "target_span": "ve Vídni", "source_span": null, "confidence": 0.97},
        {"severity": "major", "type": "accuracy/omission", "target_span": null, "source_span": "the stop-start", "confidence": 0.94},
        {"severity": "minor", "type": "terminology/inappropriate for context", "target_span": "partaje", "source_span": null, "confidence": 0.9}
      ]
    }
    """
).strip()

GEMBA_ESA_FEWSHOT_ASSISTANT_3 = dedent(
    """\
    {
      "errors": [
        {"severity": "major", "type": "accuracy/addition", "target_span": "of high-speed rail", "source_span": null, "confidence": 0.98},
        {"severity": "major", "type": "accuracy/mistranslation", "target_span": "go to the reviews", "source_span": null, "confidence": 0.96},
        {"severity": "minor", "type": "style/awkward", "target_span": "etc.,", "source_span": null, "confidence": 0.88}
      ]
    }
    """
).strip()

GEMBA_ESA_RANKING_PROMPT_TEMPLATE = (
    "Given the translation from {source_lang} to {target_lang} and the annotated errors JSON, "
    "assign a score on a continuous scale from 0 to 100. The scale has following reference points: "
    '0="No meaning preserved", 33="Some meaning preserved", '
    '66="Most meaning preserved and few grammar mistakes", '
    'up to 100="Perfect meaning and grammar".\n\n'
    "Score the following translation from {source_lang} source:\n"
    "```{source_seg}```\n"
    "{target_lang} translation:\n"
    "```{target_seg}```\n"
    "Annotated errors JSON:\n"
    "```{error_spans}```\n"
    'Respond with only valid JSON like {{"score": 83}}. Do not include any explanation or extra text.'
)

_ESA_SCORE_PATTERNS: tuple[str, ...] = (
    r"(?i)\bscore\b[^0-9]{0,30}(-?\d+(?:\.\d+)?)\s*/\s*100\b",
    r"(?i)\bscore\b[^0-9]{0,30}(-?\d+(?:\.\d+)?)\s+out\s+of\s+100\b",
    r"(?i)\bscore\b[^0-9]{0,30}(-?\d+(?:\.\d+)?)\b",
    r"(?i)\b(-?\d+(?:\.\d+)?)\s*/\s*100\b",
    r"(?i)\b(-?\d+(?:\.\d+)?)\s+out\s+of\s+100\b",
    r":\s*(-?\d+(?:\.\d+)?)\s*$",
    r"^\s*(-?\d+(?:\.\d+)?)\s*$",
)


def _gemba_esa_error_user_message(
    *,
    source_lang: str,
    target_lang: str,
    source_seg: str,
    target_seg: str,
    task_prompt: str = GEMBA_ESA_USER_TASK_PROMPT,
) -> str:
    return (
        f"{source_lang} source:\n"
        f"```{source_seg}```\n"
        f"{target_lang} translation:\n"
        f"```{target_seg}```\n\n"
        f"{task_prompt}"
        f"{_gemba_json_output_instructions(allowed_levels=('major', 'minor'))}"
    )


GEMBA_ESA_FEWSHOT_USER_1 = _gemba_esa_error_user_message(
    source_lang="English",
    target_lang="German",
    source_seg=(
        "I do apologise about this, we must gain permission from the account holder to discuss "
        "an order with another person, I apologise if this was done previously, however, I would "
        "not be able to discuss this with yourself without the account holders permission."
    ),
    target_seg=(
        "Ich entschuldige mich dafür, wir müssen die Erlaubnis einholen, um eine Bestellung mit "
        "einer anderen Person zu besprechen. Ich entschuldige mich, falls dies zuvor geschehen "
        "wäre, aber ohne die Erlaubnis des Kontoinhabers wäre ich nicht in der Lage, dies mit dir "
        "involvement."
    ),
)

GEMBA_ESA_FEWSHOT_USER_2 = _gemba_esa_error_user_message(
    source_lang="English",
    target_lang="Czech",
    source_seg=(
        "Talks have resumed in Vienna to try to revive the nuclear pact, with both sides trying "
        "to gauge the prospects of success after the latest exchanges in the stop-start negotiations."
    ),
    target_seg=(
        "Ve Vídni se ve Vídni obnovily rozhovory o oživení jaderného paktu, přičemž obě partaje "
        "se snaží posoudit vyhlídky na úspěch po posledních výměnách v jednáních."
    ),
)

GEMBA_ESA_FEWSHOT_USER_3 = _gemba_esa_error_user_message(
    source_lang="Chinese",
    target_lang="English",
    source_seg="大众点评乌鲁木齐家居卖场频道为您提供高铁居然之家地址，电话，营业时间等最新商户信息，找装修公司，就上大众点评",
    target_seg=(
        "Urumqi Home Furnishing Store Channel provides you with the latest business information "
        "such as the address, telephone number, business hours, etc., of high-speed rail, and "
        "find a decoration company, and go to the reviews."
    ),
)

_GEMBA_KO_EN_ENTERPRISE_EXTRA_GUIDANCE = dedent(
    """\
    Additional guidance for ko<->en enterprise/noisy-chat evaluation:
    - Translation evaluation only. Do not repair or normalize the source.
    - If the output responds like an assistant, asks for more input, or explains instead of translating, treat it as non-translation / severe error.
    - In enterprise or product contexts, terminology mismatch is more serious than minor stylistic awkwardness.
    - Preserve roles, honorifics, team names, organization names, and internal product naming faithfully.
    - Noisy chat, typos, abbreviations, and partial sentences must still be evaluated as translation outputs, not rewritten into a new intended meaning.
    """
).strip()

_GEMBA_KO_EN_ENTERPRISE_MQM_TASK_PROMPT = _append_gemba_guidance(
    GEMBA_USER_TASK_PROMPT,
    _GEMBA_KO_EN_ENTERPRISE_EXTRA_GUIDANCE,
)
_GEMBA_KO_EN_ENTERPRISE_ESA_TASK_PROMPT = _append_gemba_guidance(
    GEMBA_ESA_USER_TASK_PROMPT,
    _GEMBA_KO_EN_ENTERPRISE_EXTRA_GUIDANCE,
)

GEMBA_KO_EN_ENTERPRISE_MQM_FEWSHOT_ASSISTANT_1 = dedent(
    """\
    {
      "errors": [
        {"severity": "major", "type": "terminology/inappropriate for context", "target_span": "reflect", "source_span": null, "confidence": 0.97},
        {"severity": "minor", "type": "style/awkward", "target_span": "the add-in server knox/brity", "source_span": null, "confidence": 0.9}
      ]
    }
    """
).strip()

GEMBA_KO_EN_ENTERPRISE_MQM_FEWSHOT_ASSISTANT_2 = dedent(
    """\
    {
      "errors": [
        {"severity": "major", "type": "accuracy/mistranslation", "target_span": "love meeting room", "source_span": null, "confidence": 0.98},
        {"severity": "minor", "type": "terminology/inappropriate for context", "target_span": "Pro-nim", "source_span": null, "confidence": 0.93},
        {"severity": "minor", "type": "style/awkward", "target_span": "?? ^^;;", "source_span": null, "confidence": 0.88}
      ]
    }
    """
).strip()

GEMBA_KO_EN_ENTERPRISE_MQM_FEWSHOT_ASSISTANT_3 = dedent(
    """\
    {
      "errors": [
        {"severity": "major", "type": "accuracy/mistranslation", "target_span": "발신자만 볼 수 있는", "source_span": null, "confidence": 0.98},
        {"severity": "minor", "type": "terminology/inappropriate for context", "target_span": "주소", "source_span": null, "confidence": 0.91}
      ]
    }
    """
).strip()

GEMBA_KO_EN_ENTERPRISE_MQM_FEWSHOT_ASSISTANT_4 = dedent(
    """\
    {
      "errors": [
        {"severity": "critical", "type": "non-translation", "target_span": "Please provide the Korean text you would like translated.", "source_span": "@", "confidence": 0.99}
      ]
    }
    """
).strip()

GEMBA_KO_EN_ENTERPRISE_ESA_FEWSHOT_ASSISTANT_1 = dedent(
    """\
    {
      "errors": [
        {"severity": "major", "type": "terminology/inappropriate for context", "target_span": "reflect", "source_span": null, "confidence": 0.97},
        {"severity": "minor", "type": "style/awkward", "target_span": "the add-in server knox/brity", "source_span": null, "confidence": 0.9}
      ]
    }
    """
).strip()

GEMBA_KO_EN_ENTERPRISE_ESA_FEWSHOT_ASSISTANT_2 = dedent(
    """\
    {
      "errors": [
        {"severity": "major", "type": "accuracy/mistranslation", "target_span": "love meeting room", "source_span": null, "confidence": 0.98},
        {"severity": "minor", "type": "terminology/inappropriate for context", "target_span": "Pro-nim", "source_span": null, "confidence": 0.93},
        {"severity": "minor", "type": "style/awkward", "target_span": "?? ^^;;", "source_span": null, "confidence": 0.88}
      ]
    }
    """
).strip()

GEMBA_KO_EN_ENTERPRISE_ESA_FEWSHOT_ASSISTANT_3 = dedent(
    """\
    {
      "errors": [
        {"severity": "major", "type": "accuracy/mistranslation", "target_span": "발신자만 볼 수 있는", "source_span": null, "confidence": 0.98},
        {"severity": "minor", "type": "terminology/inappropriate for context", "target_span": "주소", "source_span": null, "confidence": 0.91}
      ]
    }
    """
).strip()

GEMBA_KO_EN_ENTERPRISE_ESA_FEWSHOT_ASSISTANT_4 = dedent(
    """\
    {
      "errors": [
        {"severity": "major", "type": "non-translation", "target_span": "Please provide the Korean text you would like translated.", "source_span": "@", "confidence": 0.99}
      ]
    }
    """
).strip()


def _build_generic_mqm_fewshot_turns() -> list[dict[str, str]]:
    return [
        {"role": "user", "content": GEMBA_FEWSHOT_USER_1},
        {"role": "assistant", "content": GEMBA_FEWSHOT_ASSISTANT_1},
        {"role": "user", "content": GEMBA_FEWSHOT_USER_2},
        {"role": "assistant", "content": GEMBA_FEWSHOT_ASSISTANT_2},
        {"role": "user", "content": GEMBA_FEWSHOT_USER_3},
        {"role": "assistant", "content": GEMBA_FEWSHOT_ASSISTANT_3},
    ]


def _build_generic_esa_fewshot_turns() -> list[dict[str, str]]:
    return [
        {"role": "user", "content": GEMBA_ESA_FEWSHOT_USER_1},
        {"role": "assistant", "content": GEMBA_ESA_FEWSHOT_ASSISTANT_1},
        {"role": "user", "content": GEMBA_ESA_FEWSHOT_USER_2},
        {"role": "assistant", "content": GEMBA_ESA_FEWSHOT_ASSISTANT_2},
        {"role": "user", "content": GEMBA_ESA_FEWSHOT_USER_3},
        {"role": "assistant", "content": GEMBA_ESA_FEWSHOT_ASSISTANT_3},
    ]


def _build_ko_en_enterprise_mqm_fewshot_turns() -> dict[tuple[str, str], list[dict[str, str]]]:
    return {
        ("korean", "english"): [
            {
                "role": "user",
                "content": _gemba_mqm_fewshot_user_message(
                    source_lang="Korean",
                    target_lang="English",
                    source_seg="애드인 서버 knox/brity 반영 예정입니다.",
                    target_seg="We will reflect the add-in server knox/brity.",
                    task_prompt=_GEMBA_KO_EN_ENTERPRISE_MQM_TASK_PROMPT,
                ),
            },
            {"role": "assistant", "content": GEMBA_KO_EN_ENTERPRISE_MQM_FEWSHOT_ASSISTANT_1},
            {
                "role": "user",
                "content": _gemba_mqm_fewshot_user_message(
                    source_lang="Korean",
                    target_lang="English",
                    source_seg="프로님 안녕하세요~ 사랑회의실 몇시부터 이용하시나요?? ^^;;",
                    target_seg="Hello Pro-nim, what time can I use the love meeting room?? ^^;;",
                    task_prompt=_GEMBA_KO_EN_ENTERPRISE_MQM_TASK_PROMPT,
                ),
            },
            {"role": "assistant", "content": GEMBA_KO_EN_ENTERPRISE_MQM_FEWSHOT_ASSISTANT_2},
            {
                "role": "user",
                "content": _gemba_mqm_fewshot_user_message(
                    source_lang="Korean",
                    target_lang="English",
                    source_seg="@",
                    target_seg="Please provide the Korean text you would like translated.",
                    task_prompt=_GEMBA_KO_EN_ENTERPRISE_MQM_TASK_PROMPT,
                ),
            },
            {"role": "assistant", "content": GEMBA_KO_EN_ENTERPRISE_MQM_FEWSHOT_ASSISTANT_4},
        ],
        ("english", "korean"): [
            {
                "role": "user",
                "content": _gemba_mqm_fewshot_user_message(
                    source_lang="English",
                    target_lang="Korean",
                    source_seg="This is a no-reply email address. Please do not reply to this message.",
                    target_seg="이 메일은 발신자만 볼 수 있는 주소입니다. 이 메시지에 회신하지 마세요.",
                    task_prompt=_GEMBA_KO_EN_ENTERPRISE_MQM_TASK_PROMPT,
                ),
            },
            {"role": "assistant", "content": GEMBA_KO_EN_ENTERPRISE_MQM_FEWSHOT_ASSISTANT_3},
        ],
    }


def _build_ko_en_enterprise_esa_fewshot_turns() -> dict[tuple[str, str], list[dict[str, str]]]:
    return {
        ("korean", "english"): [
            {
                "role": "user",
                "content": _gemba_esa_error_user_message(
                    source_lang="Korean",
                    target_lang="English",
                    source_seg="애드인 서버 knox/brity 반영 예정입니다.",
                    target_seg="We will reflect the add-in server knox/brity.",
                    task_prompt=_GEMBA_KO_EN_ENTERPRISE_ESA_TASK_PROMPT,
                ),
            },
            {"role": "assistant", "content": GEMBA_KO_EN_ENTERPRISE_ESA_FEWSHOT_ASSISTANT_1},
            {
                "role": "user",
                "content": _gemba_esa_error_user_message(
                    source_lang="Korean",
                    target_lang="English",
                    source_seg="프로님 안녕하세요~ 사랑회의실 몇시부터 이용하시나요?? ^^;;",
                    target_seg="Hello Pro-nim, what time can I use the love meeting room?? ^^;;",
                    task_prompt=_GEMBA_KO_EN_ENTERPRISE_ESA_TASK_PROMPT,
                ),
            },
            {"role": "assistant", "content": GEMBA_KO_EN_ENTERPRISE_ESA_FEWSHOT_ASSISTANT_2},
            {
                "role": "user",
                "content": _gemba_esa_error_user_message(
                    source_lang="Korean",
                    target_lang="English",
                    source_seg="@",
                    target_seg="Please provide the Korean text you would like translated.",
                    task_prompt=_GEMBA_KO_EN_ENTERPRISE_ESA_TASK_PROMPT,
                ),
            },
            {"role": "assistant", "content": GEMBA_KO_EN_ENTERPRISE_ESA_FEWSHOT_ASSISTANT_4},
        ],
        ("english", "korean"): [
            {
                "role": "user",
                "content": _gemba_esa_error_user_message(
                    source_lang="English",
                    target_lang="Korean",
                    source_seg="This is a no-reply email address. Please do not reply to this message.",
                    target_seg="이 메일은 발신자만 볼 수 있는 주소입니다. 이 메시지에 회신하지 마세요.",
                    task_prompt=_GEMBA_KO_EN_ENTERPRISE_ESA_TASK_PROMPT,
                ),
            },
            {"role": "assistant", "content": GEMBA_KO_EN_ENTERPRISE_ESA_FEWSHOT_ASSISTANT_3},
        ],
    }


_GEMBA_PROMPT_PACKS: dict[str, _GembaPromptPack] = {
    "generic": _GembaPromptPack(
        mqm_system_prompt=GEMBA_SYSTEM_PROMPT,
        mqm_task_prompt=GEMBA_USER_TASK_PROMPT,
        mqm_fewshot_turns={_GEMBA_PROMPT_DIRECTION_ANY: _build_generic_mqm_fewshot_turns()},
        esa_system_prompt=GEMBA_ESA_SYSTEM_PROMPT,
        esa_task_prompt=GEMBA_ESA_USER_TASK_PROMPT,
        esa_fewshot_turns={_GEMBA_PROMPT_DIRECTION_ANY: _build_generic_esa_fewshot_turns()},
    ),
    "ko_en_enterprise_v1": _GembaPromptPack(
        mqm_system_prompt=GEMBA_SYSTEM_PROMPT,
        mqm_task_prompt=_GEMBA_KO_EN_ENTERPRISE_MQM_TASK_PROMPT,
        mqm_fewshot_turns={
            _GEMBA_PROMPT_DIRECTION_ANY: _build_generic_mqm_fewshot_turns(),
            **_build_ko_en_enterprise_mqm_fewshot_turns(),
        },
        esa_system_prompt=GEMBA_ESA_SYSTEM_PROMPT,
        esa_task_prompt=_GEMBA_KO_EN_ENTERPRISE_ESA_TASK_PROMPT,
        esa_fewshot_turns={
            _GEMBA_PROMPT_DIRECTION_ANY: _build_generic_esa_fewshot_turns(),
            **_build_ko_en_enterprise_esa_fewshot_turns(),
        },
    ),
}


def _resolve_gemba_prompt_pack(prompt_pack: str) -> _GembaPromptPack:
    pack_key = str(prompt_pack or "generic").strip() or "generic"
    pack = _GEMBA_PROMPT_PACKS.get(pack_key)
    if pack is None:
        supported = ", ".join(sorted(_VALID_GEMBA_PROMPT_PACKS))
        raise ValueError(f"Unsupported GEMBA prompt_pack={pack_key!r}. Supported: {supported}")
    return pack


def build_gemba_mqm_messages(
    *,
    source_lang: str,
    target_lang: str,
    source_seg: str,
    target_seg: str,
    use_fewshot: bool = True,
    prompt_pack: str = "generic",
) -> list[dict[str, str]]:
    pack = _resolve_gemba_prompt_pack(prompt_pack)
    messages: list[dict[str, str]] = [{"role": "system", "content": pack.mqm_system_prompt}]
    if use_fewshot:
        messages.extend(
            _select_gemba_fewshot_turns(
                pack.mqm_fewshot_turns,
                source_lang=source_lang,
                target_lang=target_lang,
            )
        )
    messages.append(
        {
            "role": "user",
            "content": _gemba_eval_user_message(
                source_lang=source_lang,
                target_lang=target_lang,
                source_seg=source_seg,
                target_seg=target_seg,
                task_prompt=pack.mqm_task_prompt,
            ),
        }
    )
    return messages


def build_gemba_mqm_repair_messages(
    *,
    source_seg: str,
    target_seg: str,
    raw_output: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": GEMBA_MQM_REPAIR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": GEMBA_MQM_REPAIR_PROMPT_TEMPLATE.format(
                source_seg=source_seg,
                target_seg=target_seg,
                raw_output=raw_output,
            ),
        },
    ]


def build_gemba_esa_error_messages(
    *,
    source_lang: str,
    target_lang: str,
    source_seg: str,
    target_seg: str,
    use_fewshot: bool = True,
    prompt_pack: str = "generic",
) -> list[dict[str, str]]:
    pack = _resolve_gemba_prompt_pack(prompt_pack)
    messages: list[dict[str, str]] = [{"role": "system", "content": pack.esa_system_prompt}]
    if use_fewshot:
        messages.extend(
            _select_gemba_fewshot_turns(
                pack.esa_fewshot_turns,
                source_lang=source_lang,
                target_lang=target_lang,
            )
        )
    messages.append(
        {
            "role": "user",
            "content": _gemba_esa_error_user_message(
                source_lang=source_lang,
                target_lang=target_lang,
                source_seg=source_seg,
                target_seg=target_seg,
                task_prompt=pack.esa_task_prompt,
            ),
        }
    )
    return messages


def gemba_esa_parse_structured_errors(model_output: str) -> list[dict[str, Any]]:
    structured = _parse_gemba_json_errors(
        model_output,
        allowed_levels=("major", "minor"),
        scorer_name="ESA",
    )
    if structured is not None:
        return structured
    return _legacy_gemba_errors_to_structured(
        model_output,
        allowed_levels=("major", "minor"),
        scorer_name="ESA",
    )


def gemba_esa_parse_errors(model_output: str) -> dict[str, list[str]]:
    return _structured_gemba_errors_to_legacy_dict(
        gemba_esa_parse_structured_errors(model_output),
        allowed_levels=("major", "minor"),
    )


def gemba_esa_format_error_spans(model_output: str | None) -> str:
    if model_output is None:
        return '{"errors": []}'
    return _format_gemba_structured_errors(gemba_esa_parse_structured_errors(model_output))


def gemba_esa_parse_score(model_output: str | None) -> float | None:
    if model_output is None:
        return None
    text = str(model_output).strip()
    if not text:
        return None

    json_payload = _try_parse_json_object(text)
    if isinstance(json_payload, dict) and "score" in json_payload:
        try:
            value = float(json_payload.get("score"))
        except Exception:
            value = math.nan
        return value if math.isfinite(value) and 0.0 <= value <= 100.0 else None

    if re.match(r"^\[['\"]?-?\d+(?:\.\d+)?['\"]?\]$", text):
        inner = re.sub(r"[^\d\.\-]", "", text)
        try:
            value = float(inner)
        except Exception:
            value = math.nan
        return value if math.isfinite(value) and 0.0 <= value <= 100.0 else None

    normalized = text.replace("**", "").replace("__", "").replace("`", "")
    normalized = re.sub(r"\b0\s*[-–]\s*100\b", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    for pattern in _ESA_SCORE_PATTERNS:
        matches = list(re.finditer(pattern, normalized))
        if not matches:
            continue
        try:
            value = float(matches[-1].group(1))
        except Exception:
            value = math.nan
        if math.isfinite(value) and 0.0 <= value <= 100.0:
            return value
    return None


def build_gemba_esa_ranking_messages(
    *,
    source_lang: str,
    target_lang: str,
    source_seg: str,
    target_seg: str,
    error_spans: str,
) -> list[dict[str, str]]:
    prompt = GEMBA_ESA_RANKING_PROMPT_TEMPLATE.format(
        source_lang=source_lang,
        target_lang=target_lang,
        source_seg=source_seg,
        target_seg=target_seg,
        error_spans=error_spans or "no-error",
    )
    return [{"role": "user", "content": prompt}]


def build_gemba_esa_repair_messages(
    *,
    source_seg: str,
    target_seg: str,
    raw_output: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": GEMBA_ESA_REPAIR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": GEMBA_ESA_REPAIR_PROMPT_TEMPLATE.format(
                source_seg=source_seg,
                target_seg=target_seg,
                raw_output=raw_output,
            ),
        },
    ]


GROUP_RANK_SYSTEM_PROMPT_KO_EN_ENTERPRISE_V1 = """You are a ranking judge for machine translation candidate comparison.

You will receive:
- one source segment
- optionally one reference translation
- 2 to 4 candidate translations of the SAME source segment

Your job is to rank all candidates from best to worst.

Priority order:
1. Faithfulness to the source meaning
   - mistranslation
   - omission
   - unjustified addition
   - untranslated text
2. Terminology and context fit
   - enterprise / product / policy / workflow terminology
   - register and address terms appropriate to the context
3. Bad failure modes that must be ranked very low
   - assistant-like fallback responses
   - wrong-language output
   - non-translation
   - source-unrelated content
4. Fluency and style
   - awkwardness, punctuation, grammar, polish

Important rules:
- Judge against the PROVIDED SOURCE, not against world knowledge.
- Do NOT silently reward a candidate for correcting source facts, dates, numbers, names, or roles.
- If the source is noisy, fragmented, typo-ridden, or malformed, prefer the candidate that remains faithful to the source over a candidate that invents, normalizes too aggressively, or turns into a generic assistant reply.
- A candidate that says things like "Please provide the text" or otherwise refuses / asks for input is a severe failure and should be ranked last unless every candidate is equally bad.
- Be strict about untranslated source-language fragments left in the target when they hurt usability.
- Use the reference only as auxiliary context if it is provided. Never let the reference override the source.

You must return STRICT JSON only.
No markdown. No code fences. No explanation outside JSON.

Return exactly this schema:
{
  "ranking": [<candidate ids in best-to-worst order>],
  "critical_candidate_ids": [<subset of candidate ids with severe failures>],
  "reasons": {
    "1": "short reason",
    "2": "short reason"
  }
}

Rules for the JSON:
- candidate ids are 1-based integers
- ranking must be a strict permutation of all candidate ids exactly once
- no ties are allowed in the JSON output
- critical_candidate_ids may be empty
- keep each reason short (one sentence or short clause)
"""

GROUP_RANK_USER_TEMPLATE_KO_EN_ENTERPRISE_V1 = """Rank the following candidate translations.

Source language: {source_lang}
Target language: {target_lang}

Source:
```text
{source_seg}
```

{reference_block}Candidates:
{candidate_blocks}

Return strict JSON only with keys:
- ranking
- critical_candidate_ids
- reasons
"""

GROUP_RANK_FEWSHOT_USER_1 = """Rank the following candidate translations.

Source language: Korean
Target language: English

Source:
```text
@김삼성 프로님, 예시 템플릿이 안보이거나 파일명처럼 보이는 경우가 있는데, 확인 요청 드립니다.
```

Candidates:
Candidate 1:
```text
Mr. Kim Samsung, there are cases where the sample template is not visible or appears as a file name. Please check.
```

Candidate 2:
```text
Hello Pro Kim, the example template may be hidden or shown as a filename. Please take a look.
```

Candidate 3:
```text
Please provide the Korean text you would like me to translate.
```

Candidate 4:
```text
Hi Mr. Kim, some users report that the example template is missing or displayed like a filename. Could you please check?
```

Return strict JSON only with keys:
- ranking
- critical_candidate_ids
- reasons
"""

GROUP_RANK_FEWSHOT_ASSISTANT_1 = """{
  "ranking": [1, 4, 2, 3],
  "critical_candidate_ids": [3],
  "reasons": {
    "1": "Most faithful overall; only slightly stiff naming.",
    "4": "Natural but adds 'some users report', which is not in the source.",
    "2": "Meaning mostly preserved but 'Pro Kim' is awkward and less context-fit.",
    "3": "Assistant fallback; not a translation of the source."
  }
}"""

GROUP_RANK_FEWSHOT_USER_2 = """Rank the following candidate translations.

Source language: Korean
Target language: English

Source:
```text
2000년 AI 알파고의 승리로 주목받기 시작
```

Candidates:
Candidate 1:
```text
It began to draw attention with AI AlphaGo's victory in 2000.
```

Candidate 2:
```text
It began to draw attention with AI AlphaGo's victory in 2016.
```

Candidate 3:
```text
People started paying attention after AlphaGo won.
```

Candidate 4:
```text
AlphaGo's 2000 victory made it famous and revolutionized global AI history.
```

Return strict JSON only with keys:
- ranking
- critical_candidate_ids
- reasons
"""

GROUP_RANK_FEWSHOT_ASSISTANT_2 = """{
  "ranking": [1, 3, 4, 2],
  "critical_candidate_ids": [2],
  "reasons": {
    "1": "Best source fidelity; keeps the explicit year from the source.",
    "3": "Meaning mostly preserved but drops the explicit year and 'AI'.",
    "4": "Keeps the year but adds unsupported claims.",
    "2": "Changes an explicit source fact (2000 -> 2016)."
  }
}"""

GROUP_RANK_FEWSHOT_USER_3 = """Rank the following candidate translations.

Source language: English
Target language: Korean

Source:
```text
I'm just about finished, sir.
```

Candidates:
Candidate 1:
```text
예, 거의 다 끝났습니다.
```

Candidate 2:
```text
거의 다 끝났습니다, 대 sir.
```

Candidate 3:
```text
거의 다 끝냈습니다, 각하.
```

Candidate 4:
```text
거의 끝났습니다.
```

Return strict JSON only with keys:
- ranking
- critical_candidate_ids
- reasons
"""

GROUP_RANK_FEWSHOT_ASSISTANT_3 = """{
  "ranking": [1, 4, 3, 2],
  "critical_candidate_ids": [2],
  "reasons": {
    "1": "Best overall and context-appropriate Korean.",
    "4": "Acceptable but loses some politeness nuance.",
    "3": "Fluent but register is too strong and context-shifted.",
    "2": "Leaves English in the Korean output and is grammatically broken."
  }
}"""

GROUP_RANK_FEWSHOT_USER_4 = """Rank the following candidate translations.

Source language: English
Target language: Korean

Source:
```text
This email is for outgoing messages only.
```

Candidates:
Candidate 1:
```text
이 메일은 발신전용 메일입니다.
```

Candidate 2:
```text
이 이메일은 발신자의 눈 전용입니다.
```

Candidate 3:
```text
이 메일은 외부 발송용으로만 사용됩니다.
```

Candidate 4:
```text
도와드릴까요? 번역할 텍스트를 보내주세요.
```

Return strict JSON only with keys:
- ranking
- critical_candidate_ids
- reasons
"""

GROUP_RANK_FEWSHOT_ASSISTANT_4 = """{
  "ranking": [1, 3, 2, 4],
  "critical_candidate_ids": [4],
  "reasons": {
    "1": "Best context fit for an outgoing-only / no-reply style message.",
    "3": "Partially related but shifts the operational meaning.",
    "2": "Mistranslates the meaning into a confidentiality statement.",
    "4": "Assistant fallback; not a translation."
  }
}"""

GROUP_RANK_FEWSHOT_PACKS = {
    "ko_en_enterprise_group_rank_v1": [
        (GROUP_RANK_FEWSHOT_USER_1, GROUP_RANK_FEWSHOT_ASSISTANT_1),
        (GROUP_RANK_FEWSHOT_USER_2, GROUP_RANK_FEWSHOT_ASSISTANT_2),
        (GROUP_RANK_FEWSHOT_USER_3, GROUP_RANK_FEWSHOT_ASSISTANT_3),
        (GROUP_RANK_FEWSHOT_USER_4, GROUP_RANK_FEWSHOT_ASSISTANT_4),
    ],
}

_VALID_GROUP_RANK_PROMPT_PACKS = frozenset(GROUP_RANK_FEWSHOT_PACKS.keys())


def _format_group_rank_candidate_blocks(candidates: list[str]) -> str:
    lines: list[str] = []
    for idx, cand in enumerate(candidates, start=1):
        lines.append(f"Candidate {idx}:\n```text\n{cand}\n```")
    return "\n\n".join(lines)


def _format_group_rank_reference_block(ref: str | None) -> str:
    if not ref:
        return ""
    return f"Reference (auxiliary only):\n```text\n{ref}\n```\n\n"


def build_group_rank_messages(
    *,
    source_lang: str,
    target_lang: str,
    source_seg: str,
    candidates: list[str],
    ref: str | None = None,
    prompt_pack: str = "ko_en_enterprise_group_rank_v1",
    use_fewshot: bool = True,
) -> list[dict[str, str]]:
    pack_key = str(prompt_pack or "ko_en_enterprise_group_rank_v1").strip() or "ko_en_enterprise_group_rank_v1"
    if pack_key not in _VALID_GROUP_RANK_PROMPT_PACKS:
        supported = ", ".join(sorted(_VALID_GROUP_RANK_PROMPT_PACKS))
        raise ValueError(f"Unsupported group rank prompt_pack={pack_key!r}. Supported: {supported}")

    messages: list[dict[str, str]] = [
        {"role": "system", "content": GROUP_RANK_SYSTEM_PROMPT_KO_EN_ENTERPRISE_V1},
    ]
    if use_fewshot:
        for user_msg, assistant_msg in GROUP_RANK_FEWSHOT_PACKS[pack_key]:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})

    user_content = GROUP_RANK_USER_TEMPLATE_KO_EN_ENTERPRISE_V1.format(
        source_lang=source_lang,
        target_lang=target_lang,
        source_seg=source_seg,
        reference_block=_format_group_rank_reference_block(ref),
        candidate_blocks=_format_group_rank_candidate_blocks(candidates),
    )
    messages.append({"role": "user", "content": user_content})
    return messages


def normalize_group_rank_candidate_text(text: str, *, mode: str) -> str:
    out = str(text).replace("\r\n", "\n")
    if mode == "none":
        return out
    if mode == "strip":
        return out.strip()
    if mode == "strip_nfkc":
        return unicodedata.normalize("NFKC", out).strip()
    raise ValueError(f"Unsupported duplicate_text_normalization: {mode}")


def deduplicate_group_rank_candidates(
    candidates: list[str],
    *,
    normalization_mode: str,
) -> tuple[list[str], list[int], list[list[int]], list[str]]:
    normalized_candidates = [
        normalize_group_rank_candidate_text(candidate, mode=normalization_mode)
        for candidate in candidates
    ]

    unique_candidates: list[str] = []
    original_to_unique_idx: list[int] = []
    unique_to_original_indices: list[list[int]] = []
    seen: dict[str, int] = {}

    for orig_idx, (raw_candidate, normalized_candidate) in enumerate(zip(candidates, normalized_candidates)):
        if normalized_candidate in seen:
            unique_idx = seen[normalized_candidate]
            original_to_unique_idx.append(unique_idx)
            unique_to_original_indices[unique_idx].append(orig_idx)
            continue
        unique_idx = len(unique_candidates)
        seen[normalized_candidate] = unique_idx
        unique_candidates.append(raw_candidate)
        original_to_unique_idx.append(unique_idx)
        unique_to_original_indices.append([orig_idx])

    return unique_candidates, original_to_unique_idx, unique_to_original_indices, normalized_candidates


def _parse_group_rank_json_object(raw_text: str) -> dict[str, Any]:
    obj = _try_parse_json_object(raw_text)
    if obj is not None:
        return obj

    try:
        parsed = json.loads(raw_text)
    except Exception as exc:
        raise ValueError(str(exc)) from exc
    if not isinstance(parsed, dict):
        raise ValueError("group rank response must be a JSON object")
    return parsed


def _parse_group_rank_reason_candidate_id(raw_key: Any) -> int | None:
    if isinstance(raw_key, bool):
        return None
    if isinstance(raw_key, int):
        return int(raw_key)
    key_text = str(raw_key).strip()
    if not key_text or re.fullmatch(r"\d+", key_text) is None:
        return None
    try:
        return int(key_text)
    except Exception:
        return None


def _collect_group_rank_reasons(
    reasons_in: Any,
    *,
    candidate_count: int,
) -> dict[int, str]:
    if reasons_in is None:
        return {}
    if not isinstance(reasons_in, dict):
        raise ValueError("reasons must be a dict")

    reasons: dict[int, str] = {}
    stack: list[dict[Any, Any]] = [reasons_in]

    while stack:
        current = stack.pop()
        for raw_key, raw_value in current.items():
            candidate_id = _parse_group_rank_reason_candidate_id(raw_key)
            if candidate_id is None:
                if isinstance(raw_value, dict):
                    stack.append(raw_value)
                elif isinstance(raw_value, list):
                    for item in raw_value:
                        if isinstance(item, dict):
                            stack.append(item)
                continue

            if candidate_id < 1 or candidate_id > int(candidate_count):
                continue
            if candidate_id in reasons:
                continue

            if isinstance(raw_value, (dict, list)):
                reasons[candidate_id] = json.dumps(raw_value, ensure_ascii=False)
            else:
                reasons[candidate_id] = str(raw_value)

    return reasons


def parse_group_rank_response(
    raw_text: str,
    *,
    candidate_count: int,
) -> tuple[list[int], list[int], dict[int, str]]:
    obj = _parse_group_rank_json_object(raw_text)

    ranking = obj.get("ranking")
    if not isinstance(ranking, list):
        raise ValueError("ranking must be a list")
    if len(ranking) != int(candidate_count):
        raise ValueError("ranking length mismatch")
    if any(not isinstance(candidate_id, int) for candidate_id in ranking):
        raise ValueError("ranking must contain integers only")
    if sorted(ranking) != list(range(1, int(candidate_count) + 1)):
        raise ValueError("ranking must be a strict permutation of 1..N")

    critical_ids = obj.get("critical_candidate_ids", [])
    if critical_ids is None:
        critical_ids = []
    if not isinstance(critical_ids, list):
        raise ValueError("critical_candidate_ids must be a list")
    deduped_critical: list[int] = []
    seen_critical: set[int] = set()
    for candidate_id in critical_ids:
        if not isinstance(candidate_id, int):
            raise ValueError("critical_candidate_ids must contain integers only")
        if candidate_id < 1 or candidate_id > int(candidate_count):
            raise ValueError("critical candidate id out of range")
        if candidate_id in seen_critical:
            continue
        seen_critical.add(candidate_id)
        deduped_critical.append(candidate_id)

    reasons = _collect_group_rank_reasons(
        obj.get("reasons", {}) or {},
        candidate_count=int(candidate_count),
    )

    return list(ranking), deduped_critical, reasons


def tie_aware_centered_borda_rewards_from_unique_ranking(
    *,
    unique_ranking_ids: list[int],
    unique_to_original_indices: list[list[int]],
    original_candidate_count: int,
    critical_unique_candidate_ids: list[int] | None = None,
    critical_error_penalty: float = 0.0,
    duplicate_extra_penalty: float = 0.0,
) -> list[float]:
    candidate_count = int(original_candidate_count)
    pos_rewards = [
        (candidate_count - pos) - ((candidate_count - 1) / 2.0)
        for pos in range(1, candidate_count + 1)
    ]

    critical_ids = set(critical_unique_candidate_ids or [])
    rewards = [0.0] * candidate_count
    slot_start = 0

    for unique_id in unique_ranking_ids:
        original_indices = unique_to_original_indices[unique_id - 1]
        width = len(original_indices)
        slot_end = slot_start + width
        base_reward = sum(pos_rewards[slot_start:slot_end]) / float(width)
        if unique_id in critical_ids:
            base_reward += float(critical_error_penalty)
        for original_idx in original_indices:
            rewards[original_idx] = float(base_reward)
        if float(duplicate_extra_penalty) != 0.0 and len(original_indices) > 1:
            for original_idx in original_indices[1:]:
                rewards[original_idx] += float(duplicate_extra_penalty)
        slot_start = slot_end

    return rewards


def _expand_group_rank_ranking_to_original_ids(
    *,
    unique_ranking_ids: list[int],
    unique_to_original_indices: list[list[int]],
) -> list[int]:
    ranking: list[int] = []
    for unique_id in unique_ranking_ids:
        ranking.extend(idx + 1 for idx in unique_to_original_indices[unique_id - 1])
    return ranking


def _propagate_group_rank_reasons_to_original_ids(
    *,
    unique_reasons: dict[int, str],
    unique_to_original_indices: list[list[int]],
) -> dict[int, str]:
    reasons: dict[int, str] = {}
    for unique_id, reason in unique_reasons.items():
        for original_idx in unique_to_original_indices[unique_id - 1]:
            reasons[original_idx + 1] = str(reason)
    return reasons


def _propagate_group_rank_critical_ids_to_original_ids(
    *,
    critical_unique_candidate_ids: list[int],
    unique_to_original_indices: list[list[int]],
) -> list[int]:
    critical_ids: list[int] = []
    for unique_id in critical_unique_candidate_ids:
        critical_ids.extend(idx + 1 for idx in unique_to_original_indices[unique_id - 1])
    return critical_ids


def _record_group_rank_parse_failure(
    *,
    log_path: Path | None,
    model_name: str,
    group: GroupRankSample,
    unique_candidates: list[str],
    raw_output: str,
    error: str,
) -> None:
    if log_path is None:
        return
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "pid": int(os.getpid()),
        "scorer": "group_rank",
        "model_name": model_name,
        "group_id": group.group_id,
        "source": group.src,
        "reference": group.ref,
        "source_lang": group.source_lang,
        "target_lang": group.target_lang,
        "unique_candidates": list(unique_candidates),
        "raw_output": raw_output,
        "error": str(error),
    }
    try:
        _append_jsonl_record(log_path, payload)
    except Exception as exc:
        logger.warning("Failed to append group_rank parse failure record to %s: %s", log_path, exc)


@dataclass
class OpenAICompatibleGroupRankScorer:
    cfg: GroupRankConfig
    predict_fn: Callable[[list[list[dict[str, str]]]], list[str]] | None = None
    parse_failure_log_path: str | Path | None = None

    def __post_init__(self) -> None:
        self._chat_url: str | None = None
        self._api_key: str | None = None
        self._parse_failure_log_path = Path(self.parse_failure_log_path) if self.parse_failure_log_path else None
        if self.predict_fn is not None or not self.cfg.enabled:
            return

        if not self.cfg.base_url or not str(self.cfg.base_url).strip():
            raise ValueError("Group rank scorer requires cfg.base_url when enabled.")
        self._chat_url = self._resolve_chat_url(self.cfg.base_url)

        if self.cfg.api_key and str(self.cfg.api_key).strip():
            self._api_key = str(self.cfg.api_key).strip()
        else:
            env_name = (self.cfg.api_key_env or "OPENAI_API_KEY").strip()
            self._api_key = os.environ.get(env_name) or os.environ.get("OPENAI_API_KEY")
            if self._api_key and self._api_key.strip():
                self._api_key = self._api_key.strip()
            else:
                self._api_key = None

    @staticmethod
    def _resolve_chat_url(base_url: str) -> str:
        url = str(base_url).strip().rstrip("/")
        if not url:
            raise ValueError("group rank base_url must not be empty.")
        if url.endswith("/chat/completions"):
            return url
        if url.endswith("/v1"):
            return f"{url}/chat/completions"
        return f"{url}/v1/chat/completions"

    def score_groups(self, groups: list[GroupRankSample]) -> dict[str, Any]:
        candidate_reward_rows: list[list[float]] = []
        ranking_rows: list[list[int]] = []
        critical_candidate_rows: list[list[int]] = []
        reasons_rows: list[dict[int, str]] = []
        raw_outputs: list[str] = []
        skipped_rows: list[bool] = []
        skip_reasons: list[str | None] = []
        meta_rows: list[dict[str, Any]] = []

        if not groups:
            return {
                "candidate_reward_rows": candidate_reward_rows,
                "ranking_rows": ranking_rows,
                "critical_candidate_rows": critical_candidate_rows,
                "reasons_rows": reasons_rows,
                "raw_outputs": raw_outputs,
                "skipped_rows": skipped_rows,
                "skip_reasons": skip_reasons,
                "meta_rows": meta_rows,
            }

        prepared_rows: list[dict[str, Any]] = []
        message_rows: list[list[dict[str, str]]] = []
        prepared_group_indices: list[int] = []

        for group_idx, group in enumerate(groups):
            original_candidates = list(group.candidates)
            original_count = len(original_candidates)
            normalization_mode = str(self.cfg.duplicate_text_normalization or "strip_nfkc").strip() or "strip_nfkc"
            if bool(self.cfg.deduplicate_exact_candidates):
                (
                    unique_candidates,
                    original_to_unique_idx,
                    unique_to_original_indices,
                    normalized_candidates,
                ) = deduplicate_group_rank_candidates(
                    original_candidates,
                    normalization_mode=normalization_mode,
                )
            else:
                normalized_candidates = [
                    normalize_group_rank_candidate_text(candidate, mode=normalization_mode)
                    for candidate in original_candidates
                ]
                unique_candidates = list(original_candidates)
                original_to_unique_idx = list(range(original_count))
                unique_to_original_indices = [[idx] for idx in range(original_count)]

            unique_count = len(unique_candidates)
            duplicate_class_count = sum(1 for indices in unique_to_original_indices if len(indices) > 1)
            duplicate_candidate_count = max(0, original_count - unique_count)
            base_meta = {
                "group_id": group.group_id,
                "original_candidate_count": int(original_count),
                "unique_candidate_count": int(unique_count),
                "duplicate_class_count": int(duplicate_class_count),
                "duplicate_candidate_count": int(duplicate_candidate_count),
                "had_duplicates": bool(duplicate_candidate_count > 0),
                "all_candidates_identical_after_dedup": bool(original_count > 0 and unique_count == 1),
                "normalized_candidates": list(normalized_candidates),
                "original_to_unique_idx": list(original_to_unique_idx),
                "unique_to_original_indices": [list(indices) for indices in unique_to_original_indices],
                "parse_failed": False,
            }

            if unique_count == 1:
                candidate_reward_rows.append([0.0 for _ in original_candidates])
                ranking_rows.append([])
                critical_candidate_rows.append([])
                reasons_rows.append({})
                raw_outputs.append("")
                skipped_rows.append(True)
                skip_reasons.append("all_candidates_identical_after_dedup")
                meta_rows.append(base_meta)
                continue

            if unique_count < int(self.cfg.candidate_min) or unique_count > int(self.cfg.candidate_max):
                exc = ValueError(
                    f"group rank unique candidate count {unique_count} outside configured range "
                    f"[{self.cfg.candidate_min}, {self.cfg.candidate_max}]"
                )
                if str(self.cfg.failure_policy).strip().lower() == "raise":
                    raise exc
                candidate_reward_rows.append([0.0 for _ in original_candidates])
                ranking_rows.append([])
                critical_candidate_rows.append([])
                reasons_rows.append({})
                raw_outputs.append("")
                skipped_rows.append(True)
                skip_reasons.append(str(exc))
                meta_rows.append(base_meta)
                continue

            messages = build_group_rank_messages(
                source_lang=str(group.source_lang or "Unknown").strip() or "Unknown",
                target_lang=str(group.target_lang or "Unknown").strip() or "Unknown",
                source_seg=group.src,
                candidates=unique_candidates,
                ref=group.ref if bool(self.cfg.use_reference) else None,
                prompt_pack=str(self.cfg.prompt_pack or "ko_en_enterprise_group_rank_v1"),
                use_fewshot=bool(self.cfg.use_fewshot),
            )
            candidate_reward_rows.append([])
            ranking_rows.append([])
            critical_candidate_rows.append([])
            reasons_rows.append({})
            raw_outputs.append("")
            skipped_rows.append(False)
            skip_reasons.append(None)
            meta_rows.append(base_meta)
            prepared_rows.append(
                {
                    "group": group,
                    "messages": messages,
                    "unique_candidates": unique_candidates,
                    "unique_to_original_indices": unique_to_original_indices,
                    "meta": base_meta,
                }
            )
            message_rows.append(messages)
            prepared_group_indices.append(group_idx)

        if message_rows:
            if self.predict_fn is not None:
                batch_raw_outputs = list(self.predict_fn(message_rows))
                if len(batch_raw_outputs) != len(message_rows):
                    raise RuntimeError(
                        "Group rank predict_fn returned mismatched output length: "
                        f"requested={len(message_rows)} returned={len(batch_raw_outputs)}"
                    )
            else:
                max_workers = max(1, int(self.cfg.batch_size))
                if max_workers == 1:
                    batch_raw_outputs = [
                        self._score_one_group(prepared["messages"])
                        for prepared in prepared_rows
                    ]
                else:
                    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="group-rank-scorer") as executor:
                        batch_raw_outputs = _run_jobs_with_bounded_concurrency(
                            executor=executor,
                            jobs=[(prepared["messages"],) for prepared in prepared_rows],
                            worker_fn=lambda messages: self._score_one_group(messages),
                            max_in_flight=max_workers,
                        )

            for prepared, group_idx, raw_output in zip(prepared_rows, prepared_group_indices, batch_raw_outputs):
                group = prepared["group"]
                unique_candidates = prepared["unique_candidates"]
                unique_to_original_indices = prepared["unique_to_original_indices"]
                meta = prepared["meta"]
                if isinstance(raw_output, Exception):
                    exc = raw_output
                    if str(self.cfg.failure_policy).strip().lower() == "raise":
                        raise exc
                    candidate_reward_rows[group_idx] = [0.0 for _ in group.candidates]
                    raw_outputs[group_idx] = ""
                    skipped_rows[group_idx] = True
                    skip_reasons[group_idx] = str(exc)
                    meta_rows[group_idx] = dict(meta)
                    continue

                raw_outputs[group_idx] = str(raw_output)
                try:
                    ranking_ids, critical_unique_ids, unique_reasons = parse_group_rank_response(
                        str(raw_output),
                        candidate_count=len(unique_candidates),
                    )
                except Exception as exc:
                    _record_group_rank_parse_failure(
                        log_path=self._parse_failure_log_path,
                        model_name=self.cfg.model_name,
                        group=group,
                        unique_candidates=unique_candidates,
                        raw_output=str(raw_output),
                        error=str(exc),
                    )
                    if str(self.cfg.failure_policy).strip().lower() == "raise":
                        raise
                    candidate_reward_rows[group_idx] = [0.0 for _ in group.candidates]
                    skipped_rows[group_idx] = True
                    skip_reasons[group_idx] = str(exc)
                    failed_meta = dict(meta)
                    failed_meta["parse_failed"] = True
                    meta_rows[group_idx] = failed_meta
                    continue

                candidate_reward_rows[group_idx] = tie_aware_centered_borda_rewards_from_unique_ranking(
                    unique_ranking_ids=ranking_ids,
                    unique_to_original_indices=unique_to_original_indices,
                    original_candidate_count=len(group.candidates),
                    critical_unique_candidate_ids=critical_unique_ids,
                    critical_error_penalty=float(self.cfg.critical_error_penalty),
                    duplicate_extra_penalty=float(self.cfg.duplicate_extra_penalty),
                )
                ranking_rows[group_idx] = _expand_group_rank_ranking_to_original_ids(
                    unique_ranking_ids=ranking_ids,
                    unique_to_original_indices=unique_to_original_indices,
                )
                critical_candidate_rows[group_idx] = _propagate_group_rank_critical_ids_to_original_ids(
                    critical_unique_candidate_ids=critical_unique_ids,
                    unique_to_original_indices=unique_to_original_indices,
                )
                reasons_rows[group_idx] = _propagate_group_rank_reasons_to_original_ids(
                    unique_reasons=unique_reasons,
                    unique_to_original_indices=unique_to_original_indices,
                )

        return {
            "candidate_reward_rows": candidate_reward_rows,
            "ranking_rows": ranking_rows,
            "critical_candidate_rows": critical_candidate_rows,
            "reasons_rows": reasons_rows,
            "raw_outputs": raw_outputs,
            "skipped_rows": skipped_rows,
            "skip_reasons": skip_reasons,
            "meta_rows": meta_rows,
        }

    def _score_one_group(self, messages: list[dict[str, str]]) -> str:
        last_exc: Exception | None = None
        attempts = max(1, int(self.cfg.max_retries) + 1)
        for _ in range(attempts):
            try:
                return self._call_openai_compatible_api(messages)
            except Exception as exc:
                last_exc = exc
                continue
        if last_exc is None:
            raise RuntimeError("group rank scoring failed without an exception.")
        raise last_exc

    def _call_openai_compatible_api(self, messages: list[dict[str, str]]) -> str:
        if self._chat_url is None:
            raise RuntimeError("group rank scorer chat URL is not set.")

        payload = {
            "model": self.cfg.model_name,
            "messages": messages,
            "temperature": float(self.cfg.temperature),
            "top_p": float(self.cfg.top_p),
            "max_tokens": int(self.cfg.max_tokens),
        }
        if self.cfg.top_k is not None:
            payload["top_k"] = int(self.cfg.top_k)
        if self.cfg.presence_penalty is not None:
            payload["presence_penalty"] = float(self.cfg.presence_penalty)
        if self.cfg.repetition_penalty is not None:
            payload["repetition_penalty"] = float(self.cfg.repetition_penalty)
        if self.cfg.stop:
            payload["stop"] = list(self.cfg.stop)
        if self.cfg.chat_template_kwargs:
            payload["chat_template_kwargs"] = dict(self.cfg.chat_template_kwargs)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        req = urllib_request.Request(
            self._chat_url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        if self._api_key:
            req.add_header("Authorization", f"Bearer {self._api_key}")

        try:
            timeout = float(self.cfg.timeout_s or self.cfg.timeout_sec)
            restore_proxy_env = _temporarily_unset_proxy_env()
            try:
                opener = urllib_request.build_opener(urllib_request.ProxyHandler({}))
                with opener.open(req, timeout=timeout) as resp:
                    resp_body = resp.read().decode("utf-8")
            finally:
                restore_proxy_env()
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"group rank API HTTPError status={exc.code} body={detail}") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"group rank API URLError: {exc}") from exc

        try:
            parsed = json.loads(resp_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("group rank API response is not valid JSON.") from exc

        return _extract_openai_response_text(
            parsed=parsed,
            scorer_name="group_rank",
            log_io=False,
            log_max_chars=20000,
        )


def metricx_qe_input(src: str, mt: str) -> str:
    return f"source: {src} candidate: {mt}"


def metricx_ref_input(src: str, mt: str, ref: str) -> str:
    return f"source: {src} candidate: {mt} reference: {ref}"


def metricx_score_to_reward(metricx_score: float, offset: float = 5.0) -> float:
    return float(offset) - float(metricx_score)


@dataclass
class MetricXQEScorer:
    cfg: MetricXConfig
    predict_fn: Callable[[list[str]], list[float]] | None = None

    def __post_init__(self) -> None:
        self._worker: _ScorerSubprocessClient | None = None
        self._model = None
        self._tokenizer = None
        self._device = resolve_device(self.cfg.device)
        self._dtype = resolve_torch_dtype(self.cfg.dtype)
        self._candidate_dtypes: list[Any] = []
        self._active_dtype_idx: int = -1
        self._model_cls: Any = None
        self._model_source: str | None = None

        if self.predict_fn is not None:
            return

        if not self.cfg.enabled:
            return

        if self.cfg.python_executable:
            worker_cfg_device = self.cfg.device
            worker_env_overrides: dict[str, str] | None = collect_huggingface_worker_env() or None
            metricx_gpu_idx = _parse_cuda_device_index(self.cfg.device)
            if metricx_gpu_idx is not None:
                worker_env_overrides = merge_env_overrides(
                    worker_env_overrides,
                    {"CUDA_VISIBLE_DEVICES": str(metricx_gpu_idx)},
                )
                worker_cfg_device = "cuda:0"
            cfg_payload = {
                "model_name": self.cfg.model_name,
                "tokenizer_name": self.cfg.tokenizer_name,
                "use_reference": bool(self.cfg.use_reference),
                "batch_size": int(self.cfg.batch_size),
                "device": worker_cfg_device,
                "dtype": self.cfg.dtype,
                "max_input_length": int(self.cfg.max_input_length),
                "overflow_policy": self.cfg.overflow_policy,
            }
            logger.info(
                "Starting MetricX scorer worker init: python=%s host=%s requested_device=%s worker_device=%s",
                self.cfg.python_executable,
                self.cfg.worker_host or "local",
                self.cfg.device,
                worker_cfg_device,
            )
            self._worker = _ScorerSubprocessClient(
                backend="metricx",
                python_executable=self.cfg.python_executable,
                timeout_sec=float(self.cfg.subprocess_timeout_sec),
                config_payload=cfg_payload,
                env_overrides=worker_env_overrides,
                remote_host=self.cfg.worker_host,
                remote_workdir=self.cfg.worker_remote_workdir,
            )
            logger.info(
                "MetricX scorer will run in external python=%s host=%s (requested_device=%s worker_device=%s).",
                self.cfg.python_executable,
                self.cfg.worker_host or "local",
                self.cfg.device,
                worker_cfg_device,
            )
            return

        if torch is None or AutoTokenizer is None:
            import_errors: list[str] = []
            if _TORCH_IMPORT_ERROR is not None:
                import_errors.append(f"torch import error: {type(_TORCH_IMPORT_ERROR).__name__}: {_TORCH_IMPORT_ERROR}")
            if _TRANSFORMERS_IMPORT_ERROR is not None:
                import_errors.append(
                    "transformers import error: "
                    f"{type(_TRANSFORMERS_IMPORT_ERROR).__name__}: {_TRANSFORMERS_IMPORT_ERROR}"
                )
            detail = f" ({'; '.join(import_errors)})" if import_errors else ""
            raise RuntimeError(
                "MetricX model loading requires torch and transformers. "
                f"Use predict_fn for lightweight testing.{detail}"
            )
        try:
            from .metricx_model import MT5ForRegression
        except Exception as exc:
            raise RuntimeError(
                "Failed to import MetricX regression model class. "
                "Check transformers installation."
            ) from exc

        self._model_cls = MT5ForRegression
        model_name = self.cfg.model_name
        self._model_source = self._resolve_model_source(model_name)
        self._tokenizer = self._load_tokenizer()
        candidate_dtypes = self._build_candidate_dtypes(model_name=model_name)
        self._candidate_dtypes = list(candidate_dtypes)
        last_error: Exception | None = None
        for idx, cand_dtype in enumerate(candidate_dtypes):
            try:
                logger.info(
                    "Loading MetricX model=%s on device=%s dtype=%s",
                    self._model_source or model_name,
                    self._device,
                    cand_dtype,
                )
                self._model = self._load_metricx_model(self._model_cls, self._model_source or model_name, cand_dtype)
                self._model.to(self._device)
                self._model.eval()
                self._dtype = cand_dtype
                self._active_dtype_idx = idx
                break
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "MetricX load failed for model=%s device=%s dtype=%s: %s",
                    model_name,
                    self._device,
                    cand_dtype,
                    exc,
                )
                if torch is not None and self._device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if self._model is None:
            raise RuntimeError(
                "Failed to load MetricX model after trying multiple dtypes. "
                f"model={model_name} device={self._device} tried_dtypes={candidate_dtypes}. "
                f"last_error={last_error!r}"
            ) from last_error

    def _build_candidate_dtypes(self, model_name: str) -> list[Any]:
        if torch is None:
            return [self._dtype]

        candidates: list[Any] = []
        model_name_lc = (model_name or "").lower()
        prefer_fp16_for_xxl = self._device.startswith("cuda") and "metricx-24-hybrid-xxl" in model_name_lc
        if prefer_fp16_for_xxl and torch.float16 not in candidates:
            # Empirically xxl can become unstable with bf16 on some setups.
            candidates.append(torch.float16)
        if self._dtype is not None and self._dtype not in candidates:
            candidates.append(self._dtype)

        if self._device.startswith("cuda"):
            if torch.bfloat16 in candidates:
                try:
                    if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
                        logger.warning("CUDA bf16 not supported; removing bfloat16 from MetricX dtype candidates.")
                        candidates = [d for d in candidates if d != torch.bfloat16]
                except Exception:
                    pass
            for fallback_dtype in (torch.float16, torch.float32):
                if fallback_dtype not in candidates:
                    candidates.append(fallback_dtype)
        else:
            if torch.float32 not in candidates:
                candidates.append(torch.float32)

        if not candidates:
            candidates.append(None)
        return candidates

    def _reload_with_next_dtype(self) -> bool:
        if self._model_cls is None:
            return False
        if self._active_dtype_idx < 0:
            return False
        next_idx = self._active_dtype_idx + 1
        if next_idx >= len(self._candidate_dtypes):
            return False

        next_dtype = self._candidate_dtypes[next_idx]
        model_name = self._model_source or self.cfg.model_name
        logger.warning(
            "Retrying MetricX with safer dtype due to non-finite outputs: model=%s old_dtype=%s next_dtype=%s",
            model_name,
            self._dtype,
            next_dtype,
        )
        try:
            self._model = None
            if torch is not None and self._device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._model = self._load_metricx_model(self._model_cls, model_name, next_dtype)
            self._model.to(self._device)
            self._model.eval()
            self._dtype = next_dtype
            self._active_dtype_idx = next_idx
            return True
        except Exception as exc:
            logger.warning(
                "MetricX reload failed for model=%s device=%s dtype=%s: %s",
                model_name,
                self._device,
                next_dtype,
                exc,
            )
            if torch is not None and self._device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False

    def _resolve_model_source(self, model_name: str) -> str:
        text = (model_name or "").strip()
        if not text:
            raise ValueError("MetricX model_name must not be empty.")

        path = Path(text).expanduser()
        if path.exists():
            return str(path)

        try:
            from huggingface_hub import snapshot_download
        except Exception:
            logger.warning(
                "huggingface_hub.snapshot_download unavailable; loading MetricX directly from repo id %s",
                text,
            )
            return text

        cache_dir = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
        kwargs: dict[str, Any] = {
            "repo_id": text,
            "cache_dir": cache_dir,
            "max_workers": 1,
            "etag_timeout": 60,
        }

        # Retry once after cleaning stale incomplete downloads/locks.
        for attempt in range(2):
            try:
                try:
                    local_path = snapshot_download(resume_download=True, **kwargs)
                except TypeError:
                    kwargs.pop("etag_timeout", None)
                    local_path = snapshot_download(**kwargs)
                logger.info("MetricX snapshot ready: repo=%s local_path=%s", text, local_path)
                return str(local_path)
            except Exception as exc:
                if attempt == 0:
                    self._cleanup_stale_hf_partials(repo_id=text, cache_dir=cache_dir)
                    logger.warning("MetricX snapshot download failed once; retrying: repo=%s err=%s", text, exc)
                    continue
                raise RuntimeError(f"Failed to download MetricX snapshot: repo={text}") from exc

        return text

    @staticmethod
    def _cleanup_stale_hf_partials(repo_id: str, cache_dir: str | None) -> None:
        if not cache_dir:
            return
        repo_cache_dir = Path(cache_dir) / f"models--{repo_id.replace('/', '--')}"
        if not repo_cache_dir.exists():
            return
        for p in repo_cache_dir.glob("blobs/*.incomplete"):
            try:
                p.unlink()
            except Exception:
                pass
        for p in repo_cache_dir.glob(".locks/**/*"):
            if p.is_file():
                try:
                    p.unlink()
                except Exception:
                    pass

    @staticmethod
    def _load_metricx_model(model_cls: Any, model_name: str, dtype: Any):
        kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
        if dtype is not None:
            kwargs["dtype"] = dtype
        try:
            return model_cls.from_pretrained(model_name, **kwargs)
        except TypeError:
            kwargs.pop("low_cpu_mem_usage", None)
            if "dtype" in kwargs:
                kwargs["torch_dtype"] = kwargs.pop("dtype")
            return model_cls.from_pretrained(model_name, **kwargs)
        except Exception as exc:
            # Some environments don't have accelerate installed, which makes
            # `low_cpu_mem_usage=True` unavailable. Retry without it.
            msg = str(exc).lower()
            if "low_cpu_mem_usage" in msg or "accelerate" in msg:
                kwargs.pop("low_cpu_mem_usage", None)
                if "dtype" in kwargs:
                    kwargs["torch_dtype"] = kwargs.pop("dtype")
                return model_cls.from_pretrained(model_name, **kwargs)
            raise

    def _load_tokenizer(self):
        candidates: list[str] = []
        if self.cfg.tokenizer_name and self.cfg.tokenizer_name.strip():
            candidates.append(self.cfg.tokenizer_name.strip())
        if self.cfg.model_name and self.cfg.model_name.strip():
            candidates.append(self.cfg.model_name.strip())
        candidates.extend(["google/mt5-xl", "google/mt5-large"])

        tried: list[str] = []
        for tok_name in candidates:
            if tok_name in tried:
                continue
            tried.append(tok_name)
            try:
                # MetricX's MT5 tokenizer path is more stable with slow tokenizer.
                return AutoTokenizer.from_pretrained(tok_name, use_fast=False)
            except Exception as exc_slow:
                logger.warning("MetricX tokenizer load failed (%s, slow): %s", tok_name, exc_slow)
                try:
                    return AutoTokenizer.from_pretrained(tok_name, use_fast=True)
                except Exception as exc_fast:
                    logger.warning("MetricX tokenizer load failed (%s, fast): %s", tok_name, exc_fast)

        raise RuntimeError(
            "Failed to load tokenizer for MetricX. "
            f"tried={tried}. Set reward.metricx.tokenizer_name (recommended: google/mt5-xl)."
        )

    def score_batch(self, samples: list[SampleForScoring]) -> RewardOutput:
        if not samples:
            return RewardOutput(sequence_scores=[])
        if self._worker is not None:
            worker_samples = [{"src": s.src, "mt": s.mt, "ref": s.ref} for s in samples]
            resp = self._worker.request({"type": "score", "samples": worker_samples})
            if not bool(resp.get("ok", False)):
                err = resp.get("error", "unknown error")
                worker_tb = resp.get("traceback")
                worker_runtime = resp.get("runtime")
                tb_text = f"\nworker_traceback:\n{worker_tb}" if worker_tb else ""
                runtime_text = f"\nworker_runtime:\n{worker_runtime}" if worker_runtime else ""
                raise RuntimeError(f"MetricX worker scoring failed: {err}{tb_text}{runtime_text}")
            scores = [float(v) for v in list(resp.get("scores", []))]
            if len(scores) != len(samples):
                raise RuntimeError(
                    f"MetricX worker returned mismatched score length: expected={len(samples)} got={len(scores)}"
                )
            metadata = resp.get("metadata")
            return RewardOutput(sequence_scores=scores, metadata=metadata)

        inputs: list[str] = []
        for sample in samples:
            ref_text = (sample.ref or "").strip()
            if self.cfg.use_reference and ref_text:
                inputs.append(metricx_ref_input(sample.src, sample.mt, ref_text))
            else:
                inputs.append(metricx_qe_input(sample.src, sample.mt))
        if self.predict_fn is not None:
            scores = [float(v) for v in self.predict_fn(inputs)]
            return RewardOutput(sequence_scores=scores, metadata={"inputs": inputs})

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("MetricXQEScorer is not initialized with a model.")

        all_scores: list[float] = []
        skipped_samples = 0
        for i in range(0, len(inputs), self.cfg.batch_size):
            batch = inputs[i : i + self.cfg.batch_size]

            if self.cfg.overflow_policy == "skip":
                raw_ids = self._tokenizer(batch, truncation=False, padding=False)["input_ids"]
                lengths = [len(ids) for ids in raw_ids]
                kept_batch = []
                filtered_idx = []
                for j, length in enumerate(lengths):
                    if int(length) <= self.cfg.max_input_length:
                        kept_batch.append(batch[j])
                        filtered_idx.append(j)
                batch_scores = [math.nan] * len(batch)
                if kept_batch:
                    pred_scores = self._predict_scores(kept_batch)
                    for pos, score in zip(filtered_idx, pred_scores):
                        batch_scores[pos] = score
                skipped_samples += sum(1 for score in batch_scores if math.isnan(score))
                all_scores.extend(batch_scores)
                continue

            all_scores.extend(self._predict_scores(batch))

        return RewardOutput(
            sequence_scores=all_scores,
            metadata={
                "inputs": inputs,
                "skipped_count": skipped_samples,
                "use_reference": bool(self.cfg.use_reference),
            },
        )

    def _predict_scores(self, batch_inputs: list[str]) -> list[float]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("MetricXQEScorer is not initialized with a model.")

        tokenized = self._tokenizer(
            batch_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_input_length,
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        input_ids = input_ids.to(self._device)
        attention_mask = attention_mask.to(self._device)

        for _ in range(max(1, len(self._candidate_dtypes) + 1)):
            with torch.no_grad():
                outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.predictions
            if torch.isfinite(preds).all():
                return [float(v) for v in preds.detach().float().cpu().tolist()]

            bad_count = int((~torch.isfinite(preds)).sum().item())
            logger.warning(
                "MetricX produced non-finite predictions (count=%s) model=%s dtype=%s device=%s.",
                bad_count,
                self.cfg.model_name,
                self._dtype,
                self._device,
            )
            if not self._reload_with_next_dtype():
                break

        raise RuntimeError(
            "MetricX produced non-finite predictions and recovery failed. "
            f"model={self.cfg.model_name} device={self._device} dtype={self._dtype}"
        )

    def _strip_final_eos(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._tokenizer is None:
            return input_ids, attention_mask
        eos_token_id = self._tokenizer.eos_token_id
        pad_token_id = self._tokenizer.pad_token_id or 0
        if eos_token_id is None:
            return input_ids, attention_mask

        ids = input_ids.clone()
        mask = attention_mask.clone()
        for row_idx in range(ids.size(0)):
            length = int(mask[row_idx].sum().item())
            if length <= 1:
                continue
            last_idx = length - 1
            if int(ids[row_idx, last_idx].item()) == int(eos_token_id):
                mask[row_idx, last_idx] = 0
                ids[row_idx, last_idx] = int(pad_token_id)
        return ids, mask

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            if getattr(self, "_worker", None) is not None:
                self._worker.close()
        except Exception:
            pass


@dataclass
class XCometXLScorer:
    cfg: XCometConfig
    predict_fn: Callable[[list[dict[str, str]]], Any] | None = None

    def __post_init__(self) -> None:
        self._worker: _ScorerSubprocessClient | None = None
        self._model = None
        self._device = resolve_device(self.cfg.device)
        if self.predict_fn is not None or not self.cfg.enabled:
            return

        if self.cfg.python_executable:
            worker_cfg_device = self.cfg.device
            worker_env_overrides: dict[str, str] | None = collect_huggingface_worker_env() or None
            xcomet_gpu_idx = _parse_cuda_device_index(self.cfg.device)
            if xcomet_gpu_idx is not None:
                worker_env_overrides = merge_env_overrides(
                    worker_env_overrides,
                    {"CUDA_VISIBLE_DEVICES": str(xcomet_gpu_idx)},
                )
                worker_cfg_device = "cuda:0"
            cfg_payload = {
                "model_name": self.cfg.model_name,
                "batch_size": int(self.cfg.batch_size),
                "device": worker_cfg_device,
                "use_reference": bool(self.cfg.use_reference),
            }
            logger.info(
                "Starting xCOMET scorer worker init: python=%s host=%s requested_device=%s worker_device=%s",
                self.cfg.python_executable,
                self.cfg.worker_host or "local",
                self.cfg.device,
                worker_cfg_device,
            )
            self._worker = _ScorerSubprocessClient(
                backend="xcomet",
                python_executable=self.cfg.python_executable,
                timeout_sec=float(self.cfg.subprocess_timeout_sec),
                config_payload=cfg_payload,
                env_overrides=worker_env_overrides,
                remote_host=self.cfg.worker_host,
                remote_workdir=self.cfg.worker_remote_workdir,
            )
            logger.info(
                "xCOMET scorer will run in external python=%s host=%s (requested_device=%s worker_device=%s).",
                self.cfg.python_executable,
                self.cfg.worker_host or "local",
                self.cfg.device,
                worker_cfg_device,
            )
            return

        try:
            from comet import download_model, load_from_checkpoint
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "XCometXLScorer requires unbabel-comet>=2.2.0. Install it before running RL."
            ) from exc

        logger.info(
            "xCOMET loading start: model=%s device=%s batch_size=%s use_reference=%s",
            self.cfg.model_name,
            self._device,
            self.cfg.batch_size,
            self.cfg.use_reference,
        )
        model_path = download_model(self.cfg.model_name)
        logger.info("xCOMET model path resolved/downloaded: %s", model_path)
        self._model = load_from_checkpoint(model_path)
        logger.info("xCOMET checkpoint loaded: %s", self.cfg.model_name)
        if self._device.startswith("cuda") and torch is not None:
            logger.info("xCOMET moving model to device: %s", self._device)
            self._model.to(torch.device(self._device))
        self._model.eval()
        logger.info("xCOMET model ready on device: %s", self._device)

    def score_batch(self, samples: list[SampleForScoring]) -> RewardOutput:
        if not samples:
            return RewardOutput(sequence_scores=[], metadata={"error_spans": []})

        payload: list[dict[str, str]] = []
        for sample in samples:
            record = {"src": sample.src, "mt": sample.mt}
            if self.cfg.use_reference and sample.ref:
                record["ref"] = sample.ref
            payload.append(record)

        if self._worker is not None:
            resp = self._worker.request({"type": "score", "payload": payload})
            if not bool(resp.get("ok", False)):
                err = resp.get("error", "unknown error")
                worker_tb = resp.get("traceback")
                worker_runtime = resp.get("runtime")
                tb_text = f"\nworker_traceback:\n{worker_tb}" if worker_tb else ""
                runtime_text = f"\nworker_runtime:\n{worker_runtime}" if worker_runtime else ""
                raise RuntimeError(f"xCOMET worker scoring failed: {err}{tb_text}{runtime_text}")
            scores = [float(v) for v in list(resp.get("scores", []))]
            if len(scores) != len(samples):
                raise RuntimeError(
                    f"xCOMET worker returned mismatched score length: expected={len(samples)} got={len(scores)}"
                )
            if "error_spans" in resp:
                try:
                    spans = extract_error_spans(resp, expected=len(samples), source="xCOMET worker")
                except ValueError as exc:
                    raise RuntimeError(str(exc)) from exc
            else:
                spans = [[] for _ in scores]
            return RewardOutput(sequence_scores=scores, metadata={"error_spans": spans})

        if self.predict_fn is not None:
            result = self.predict_fn(payload)
            scores, spans = self._parse_prediction(result, expected=len(samples))
            return RewardOutput(sequence_scores=scores, metadata={"error_spans": spans})

        if self._model is None:
            raise RuntimeError("XCometXLScorer is not initialized with a model.")

        scores, spans = self._predict_with_loaded_model(payload)
        return RewardOutput(sequence_scores=scores, metadata={"error_spans": spans})

    def _predict_with_loaded_model(self, payload: list[dict[str, str]]) -> tuple[list[float], list[list[dict[str, Any]]]]:
        if self._model is None:
            raise RuntimeError("XCometXLScorer is not initialized with a model.")
        if torch is None:
            detail = ""
            if _TORCH_IMPORT_ERROR is not None:
                detail = f" (torch import error: {type(_TORCH_IMPORT_ERROR).__name__}: {_TORCH_IMPORT_ERROR})"
            raise RuntimeError(f"XCometXLScorer requires torch for model inference.{detail}")

        all_scores: list[float] = []
        all_spans: list[list[dict[str, Any]]] = []
        for i in range(0, len(payload), self.cfg.batch_size):
            batch_payload = payload[i : i + self.cfg.batch_size]
            batch_inputs = self._model.prepare_for_inference(batch_payload)
            batch_inputs = self._move_to_device(batch_inputs, self._device)
            with torch.no_grad():
                pred = self._model.predict_step(batch_inputs)

            batch_scores_tensor = pred.get("scores") if isinstance(pred, dict) else getattr(pred, "scores", None)
            if batch_scores_tensor is None:
                raise ValueError("xCOMET predict_step returned no scores.")
            batch_scores = torch.as_tensor(batch_scores_tensor).detach().float().cpu().tolist()
            batch_size = len(batch_scores)

            if isinstance(pred, dict):
                metadata = pred.get("metadata")
            else:
                metadata = getattr(pred, "metadata", None)
            batch_spans = extract_error_spans(
                metadata,
                expected=batch_size,
                source="xCOMET predict_step metadata",
            )

            all_scores.extend(float(v) for v in batch_scores)
            all_spans.extend(batch_spans)

        return all_scores, all_spans

    @staticmethod
    def _move_to_device(batch: Any, device: str) -> Any:
        if torch is None:
            return batch
        if torch.is_tensor(batch):
            return batch.to(device)
        if isinstance(batch, dict):
            return {k: XCometXLScorer._move_to_device(v, device) for k, v in batch.items()}
        if isinstance(batch, tuple):
            return tuple(XCometXLScorer._move_to_device(v, device) for v in batch)
        if isinstance(batch, list):
            return [XCometXLScorer._move_to_device(v, device) for v in batch]
        return batch

    @staticmethod
    def _parse_prediction(result: Any, expected: int) -> tuple[list[float], list[list[dict[str, Any]]]]:
        scores: list[float] | None = None
        metadata: Any = None

        if hasattr(result, "scores"):
            scores = [float(v) for v in list(result.scores)]
            metadata = getattr(result, "metadata", None)
        elif isinstance(result, dict):
            if "scores" in result:
                scores = [float(v) for v in list(result["scores"])]
            metadata = result.get("metadata", result)
        elif isinstance(result, tuple):
            if len(result) >= 1:
                scores = [float(v) for v in list(result[0])]
            if len(result) >= 2:
                metadata = result[1]

        if scores is None:
            raise ValueError("Unsupported xCOMET prediction output format.")

        spans = extract_error_spans(metadata, expected=expected, source="xCOMET prediction")
        if len(scores) != expected:
            raise ValueError(f"xCOMET score length mismatch expected={expected} got={len(scores)}")
        return scores, spans

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            if getattr(self, "_worker", None) is not None:
                self._worker.close()
        except Exception:
            pass


def _mqm_failure_fallback_score(cfg: MQMConfig) -> float:
    failure_policy = str(cfg.failure_policy).strip().lower()
    if failure_policy == "neutral_zero":
        return 0.0
    if failure_policy == "worst_score":
        return float(cfg.score_min)
    raise RuntimeError("failure_policy=raise should not request fallback")


@dataclass
class OpenAICompatibleMQMScorer:
    cfg: MQMConfig
    predict_fn: Callable[[list[list[dict[str, str]]]], list[float]] | None = None
    parse_failure_log_path: str | Path | None = None

    def __post_init__(self) -> None:
        self._chat_url: str | None = None
        self._api_key: str | None = None
        self._parse_failure_log_path = Path(self.parse_failure_log_path) if self.parse_failure_log_path else None
        if self.predict_fn is not None or not self.cfg.enabled:
            return

        if not self.cfg.base_url or not str(self.cfg.base_url).strip():
            raise ValueError("MQM scorer requires cfg.base_url when enabled.")
        self._chat_url = self._resolve_chat_url(self.cfg.base_url)

        if self.cfg.api_key and str(self.cfg.api_key).strip():
            self._api_key = str(self.cfg.api_key).strip()
        else:
            env_name = (self.cfg.api_key_env or "OPENAI_API_KEY").strip()
            self._api_key = os.environ.get(env_name) or os.environ.get("OPENAI_API_KEY")
            if self._api_key and self._api_key.strip():
                self._api_key = self._api_key.strip()
            else:
                self._api_key = None

    @staticmethod
    def _resolve_chat_url(base_url: str) -> str:
        url = str(base_url).strip().rstrip("/")
        if not url:
            raise ValueError("MQM base_url must not be empty.")
        if url.endswith("/chat/completions"):
            return url
        if url.endswith("/v1"):
            return f"{url}/chat/completions"
        return f"{url}/v1/chat/completions"

    def score_batch(self, samples: list[SampleForScoring]) -> RewardOutput:
        if not samples:
            return RewardOutput(
                sequence_scores=[],
                metadata={
                    "raw_outputs": [],
                    "error_spans": [],
                    "unanchored_errors": [],
                    "skipped_rows": [],
                    "skip_reasons": [],
                    "failure_rows": [],
                },
            )

        message_rows = [
            build_gemba_mqm_messages(
                source_lang=source_lang,
                target_lang=target_lang,
                source_seg=sample.src,
                target_seg=sample.mt,
                use_fewshot=bool(self.cfg.use_fewshot),
                prompt_pack=str(self.cfg.prompt_pack or "generic"),
            )
            for sample in samples
            for source_lang, target_lang in [
                _resolve_sample_lang_pair(
                    sample,
                    default_source_lang=self.cfg.source_lang,
                    default_target_lang=self.cfg.target_lang,
                )
            ]
        ]
        if self.predict_fn is not None:
            scores = [float(v) for v in self.predict_fn(message_rows)]
            return RewardOutput(
                sequence_scores=scores,
                metadata={
                    "raw_outputs": [],
                    "error_spans": [[] for _ in samples],
                    "unanchored_errors": [[] for _ in samples],
                    "skipped_rows": [False for _ in samples],
                    "skip_reasons": [None for _ in samples],
                    "failure_rows": [False for _ in samples],
                },
            )

        if self._chat_url is None:
            raise RuntimeError("OpenAICompatibleMQMScorer is not initialized.")

        sequence_scores: list[float] = []
        raw_outputs: list[str] = []
        error_spans: list[list[dict[str, Any]]] = []
        unanchored_errors: list[list[dict[str, Any]]] = []
        skipped_rows: list[bool] = []
        skip_reasons: list[str | None] = []
        failure_rows: list[bool] = []
        max_workers = max(1, int(self.cfg.batch_size))
        if max_workers == 1:
            for sample, messages in zip(samples, message_rows):
                try:
                    score, raw_text, spans, unanchored = self._score_one_sample(sample, messages)
                    sequence_scores.append(score)
                    raw_outputs.append(raw_text)
                    error_spans.append(spans)
                    unanchored_errors.append(unanchored)
                    skipped_rows.append(False)
                    skip_reasons.append(None)
                    failure_rows.append(False)
                except Exception as exc:
                    if str(self.cfg.failure_policy).strip().lower() == "raise":
                        raise
                    fallback_score = _mqm_failure_fallback_score(self.cfg)
                    logger.warning(
                        "MQM scoring failed after repeated failures; using fallback score=%s and empty spans: error=%s",
                        fallback_score,
                        exc,
                    )
                    sequence_scores.append(float(fallback_score))
                    raw_outputs.append("")
                    error_spans.append([])
                    unanchored_errors.append([])
                    skipped_rows.append(False)
                    skip_reasons.append(str(exc))
                    failure_rows.append(True)
        else:
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="mqm-scorer") as executor:
                batch_results = _run_jobs_with_bounded_concurrency(
                    executor=executor,
                    jobs=[(sample, messages) for sample, messages in zip(samples, message_rows)],
                    worker_fn=lambda sample, messages: _capture_exception(self._score_one_sample, sample, messages),
                    max_in_flight=max_workers,
                )
                for result in batch_results:
                    if isinstance(result, Exception):
                        if str(self.cfg.failure_policy).strip().lower() == "raise":
                            raise result
                        fallback_score = _mqm_failure_fallback_score(self.cfg)
                        logger.warning(
                            "MQM scoring failed after repeated failures; using fallback score=%s and empty spans: error=%s",
                            fallback_score,
                            result,
                        )
                        sequence_scores.append(float(fallback_score))
                        raw_outputs.append("")
                        error_spans.append([])
                        unanchored_errors.append([])
                        skipped_rows.append(False)
                        skip_reasons.append(str(result))
                        failure_rows.append(True)
                        continue
                    score, raw_text, spans, unanchored = result
                    sequence_scores.append(score)
                    raw_outputs.append(raw_text)
                    error_spans.append(spans)
                    unanchored_errors.append(unanchored)
                    skipped_rows.append(False)
                    skip_reasons.append(None)
                    failure_rows.append(False)

        return RewardOutput(
            sequence_scores=sequence_scores,
            metadata={
                "raw_outputs": raw_outputs,
                "error_spans": error_spans,
                "unanchored_errors": unanchored_errors,
                "skipped_rows": skipped_rows,
                "skip_reasons": skip_reasons,
                "failure_rows": failure_rows,
            },
        )

    def _score_one_sample(
        self,
        sample: SampleForScoring,
        messages: list[dict[str, str]],
    ) -> tuple[float, str, list[dict[str, Any]], list[dict[str, Any]]]:
        last_exc: Exception | None = None
        for enable_thinking, attempts in _mqm_parse_phase_specs(self.cfg.chat_template_kwargs):
            for _ in range(attempts):
                try:
                    raw_text = self._call_openai_compatible_api(
                        messages,
                        chat_template_kwargs_override=_override_enable_thinking(
                            self.cfg.chat_template_kwargs,
                            enable_thinking=enable_thinking,
                        ),
                    )
                    parsed_text = self._repair_mqm_output_if_needed(
                        sample=sample,
                        raw_text=raw_text,
                        enable_thinking=enable_thinking,
                    )
                    try:
                        raw_score = gemba_mqm_score(parsed_text)
                    except Exception as exc:
                        _record_scorer_parse_failure(
                            log_path=self._parse_failure_log_path,
                            scorer_name="mqm",
                            model_name=self.cfg.model_name,
                            sample=sample,
                            enable_thinking=enable_thinking,
                            stage="repair_output_parse_failed",
                            error=str(exc),
                            details={
                                "raw_text": raw_text,
                                "parsed_text": parsed_text,
                            },
                        )
                        raise
                    if raw_score is None:
                        _record_scorer_parse_failure(
                            log_path=self._parse_failure_log_path,
                            scorer_name="mqm",
                            model_name=self.cfg.model_name,
                            sample=sample,
                            enable_thinking=enable_thinking,
                            stage="repair_output_parse_failed",
                            error="GEMBA-MQM score parse returned None.",
                            details={
                                "raw_text": raw_text,
                                "parsed_text": parsed_text,
                            },
                        )
                        raise GembaParseError("GEMBA-MQM score parse returned None.")
                    try:
                        spans, unanchored = gemba_mqm_extract_error_annotations(parsed_text, sample.mt)
                    except Exception:
                        spans = []
                        unanchored = []
                    scaled = self._scale_score(float(raw_score))
                    return float(scaled), parsed_text, spans, unanchored
                except Exception as exc:
                    last_exc = exc
                    continue

        if last_exc is None:
            raise RuntimeError("MQM API scoring failed without an exception.")
        raise last_exc

    def _repair_mqm_output_if_needed(self, *, sample: SampleForScoring, raw_text: str, enable_thinking: bool) -> str:
        try:
            structured_errors = gemba_mqm_parse_structured_errors(raw_text)
        except Exception as exc:
            structured_errors = None
            error_text = str(exc)
        else:
            return _format_gemba_structured_errors(structured_errors)
        _record_scorer_parse_failure(
            log_path=self._parse_failure_log_path,
            scorer_name="mqm",
            model_name=self.cfg.model_name,
            sample=sample,
            enable_thinking=enable_thinking,
            stage="raw_output_parse_failed",
            error=error_text,
            details={"raw_text": raw_text},
        )
        repaired_text = self._call_openai_compatible_api(
            build_gemba_mqm_repair_messages(
                source_seg=sample.src,
                target_seg=sample.mt,
                raw_output=raw_text,
            ),
            max_tokens=min(1024, max(256, int(self.cfg.max_tokens))),
            chat_template_kwargs_override=_override_enable_thinking(
                self.cfg.chat_template_kwargs,
                enable_thinking=enable_thinking,
            ),
        )
        try:
            structured_errors = gemba_mqm_parse_structured_errors(repaired_text)
        except Exception as exc:
            _record_scorer_parse_failure(
                log_path=self._parse_failure_log_path,
                scorer_name="mqm",
                model_name=self.cfg.model_name,
                sample=sample,
                enable_thinking=enable_thinking,
                stage="repair_output_parse_failed",
                error=str(exc),
                details={
                    "raw_text": raw_text,
                    "parsed_text": repaired_text,
                },
            )
            raise
        return _format_gemba_structured_errors(structured_errors)

    def _call_openai_compatible_api(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        chat_template_kwargs_override: dict[str, Any] | None = None,
    ) -> str:
        if self._chat_url is None:
            raise RuntimeError("MQM scorer chat URL is not set.")

        log_io = _env_flag("GEMMA27_RL_LOG_MQM_IO", default=False)
        log_max_chars = _env_int("GEMMA27_RL_LOG_MQM_IO_MAX_CHARS", default=20000, minimum=256)
        payload = {
            "model": self.cfg.model_name,
            "messages": messages,
            "temperature": float(self.cfg.temperature),
            "top_p": float(self.cfg.top_p),
            "max_tokens": int(self.cfg.max_tokens if max_tokens is None else max_tokens),
        }
        if self.cfg.top_k is not None:
            payload["top_k"] = int(self.cfg.top_k)
        if self.cfg.presence_penalty is not None:
            payload["presence_penalty"] = float(self.cfg.presence_penalty)
        if self.cfg.repetition_penalty is not None:
            payload["repetition_penalty"] = float(self.cfg.repetition_penalty)
        if self.cfg.stop:
            payload["stop"] = list(self.cfg.stop)
        chat_template_kwargs = (
            chat_template_kwargs_override
            if chat_template_kwargs_override is not None
            else self.cfg.chat_template_kwargs
        )
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = dict(chat_template_kwargs)
        if log_io:
            try:
                payload_text = json.dumps(payload, ensure_ascii=False)
            except Exception:
                payload_text = repr(payload)
            logger.info(
                "[mqm-io] request url=%s payload=%s",
                self._chat_url,
                _truncate_for_log(payload_text, log_max_chars),
            )
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        req = urllib_request.Request(
            self._chat_url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        if self._api_key:
            req.add_header("Authorization", f"Bearer {self._api_key}")

        try:
            timeout = float(self.cfg.timeout_s or self.cfg.timeout_sec)
            restore_proxy_env = _temporarily_unset_proxy_env()
            try:
                opener = urllib_request.build_opener(urllib_request.ProxyHandler({}))
                with opener.open(req, timeout=timeout) as resp:
                    resp_body = resp.read().decode("utf-8")
                    if log_io:
                        logger.info(
                            "[mqm-io] response_body=%s",
                            _truncate_for_log(resp_body, log_max_chars),
                        )
            finally:
                restore_proxy_env()
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if log_io:
                logger.error(
                    "[mqm-io] http_error status=%s body=%s",
                    exc.code,
                    _truncate_for_log(detail, log_max_chars),
                )
            raise RuntimeError(f"MQM API HTTPError status={exc.code} body={detail}") from exc
        except urllib_error.URLError as exc:
            if log_io:
                logger.error("[mqm-io] url_error=%s", exc)
            raise RuntimeError(f"MQM API URLError: {exc}") from exc

        try:
            parsed = json.loads(resp_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("MQM API response is not valid JSON.") from exc

        return _extract_openai_response_text(
            parsed=parsed,
            scorer_name="MQM",
            log_io=log_io,
            log_max_chars=log_max_chars,
        )

    def _scale_score(self, score: float) -> float:
        lo = float(self.cfg.score_min)
        hi = float(self.cfg.score_max)
        clipped = min(max(float(score), lo), hi)
        if not self.cfg.scale_to_unit_interval:
            return clipped
        return (clipped - lo) / max(1e-8, hi - lo)


@dataclass
class OpenAICompatibleESAScorer:
    cfg: ESAConfig
    predict_fn: Callable[[list[SampleForScoring]], list[float]] | None = None
    parse_failure_log_path: str | Path | None = None

    def __post_init__(self) -> None:
        self._chat_url: str | None = None
        self._api_key: str | None = None
        self._parse_failure_log_path = Path(self.parse_failure_log_path) if self.parse_failure_log_path else None
        if self.predict_fn is not None or not self.cfg.enabled:
            return

        if not self.cfg.base_url or not str(self.cfg.base_url).strip():
            raise ValueError("ESA scorer requires cfg.base_url when enabled.")
        self._chat_url = self._resolve_chat_url(self.cfg.base_url)

        if self.cfg.api_key and str(self.cfg.api_key).strip():
            self._api_key = str(self.cfg.api_key).strip()
        else:
            env_name = (self.cfg.api_key_env or "OPENAI_API_KEY").strip()
            self._api_key = os.environ.get(env_name) or os.environ.get("OPENAI_API_KEY")
            if self._api_key and self._api_key.strip():
                self._api_key = self._api_key.strip()
            else:
                self._api_key = None

    @staticmethod
    def _resolve_chat_url(base_url: str) -> str:
        url = str(base_url).strip().rstrip("/")
        if not url:
            raise ValueError("ESA base_url must not be empty.")
        if url.endswith("/chat/completions"):
            return url
        if url.endswith("/v1"):
            return f"{url}/chat/completions"
        return f"{url}/v1/chat/completions"

    def score_batch(self, samples: list[SampleForScoring]) -> RewardOutput:
        if not samples:
            return RewardOutput(
                sequence_scores=[],
                metadata={"raw_error_outputs": [], "raw_score_outputs": [], "skipped_rows": [], "skip_reasons": []},
            )

        if self.predict_fn is not None:
            scores = [float(v) for v in self.predict_fn(samples)]
            return RewardOutput(
                sequence_scores=scores,
                metadata={
                    "raw_error_outputs": ["" for _ in samples],
                    "raw_score_outputs": ["" for _ in samples],
                    "skipped_rows": [False for _ in samples],
                    "skip_reasons": [None for _ in samples],
                },
            )

        if self._chat_url is None:
            raise RuntimeError("OpenAICompatibleESAScorer is not initialized.")

        sequence_scores: list[float] = []
        raw_error_outputs: list[str] = []
        raw_score_outputs: list[str] = []
        skipped_rows: list[bool] = []
        skip_reasons: list[str | None] = []
        max_workers = max(1, int(self.cfg.batch_size))
        if max_workers == 1:
            for sample in samples:
                try:
                    score, raw_error_text, raw_score_text = self._score_one_sample(sample)
                    sequence_scores.append(score)
                    raw_error_outputs.append(raw_error_text)
                    raw_score_outputs.append(raw_score_text)
                    skipped_rows.append(False)
                    skip_reasons.append(None)
                except Exception as exc:
                    logger.warning("Skipping ESA-scoring sample after repeated failures: error=%s", exc)
                    sequence_scores.append(0.0)
                    raw_error_outputs.append("")
                    raw_score_outputs.append("")
                    skipped_rows.append(True)
                    skip_reasons.append(str(exc))
        else:
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="esa-scorer") as executor:
                batch_results = _run_jobs_with_bounded_concurrency(
                    executor=executor,
                    jobs=[(sample,) for sample in samples],
                    worker_fn=lambda sample: _capture_exception(self._score_one_sample, sample),
                    max_in_flight=max_workers,
                )
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.warning("Skipping ESA-scoring sample after repeated failures: error=%s", result)
                        sequence_scores.append(0.0)
                        raw_error_outputs.append("")
                        raw_score_outputs.append("")
                        skipped_rows.append(True)
                        skip_reasons.append(str(result))
                        continue
                    score, raw_error_text, raw_score_text = result
                    sequence_scores.append(score)
                    raw_error_outputs.append(raw_error_text)
                    raw_score_outputs.append(raw_score_text)
                    skipped_rows.append(False)
                    skip_reasons.append(None)

        return RewardOutput(
            sequence_scores=sequence_scores,
            metadata={
                "raw_error_outputs": raw_error_outputs,
                "raw_score_outputs": raw_score_outputs,
                "skipped_rows": skipped_rows,
                "skip_reasons": skip_reasons,
            },
        )

    def _score_one_sample(self, sample: SampleForScoring) -> tuple[float, str, str]:
        last_exc: Exception | None = None
        source_lang, target_lang = _resolve_sample_lang_pair(
            sample,
            default_source_lang=self.cfg.source_lang,
            default_target_lang=self.cfg.target_lang,
        )
        for enable_thinking, attempts in _esa_score_phase_specs(self.cfg.chat_template_kwargs):
            for _ in range(attempts):
                try:
                    raw_error_text = self._call_openai_compatible_api(
                        build_gemba_esa_error_messages(
                            source_lang=source_lang,
                            target_lang=target_lang,
                            source_seg=sample.src,
                            target_seg=sample.mt,
                            use_fewshot=bool(self.cfg.use_fewshot),
                            prompt_pack=str(self.cfg.prompt_pack or "generic"),
                        ),
                        max_tokens=int(self.cfg.max_tokens_error_spans),
                        chat_template_kwargs_override=_override_enable_thinking(
                            self.cfg.chat_template_kwargs,
                            enable_thinking=enable_thinking,
                        ),
                    )
                    parsed_error_text = self._repair_esa_output_if_needed(
                        sample=sample,
                        raw_text=raw_error_text,
                        enable_thinking=enable_thinking,
                    )
                    raw_score_text = self._call_openai_compatible_api(
                        build_gemba_esa_ranking_messages(
                            source_lang=source_lang,
                            target_lang=target_lang,
                            source_seg=sample.src,
                            target_seg=sample.mt,
                            error_spans=parsed_error_text,
                        ),
                        max_tokens=int(self.cfg.max_tokens_score),
                        chat_template_kwargs_override=_override_enable_thinking(
                            self.cfg.chat_template_kwargs,
                            enable_thinking=enable_thinking,
                        ),
                    )
                    raw_score = gemba_esa_parse_score(raw_score_text)
                    if raw_score is None:
                        _record_scorer_parse_failure(
                            log_path=self._parse_failure_log_path,
                            scorer_name="esa",
                            model_name=self.cfg.model_name,
                            sample=sample,
                            enable_thinking=enable_thinking,
                            stage="score_parse_failed",
                            error="GEMBA-ESA score parse returned None.",
                            details={
                                "raw_error_text": parsed_error_text,
                                "raw_score_text": raw_score_text,
                            },
                        )
                        raise GembaParseError("GEMBA-ESA score parse returned None.")
                    scaled = self._scale_score(float(raw_score))
                    return float(scaled), parsed_error_text, raw_score_text
                except Exception as exc:
                    last_exc = exc
                    continue

        if last_exc is None:
            raise RuntimeError("ESA API scoring failed without an exception.")
        raise last_exc

    def _repair_esa_output_if_needed(self, *, sample: SampleForScoring, raw_text: str, enable_thinking: bool) -> str:
        try:
            structured_errors = gemba_esa_parse_structured_errors(raw_text)
        except Exception as exc:
            structured_errors = None
            error_text = str(exc)
        else:
            return _format_gemba_structured_errors(structured_errors)
        _record_scorer_parse_failure(
            log_path=self._parse_failure_log_path,
            scorer_name="esa",
            model_name=self.cfg.model_name,
            sample=sample,
            enable_thinking=enable_thinking,
            stage="raw_error_output_parse_failed",
            error=error_text,
            details={"raw_error_text": raw_text},
        )
        repaired_text = self._call_openai_compatible_api(
            build_gemba_esa_repair_messages(
                source_seg=sample.src,
                target_seg=sample.mt,
                raw_output=raw_text,
            ),
            max_tokens=min(1024, max(256, int(self.cfg.max_tokens_error_spans))),
            chat_template_kwargs_override=_override_enable_thinking(
                self.cfg.chat_template_kwargs,
                enable_thinking=enable_thinking,
            ),
        )
        try:
            structured_errors = gemba_esa_parse_structured_errors(repaired_text)
        except Exception as exc:
            _record_scorer_parse_failure(
                log_path=self._parse_failure_log_path,
                scorer_name="esa",
                model_name=self.cfg.model_name,
                sample=sample,
                enable_thinking=enable_thinking,
                stage="repair_output_parse_failed",
                error=str(exc),
                details={
                    "raw_error_text": raw_text,
                    "parsed_error_text": repaired_text,
                },
            )
            raise
        return _format_gemba_structured_errors(structured_errors)

    def _call_openai_compatible_api(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        chat_template_kwargs_override: dict[str, Any] | None = None,
    ) -> str:
        if self._chat_url is None:
            raise RuntimeError("ESA scorer chat URL is not set.")

        log_io = _env_flag("GEMMA27_RL_LOG_ESA_IO", default=False)
        log_max_chars = _env_int("GEMMA27_RL_LOG_ESA_IO_MAX_CHARS", default=20000, minimum=256)
        payload = {
            "model": self.cfg.model_name,
            "messages": messages,
            "temperature": float(self.cfg.temperature),
            "top_p": float(self.cfg.top_p),
            "max_tokens": int(max_tokens),
        }
        if self.cfg.top_k is not None:
            payload["top_k"] = int(self.cfg.top_k)
        if self.cfg.presence_penalty is not None:
            payload["presence_penalty"] = float(self.cfg.presence_penalty)
        if self.cfg.repetition_penalty is not None:
            payload["repetition_penalty"] = float(self.cfg.repetition_penalty)
        if self.cfg.stop:
            payload["stop"] = list(self.cfg.stop)
        chat_template_kwargs = (
            chat_template_kwargs_override
            if chat_template_kwargs_override is not None
            else self.cfg.chat_template_kwargs
        )
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = dict(chat_template_kwargs)
        if log_io:
            try:
                payload_text = json.dumps(payload, ensure_ascii=False)
            except Exception:
                payload_text = repr(payload)
            logger.info(
                "[esa-io] request url=%s payload=%s",
                self._chat_url,
                _truncate_for_log(payload_text, log_max_chars),
            )
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        req = urllib_request.Request(
            self._chat_url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        if self._api_key:
            req.add_header("Authorization", f"Bearer {self._api_key}")

        try:
            timeout = float(self.cfg.timeout_s or self.cfg.timeout_sec)
            restore_proxy_env = _temporarily_unset_proxy_env()
            try:
                opener = urllib_request.build_opener(urllib_request.ProxyHandler({}))
                with opener.open(req, timeout=timeout) as resp:
                    resp_body = resp.read().decode("utf-8")
                    if log_io:
                        logger.info(
                            "[esa-io] response_body=%s",
                            _truncate_for_log(resp_body, log_max_chars),
                        )
            finally:
                restore_proxy_env()
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if log_io:
                logger.error(
                    "[esa-io] http_error status=%s body=%s",
                    exc.code,
                    _truncate_for_log(detail, log_max_chars),
                )
            raise RuntimeError(f"ESA API HTTPError status={exc.code} body={detail}") from exc
        except urllib_error.URLError as exc:
            if log_io:
                logger.error("[esa-io] url_error=%s", exc)
            raise RuntimeError(f"ESA API URLError: {exc}") from exc

        try:
            parsed = json.loads(resp_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("ESA API response is not valid JSON.") from exc

        return _extract_openai_response_text(
            parsed=parsed,
            scorer_name="ESA",
            log_io=log_io,
            log_max_chars=log_max_chars,
        )

    def _scale_score(self, score: float) -> float:
        lo = float(self.cfg.score_min)
        hi = float(self.cfg.score_max)
        clipped = min(max(float(score), lo), hi)
        if not self.cfg.scale_to_unit_interval:
            return clipped
        return (clipped - lo) / max(1e-8, hi - lo)


def extract_error_spans(
    metadata: Any,
    expected: int,
    *,
    source: str = "xCOMET metadata",
) -> list[list[dict[str, Any]]]:
    if metadata is None:
        return [[] for _ in range(expected)]

    def _normalize_one(item: Any) -> list[dict[str, Any]]:
        if item is None:
            return []
        if isinstance(item, dict):
            spans = item.get("error_spans")
            if isinstance(spans, list):
                return [span for span in spans if isinstance(span, dict)]
            return []
        if isinstance(item, list):
            return [span for span in item if isinstance(span, dict)]
        return []

    def _length_mismatch(actual: int) -> ValueError:
        return ValueError(f"{source} returned mismatched error_spans length: expected={expected} got={actual}")

    # Common format: metadata is list of per-sample dicts.
    if isinstance(metadata, list):
        if len(metadata) != expected:
            raise _length_mismatch(len(metadata))
        out = [_normalize_one(item) for item in metadata]
        return out

    # Dict with direct `error_spans` entry.
    if isinstance(metadata, dict):
        if "error_spans" in metadata:
            spans = metadata.get("error_spans")
            if not isinstance(spans, list):
                raise ValueError(
                    f"{source} returned non-list error_spans (type={type(spans).__name__})."
                )
            if spans and isinstance(spans[0], list):
                if len(spans) != expected:
                    raise _length_mismatch(len(spans))
                out = [_normalize_one(item) for item in spans]
            elif spans and isinstance(spans[0], dict):
                if expected != 1:
                    raise _length_mismatch(1)
                out = [_normalize_one(spans)]
            else:
                if expected != 1:
                    raise _length_mismatch(1)
                out = [[]]
            return out

        if "samples" in metadata:
            samples = metadata["samples"]
            if not isinstance(samples, list):
                raise ValueError(
                    f"{source} returned non-list samples metadata (type={type(samples).__name__})."
                )
            if len(samples) != expected:
                raise _length_mismatch(len(samples))
            out = [_normalize_one(item) for item in samples]
            return out

    return [[] for _ in range(expected)]


def _resolve_mqm_token_type_weight(error_type: str | None, weights: dict[str, float]) -> float:
    normalized = _normalize_gemba_error_type(error_type)
    if not normalized or not weights:
        return 1.0
    if normalized in weights:
        return float(weights[normalized])
    candidate = normalized
    while "/" in candidate:
        candidate = candidate.rsplit("/", 1)[0].strip()
        if candidate in weights:
            return float(weights[candidate])
    return 1.0


def _mqm_error_type_matches_allowed(error_type: str | None, allowed_types: list[str]) -> bool:
    normalized = _normalize_gemba_error_type(error_type)
    if not normalized:
        return False
    for allowed in allowed_types:
        allowed_normalized = _normalize_gemba_error_type(allowed)
        if not allowed_normalized:
            continue
        if normalized == allowed_normalized or normalized.startswith(f"{allowed_normalized}/"):
            return True
    return False


def _compute_mqm_unanchored_seq_penalty(
    *,
    unanchored_errors: list[dict[str, Any]],
    severity_weights: dict[str, float],
    type_weights: dict[str, float],
    allowed_types: list[str],
    scale: float,
) -> float:
    scale_f = float(scale)
    if scale_f == 0.0:
        return 0.0
    penalty_total = 0.0
    for error in unanchored_errors:
        if not isinstance(error, dict):
            continue
        error_type = str(error.get("error_type") or "").strip() or None
        if not _mqm_error_type_matches_allowed(error_type, allowed_types):
            continue
        severity = str(error.get("severity", "")).strip().upper()
        severity_weight = float(severity_weights.get(severity, 0.0))
        type_weight = _resolve_mqm_token_type_weight(error_type, type_weights)
        penalty_total += severity_weight * type_weight * scale_f
    return float(penalty_total)


def spans_to_token_rewards(
    mt_text: str,
    token_char_offsets: list[tuple[int, int]],
    error_spans: list[dict[str, Any]],
    severity_weights: dict[str, float],
    mqm_token_type_weights: dict[str, float] | None = None,
    overlap_policy: str = "any_overlap",
    majority_threshold: float = 0.5,
    use_confidence: bool = False,
    combine_policy: str = "sum",
) -> list[float]:
    del mt_text  # Reserved for debug extension.

    if overlap_policy not in {"any_overlap", "majority_overlap"}:
        raise ValueError("overlap_policy must be any_overlap or majority_overlap")
    if combine_policy not in {"sum", "min", "max"}:
        raise ValueError("combine_policy must be sum|min|max")

    normalized_mqm_type_weights: dict[str, float] = {}
    if mqm_token_type_weights:
        for raw_key, raw_value in mqm_token_type_weights.items():
            key = _normalize_gemba_error_type(raw_key)
            if key:
                normalized_mqm_type_weights[key] = float(raw_value)

    rewards = [0.0 for _ in token_char_offsets]
    initialized = [False for _ in token_char_offsets]

    for span in error_spans:
        try:
            start = int(span.get("start", 0))
            end = int(span.get("end", 0))
        except Exception:
            continue
        if end <= start:
            continue

        severity = str(span.get("severity", "")).strip().upper()
        penalty = float(severity_weights.get(severity, 0.0))
        if str(span.get("source", "")).strip().lower() == "mqm":
            error_type = str(span.get("error_type") or span.get("type") or "").strip() or None
            penalty *= _resolve_mqm_token_type_weight(error_type, normalized_mqm_type_weights)
        if use_confidence and "confidence" in span:
            try:
                penalty *= float(span.get("confidence", 1.0))
            except Exception:
                pass
        if penalty == 0.0:
            continue

        for token_idx, (tok_s, tok_e) in enumerate(token_char_offsets):
            if tok_e <= tok_s:
                continue
            overlap = max(0, min(tok_e, end) - max(tok_s, start))
            if overlap <= 0:
                continue

            apply = False
            if overlap_policy == "any_overlap":
                apply = True
            else:
                ratio = overlap / max(1, tok_e - tok_s)
                apply = ratio >= majority_threshold
            if not apply:
                continue

            if combine_policy == "sum":
                rewards[token_idx] += penalty
            elif combine_policy == "min":
                rewards[token_idx] = penalty if not initialized[token_idx] else min(rewards[token_idx], penalty)
            elif combine_policy == "max":
                if not initialized[token_idx]:
                    rewards[token_idx] = penalty
                else:
                    current = rewards[token_idx]
                    current_abs = abs(float(current))
                    penalty_abs = abs(float(penalty))
                    if penalty_abs > current_abs:
                        rewards[token_idx] = penalty
                    elif penalty_abs == current_abs:
                        # Tie-break toward stronger (more negative) penalty for
                        # common negative-penalty configurations.
                        rewards[token_idx] = min(current, penalty)
            initialized[token_idx] = True

    return rewards
