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

from .config import ESAConfig, MQMConfig, MetricXConfig, XCometConfig
from .rl_types import RewardOutput, SampleForScoring
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
    "You normalize machine translation MQM annotations into a strict canonical format."
)

GEMBA_MQM_REPAIR_PROMPT_TEMPLATE = (
    "Rewrite the evaluator output below into the exact canonical MQM format.\n\n"
    "Return only:\n"
    "Critical:\n"
    "<error lines or no-error>\n"
    "Major:\n"
    "<error lines or no-error>\n"
    "Minor:\n"
    "<error lines or no-error>\n\n"
    "Rules:\n"
    '- Each error line must look like category/subcategory - "exact target span"\n'
    "- Copy quoted target spans exactly from the translation when possible.\n"
    "- If a severity has no errors, output no-error under that severity.\n"
    "- If there are no errors at all, output no-error under all three severities.\n"
    "- Do not include explanations or any text outside the canonical format.\n\n"
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
    Critical:
    no-error
    Major:
    accuracy/mistranslation - "involvement"
    accuracy/omission - "the account holder"
    Minor:
    fluency/grammar - "wäre"
    fluency/register - "dir"
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
    Critical:
    no-error
    Major:
    accuracy/addition - "ve Vídni"
    accuracy/omission - "the stop-start"
    Minor:
    terminology/inappropriate for context - "partaje"
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
    Critical:
    accuracy/addition - "of high-speed rail"
    Major:
    accuracy/mistranslation - "go to the reviews"
    Minor:
    style/awkward - "etc.,"
    """
).strip()


_GEMBA_ERROR_LINE_PATTERN = re.compile(
    r"^(accuracy|fluency|style|terminology|non-translation|other)"
    r"(?:\s*/\s*[^:]+?)?\s*(?:-|:|–|—)\s*(.+)$",
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
    match = _GEMBA_ERROR_LINE_PATTERN.match(str(line).strip())
    if match is None:
        return False
    detail = str(match.group(2)).strip()
    if not detail:
        return False
    return True


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


def gemba_mqm_parse_errors(model_output: str) -> dict[str, list[str]]:
    return _parse_gemba_error_output(
        model_output,
        allowed_levels=("critical", "major", "minor"),
        scorer_name="MQM",
    )


def gemba_mqm_score(model_output: str | None) -> int | None:
    if model_output is None:
        return None
    errors = gemba_mqm_parse_errors(model_output)

    penalty = 0
    count = 0
    for lvl in ["critical", "major", "minor"]:
        for _err in errors.get(lvl, []):
            if count >= 5:
                break
            penalty += 25 if lvl == "critical" else 5 if lvl == "major" else 1
            count += 1
    if penalty > 25:
        penalty = 25
    return -penalty


def _extract_mqm_quoted_text(line: str) -> str | None:
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


def _extract_mqm_error_detail(line: str) -> str | None:
    match = _GEMBA_ERROR_LINE_PATTERN.match(str(line).strip())
    if match is None:
        return None
    detail = str(match.group(2)).strip()
    return detail or None


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


def gemba_mqm_extract_error_spans(model_output: str | None, mt_text: str) -> list[dict[str, Any]]:
    if model_output is None or not mt_text:
        return []

    parsed = gemba_mqm_parse_errors(model_output)
    out: list[dict[str, Any]] = []
    used_spans: list[tuple[int, int]] = []
    max_items = 5

    for severity in ("critical", "major", "minor"):
        for line in parsed.get(severity, []):
            if len(out) >= max_items:
                return out
            span = None
            for candidate in _mqm_error_text_candidates(line):
                span = _find_text_span(mt_text, candidate, used_spans)
                if span is not None:
                    break
            if span is None:
                continue
            start, end = span
            used_spans.append(span)
            out.append(
                {
                    "text": mt_text[start:end],
                    "start": int(start),
                    "end": int(end),
                    "severity": severity.upper(),
                    "confidence": 1.0,
                    "source": "mqm",
                    "label": line,
                }
            )

    return out


def _gemba_eval_user_message(
    *,
    source_lang: str,
    target_lang: str,
    source_seg: str,
    target_seg: str,
) -> str:
    return (
        f"{source_lang} source:\n"
        f"```{source_seg}```\n"
        f"{target_lang} translation:\n"
        f"```{target_seg}```\n\n"
        f"{GEMBA_USER_TASK_PROMPT}"
    )


def build_gemba_mqm_messages(
    *,
    source_lang: str,
    target_lang: str,
    source_seg: str,
    target_seg: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": GEMBA_SYSTEM_PROMPT},
        {"role": "user", "content": GEMBA_FEWSHOT_USER_1},
        {"role": "assistant", "content": GEMBA_FEWSHOT_ASSISTANT_1},
        {"role": "user", "content": GEMBA_FEWSHOT_USER_2},
        {"role": "assistant", "content": GEMBA_FEWSHOT_ASSISTANT_2},
        {"role": "user", "content": GEMBA_FEWSHOT_USER_3},
        {"role": "assistant", "content": GEMBA_FEWSHOT_ASSISTANT_3},
        {
            "role": "user",
            "content": _gemba_eval_user_message(
                source_lang=source_lang,
                target_lang=target_lang,
                source_seg=source_seg,
                target_seg=target_seg,
            ),
        },
    ]


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
    "You normalize machine translation ESA annotations into a strict canonical format."
)

GEMBA_ESA_REPAIR_PROMPT_TEMPLATE = (
    "Rewrite the evaluator output below into the exact canonical ESA error format.\n\n"
    "Return only:\n"
    "Major:\n"
    "<error lines or no-error>\n"
    "Minor:\n"
    "<error lines or no-error>\n\n"
    "Rules:\n"
    '- Each error line must look like category/subcategory - "exact target span"\n'
    "- Copy quoted target spans exactly from the translation when possible.\n"
    "- If a severity has no errors, output no-error under that severity.\n"
    "- If there are no errors at all, output no-error under both severities.\n"
    "- Do not include explanations or any text outside the canonical format.\n\n"
    "Source:\n"
    "```{source_seg}```\n"
    "Translation:\n"
    "```{target_seg}```\n"
    "Evaluator output:\n"
    "```{raw_output}```"
)

GEMBA_ESA_FEWSHOT_ASSISTANT_1 = dedent(
    """\
    Major:
    accuracy/mistranslation - "involvement"
    accuracy/omission - "the account holder"
    Minor:
    fluency/grammar - "wäre"
    fluency/register - "dir"
    """
).strip()

GEMBA_ESA_FEWSHOT_ASSISTANT_2 = dedent(
    """\
    Major:
    accuracy/addition - "ve Vídni"
    accuracy/omission - "the stop-start"
    Minor:
    terminology/inappropriate for context - "partaje"
    """
).strip()

GEMBA_ESA_FEWSHOT_ASSISTANT_3 = dedent(
    """\
    Major:
    accuracy/addition - "of high-speed rail"
    accuracy/mistranslation - "go to the reviews"
    Minor:
    style/awkward - "etc.,"
    """
).strip()

GEMBA_ESA_RANKING_PROMPT_TEMPLATE = (
    "Given the translation from {source_lang} to {target_lang} and the annotated error spans, "
    "assign a score on a continuous scale from 0 to 100. The scale has following reference points: "
    '0="No meaning preserved", 33="Some meaning preserved", '
    '66="Most meaning preserved and few grammar mistakes", '
    'up to 100="Perfect meaning and grammar".\n\n'
    "Score the following translation from {source_lang} source:\n"
    "```{source_seg}```\n"
    "{target_lang} translation:\n"
    "```{target_seg}```\n"
    "Annotated error spans:\n"
    "```{error_spans}```\n"
    "Respond with only one integer from 0 to 100. Do not include any explanation or extra text.\n"
    "Score (0-100):"
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
) -> str:
    return (
        f"{source_lang} source:\n"
        f"```{source_seg}```\n"
        f"{target_lang} translation:\n"
        f"```{target_seg}```\n\n"
        f"{GEMBA_ESA_USER_TASK_PROMPT}"
    )


def build_gemba_esa_error_messages(
    *,
    source_lang: str,
    target_lang: str,
    source_seg: str,
    target_seg: str,
    use_fewshot: bool = True,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": GEMBA_ESA_SYSTEM_PROMPT}]
    if use_fewshot:
        messages.extend(
            [
                {"role": "user", "content": _gemba_esa_error_user_message(
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
                )},
                {"role": "assistant", "content": GEMBA_ESA_FEWSHOT_ASSISTANT_1},
                {"role": "user", "content": _gemba_esa_error_user_message(
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
                )},
                {"role": "assistant", "content": GEMBA_ESA_FEWSHOT_ASSISTANT_2},
                {"role": "user", "content": _gemba_esa_error_user_message(
                    source_lang="Chinese",
                    target_lang="English",
                    source_seg=(
                        "大众点评乌鲁木齐家居卖场频道为您提供高铁居然之家地址，电话，营业时间等最新商户信息，找装修公司，就上大众点评"
                    ),
                    target_seg=(
                        "Urumqi Home Furnishing Store Channel provides you with the latest business information "
                        "such as the address, telephone number, business hours, etc., of high-speed rail, and "
                        "find a decoration company, and go to the reviews."
                    ),
                )},
                {"role": "assistant", "content": GEMBA_ESA_FEWSHOT_ASSISTANT_3},
            ]
        )
    messages.append(
        {
            "role": "user",
            "content": _gemba_esa_error_user_message(
                source_lang=source_lang,
                target_lang=target_lang,
                source_seg=source_seg,
                target_seg=target_seg,
            ),
        }
    )
    return messages


def gemba_esa_parse_errors(model_output: str) -> dict[str, list[str]]:
    return _parse_gemba_error_output(
        model_output,
        allowed_levels=("major", "minor"),
        scorer_name="ESA",
    )


def gemba_esa_format_error_spans(model_output: str | None) -> str:
    if model_output is None:
        return "no-error"
    parsed = gemba_esa_parse_errors(model_output)
    lines: list[str] = []
    if parsed["major"]:
        lines.append("Major:")
        lines.extend(parsed["major"])
    if parsed["minor"]:
        lines.append("Minor:")
        lines.extend(parsed["minor"])
    if not lines:
        return "no-error"
    return "\n".join(lines)


def gemba_esa_parse_score(model_output: str | None) -> float | None:
    if model_output is None:
        return None
    text = str(model_output).strip()
    if not text:
        return None

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
                metadata={"raw_outputs": [], "error_spans": [], "skipped_rows": [], "skip_reasons": []},
            )

        message_rows = [
            build_gemba_mqm_messages(
                source_lang=source_lang,
                target_lang=target_lang,
                source_seg=sample.src,
                target_seg=sample.mt,
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
                    "skipped_rows": [False for _ in samples],
                    "skip_reasons": [None for _ in samples],
                },
            )

        if self._chat_url is None:
            raise RuntimeError("OpenAICompatibleMQMScorer is not initialized.")

        sequence_scores: list[float] = []
        raw_outputs: list[str] = []
        error_spans: list[list[dict[str, Any]]] = []
        skipped_rows: list[bool] = []
        skip_reasons: list[str | None] = []
        max_workers = max(1, int(self.cfg.batch_size))
        if max_workers == 1:
            for sample, messages in zip(samples, message_rows):
                try:
                    score, raw_text, spans = self._score_one_sample(sample, messages)
                    sequence_scores.append(score)
                    raw_outputs.append(raw_text)
                    error_spans.append(spans)
                    skipped_rows.append(False)
                    skip_reasons.append(None)
                except Exception as exc:
                    logger.warning(
                        "MQM scoring failed after repeated failures; using fallback score=0.0 and empty spans: error=%s",
                        exc,
                    )
                    sequence_scores.append(0.0)
                    raw_outputs.append("")
                    error_spans.append([])
                    skipped_rows.append(False)
                    skip_reasons.append(None)
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
                        logger.warning(
                            "MQM scoring failed after repeated failures; using fallback score=0.0 and empty spans: error=%s",
                            result,
                        )
                        sequence_scores.append(0.0)
                        raw_outputs.append("")
                        error_spans.append([])
                        skipped_rows.append(False)
                        skip_reasons.append(None)
                        continue
                    score, raw_text, spans = result
                    sequence_scores.append(score)
                    raw_outputs.append(raw_text)
                    error_spans.append(spans)
                    skipped_rows.append(False)
                    skip_reasons.append(None)

        return RewardOutput(
            sequence_scores=sequence_scores,
            metadata={
                "raw_outputs": raw_outputs,
                "error_spans": error_spans,
                "skipped_rows": skipped_rows,
                "skip_reasons": skip_reasons,
            },
        )

    def _score_one_sample(
        self,
        sample: SampleForScoring,
        messages: list[dict[str, str]],
    ) -> tuple[float, str, list[dict[str, Any]]]:
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
                        spans = gemba_mqm_extract_error_spans(parsed_text, sample.mt)
                    except Exception:
                        spans = []
                    scaled = self._scale_score(float(raw_score))
                    return float(scaled), parsed_text, spans
                except Exception as exc:
                    last_exc = exc
                    continue

        if last_exc is None:
            raise RuntimeError("MQM API scoring failed without an exception.")
        raise last_exc

    def _repair_mqm_output_if_needed(self, *, sample: SampleForScoring, raw_text: str, enable_thinking: bool) -> str:
        try:
            raw_score = gemba_mqm_score(raw_text)
        except Exception as exc:
            raw_score = None
            error_text = str(exc)
        else:
            error_text = "GEMBA-MQM score parse returned None."
        if raw_score is not None:
            return raw_text
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
        return self._call_openai_compatible_api(
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
                        ),
                        max_tokens=int(self.cfg.max_tokens_error_spans),
                        chat_template_kwargs_override=_override_enable_thinking(
                            self.cfg.chat_template_kwargs,
                            enable_thinking=enable_thinking,
                        ),
                    )
                    raw_score_text = self._call_openai_compatible_api(
                        build_gemba_esa_ranking_messages(
                            source_lang=source_lang,
                            target_lang=target_lang,
                            source_seg=sample.src,
                            target_seg=sample.mt,
                            error_spans=raw_error_text,
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
                                "raw_error_text": raw_error_text,
                                "raw_score_text": raw_score_text,
                            },
                        )
                        raise GembaParseError("GEMBA-ESA score parse returned None.")
                    scaled = self._scale_score(float(raw_score))
                    return float(scaled), raw_error_text, raw_score_text
                except Exception as exc:
                    last_exc = exc
                    continue

        if last_exc is None:
            raise RuntimeError("ESA API scoring failed without an exception.")
        raise last_exc

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


def spans_to_token_rewards(
    mt_text: str,
    token_char_offsets: list[tuple[int, int]],
    error_spans: list[dict[str, Any]],
    severity_weights: dict[str, float],
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
