from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
import os
from pathlib import Path
import re
import select
import subprocess
from textwrap import dedent
from typing import Any, Callable
from urllib import error as urllib_error
from urllib import request as urllib_request

try:
    import torch
except Exception:  # pragma: no cover - optional during lightweight tests
    torch = None  # type: ignore[assignment]

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover - optional during lightweight tests
    AutoTokenizer = None  # type: ignore[assignment]

from .config import MQMConfig, MetricXConfig, XCometConfig
from .types import RewardOutput, SampleForScoring
from .utils import resolve_device, resolve_torch_dtype


logger = logging.getLogger(__name__)


class _ScorerSubprocessClient:
    def __init__(
        self,
        *,
        backend: str,
        python_executable: str,
        timeout_sec: float,
        config_payload: dict[str, Any],
    ) -> None:
        self._backend = backend
        self._timeout_sec = float(timeout_sec)
        worker_script = Path(__file__).resolve().with_name("scorer_worker.py")
        if not worker_script.exists():
            raise FileNotFoundError(f"scorer worker script not found: {worker_script}")

        cmd = [python_executable, str(worker_script), "--backend", backend]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to start {backend} scorer worker: cmd={cmd}") from exc

        try:
            init_resp = self.request({"type": "init", "config": config_payload})
        except Exception:
            self.close()
            raise
        if not bool(init_resp.get("ok", False)):
            self.close()
            raise RuntimeError(
                f"{backend} scorer worker init failed: {init_resp.get('error', 'unknown error')}"
            )

    def _assert_alive(self) -> None:
        if self._proc.poll() is not None:
            raise RuntimeError(f"{self._backend} scorer worker exited unexpectedly with code={self._proc.returncode}")

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._assert_alive()
        assert self._proc.stdin is not None
        assert self._proc.stdout is not None

        try:
            self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
        except Exception as exc:
            raise RuntimeError(f"Failed to send request to {self._backend} scorer worker.") from exc

        ready, _, _ = select.select([self._proc.stdout], [], [], self._timeout_sec)
        if not ready:
            raise TimeoutError(f"{self._backend} scorer worker timed out after {self._timeout_sec}s")
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


def gemba_mqm_parse_errors(model_output: str) -> dict[str, list[str]]:
    errors: dict[str, list[str]] = {"critical": [], "major": [], "minor": []}
    level: str | None = None

    for raw_line in str(model_output).splitlines():
        line = raw_line.strip()
        line_l = line.lower()
        if (not line) or ("no-error" in line_l) or ("no error" in line_l):
            continue
        if line_l == "critical:":
            level = "critical"
            continue
        if line_l == "major:":
            level = "major"
            continue
        if line_l == "minor:":
            level = "minor"
            continue
        if level is None:
            continue
        if "non-translation" in line_l:
            errors["critical"].append(line)
        else:
            errors[level].append(line)

    return errors


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


_MQM_QUOTED_TEXT_PATTERNS: tuple[str, ...] = (
    r'"([^"\n]+)"',
    r"“([^”\n]+)”",
    r"'([^'\n]+)'",
    r"`([^`\n]+)`",
)


def _extract_mqm_quoted_text(line: str) -> str | None:
    matches: list[str] = []
    for pattern in _MQM_QUOTED_TEXT_PATTERNS:
        for found in re.findall(pattern, line):
            value = str(found).strip()
            if not value:
                continue
            if value.lower() in {"no-error", "no error"}:
                continue
            matches.append(value)
    if not matches:
        return None
    return max(matches, key=len)


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
            quoted = _extract_mqm_quoted_text(line)
            if quoted is None:
                continue
            span = _find_text_span(mt_text, quoted, used_spans)
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
            cfg_payload = {
                "model_name": self.cfg.model_name,
                "tokenizer_name": self.cfg.tokenizer_name,
                "use_reference": bool(self.cfg.use_reference),
                "batch_size": int(self.cfg.batch_size),
                "device": self.cfg.device,
                "dtype": self.cfg.dtype,
                "max_input_length": int(self.cfg.max_input_length),
                "overflow_policy": self.cfg.overflow_policy,
            }
            self._worker = _ScorerSubprocessClient(
                backend="metricx",
                python_executable=self.cfg.python_executable,
                timeout_sec=float(self.cfg.subprocess_timeout_sec),
                config_payload=cfg_payload,
            )
            logger.info(
                "MetricX scorer will run in external python=%s (device=%s).",
                self.cfg.python_executable,
                self.cfg.device,
            )
            return

        if torch is None or AutoTokenizer is None:
            raise RuntimeError(
                "MetricX model loading requires torch and transformers. "
                "Use predict_fn for lightweight testing."
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
                raise RuntimeError(f"MetricX worker scoring failed: {resp.get('error', 'unknown error')}")
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
            cfg_payload = {
                "model_name": self.cfg.model_name,
                "batch_size": int(self.cfg.batch_size),
                "device": self.cfg.device,
                "use_reference": bool(self.cfg.use_reference),
            }
            self._worker = _ScorerSubprocessClient(
                backend="xcomet",
                python_executable=self.cfg.python_executable,
                timeout_sec=float(self.cfg.subprocess_timeout_sec),
                config_payload=cfg_payload,
            )
            logger.info(
                "xCOMET scorer will run in external python=%s (device=%s).",
                self.cfg.python_executable,
                self.cfg.device,
            )
            return

        try:
            from comet import download_model, load_from_checkpoint
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "XCometXLScorer requires unbabel-comet>=2.2.0. Install it before running RL."
            ) from exc

        model_path = download_model(self.cfg.model_name)
        self._model = load_from_checkpoint(model_path)
        if self._device.startswith("cuda") and torch is not None:
            self._model.to(torch.device(self._device))
        self._model.eval()

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
                raise RuntimeError(f"xCOMET worker scoring failed: {resp.get('error', 'unknown error')}")
            scores = [float(v) for v in list(resp.get("scores", []))]
            spans = resp.get("error_spans", [[] for _ in scores])
            if len(scores) != len(samples):
                raise RuntimeError(
                    f"xCOMET worker returned mismatched score length: expected={len(samples)} got={len(scores)}"
                )
            if not isinstance(spans, list) or len(spans) != len(samples):
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
            raise RuntimeError("XCometXLScorer requires torch for model inference.")

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
            batch_spans = extract_error_spans(metadata, expected=batch_size)

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

        spans = extract_error_spans(metadata, expected=expected)
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

    def __post_init__(self) -> None:
        self._chat_url: str | None = None
        self._api_key: str | None = None
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
            return RewardOutput(sequence_scores=[], metadata={"raw_outputs": [], "error_spans": []})

        message_rows = [
            build_gemba_mqm_messages(
                source_lang=self.cfg.source_lang,
                target_lang=self.cfg.target_lang,
                source_seg=sample.src,
                target_seg=sample.mt,
            )
            for sample in samples
        ]
        if self.predict_fn is not None:
            scores = [float(v) for v in self.predict_fn(message_rows)]
            return RewardOutput(
                sequence_scores=scores,
                metadata={"raw_outputs": [], "error_spans": [[] for _ in samples]},
            )

        if self._chat_url is None:
            raise RuntimeError("OpenAICompatibleMQMScorer is not initialized.")

        sequence_scores: list[float] = []
        raw_outputs: list[str] = []
        error_spans: list[list[dict[str, Any]]] = []
        for i in range(0, len(message_rows), max(1, int(self.cfg.batch_size))):
            batch_messages = message_rows[i : i + max(1, int(self.cfg.batch_size))]
            batch_samples = samples[i : i + max(1, int(self.cfg.batch_size))]
            for sample, messages in zip(batch_samples, batch_messages):
                score, raw_text = self._score_one_messages(messages)
                sequence_scores.append(score)
                raw_outputs.append(raw_text)
                error_spans.append(gemba_mqm_extract_error_spans(raw_text, sample.mt))

        return RewardOutput(
            sequence_scores=sequence_scores,
            metadata={"raw_outputs": raw_outputs, "error_spans": error_spans},
        )

    def _score_one_messages(self, messages: list[dict[str, str]]) -> tuple[float, str]:
        retries = max(0, int(self.cfg.max_retries))
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                raw_text = self._call_openai_compatible_api(messages)
                raw_score = gemba_mqm_score(raw_text)
                if raw_score is None:
                    raise RuntimeError("GEMBA-MQM score parse returned None.")
                scaled = self._scale_score(float(raw_score))
                return float(scaled), raw_text
            except Exception as exc:
                last_exc = exc
                if attempt < retries:
                    continue
                break

        try:
            raise RuntimeError("MQM API scoring failed.") from last_exc
        except Exception as exc:
            if self.cfg.error_policy == "zero":
                logger.warning("MQM API scoring failed; fallback score=0.0 due to error_policy=zero: %s", exc)
                return 0.0, ""
            raise

    def _call_openai_compatible_api(self, messages: list[dict[str, str]]) -> str:
        if self._chat_url is None:
            raise RuntimeError("MQM scorer chat URL is not set.")

        payload = {
            "model": self.cfg.model_name,
            "messages": messages,
            "temperature": float(self.cfg.temperature),
            "top_p": float(self.cfg.top_p),
            "max_tokens": int(self.cfg.max_tokens),
        }
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
            with urllib_request.urlopen(req, timeout=timeout) as resp:
                resp_body = resp.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"MQM API HTTPError status={exc.code} body={detail}") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"MQM API URLError: {exc}") from exc

        try:
            parsed = json.loads(resp_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("MQM API response is not valid JSON.") from exc

        choices = parsed.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("MQM API response has no choices.")
        first = choices[0]
        if isinstance(first, dict):
            msg = first.get("message")
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and isinstance(item.get("text"), str):
                            text_parts.append(item["text"])
                    if text_parts:
                        return "\n".join(text_parts)
            text = first.get("text")
            if isinstance(text, str):
                return text

        raise RuntimeError("MQM API response format is unsupported.")

    def _scale_score(self, score: float) -> float:
        lo = float(self.cfg.score_min)
        hi = float(self.cfg.score_max)
        clipped = min(max(float(score), lo), hi)
        if not self.cfg.scale_to_unit_interval:
            return clipped
        return (clipped - lo) / max(1e-8, hi - lo)


def extract_error_spans(metadata: Any, expected: int) -> list[list[dict[str, Any]]]:
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

    # Common format: metadata is list of per-sample dicts.
    if isinstance(metadata, list):
        out = [_normalize_one(item) for item in metadata]
        if len(out) < expected:
            out.extend([[] for _ in range(expected - len(out))])
        return out[:expected]

    # Dict with direct `error_spans` entry.
    if isinstance(metadata, dict):
        spans = metadata.get("error_spans")
        if isinstance(spans, list):
            if spans and isinstance(spans[0], list):
                out = [_normalize_one(item) for item in spans]
            elif spans and isinstance(spans[0], dict):
                out = [_normalize_one(spans)] if expected == 1 else [[*spans]] + [[] for _ in range(expected - 1)]
            else:
                out = [[] for _ in range(expected)]
            if len(out) < expected:
                out.extend([[] for _ in range(expected - len(out))])
            return out[:expected]

        if "samples" in metadata and isinstance(metadata["samples"], list):
            out = [_normalize_one(item) for item in metadata["samples"]]
            if len(out) < expected:
                out.extend([[] for _ in range(expected - len(out))])
            return out[:expected]

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
                rewards[token_idx] = penalty if not initialized[token_idx] else max(rewards[token_idx], penalty)
            initialized[token_idx] = True

    return rewards
