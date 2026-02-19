from __future__ import annotations

import logging
import os
import random
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable

import httpx
from openai import APIStatusError, OpenAI

from .caches import SQLiteKVCache
from .config import TeacherConfig
from .stats import StatsCollector
from .utils import stable_hash


@dataclass
class CompletionTask:
    item_id: str
    messages: list[dict[str, str]]
    temperature: float
    top_p: float
    max_tokens: int
    model: str | None = None
    presence_penalty: float | None = None
    extra_body: dict[str, Any] | None = None


class TeacherClient:
    def __init__(
        self,
        cfg: TeacherConfig,
        cache_db_path: str,
        stats: StatsCollector,
        logger: logging.Logger | None = None,
    ):
        self.cfg = cfg
        self.stats = stats
        self.logger = logger or logging.getLogger(__name__)
        api_key = os.getenv(cfg.api_key_env, "token-placeholder")
        self._http_client: httpx.Client | None = None
        if cfg.unset_proxy_env:
            # Ensure API calls bypass HTTP(S)_PROXY env vars.
            self._http_client = httpx.Client(timeout=cfg.request_timeout_s, trust_env=False)
            self.client = OpenAI(
                base_url=cfg.base_url,
                api_key=api_key,
                timeout=cfg.request_timeout_s,
                max_retries=cfg.sdk_max_retries,
                http_client=self._http_client,
            )
            self.logger.info(
                "Teacher client created with trust_env=False (proxy env disabled), sdk_max_retries=%s.",
                cfg.sdk_max_retries,
            )
        else:
            self.client = OpenAI(
                base_url=cfg.base_url,
                api_key=api_key,
                timeout=cfg.request_timeout_s,
                max_retries=cfg.sdk_max_retries,
            )
        self.cache = SQLiteKVCache(cache_db_path, table_name="teacher_cache")
        self.random = random.Random()

    def _request_extra_body(self) -> dict[str, Any]:
        raw = self.cfg.generation.extra_body
        if not raw:
            return {}
        if isinstance(raw, dict):
            return raw
        self.logger.warning("Ignoring teacher.generation.extra_body because it is not a dict: %s", type(raw).__name__)
        return {}

    def _merge_extra_body(self, call_extra_body: dict[str, Any] | None) -> dict[str, Any]:
        merged = dict(self._request_extra_body())
        if not call_extra_body:
            return merged
        if not isinstance(call_extra_body, dict):
            self.logger.warning("Ignoring call extra_body because it is not a dict: %s", type(call_extra_body).__name__)
            return merged
        merged.update(call_extra_body)
        return merged

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, bytes):
            return content.decode("utf-8", errors="ignore")
        if isinstance(content, (int, float, bool)):
            return str(content)
        if isinstance(content, list):
            return "".join(TeacherClient._content_to_text(part) for part in content)
        if isinstance(content, dict):
            for key in ("text", "output_text", "content", "reasoning_content", "value"):
                if key in content:
                    return TeacherClient._content_to_text(content.get(key))
            return ""

        for attr in ("text", "output_text", "content", "reasoning_content", "value"):
            value = getattr(content, attr, None)
            if value is not None:
                text = TeacherClient._content_to_text(value)
                if text:
                    return text

        dump = getattr(content, "model_dump", None)
        if callable(dump):
            try:
                return TeacherClient._content_to_text(dump())
            except Exception:  # pylint: disable=broad-except
                return ""

        return ""

    @classmethod
    def _extract_response_text(cls, response: Any) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        choice = choices[0]
        message = getattr(choice, "message", None)

        candidates = [
            cls._content_to_text(getattr(message, "content", None)),
            cls._content_to_text(getattr(message, "output_text", None)),
            cls._content_to_text(getattr(message, "text", None)),
            cls._content_to_text(getattr(choice, "text", None)),
            cls._content_to_text(getattr(choice, "output_text", None)),
        ]
        for candidate in candidates:
            text = candidate.strip()
            if text:
                return text
        return ""

    def _cache_key(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        presence_penalty: float | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        merged_extra_body = self._merge_extra_body(extra_body)
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "presence_penalty": presence_penalty,
            "extra_body": merged_extra_body,
        }
        return stable_hash(payload)

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        model: str | None = None,
        presence_penalty: float | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        chosen_model = model or self.cfg.model
        merged_extra_body = self._merge_extra_body(extra_body)
        cache_key = self._cache_key(
            chosen_model,
            messages,
            temperature,
            top_p,
            max_tokens,
            presence_penalty=presence_penalty,
            extra_body=merged_extra_body,
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            cached_text = str(cached.get("text", "")).strip()
            if cached_text:
                self.stats.inc("teacher.cache_hit")
                return cached_text
            self.stats.inc("teacher.cache_stale_empty")
            self.logger.warning("Ignoring empty teacher cache entry and refetching key=%s", cache_key[:12])

        self.stats.inc("teacher.cache_miss")
        backoff = self.cfg.retry.backoff_s
        max_attempts = max(1, self.cfg.retry.max_attempts)

        for attempt in range(max_attempts):
            try:
                with self.stats.time_block("teacher.request"):
                    request_kwargs: dict[str, Any] = {
                        "model": chosen_model,
                        "messages": messages,
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_tokens": max_tokens,
                        "extra_headers": {"Idempotency-Key": cache_key},
                    }
                    if presence_penalty is not None:
                        request_kwargs["presence_penalty"] = presence_penalty
                    if merged_extra_body:
                        request_kwargs["extra_body"] = merged_extra_body
                    response = self.client.chat.completions.create(
                        **request_kwargs,
                    )
                text = self._extract_response_text(response)
                if not text:
                    self.stats.inc("teacher.empty_response")
                    choices = getattr(response, "choices", None) or []
                    finish_reason = getattr(choices[0], "finish_reason", None) if choices else None
                    raise RuntimeError(
                        "Teacher returned empty completion text "
                        f"(finish_reason={finish_reason}, model={chosen_model})"
                    )

                self.cache.set(cache_key, {"text": text, "model": chosen_model})
                self.stats.inc("teacher.success")
                return text
            except Exception as exc:  # pylint: disable=broad-except
                self.stats.inc("teacher.error")
                wait = backoff[min(attempt, len(backoff) - 1)] if backoff else (2**attempt)
                wait = wait + self.random.random() * 0.25

                no_retry = False
                timeout_like = isinstance(exc, httpx.TimeoutException)
                if not timeout_like:
                    msg_lower = str(exc).lower()
                    timeout_like = "timeout" in type(exc).__name__.lower() or "timed out" in msg_lower

                if isinstance(exc, APIStatusError):
                    status = int(getattr(exc, "status_code", 0) or 0)
                    resp_text = getattr(getattr(exc, "response", None), "text", None)
                    self.logger.error(
                        "Teacher HTTP error status=%s type=%s body=%s",
                        status,
                        type(exc).__name__,
                        (resp_text[:500] if isinstance(resp_text, str) else None),
                    )
                    # Fast-fail for non-retriable client errors.
                    if status in {400, 401, 403, 404, 422}:
                        no_retry = True
                else:
                    self.logger.error(
                        "Teacher error type=%s detail=%r",
                        type(exc).__name__,
                        exc,
                    )
                if timeout_like:
                    self.logger.error(
                        "Timeout hint: increase teacher.request_timeout_s or lower "
                        "teacher.max_concurrency / teacher.generation.max_tokens "
                        "(current timeout=%s, concurrency=%s, max_tokens=%s).",
                        self.cfg.request_timeout_s,
                        self.cfg.max_concurrency,
                        max_tokens,
                    )

                self.logger.warning(
                    "Teacher call failed (attempt=%s/%s): %s",
                    attempt + 1,
                    max_attempts,
                    exc,
                )
                # Common vLLM misconfiguration hint.
                msg = str(exc).lower()
                if "chat template" in msg:
                    self.logger.error(
                        "vLLM chat template error detected. Start server with --chat-template if needed."
                    )
                if no_retry or attempt == max_attempts - 1:
                    raise
                self.stats.inc("teacher.retry")
                time.sleep(wait)

        raise RuntimeError("Unreachable teacher retry logic")

    def run_tasks(
        self,
        tasks: list[CompletionTask],
        worker_fn: Callable[[CompletionTask, str], dict[str, Any]],
        max_workers: int | None = None,
    ) -> list[dict[str, Any]]:
        workers = max_workers or self.cfg.max_concurrency
        results: list[dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map: dict[Future[dict[str, Any]], CompletionTask] = {}
            for task in tasks:
                future = executor.submit(
                    self._run_single_task,
                    task,
                    worker_fn,
                )
                future_map[future] = task

            for future in as_completed(future_map):
                task = future_map[future]
                try:
                    results.append(future.result())
                except Exception as exc:  # pylint: disable=broad-except
                    self.logger.exception("Task failed for item_id=%s: %s", task.item_id, exc)
                    raise

        return results

    def _run_single_task(
        self,
        task: CompletionTask,
        worker_fn: Callable[[CompletionTask, str], dict[str, Any]],
    ) -> dict[str, Any]:
        text = self.complete(
            messages=task.messages,
            temperature=task.temperature,
            top_p=task.top_p,
            max_tokens=task.max_tokens,
            model=task.model,
            presence_penalty=task.presence_penalty,
            extra_body=task.extra_body,
        )
        return worker_fn(task, text)

    def close(self) -> None:
        self.cache.close()
        if self._http_client is not None:
            self._http_client.close()
