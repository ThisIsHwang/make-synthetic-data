from __future__ import annotations

import logging
import os
import random
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable

from openai import OpenAI

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
        self.client = OpenAI(base_url=cfg.base_url, api_key=api_key, timeout=cfg.request_timeout_s)
        self.cache = SQLiteKVCache(cache_db_path, table_name="teacher_cache")
        self.random = random.Random()

    def _cache_key(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        return stable_hash(payload)

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        model: str | None = None,
    ) -> str:
        chosen_model = model or self.cfg.model
        cache_key = self._cache_key(chosen_model, messages, temperature, top_p, max_tokens)
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.stats.inc("teacher.cache_hit")
            return str(cached.get("text", ""))

        self.stats.inc("teacher.cache_miss")
        backoff = self.cfg.retry.backoff_s
        max_attempts = max(1, self.cfg.retry.max_attempts)

        for attempt in range(max_attempts):
            try:
                with self.stats.time_block("teacher.request"):
                    response = self.client.chat.completions.create(
                        model=chosen_model,
                        messages=messages,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        extra_headers={"Idempotency-Key": cache_key},
                    )
                content = response.choices[0].message.content
                if isinstance(content, list):
                    text = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    )
                else:
                    text = str(content or "")

                self.cache.set(cache_key, {"text": text, "model": chosen_model})
                self.stats.inc("teacher.success")
                return text.strip()
            except Exception as exc:  # pylint: disable=broad-except
                self.stats.inc("teacher.error")
                wait = backoff[min(attempt, len(backoff) - 1)] if backoff else (2**attempt)
                wait = wait + self.random.random() * 0.25
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
                if attempt == max_attempts - 1:
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
        )
        return worker_fn(task, text)

    def close(self) -> None:
        self.cache.close()
