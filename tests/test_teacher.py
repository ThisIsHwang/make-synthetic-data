from types import SimpleNamespace

import pytest

from synth_parallel.config import TeacherConfig
from synth_parallel.stats import StatsCollector
from synth_parallel.teacher import TeacherClient


def _response(
    *,
    message_content=None,
    message_output_text=None,
    choice_text=None,
    finish_reason="stop",
):
    message = SimpleNamespace(content=message_content, output_text=message_output_text)
    choice = SimpleNamespace(message=message, text=choice_text, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


class _FakeCompletions:
    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    def create(self, **kwargs):
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


class _FakeOpenAI:
    def __init__(self, responses):
        self.completions = _FakeCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)


def _build_client(tmp_path, responses, max_attempts=1):
    cfg = TeacherConfig(base_url="http://example.local/v1", unset_proxy_env=False)
    cfg.retry.max_attempts = max_attempts
    stats = StatsCollector(tmp_path / "stats.json")
    client = TeacherClient(
        cfg=cfg,
        cache_db_path=str(tmp_path / "teacher_cache.sqlite"),
        stats=stats,
    )
    client.client = _FakeOpenAI(responses)
    return client


def test_teacher_extracts_text_from_choice_text_fallback(tmp_path):
    client = _build_client(tmp_path, [_response(message_content=None, choice_text="translated")])
    try:
        text = client.complete(
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.0,
            top_p=1.0,
            max_tokens=64,
        )
        assert text == "translated"
    finally:
        client.close()


def test_teacher_ignores_empty_cached_text_and_refetches(tmp_path):
    messages = [{"role": "user", "content": "hello"}]
    client = _build_client(tmp_path, [_response(message_content="fresh text")])
    try:
        cache_key = client._cache_key(client.cfg.model, messages, 0.0, 1.0, 64)
        client.cache.set(cache_key, {"text": "", "model": client.cfg.model})

        text = client.complete(
            messages=messages,
            temperature=0.0,
            top_p=1.0,
            max_tokens=64,
        )
        assert text == "fresh text"
        assert client.client.completions.calls == 1
    finally:
        client.close()


def test_teacher_raises_on_empty_completion(tmp_path):
    messages = [{"role": "user", "content": "hello"}]
    client = _build_client(tmp_path, [_response(message_content="", choice_text=None)], max_attempts=1)
    try:
        with pytest.raises(RuntimeError, match="empty completion text"):
            client.complete(
                messages=messages,
                temperature=0.0,
                top_p=1.0,
                max_tokens=64,
            )

        cache_key = client._cache_key(client.cfg.model, messages, 0.0, 1.0, 64)
        assert client.cache.get(cache_key) is None
    finally:
        client.close()
