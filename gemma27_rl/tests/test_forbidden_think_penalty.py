from __future__ import annotations

import pytest

pytest.importorskip("torch")

from gemma27_rl.trainer import _apply_forbidden_think_tag_penalty, _find_forbidden_think_tag_spans


def test_find_forbidden_think_tag_spans_detects_both_tags_case_insensitive() -> None:
    text = "A <THINK>reason</think> B"
    matches = _find_forbidden_think_tag_spans(text)

    assert [text[start:end].lower() for start, end, _ in matches] == ["<think>", "</think>"]


def test_apply_forbidden_think_tag_penalty_penalizes_token_and_sequence_rewards() -> None:
    completion = "x<think>y</think>z"
    token_rewards = [0.0]
    token_offsets = [(0, len(completion))]

    adjusted_tokens, adjusted_seq, tag_count, token_hits = _apply_forbidden_think_tag_penalty(
        completion_text=completion,
        token_char_offsets=token_offsets,
        token_rewards=token_rewards,
        seq_reward=1.5,
        token_penalty=-100.0,
        seq_penalty_per_match=-30.0,
    )

    assert adjusted_tokens == [-200.0]
    assert adjusted_seq == pytest.approx(-58.5)
    assert tag_count == 2
    assert token_hits == 1
