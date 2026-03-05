from __future__ import annotations

import pytest

pytest.importorskip("torch")

from gemma27_rl.trainer import (
    _apply_forbidden_think_tag_penalty,
    _apply_ngram_repeat_penalty,
    _apply_repeated_token_penalty,
    _apply_special_token_penalty,
    _find_forbidden_think_tag_spans,
    _sanitize_text_for_mqm_esa,
)


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


def test_apply_repeated_token_penalty_penalizes_consecutive_repeat_tokens() -> None:
    adjusted_tokens, adjusted_seq, repeat_count, repeat_runs = _apply_repeated_token_penalty(
        completion_token_ids=[10, 20, 20, 20, 30],
        token_rewards=[0.0, 0.0, 0.0, 0.0, 0.0],
        seq_reward=0.0,
        token_penalty=-2.0,
        seq_penalty_per_repeat=-1.0,
        min_repeat_run_length=2,
    )

    assert adjusted_tokens == [0.0, 0.0, -2.0, -2.0, 0.0]
    assert adjusted_seq == pytest.approx(-2.0)
    assert repeat_count == 2
    assert repeat_runs == 1


def test_apply_repeated_token_penalty_penalizes_periodic_non_consecutive_repeats() -> None:
    adjusted_tokens, adjusted_seq, repeat_count, repeat_runs = _apply_repeated_token_penalty(
        completion_token_ids=[10, 20, 10, 20, 10, 20],
        token_rewards=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        seq_reward=0.0,
        token_penalty=-1.5,
        seq_penalty_per_repeat=-0.5,
        min_repeat_run_length=2,
        max_repeat_pattern_length=4,
    )

    assert adjusted_tokens == [0.0, 0.0, -1.5, -1.5, -1.5, -1.5]
    assert adjusted_seq == pytest.approx(-2.0)
    assert repeat_count == 4
    assert repeat_runs == 1


def test_apply_ngram_repeat_penalty_penalizes_repeated_ngrams() -> None:
    adjusted_tokens, adjusted_seq, token_hits, repeat_occurrences = _apply_ngram_repeat_penalty(
        completion_token_ids=[1, 2, 3, 1, 2, 3, 4],
        token_rewards=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        seq_reward=0.0,
        token_penalty=-2.0,
        seq_penalty_per_repeat=-0.5,
        ngram_size=3,
        min_occurrences=2,
    )

    assert adjusted_tokens == [0.0, 0.0, 0.0, -2.0, -2.0, -2.0, 0.0]
    assert adjusted_seq == pytest.approx(-0.5)
    assert token_hits == 3
    assert repeat_occurrences == 1


def test_sanitize_text_for_mqm_esa_removes_special_and_think_markers() -> None:
    raw = "<bos>hello <think>hidden</think> world <|assistant|>"
    sanitized, replacement_count = _sanitize_text_for_mqm_esa(
        raw,
        special_tokens=["<bos>", "<|assistant|>"],
    )

    assert "<bos>" not in sanitized
    assert "<think>" not in sanitized
    assert "</think>" not in sanitized
    assert "<|assistant|>" not in sanitized
    assert "hidden" not in sanitized
    assert replacement_count >= 3


def test_apply_special_token_penalty_penalizes_special_token_id_hits() -> None:
    adjusted_tokens, adjusted_seq, occurrences, token_hits = _apply_special_token_penalty(
        completion_text="abc",
        completion_token_ids=[11, 99, 12],
        token_char_offsets=[(0, 1), (1, 2), (2, 3)],
        token_rewards=[0.0, 0.0, 0.0],
        seq_reward=0.0,
        special_token_ids={99},
        special_token_strings=[],
        token_penalty=-5.0,
        seq_penalty_per_occurrence=-2.0,
    )

    assert adjusted_tokens == [0.0, -5.0, 0.0]
    assert adjusted_seq == pytest.approx(-2.0)
    assert occurrences == 1
    assert token_hits == 1


def test_apply_special_token_penalty_penalizes_special_token_text_hits() -> None:
    text = "hello <|assistant|> world"
    adjusted_tokens, adjusted_seq, occurrences, token_hits = _apply_special_token_penalty(
        completion_text=text,
        completion_token_ids=[11],
        token_char_offsets=[(0, len(text))],
        token_rewards=[0.0],
        seq_reward=1.0,
        special_token_ids=set(),
        special_token_strings=["<|assistant|>"],
        token_penalty=-4.0,
        seq_penalty_per_occurrence=-3.0,
    )

    assert adjusted_tokens == [-4.0]
    assert adjusted_seq == pytest.approx(-2.0)
    assert occurrences == 1
    assert token_hits == 1


def test_apply_special_token_penalty_skips_final_exempt_end_of_turn_token_id() -> None:
    adjusted_tokens, adjusted_seq, occurrences, token_hits = _apply_special_token_penalty(
        completion_text="abc",
        completion_token_ids=[11, 22, 99],
        token_char_offsets=[(0, 1), (1, 2), (2, 3)],
        token_rewards=[0.0, 0.0, 0.0],
        seq_reward=0.5,
        special_token_ids={99},
        special_token_strings=[],
        token_penalty=-5.0,
        seq_penalty_per_occurrence=-2.0,
        exempt_final_token_ids={99},
    )

    assert adjusted_tokens == [0.0, 0.0, 0.0]
    assert adjusted_seq == pytest.approx(0.5)
    assert occurrences == 0
    assert token_hits == 0


def test_apply_special_token_penalty_skips_final_exempt_end_of_turn_text() -> None:
    text = "hello <end_of_turn>   "
    adjusted_tokens, adjusted_seq, occurrences, token_hits = _apply_special_token_penalty(
        completion_text=text,
        completion_token_ids=[11],
        token_char_offsets=[(0, len(text))],
        token_rewards=[0.0],
        seq_reward=1.0,
        special_token_ids=set(),
        special_token_strings=["<end_of_turn>"],
        token_penalty=-4.0,
        seq_penalty_per_occurrence=-3.0,
        exempt_final_token_strings={"<end_of_turn>"},
    )

    assert adjusted_tokens == [0.0]
    assert adjusted_seq == pytest.approx(1.0)
    assert occurrences == 0
    assert token_hits == 0


def test_apply_special_token_penalty_uses_raw_ids_for_sequence_penalty() -> None:
    adjusted_tokens, adjusted_seq, occurrences, token_hits = _apply_special_token_penalty(
        completion_text="clean output",
        completion_token_ids=[10, 20, 30],
        penalty_token_ids=[10, 99, 30],
        token_char_offsets=[(0, 4), (4, 10), (10, 12)],
        token_rewards=[0.0, 0.0, 0.0],
        seq_reward=0.0,
        special_token_ids={99},
        special_token_strings=[],
        token_penalty=-4.0,
        seq_penalty_per_occurrence=-3.0,
    )

    # Special token was present only in raw ids, so sequence gets penalized
    # while aligned token-level reward stays unchanged.
    assert adjusted_tokens == [0.0, 0.0, 0.0]
    assert adjusted_seq == pytest.approx(-3.0)
    assert occurrences == 1
    assert token_hits == 0
