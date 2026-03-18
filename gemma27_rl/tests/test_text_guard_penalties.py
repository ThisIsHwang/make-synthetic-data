from __future__ import annotations

import pytest

pytest.importorskip("torch")

from gemma27_rl.trainer import (
    _apply_assistant_fallback_penalty,
    _apply_script_mismatch_penalty,
    _detect_script_mismatch,
)


def test_apply_assistant_fallback_penalty_hits_matching_tokens() -> None:
    completion = "Could you provide this translation?"
    token_rewards = [0.0, 0.0, 0.0, 0.0]
    token_offsets = [(0, 5), (6, 9), (10, 17), (18, 22)]

    adjusted_tokens, adjusted_seq, match_count, token_hit_count = _apply_assistant_fallback_penalty(
        completion_text=completion,
        token_char_offsets=token_offsets,
        token_rewards=token_rewards,
        seq_reward=0.0,
        patterns=[r"(?i)^could you provide"],
        token_penalty=-1.0,
        seq_penalty_per_match=-4.0,
    )

    assert adjusted_tokens == [-1.0, -1.0, -1.0, 0.0]
    assert adjusted_seq == pytest.approx(-4.0)
    assert match_count == 1
    assert token_hit_count == 3


def test_apply_assistant_fallback_penalty_no_match_no_penalty() -> None:
    adjusted_tokens, adjusted_seq, match_count, token_hit_count = _apply_assistant_fallback_penalty(
        completion_text="This is an actual translation.",
        token_char_offsets=[(0, 4), (5, 7), (8, 10)],
        token_rewards=[0.0, 0.0, 0.0],
        seq_reward=1.5,
        patterns=[r"(?i)^as an ai", r"(?i)^could you provide"],
        token_penalty=-1.0,
        seq_penalty_per_match=-4.0,
    )

    assert adjusted_tokens == [0.0, 0.0, 0.0]
    assert adjusted_seq == pytest.approx(1.5)
    assert match_count == 0
    assert token_hit_count == 0


def test_detect_script_mismatch_en_target_with_hangul_heavy_output() -> None:
    assert _detect_script_mismatch(
        text="이것은 한국어 출력이며 전체 문장이 거의 한글로만 구성되어 있습니다 abc",
        target_lang="English",
        target_lang_code="en",
        min_letters=6,
        ratio_threshold=0.35,
    )


def test_detect_script_mismatch_ko_target_with_low_hangul_ratio() -> None:
    assert _detect_script_mismatch(
        text="This output is mostly English text with brand names",
        target_lang="Korean",
        target_lang_code="ko",
        min_letters=6,
        ratio_threshold=0.35,
    )


def test_detect_script_mismatch_ignores_short_text() -> None:
    assert not _detect_script_mismatch(
        text="한글",
        target_lang="English",
        target_lang_code="en",
        min_letters=6,
        ratio_threshold=0.35,
    )


def test_apply_script_mismatch_penalty_applies_sequence_penalty() -> None:
    adjusted_seq, mismatch = _apply_script_mismatch_penalty(
        text="이것은 한국어 출력이며 전체 문장이 거의 한글로만 구성되어 있습니다 abc",
        target_lang="English",
        target_lang_code="en",
        seq_reward=2.0,
        seq_penalty=-2.5,
        min_letters=6,
        ratio_threshold=0.35,
    )

    assert mismatch is True
    assert adjusted_seq == pytest.approx(-0.5)
