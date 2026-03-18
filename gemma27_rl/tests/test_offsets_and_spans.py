from __future__ import annotations

from gemma27_rl.rewards import spans_to_token_rewards
from gemma27_rl.rollout import TokenDecodeConfig, compute_token_char_offsets


class FakeTokenizer:
    is_fast = False

    def __init__(self, token_map: dict[int, str]):
        self._token_map = token_map

    def decode(self, ids, clean_up_tokenization_spaces=False, skip_special_tokens=False):
        del clean_up_tokenization_spaces
        del skip_special_tokens
        if isinstance(ids, int):
            ids = [ids]
        return "".join(self._token_map[int(i)] for i in ids)


def test_compute_token_char_offsets_reconstructs_text() -> None:
    tokenizer = FakeTokenizer({1: "Hel", 2: "lo", 3: "!"})
    offsets = compute_token_char_offsets(
        tokenizer=tokenizer,
        completion_token_ids=[1, 2, 3],
        decode_cfg=TokenDecodeConfig(),
        completion_text="Hello!",
    )
    assert offsets == [(0, 3), (3, 5), (5, 6)]


def test_spans_to_token_rewards_maps_penalty_to_matching_token() -> None:
    rewards = spans_to_token_rewards(
        mt_text="hello world",
        token_char_offsets=[(0, 5), (5, 6), (6, 11)],
        error_spans=[{"start": 6, "end": 11, "severity": "MAJOR"}],
        severity_weights={"MINOR": -1.0, "MAJOR": -5.0, "CRITICAL": -10.0},
        overlap_policy="any_overlap",
        use_confidence=False,
        combine_policy="sum",
    )
    assert rewards == [0.0, 0.0, -5.0]


def test_spans_to_token_rewards_max_prefers_stronger_penalty_by_magnitude() -> None:
    rewards = spans_to_token_rewards(
        mt_text="hello world",
        token_char_offsets=[(0, 5), (5, 6), (6, 11)],
        error_spans=[
            {"start": 6, "end": 11, "severity": "MAJOR"},
            {"start": 6, "end": 11, "severity": "CRITICAL"},
        ],
        severity_weights={"MINOR": -1.0, "MAJOR": -5.0, "CRITICAL": -10.0},
        overlap_policy="any_overlap",
        use_confidence=False,
        combine_policy="max",
    )
    assert rewards == [0.0, 0.0, -10.0]


def test_spans_to_token_rewards_applies_confidence_scaling() -> None:
    rewards = spans_to_token_rewards(
        mt_text="hello world",
        token_char_offsets=[(0, 5), (5, 6), (6, 11)],
        error_spans=[{"start": 6, "end": 11, "severity": "MAJOR", "confidence": 0.4}],
        severity_weights={"MINOR": -1.0, "MAJOR": -5.0, "CRITICAL": -10.0},
        overlap_policy="any_overlap",
        use_confidence=True,
        combine_policy="sum",
    )
    assert rewards == [0.0, 0.0, -2.0]
