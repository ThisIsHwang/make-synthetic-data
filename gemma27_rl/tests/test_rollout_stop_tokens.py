from __future__ import annotations

from gemma27_rl.rollout import (
    _collect_end_of_turn_token_ids,
    _resolve_eos_token_ids,
    _trim_completion_ids,
)


class _FakeTokenizer:
    def __init__(self) -> None:
        self.all_special_tokens = ["<bos>", "<end_of_turn>", "<|eot_id|>"]
        self.additional_special_tokens = ["<|im_end|>"]
        self.special_tokens_map = {"additional_special_tokens": ["<end_of_turn>"]}
        self._token_to_id = {
            "<end_of_turn>": 71,
            "<|eot_id|>": 81,
            "<|im_end|>": 91,
        }

    def convert_tokens_to_ids(self, token: str) -> int:
        return int(self._token_to_id.get(str(token), -1))

    def get_added_vocab(self) -> dict[str, int]:
        return {"<|eot_id|>": 81, "<|assistant|>": 777}


def test_resolve_eos_token_ids_includes_eot_extra_ids() -> None:
    resolved = _resolve_eos_token_ids(
        tokenizer_eos_token_id=2,
        model_eos_token_id=[1, 2],
        extra_token_ids=[9, 1],
    )

    assert resolved == [1, 2, 9]


def test_collect_end_of_turn_token_ids_detects_model_specific_markers() -> None:
    tokenizer = _FakeTokenizer()

    token_ids = _collect_end_of_turn_token_ids(tokenizer)

    assert set(token_ids) == {71, 81, 91}
    assert len(token_ids) == 3


def test_trim_completion_ids_stops_on_eot_id() -> None:
    trimmed = _trim_completion_ids(
        ids=[10, 11, 81, 12, 13],
        eos_token_ids=[2, 81],
        pad_token_id=None,
    )

    assert trimmed == [10, 11]
