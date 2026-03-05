from __future__ import annotations

import logging

from gemma27_rl.rollout import compute_token_char_offsets


class _ContextualDecodeTokenizer:
    # Force non-fast path to exercise fallback behavior.
    is_fast = False

    def decode(self, token_ids, *, clean_up_tokenization_spaces=False, skip_special_tokens=False):
        del clean_up_tokenization_spaces, skip_special_tokens
        key = tuple(int(v) for v in token_ids)
        table = {
            (10,): "Hello ",
            (11,): " world",
            (10, 11): "Hello world",
        }
        if key not in table:
            raise ValueError(f"unexpected token ids: {key!r}")
        return table[key]


def test_compute_token_char_offsets_uses_prefix_decode_when_single_decode_concat_mismatches(caplog) -> None:
    tokenizer = _ContextualDecodeTokenizer()
    completion_ids = [10, 11]
    completion_text = "Hello world"

    # Old fallback emitted reconstruction mismatch warning here.
    with caplog.at_level(logging.WARNING):
        offsets = compute_token_char_offsets(
            tokenizer=tokenizer,
            completion_token_ids=completion_ids,
            completion_text=completion_text,
        )

    assert offsets == [(0, 6), (6, 11)]
    assert not any(
        "Token offset reconstruction mismatch" in rec.getMessage()
        for rec in caplog.records
    )
