from __future__ import annotations

import pytest

pytest.importorskip("torch")

from gemma27_rl.rl_types import Example
from gemma27_rl.trainer import _DirectionBatchSampler


def _make_examples() -> list[Example]:
    return [
        Example(example_id="en-0", src_text="a", src_lang="English", tgt_lang="Korean", src_lang_code="en", tgt_lang_code="ko"),
        Example(example_id="en-1", src_text="b", src_lang="English", tgt_lang="Korean", src_lang_code="en", tgt_lang_code="ko"),
        Example(example_id="ko-0", src_text="c", src_lang="Korean", tgt_lang="English", src_lang_code="ko", tgt_lang_code="en"),
        Example(example_id="ko-1", src_text="d", src_lang="Korean", tgt_lang="English", src_lang_code="ko", tgt_lang_code="en"),
    ]


def test_direction_batch_sampler_keeps_each_batch_single_direction() -> None:
    import random

    examples = _make_examples()
    sampler = _DirectionBatchSampler(examples=examples, rng=random.Random(0))

    direction_a, batch_a = sampler.next_batch(2)
    direction_b, batch_b = sampler.next_batch(2)

    assert len(batch_a) == 2
    assert len(batch_b) == 2
    assert {f"{examples[idx].src_lang_code}->{examples[idx].tgt_lang_code}" for idx in batch_a} == {direction_a}
    assert {f"{examples[idx].src_lang_code}->{examples[idx].tgt_lang_code}" for idx in batch_b} == {direction_b}
    assert {direction_a, direction_b} == {"en->ko", "ko->en"}


def test_direction_batch_sampler_wraps_with_same_direction_when_batch_exceeds_remaining() -> None:
    import random

    examples = [
        Example(example_id="en-0", src_text="a", src_lang="English", tgt_lang="Korean", src_lang_code="en", tgt_lang_code="ko"),
        Example(example_id="en-1", src_text="b", src_lang="English", tgt_lang="Korean", src_lang_code="en", tgt_lang_code="ko"),
    ]
    sampler = _DirectionBatchSampler(examples=examples, rng=random.Random(0))

    direction, batch = sampler.next_batch(3)

    assert direction == "en->ko"
    assert len(batch) == 3
    assert {f"{examples[idx].src_lang_code}->{examples[idx].tgt_lang_code}" for idx in batch} == {"en->ko"}
