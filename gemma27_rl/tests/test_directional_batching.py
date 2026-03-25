from __future__ import annotations

import pytest

pytest.importorskip("torch")

from gemma27_rl.rl_types import Example
from gemma27_rl.trainer import _DirectionBatchSampler, _DirectionDomainLengthBatchSampler


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


def test_direction_domain_length_sampler_keeps_batch_single_direction_and_domain() -> None:
    import random

    examples = [
        Example(
            example_id="casual-0",
            src_text="a",
            src_lang="English",
            tgt_lang="Korean",
            src_lang_code="en",
            tgt_lang_code="ko",
            domain="casual",
        ),
        Example(
            example_id="casual-1",
            src_text="b",
            src_lang="English",
            tgt_lang="Korean",
            src_lang_code="en",
            tgt_lang_code="ko",
            domain="casual",
        ),
        Example(
            example_id="expert-0",
            src_text="c",
            src_lang="English",
            tgt_lang="Korean",
            src_lang_code="en",
            tgt_lang_code="ko",
            domain="expert",
        ),
        Example(
            example_id="expert-1",
            src_text="d",
            src_lang="English",
            tgt_lang="Korean",
            src_lang_code="en",
            tgt_lang_code="ko",
            domain="expert",
        ),
    ]
    sampler = _DirectionDomainLengthBatchSampler(
        examples=examples,
        prompt_token_lengths=[10, 11, 50, 51],
        effective_batch_size=2,
        rng=random.Random(0),
    )

    _, batch_a = sampler.next_batch(2)
    _, batch_b = sampler.next_batch(2)

    for batch in (batch_a, batch_b):
        assert {f"{examples[idx].src_lang_code}->{examples[idx].tgt_lang_code}" for idx in batch} == {"en->ko"}
        assert len({examples[idx].domain for idx in batch}) == 1

    assert {examples[batch_a[0]].domain, examples[batch_b[0]].domain} == {"casual", "expert"}


def test_direction_domain_length_sampler_batches_nearby_prompt_lengths() -> None:
    import random

    examples = [
        Example(
            example_id=f"e-{idx}",
            src_text=str(idx),
            src_lang="English",
            tgt_lang="Korean",
            src_lang_code="en",
            tgt_lang_code="ko",
            domain="casual",
        )
        for idx in range(6)
    ]
    prompt_lengths = [5, 6, 7, 40, 41, 42]
    sampler = _DirectionDomainLengthBatchSampler(
        examples=examples,
        prompt_token_lengths=prompt_lengths,
        effective_batch_size=3,
        rng=random.Random(0),
    )

    _, batch_a = sampler.next_batch(3)
    _, batch_b = sampler.next_batch(3)

    spreads = []
    for batch in (batch_a, batch_b):
        batch_lengths = [prompt_lengths[idx] for idx in batch]
        spreads.append(max(batch_lengths) - min(batch_lengths))
    assert sorted(spreads) == [2, 2]


def test_direction_domain_length_sampler_round_robins_across_groups() -> None:
    import random

    examples = [
        Example(
            example_id="casual-0",
            src_text="a",
            src_lang="English",
            tgt_lang="Korean",
            src_lang_code="en",
            tgt_lang_code="ko",
            domain="casual",
        ),
        Example(
            example_id="casual-1",
            src_text="b",
            src_lang="English",
            tgt_lang="Korean",
            src_lang_code="en",
            tgt_lang_code="ko",
            domain="casual",
        ),
        Example(
            example_id="expert-0",
            src_text="c",
            src_lang="English",
            tgt_lang="Korean",
            src_lang_code="en",
            tgt_lang_code="ko",
            domain="expert",
        ),
        Example(
            example_id="expert-1",
            src_text="d",
            src_lang="English",
            tgt_lang="Korean",
            src_lang_code="en",
            tgt_lang_code="ko",
            domain="expert",
        ),
        Example(
            example_id="expert-2",
            src_text="e",
            src_lang="English",
            tgt_lang="Korean",
            src_lang_code="en",
            tgt_lang_code="ko",
            domain="expert",
        ),
        Example(
            example_id="expert-3",
            src_text="f",
            src_lang="English",
            tgt_lang="Korean",
            src_lang_code="en",
            tgt_lang_code="ko",
            domain="expert",
        ),
    ]
    sampler = _DirectionDomainLengthBatchSampler(
        examples=examples,
        prompt_token_lengths=[5, 6, 20, 21, 22, 23],
        effective_batch_size=2,
        rng=random.Random(0),
    )

    _, batch_a = sampler.next_batch(2)
    _, batch_b = sampler.next_batch(2)
    _, batch_c = sampler.next_batch(2)

    domains = [examples[batch[0]].domain for batch in (batch_a, batch_b, batch_c)]
    assert domains[0] != domains[1]
    assert domains.count("expert") == 2
    assert domains.count("casual") == 1


def test_direction_domain_length_sampler_global_batch_stays_single_bucket_before_sharding() -> None:
    import random

    examples = [
        Example(
            example_id="casual-0",
            src_text="a",
            src_lang="English",
            tgt_lang="Korean",
            src_lang_code="en",
            tgt_lang_code="ko",
            domain="casual",
        ),
        Example(
            example_id="casual-1",
            src_text="b",
            src_lang="English",
            tgt_lang="Korean",
            src_lang_code="en",
            tgt_lang_code="ko",
            domain="casual",
        ),
        Example(
            example_id="casual-2",
            src_text="c",
            src_lang="English",
            tgt_lang="Korean",
            src_lang_code="en",
            tgt_lang_code="ko",
            domain="casual",
        ),
        Example(
            example_id="casual-3",
            src_text="d",
            src_lang="English",
            tgt_lang="Korean",
            src_lang_code="en",
            tgt_lang_code="ko",
            domain="casual",
        ),
        Example(
            example_id="expert-0",
            src_text="e",
            src_lang="English",
            tgt_lang="Korean",
            src_lang_code="en",
            tgt_lang_code="ko",
            domain="expert",
        ),
        Example(
            example_id="expert-1",
            src_text="f",
            src_lang="English",
            tgt_lang="Korean",
            src_lang_code="en",
            tgt_lang_code="ko",
            domain="expert",
        ),
        Example(
            example_id="expert-2",
            src_text="g",
            src_lang="English",
            tgt_lang="Korean",
            src_lang_code="en",
            tgt_lang_code="ko",
            domain="expert",
        ),
        Example(
            example_id="expert-3",
            src_text="h",
            src_lang="English",
            tgt_lang="Korean",
            src_lang_code="en",
            tgt_lang_code="ko",
            domain="expert",
        ),
    ]
    sampler = _DirectionDomainLengthBatchSampler(
        examples=examples,
        prompt_token_lengths=[5, 6, 7, 8, 30, 31, 32, 33],
        effective_batch_size=4,
        rng=random.Random(1),
    )

    _, global_batch = sampler.next_batch(4)
    per_rank_batches = [global_batch[:2], global_batch[2:]]

    for shard in per_rank_batches:
        assert {f"{examples[idx].src_lang_code}->{examples[idx].tgt_lang_code}" for idx in shard} == {"en->ko"}
        assert len({examples[idx].domain for idx in shard}) == 1
