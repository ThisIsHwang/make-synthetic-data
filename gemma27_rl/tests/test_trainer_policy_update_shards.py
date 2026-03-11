from __future__ import annotations

import pytest

pytest.importorskip("torch")

from gemma27_rl.rl_types import Rollout
from gemma27_rl.trainer import _should_use_per_rank_policy_update_shards


def _make_rollout(example_id: str, completion_ids: list[int]) -> Rollout:
    return Rollout(
        example_id=example_id,
        prompt_text="p",
        prompt_input_ids=[1, 2],
        completion_text="c" if completion_ids else "",
        completion_token_ids=list(completion_ids),
        old_logprobs=[0.0 for _ in completion_ids],
        ref_logprobs=None,
        token_char_offsets=[(0, 1) for _ in completion_ids],
        src_text="src",
        ref_text=None,
    )


def test_should_use_per_rank_policy_update_shards_accepts_uniform_nonempty_shards() -> None:
    per_rank_rollouts = [
        [_make_rollout("r0-0", [3, 4]), _make_rollout("r0-1", [5])],
        [_make_rollout("r1-0", [6]), _make_rollout("r1-1", [7, 8])],
    ]
    merged_rollouts = [rollout for shard in per_rank_rollouts for rollout in shard]
    merged_advantages = [[0.1 for _ in rollout.completion_token_ids] for rollout in merged_rollouts]

    ok, reason = _should_use_per_rank_policy_update_shards(
        per_rank_rollouts=per_rank_rollouts,
        merged_rollouts=merged_rollouts,
        merged_advantages=merged_advantages,
        reward_stats={},
    )

    assert ok is True
    assert reason == ""


def test_should_use_per_rank_policy_update_shards_rejects_zero_token_shard() -> None:
    per_rank_rollouts = [
        [_make_rollout("r0-0", []), _make_rollout("r0-1", [])],
        [_make_rollout("r1-0", [6]), _make_rollout("r1-1", [7, 8])],
    ]
    merged_rollouts = [rollout for shard in per_rank_rollouts for rollout in shard]
    merged_advantages = [[0.1 for _ in rollout.completion_token_ids] for rollout in merged_rollouts]

    ok, reason = _should_use_per_rank_policy_update_shards(
        per_rank_rollouts=per_rank_rollouts,
        merged_rollouts=merged_rollouts,
        merged_advantages=merged_advantages,
        reward_stats={},
    )

    assert ok is False
    assert reason == "zero_token_shard"
