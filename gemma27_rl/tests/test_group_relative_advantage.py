from __future__ import annotations

import pytest

from gemma27_rl.advantage import apply_group_relative_advantage


def test_apply_group_relative_advantage_uses_explicit_rollout_scalars() -> None:
    raw_advantages = [
        [10.0, 0.0],  # mean=5.0
        [0.0, 0.0],   # mean=0.0
    ]
    group_ids = ["g", "g"]
    rollout_scalars = [1.0, 3.0]

    adjusted, group_z = apply_group_relative_advantage(
        raw_advantages=raw_advantages,
        group_ids=group_ids,
        rollout_scalars=rollout_scalars,
        coef=1.0,
        eps=0.0,
    )

    # z-score computed from explicit rollout_scalars [1, 3]:
    # mean=2, std=1 => z=[-1, +1]
    assert group_z == pytest.approx([-1.0, 1.0])
    assert adjusted[0] == pytest.approx([9.0, -1.0])
    assert adjusted[1] == pytest.approx([1.0, 1.0])


def test_apply_group_relative_advantage_rollout_scalars_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="rollout_scalars and raw_advantages length mismatch"):
        _ = apply_group_relative_advantage(
            raw_advantages=[[1.0], [2.0]],
            group_ids=["a", "a"],
            rollout_scalars=[1.0],
        )
