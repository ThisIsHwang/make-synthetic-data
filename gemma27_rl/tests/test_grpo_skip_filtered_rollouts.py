from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from gemma27_rl.config import RLPostTrainConfig
from gemma27_rl.rl_types import RewardOutput, Rollout
from gemma27_rl.trainer import _prepare_rewards_and_advantages


def _make_rollouts() -> list[Rollout]:
    return [
        Rollout(
            example_id="ex-0",
            prompt_text="prompt",
            prompt_input_ids=[1, 2],
            completion_text="ab",
            completion_token_ids=[3, 4],
            old_logprobs=[-0.1, -0.2],
            ref_logprobs=None,
            token_char_offsets=[(0, 1), (1, 2)],
            src_text="src-0",
            ref_text="ref-0",
            raw_completion_token_ids=[3, 4],
            completion_raw_text="ab",
            completion_clean_text="ab",
        ),
        Rollout(
            example_id="ex-1",
            prompt_text="prompt",
            prompt_input_ids=[1, 2],
            completion_text="cd",
            completion_token_ids=[5, 6],
            old_logprobs=[-0.3, -0.4],
            ref_logprobs=None,
            token_char_offsets=[(0, 1), (1, 2)],
            src_text="src-1",
            ref_text="ref-1",
            raw_completion_token_ids=[5, 6],
            completion_raw_text="cd",
            completion_clean_text="cd",
        ),
    ]


class _MQMSkipSecondScorer:
    cfg = SimpleNamespace(use_reference=False)

    def score_batch(self, samples):  # type: ignore[no-untyped-def]
        assert len(samples) == 2
        return RewardOutput(
            sequence_scores=[-1.0, -2.0],
            metadata={"error_spans": [[], []], "skipped_rows": [False, True]},
        )


class _MQMSkipAllScorer:
    cfg = SimpleNamespace(use_reference=False)

    def score_batch(self, samples):  # type: ignore[no-untyped-def]
        assert len(samples) == 2
        return RewardOutput(
            sequence_scores=[0.0, 0.0],
            metadata={"error_spans": [[], []], "skipped_rows": [True, True]},
        )


class _MQMFailureSecondScorer:
    cfg = SimpleNamespace(use_reference=False)

    def score_batch(self, samples):  # type: ignore[no-untyped-def]
        assert len(samples) == 2
        return RewardOutput(
            sequence_scores=[0.0, 0.0],
            metadata={
                "error_spans": [[], []],
                "skipped_rows": [False, False],
                "failure_rows": [False, True],
            },
        )


class _ESASkipSecondScorer:
    cfg = SimpleNamespace(use_reference=False)

    def score_batch(self, samples):  # type: ignore[no-untyped-def]
        assert len(samples) == 2
        return RewardOutput(
            sequence_scores=[80.0, 0.0],
            metadata={"skipped_rows": [False, True]},
        )


class _ESASkipAllScorer:
    cfg = SimpleNamespace(use_reference=False)

    def score_batch(self, samples):  # type: ignore[no-untyped-def]
        assert len(samples) == 2
        return RewardOutput(
            sequence_scores=[0.0, 0.0],
            metadata={"skipped_rows": [True, True]},
        )


def _base_cfg() -> RLPostTrainConfig:
    cfg = RLPostTrainConfig()
    cfg.reward.metricx.enabled = False
    cfg.reward.xcomet.enabled = False
    cfg.reward.mqm.enabled = True
    cfg.reward.esa.enabled = False
    cfg.reward.cache_enabled = False
    cfg.reward.w_metricx = 0.0
    cfg.reward.w_mqm_seq = 0.2
    cfg.rl.group_normalize = True
    cfg.generation.num_samples_per_prompt = 2
    return cfg


def test_prepare_rewards_and_advantages_keeps_rollouts_when_only_mqm_reports_skips() -> None:
    filtered_rollouts, advantages, reward_stats, adv_stats = _prepare_rewards_and_advantages(
        rollouts=_make_rollouts(),
        cfg=_base_cfg(),
        metricx_scorer=None,
        xcomet_scorer=None,
        mqm_scorer=_MQMSkipSecondScorer(),  # type: ignore[arg-type]
        esa_scorer=None,
        metricx_cache={},
        xcomet_cache={},
        mqm_cache={},
        esa_cache={},
        tokenizer=None,
    )

    assert [rollout.example_id for rollout in filtered_rollouts] == ["ex-0", "ex-1"]
    assert len(advantages) == 2
    assert reward_stats["mqm_skipped_count"] == 1.0
    assert reward_stats["grpo_dropped_rollouts_count"] == 0.0
    assert adv_stats["raw_std"] > 0.0


def test_prepare_rewards_and_advantages_keeps_rollouts_when_all_mqm_rows_report_skips() -> None:
    filtered_rollouts, advantages, reward_stats, adv_stats = _prepare_rewards_and_advantages(
        rollouts=_make_rollouts(),
        cfg=_base_cfg(),
        metricx_scorer=None,
        xcomet_scorer=None,
        mqm_scorer=_MQMSkipAllScorer(),  # type: ignore[arg-type]
        esa_scorer=None,
        metricx_cache={},
        xcomet_cache={},
        mqm_cache={},
        esa_cache={},
        tokenizer=None,
    )

    assert [rollout.example_id for rollout in filtered_rollouts] == ["ex-0", "ex-1"]
    assert len(advantages) == 2
    assert reward_stats["mqm_skipped_count"] == 2.0
    assert reward_stats["grpo_dropped_rollouts_count"] == 0.0
    assert adv_stats["raw_mean"] == pytest.approx(0.0)


def test_prepare_rewards_and_advantages_applies_mqm_failure_seq_penalty() -> None:
    cfg = _base_cfg()
    cfg.reward.w_mqm_seq = 1.0
    cfg.reward.mqm.failure_seq_penalty = -2.0
    cfg.rl.group_normalize = False
    cfg.rl.normalize_advantage = False

    filtered_rollouts, advantages, reward_stats, adv_stats = _prepare_rewards_and_advantages(
        rollouts=_make_rollouts(),
        cfg=cfg,
        metricx_scorer=None,
        xcomet_scorer=None,
        mqm_scorer=_MQMFailureSecondScorer(),  # type: ignore[arg-type]
        esa_scorer=None,
        metricx_cache={},
        xcomet_cache={},
        mqm_cache={},
        esa_cache={},
        tokenizer=None,
    )

    assert [rollout.example_id for rollout in filtered_rollouts] == ["ex-0", "ex-1"]
    assert advantages == [[0.0, 0.0], [-2.0, -2.0]]
    assert reward_stats["mqm_failure_count"] == 1.0
    assert reward_stats["mqm_failure_penalty_total"] == -2.0
    assert reward_stats["mqm_failure_penalty_mean"] == -1.0
    assert adv_stats["raw_mean"] == pytest.approx(-1.0)


def test_prepare_rewards_and_advantages_drops_skipped_esa_rollouts() -> None:
    cfg = _base_cfg()
    cfg.reward.mqm.enabled = False
    cfg.reward.esa.enabled = True
    cfg.reward.w_mqm_seq = 0.0
    cfg.reward.w_esa_seq = 0.2

    filtered_rollouts, advantages, reward_stats, adv_stats = _prepare_rewards_and_advantages(
        rollouts=_make_rollouts(),
        cfg=cfg,
        metricx_scorer=None,
        xcomet_scorer=None,
        mqm_scorer=None,
        esa_scorer=_ESASkipSecondScorer(),  # type: ignore[arg-type]
        metricx_cache={},
        xcomet_cache={},
        mqm_cache={},
        esa_cache={},
        tokenizer=None,
    )

    assert [rollout.example_id for rollout in filtered_rollouts] == ["ex-0"]
    assert len(advantages) == 1
    assert reward_stats["esa_skipped_count"] == 1.0
    assert reward_stats["grpo_dropped_rollouts_count"] == 1.0
    assert adv_stats["raw_std"] == pytest.approx(0.0)


def test_prepare_rewards_and_advantages_handles_all_skipped_esa_rollouts() -> None:
    cfg = _base_cfg()
    cfg.reward.mqm.enabled = False
    cfg.reward.esa.enabled = True
    cfg.reward.w_mqm_seq = 0.0
    cfg.reward.w_esa_seq = 0.2

    filtered_rollouts, advantages, reward_stats, adv_stats = _prepare_rewards_and_advantages(
        rollouts=_make_rollouts(),
        cfg=cfg,
        metricx_scorer=None,
        xcomet_scorer=None,
        mqm_scorer=None,
        esa_scorer=_ESASkipAllScorer(),  # type: ignore[arg-type]
        metricx_cache={},
        xcomet_cache={},
        mqm_cache={},
        esa_cache={},
        tokenizer=None,
    )

    assert filtered_rollouts == []
    assert advantages == []
    assert reward_stats["esa_skipped_count"] == 2.0
    assert reward_stats["grpo_dropped_rollouts_count"] == 2.0
    assert adv_stats["raw_mean"] == pytest.approx(0.0)
