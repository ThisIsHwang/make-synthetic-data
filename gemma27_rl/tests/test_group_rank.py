from __future__ import annotations

import pytest

pytest.importorskip("torch")

from gemma27_rl.config import GroupRankConfig, RLPostTrainConfig
from gemma27_rl.rewards import (
    OpenAICompatibleGroupRankScorer,
    build_group_rank_messages,
    deduplicate_group_rank_candidates,
    parse_group_rank_response,
    tie_aware_centered_borda_rewards_from_unique_ranking,
)
from gemma27_rl.rl_types import GroupRankSample, Rollout
from gemma27_rl.trainer import _score_group_rank_rollouts


def _make_rollout(
    *,
    example_id: str,
    completion_text: str,
    prompt_instance_id: str,
    src_text: str = "source",
) -> Rollout:
    return Rollout(
        example_id=example_id,
        prompt_text="prompt",
        prompt_input_ids=[1, 2],
        completion_text=completion_text,
        completion_token_ids=[3, 4],
        old_logprobs=[-0.1, -0.1],
        ref_logprobs=None,
        token_char_offsets=[(0, 1), (1, 2)],
        src_text=src_text,
        src_lang="Korean",
        tgt_lang="English",
        ref_text="reference",
        prompt_instance_id=prompt_instance_id,
    )


def _group_rank_cfg(**overrides: object) -> GroupRankConfig:
    cfg = GroupRankConfig(enabled=True, use_fewshot=False)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_build_group_rank_messages_contains_numbered_candidates() -> None:
    messages = build_group_rank_messages(
        source_lang="Korean",
        target_lang="English",
        source_seg="원문",
        candidates=["번역 1", "번역 2"],
        ref="참고 번역",
        use_fewshot=False,
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "ranking judge for machine translation candidate comparison" in messages[0]["content"]
    assert "Candidate 1:" in messages[1]["content"]
    assert "Candidate 2:" in messages[1]["content"]
    assert "Reference (auxiliary only):" in messages[1]["content"]


def test_parse_group_rank_response_accepts_valid_json() -> None:
    ranking, critical_ids, reasons = parse_group_rank_response(
        """{
  "ranking": [2, 1, 3],
  "critical_candidate_ids": [3],
  "reasons": {
    "1": "slightly awkward",
    "3": "assistant fallback"
  }
}""",
        candidate_count=3,
    )

    assert ranking == [2, 1, 3]
    assert critical_ids == [3]
    assert reasons == {1: "slightly awkward", 3: "assistant fallback"}


def test_parse_group_rank_response_accepts_json_with_trailing_explanation() -> None:
    ranking, critical_ids, reasons = parse_group_rank_response(
        """{
  "ranking": [3, 1, 2],
  "critical_candidate_ids": [],
  "reasons": {
    "3": "Most faithful; captures the uncertainty ('maybe').",
    "4": "Candidate 4 is missing from the input list."
  }
}

Wait, I need to re-evaluate the title translation.""",
        candidate_count=3,
    )

    assert ranking == [3, 1, 2]
    assert critical_ids == []
    assert reasons == {3: "Most faithful; captures the uncertainty ('maybe')."}


def test_parse_group_rank_response_flattens_nested_reasons_and_ignores_aux_keys() -> None:
    ranking, critical_ids, reasons = parse_group_rank_response(
        """{
  "ranking": [2, 4, 1, 3],
  "critical_candidate_ids": [],
  "reasons": {
    "2": "Top-level reason for candidate 2.",
    "4": "Top-level reason for candidate 4.",
    "reasons": {
      "1": "Nested reason for candidate 1.",
      "3": "Nested reason for candidate 3.",
      "note": "Ignore this note.",
      "5": "Ignore this out-of-range candidate."
    }
  }
}""",
        candidate_count=4,
    )

    assert ranking == [2, 4, 1, 3]
    assert critical_ids == []
    assert reasons == {
        1: "Nested reason for candidate 1.",
        2: "Top-level reason for candidate 2.",
        3: "Nested reason for candidate 3.",
        4: "Top-level reason for candidate 4.",
    }


def test_parse_group_rank_response_repairs_unescaped_quotes_in_reason_strings() -> None:
    ranking, critical_ids, reasons = parse_group_rank_response(
        """{
  "ranking": [2, 1, 3],
  "critical_candidate_ids": [],
  "reasons": {
    "3": "Slightly less polished article formatting (as "School Sports Promotion Plan" vs as the "School Sports Promotion Plan") and uses law."
  }
}""",
        candidate_count=3,
    )

    assert ranking == [2, 1, 3]
    assert critical_ids == []
    assert '"School Sports Promotion Plan"' in reasons[3]


@pytest.mark.parametrize(
    "raw_text",
    [
        '{"ranking": [1, 1, 3], "critical_candidate_ids": [], "reasons": {}}',
        '{"ranking": [1, 3], "critical_candidate_ids": [], "reasons": {}}',
    ],
)
def test_parse_group_rank_response_rejects_duplicate_or_missing_candidate_ids(raw_text: str) -> None:
    with pytest.raises(ValueError, match="permutation|length mismatch"):
        _ = parse_group_rank_response(raw_text, candidate_count=3)


def test_deduplicate_group_rank_candidates_preserves_first_seen_order() -> None:
    unique_candidates, original_to_unique_idx, unique_to_original_indices, normalized_candidates = (
        deduplicate_group_rank_candidates(
            ["A", "B", "A", "C", "B"],
            normalization_mode="none",
        )
    )

    assert unique_candidates == ["A", "B", "C"]
    assert original_to_unique_idx == [0, 1, 0, 2, 1]
    assert unique_to_original_indices == [[0, 2], [1, 4], [3]]
    assert normalized_candidates == ["A", "B", "A", "C", "B"]


def test_deduplicate_group_rank_candidates_respects_normalization_mode() -> None:
    candidates = ["  ABC  ", "ABC", "ＡＢＣ"]

    none_unique, *_ = deduplicate_group_rank_candidates(candidates, normalization_mode="none")
    strip_unique, *_ = deduplicate_group_rank_candidates(candidates, normalization_mode="strip")
    nfkc_unique, *_ = deduplicate_group_rank_candidates(candidates, normalization_mode="strip_nfkc")

    assert none_unique == ["  ABC  ", "ABC", "ＡＢＣ"]
    assert strip_unique == ["  ABC  ", "ＡＢＣ"]
    assert nfkc_unique == ["  ABC  "]


def test_tie_aware_centered_borda_rewards_partial_duplicate_case() -> None:
    rewards = tie_aware_centered_borda_rewards_from_unique_ranking(
        unique_ranking_ids=[1, 2, 3],
        unique_to_original_indices=[[0, 1], [2], [3]],
        original_candidate_count=4,
    )

    assert rewards == pytest.approx([1.0, 1.0, -0.5, -1.5])
    assert sum(rewards) == pytest.approx(0.0)


def test_all_identical_group_skips_judge_and_returns_zero_rewards() -> None:
    call_count = 0

    def fake_predict(rows: list[list[dict[str, str]]]) -> list[str]:
        nonlocal call_count
        call_count += 1
        return ['{"ranking": [1], "critical_candidate_ids": [], "reasons": {}}']

    scorer = OpenAICompatibleGroupRankScorer(
        cfg=_group_rank_cfg(),
        predict_fn=fake_predict,
    )
    result = scorer.score_groups(
        [
            GroupRankSample(
                group_id="g1",
                src="source",
                candidates=["same", "same", "same", "same"],
                source_lang="Korean",
                target_lang="English",
            )
        ]
    )

    assert call_count == 0
    assert result["candidate_reward_rows"] == [[0.0, 0.0, 0.0, 0.0]]
    assert result["skipped_rows"] == [True]
    assert result["skip_reasons"] == ["all_candidates_identical_after_dedup"]


def test_group_rank_scorer_deduplicates_before_prompting() -> None:
    captured_rows: list[list[dict[str, str]]] = []

    def fake_predict(rows: list[list[dict[str, str]]]) -> list[str]:
        captured_rows.extend(rows)
        return ['{"ranking": [1, 2, 3], "critical_candidate_ids": [], "reasons": {}}']

    scorer = OpenAICompatibleGroupRankScorer(
        cfg=_group_rank_cfg(),
        predict_fn=fake_predict,
    )
    result = scorer.score_groups(
        [
            GroupRankSample(
                group_id="g1",
                src="source",
                candidates=["Apply to Knox.", "Apply to Knox.", "Deploy to Knox.", "Reflect to Knox."],
                source_lang="English",
                target_lang="Korean",
            )
        ]
    )

    prompt = captured_rows[0][-1]["content"]
    assert "Candidate 1:" in prompt
    assert "Candidate 2:" in prompt
    assert "Candidate 3:" in prompt
    assert "Candidate 4:" not in prompt
    assert prompt.count("Apply to Knox.") == 1
    assert result["candidate_reward_rows"][0] == pytest.approx([1.0, 1.0, -0.5, -1.5])


def test_group_rank_scorer_propagates_critical_flag_to_duplicate_class() -> None:
    scorer = OpenAICompatibleGroupRankScorer(
        cfg=_group_rank_cfg(critical_error_penalty=-1.0),
        predict_fn=lambda rows: [
            '{"ranking": [1, 2, 3], "critical_candidate_ids": [1], "reasons": {"1": "assistant fallback"}}'
        ],
    )
    result = scorer.score_groups(
        [
            GroupRankSample(
                group_id="g1",
                src="source",
                candidates=["same", "same", "alt", "worse"],
                source_lang="English",
                target_lang="Korean",
            )
        ]
    )

    assert result["candidate_reward_rows"][0] == pytest.approx([0.0, 0.0, -0.5, -1.5])
    assert result["critical_candidate_rows"][0] == [1, 2]
    assert result["reasons_rows"][0] == {1: "assistant fallback", 2: "assistant fallback"}


def test_group_rank_scorer_zero_failure_policy_returns_zero_reward_row() -> None:
    scorer = OpenAICompatibleGroupRankScorer(
        cfg=_group_rank_cfg(failure_policy="zero"),
        predict_fn=lambda rows: ["not-json"],
    )
    result = scorer.score_groups(
        [
            GroupRankSample(
                group_id="g1",
                src="source",
                candidates=["a", "b"],
                source_lang="English",
                target_lang="Korean",
            )
        ]
    )

    assert result["candidate_reward_rows"] == [[0.0, 0.0]]
    assert result["skipped_rows"] == [True]
    assert result["raw_outputs"] == ["not-json"]
    assert result["meta_rows"][0]["parse_failed"] is True


def test_score_group_rank_rollouts_maps_scores_back_to_rollout_order() -> None:
    cfg = RLPostTrainConfig()
    cfg.reward.group_rank.enabled = True
    cfg.reward.group_rank.use_fewshot = False
    cfg.generation.num_samples_per_prompt = 3
    scorer = OpenAICompatibleGroupRankScorer(
        cfg=cfg.reward.group_rank,
        predict_fn=lambda rows: [
            '{"ranking": [2, 1], "critical_candidate_ids": [], "reasons": {}}',
            '{"ranking": [2, 1], "critical_candidate_ids": [], "reasons": {}}',
        ],
    )
    rollouts = [
        _make_rollout(example_id="a1", completion_text="A1", prompt_instance_id="group-a"),
        _make_rollout(example_id="a2", completion_text="A2", prompt_instance_id="group-a"),
        _make_rollout(example_id="b1", completion_text="B1", prompt_instance_id="group-b"),
        _make_rollout(example_id="b2", completion_text="B1", prompt_instance_id="group-b"),
        _make_rollout(example_id="b3", completion_text="B2", prompt_instance_id="group-b"),
    ]
    scores, _ = _score_group_rank_rollouts(
        rollouts=rollouts,
        span_reward_texts=["A1", "A2", "B1", "B1", "B2"],
        cfg=cfg,
        group_rank_scorer=scorer,
    )

    assert scores == pytest.approx([-0.5, 0.5, -0.5, -0.5, 1.0])


def test_score_group_rank_rollouts_reports_duplicate_stats() -> None:
    cfg = RLPostTrainConfig()
    cfg.reward.group_rank.enabled = True
    cfg.reward.group_rank.use_fewshot = False
    cfg.generation.num_samples_per_prompt = 3
    scorer = OpenAICompatibleGroupRankScorer(
        cfg=cfg.reward.group_rank,
        predict_fn=lambda rows: ['{"ranking": [2, 1], "critical_candidate_ids": [], "reasons": {}}'],
    )
    rollouts = [
        _make_rollout(example_id="a1", completion_text="A1", prompt_instance_id="group-a"),
        _make_rollout(example_id="a2", completion_text="A1", prompt_instance_id="group-a"),
        _make_rollout(example_id="a3", completion_text="A2", prompt_instance_id="group-a"),
        _make_rollout(example_id="b1", completion_text="B1", prompt_instance_id="group-b"),
        _make_rollout(example_id="b2", completion_text="B1", prompt_instance_id="group-b"),
    ]
    _, stats = _score_group_rank_rollouts(
        rollouts=rollouts,
        span_reward_texts=["A1", "A1", "A2", "B1", "B1"],
        cfg=cfg,
        group_rank_scorer=scorer,
    )

    assert stats["group_rank_group_count"] == pytest.approx(2.0)
    assert stats["group_rank_scored_group_count"] == pytest.approx(1.0)
    assert stats["group_rank_skipped_group_count"] == pytest.approx(1.0)
    assert stats["group_rank_duplicate_group_count"] == pytest.approx(2.0)
    assert stats["group_rank_duplicate_candidate_count_total"] == pytest.approx(2.0)
    assert stats["group_rank_all_duplicate_group_count"] == pytest.approx(1.0)
    assert stats["group_rank_unique_candidate_count_mean"] == pytest.approx(1.5)
    assert stats["group_rank_unique_candidate_count_min"] == pytest.approx(1.0)
    assert stats["group_rank_unique_candidate_count_max"] == pytest.approx(2.0)
