from __future__ import annotations

from gemma27_rl import eval as eval_mod
from gemma27_rl.config import RLPostTrainConfig
from gemma27_rl.rl_types import Example, RewardOutput, Rollout
from gemma27_rl.trainer import _compute_eval_selection_score


class _MetricXStub:
    def score_batch(self, samples):  # type: ignore[no-untyped-def]
        assert len(samples) == 3
        return RewardOutput(sequence_scores=[1.0, 1.0, 5.0], metadata={})


def _cfg() -> RLPostTrainConfig:
    cfg = RLPostTrainConfig()
    cfg.reward.metricx.enabled = True
    cfg.reward.xcomet.enabled = False
    cfg.reward.mqm.enabled = False
    cfg.reward.esa.enabled = False
    cfg.reward.w_metricx = 1.0
    return cfg


def _examples() -> list[Example]:
    return [
        Example(
            example_id="ex-1",
            src_text="hello-1",
            src_lang="English",
            tgt_lang="Korean",
            src_lang_code="en",
            tgt_lang_code="ko",
            ref_text=None,
        ),
        Example(
            example_id="ex-2",
            src_text="hello-2",
            src_lang="English",
            tgt_lang="Korean",
            src_lang_code="en",
            tgt_lang_code="ko",
            ref_text=None,
        ),
        Example(
            example_id="ex-3",
            src_text="안녕하세요",
            src_lang="Korean",
            tgt_lang="English",
            src_lang_code="ko",
            tgt_lang_code="en",
            ref_text=None,
        ),
    ]


def _rollouts() -> list[Rollout]:
    return [
        Rollout(
            example_id="ex-1",
            prompt_text="p1",
            prompt_input_ids=[1],
            completion_text="c1",
            completion_token_ids=[2],
            old_logprobs=[],
            ref_logprobs=None,
            token_char_offsets=[],
            src_text="hello-1",
            src_lang="English",
            tgt_lang="Korean",
            src_lang_code="en",
            tgt_lang_code="ko",
            ref_text=None,
        ),
        Rollout(
            example_id="ex-2",
            prompt_text="p2",
            prompt_input_ids=[1],
            completion_text="c2",
            completion_token_ids=[2],
            old_logprobs=[],
            ref_logprobs=None,
            token_char_offsets=[],
            src_text="hello-2",
            src_lang="English",
            tgt_lang="Korean",
            src_lang_code="en",
            tgt_lang_code="ko",
            ref_text=None,
        ),
        Rollout(
            example_id="ex-3",
            prompt_text="p3",
            prompt_input_ids=[1],
            completion_text="c3",
            completion_token_ids=[2],
            old_logprobs=[],
            ref_logprobs=None,
            token_char_offsets=[],
            src_text="안녕하세요",
            src_lang="Korean",
            tgt_lang="English",
            src_lang_code="ko",
            tgt_lang_code="en",
            ref_text=None,
        ),
    ]


def test_evaluate_reports_direction_metrics_and_rows(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(eval_mod, "generate_rollouts", lambda **_: _rollouts())  # type: ignore[assignment]

    report = eval_mod.evaluate_on_dataset(
        examples=_examples(),
        policy_model=object(),
        tokenizer=None,
        cfg=_cfg(),
        device="cpu",
        metricx_scorer=_MetricXStub(),  # type: ignore[arg-type]
        collect_outputs=True,
    )

    assert set(report["direction_metrics"].keys()) == {"en->ko", "ko->en"}
    assert report["direction_metrics"]["en->ko"]["metricx_score_mean"] == 1.0
    assert report["direction_metrics"]["ko->en"]["metricx_score_mean"] == 5.0
    assert [row["direction"] for row in report["eval_rows"]] == ["en->ko", "en->ko", "ko->en"]


def test_eval_selection_averages_direction_scores() -> None:
    cfg = _cfg()
    report = {
        "metricx_reward_mean": 2.6666666667,
        "direction_metrics": {
            "en->ko": {"metricx_reward_mean": 4.0},
            "ko->en": {"metricx_reward_mean": 0.0},
        },
    }

    score = _compute_eval_selection_score(report, cfg)

    assert score == 2.0
