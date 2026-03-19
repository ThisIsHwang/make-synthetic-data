from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from gemma27_rl import eval as eval_mod
from gemma27_rl import trainer as trainer_mod
from gemma27_rl.config import RLPostTrainConfig
from gemma27_rl.rl_types import Example, RewardOutput, Rollout


class _DummyTokenizer:
    def __init__(self) -> None:
        self.all_special_tokens = ["<bos>", "<|assistant|>"]
        self.additional_special_tokens = ["<|assistant|>"]
        self.special_tokens_map = {
            "bos_token": "<bos>",
            "additional_special_tokens": ["<|assistant|>"],
        }


class _MQMRecorder:
    cfg = SimpleNamespace(use_reference=False)

    def __init__(self, scores_by_src: dict[str, float]) -> None:
        self._scores_by_src = dict(scores_by_src)
        self.calls: list[list[str]] = []

    def score_batch(self, samples):  # type: ignore[no-untyped-def]
        self.calls.append([str(sample.src) for sample in samples])
        return RewardOutput(
            sequence_scores=[float(self._scores_by_src[str(sample.src)]) for sample in samples],
            metadata={
                "error_spans": [[] for _ in samples],
                "skipped_rows": [False for _ in samples],
            },
        )


class _ESARecorder:
    def __init__(self, scores_by_src: dict[str, float]) -> None:
        self._scores_by_src = dict(scores_by_src)
        self.calls: list[list[str]] = []

    def score_batch(self, samples):  # type: ignore[no-untyped-def]
        self.calls.append([str(sample.src) for sample in samples])
        return RewardOutput(
            sequence_scores=[float(self._scores_by_src[str(sample.src)]) for sample in samples],
            metadata={
                "skipped_rows": [False for _ in samples],
            },
        )


def _cfg() -> RLPostTrainConfig:
    cfg = RLPostTrainConfig()
    cfg.reward.metricx.enabled = False
    cfg.reward.xcomet.enabled = False
    cfg.reward.mqm.enabled = True
    cfg.reward.esa.enabled = False
    cfg.reward.cache_enabled = False
    cfg.reward.w_mqm_seq = 1.0
    cfg.rl.group_normalize = False
    cfg.generation.num_samples_per_prompt = 1
    return cfg


def _examples() -> list[Example]:
    return [
        Example(
            example_id="ex-0",
            src_text="ex-0",
            src_lang="English",
            tgt_lang="Korean",
            src_lang_code="en",
            tgt_lang_code="ko",
            ref_text="ref-0",
        ),
        Example(
            example_id="ex-1",
            src_text="ex-1",
            src_lang="English",
            tgt_lang="Korean",
            src_lang_code="en",
            tgt_lang_code="ko",
            ref_text="ref-1",
        ),
    ]


def _rollout(example: Example, *, prompt_instance_id: str | None = None) -> Rollout:
    return Rollout(
        example_id=example.example_id,
        prompt_text=f"prompt::{example.example_id}",
        prompt_input_ids=[1, 2],
        completion_text=f"completion::{example.example_id}",
        completion_token_ids=[3, 4],
        old_logprobs=[-0.1, -0.2],
        ref_logprobs=None,
        token_char_offsets=[(0, 1), (1, 2)],
        src_text=example.src_text,
        src_lang=example.src_lang,
        tgt_lang=example.tgt_lang,
        src_lang_code=example.src_lang_code,
        tgt_lang_code=example.tgt_lang_code,
        ref_text=example.ref_text,
        raw_completion_token_ids=[3, 4],
        completion_raw_text=f"completion::{example.example_id}",
        completion_clean_text=f"completion::{example.example_id}",
        prompt_instance_id=prompt_instance_id,
    )


def test_training_pipeline_chunks_rollouts_and_merges_scores(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    cfg = _cfg()
    examples = _examples()
    prompt_instance_ids = [f"prompt-{idx}" for idx in range(len(examples))]
    scorer = _MQMRecorder({"ex-0": -1.0, "ex-1": -3.0})
    rollout_calls: list[list[str]] = []

    def _fake_generate_rollouts(**kwargs):  # type: ignore[no-untyped-def]
        chunk_examples = list(kwargs["examples"])
        chunk_prompt_ids = list(kwargs.get("prompt_instance_ids") or [])
        rollout_calls.append([str(example.example_id) for example in chunk_examples])
        return [
            _rollout(example, prompt_instance_id=(chunk_prompt_ids[idx] if idx < len(chunk_prompt_ids) else None))
            for idx, example in enumerate(chunk_examples)
        ]

    monkeypatch.setenv("GEMMA27_RL_ROLLOUT_PIPELINE_CHUNK", "1")
    monkeypatch.setattr(trainer_mod, "generate_rollouts", _fake_generate_rollouts)

    rollouts, advantages, reward_stats, _ = trainer_mod._prepare_training_batch_rollouts_and_advantages(
        batch_examples=examples,
        prompt_instance_ids=prompt_instance_ids,
        update_idx=7,
        policy_model=object(),
        tokenizer=None,
        cfg=cfg,
        device="cpu",
        metricx_scorer=None,
        xcomet_scorer=None,
        mqm_scorer=scorer,  # type: ignore[arg-type]
        esa_scorer=None,
        group_rank_scorer=None,
        metricx_cache={},
        xcomet_cache={},
        mqm_cache={},
        esa_cache={},
    )

    assert rollout_calls == [["ex-0"], ["ex-1"]]
    assert scorer.calls == [["ex-0"], ["ex-1"]]
    assert [rollout.example_id for rollout in rollouts] == ["ex-0", "ex-1"]
    assert [rollout.prompt_instance_id for rollout in rollouts] == prompt_instance_ids
    assert len(advantages) == 2
    assert reward_stats["mqm_score_mean"] == pytest.approx(-2.0)


def test_eval_pipeline_chunks_rollouts_and_merges_scores(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    cfg = _cfg()
    examples = _examples()
    scorer = _MQMRecorder({"ex-0": -1.0, "ex-1": -3.0})
    rollout_calls: list[list[str]] = []

    def _fake_generate_rollouts(**kwargs):  # type: ignore[no-untyped-def]
        chunk_examples = list(kwargs["examples"])
        rollout_calls.append([str(example.example_id) for example in chunk_examples])
        return [_rollout(example) for example in chunk_examples]

    monkeypatch.setenv("GEMMA27_RL_ROLLOUT_PIPELINE_CHUNK", "1")
    monkeypatch.setattr(eval_mod, "generate_rollouts", _fake_generate_rollouts)

    report = eval_mod.evaluate_on_dataset(
        examples=examples,
        policy_model=object(),
        tokenizer=_DummyTokenizer(),
        cfg=cfg,
        device="cpu",
        mqm_scorer=scorer,  # type: ignore[arg-type]
        collect_outputs=True,
    )

    assert rollout_calls == [["ex-0"], ["ex-1"]]
    assert scorer.calls == [["ex-0"], ["ex-1"]]
    assert report["mqm_score_mean"] == pytest.approx(-2.0)
    assert [row["example_id"] for row in report["eval_rows"]] == ["ex-0", "ex-1"]


def test_eval_pipeline_can_use_esa_without_training_esa_reward(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    cfg = _cfg()
    cfg.reward.mqm.enabled = False
    cfg.reward.esa.enabled = False
    cfg.eval.use_esa = True
    scorer = _ESARecorder({"ex-0": 81.0, "ex-1": 79.0})
    rollout_calls: list[list[str]] = []

    def _fake_generate_rollouts(**kwargs):  # type: ignore[no-untyped-def]
        chunk_examples = list(kwargs["examples"])
        rollout_calls.append([str(example.example_id) for example in chunk_examples])
        return [_rollout(example) for example in chunk_examples]

    monkeypatch.setenv("GEMMA27_RL_ROLLOUT_PIPELINE_CHUNK", "1")
    monkeypatch.setattr(eval_mod, "generate_rollouts", _fake_generate_rollouts)

    report = eval_mod.evaluate_on_dataset(
        examples=_examples(),
        policy_model=object(),
        tokenizer=_DummyTokenizer(),
        cfg=cfg,
        device="cpu",
        esa_scorer=scorer,  # type: ignore[arg-type]
        collect_outputs=True,
    )

    assert rollout_calls == [["ex-0"], ["ex-1"]]
    assert scorer.calls == [["ex-0"], ["ex-1"]]
    assert report["esa_score_mean"] == pytest.approx(80.0)
    assert [row["esa_score"] for row in report["eval_rows"]] == [81.0, 79.0]
