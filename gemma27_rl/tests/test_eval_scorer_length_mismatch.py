from __future__ import annotations

from gemma27_rl import eval as eval_mod
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


class _MetricXShortScorer:
    def score_batch(self, samples):  # type: ignore[no-untyped-def]
        del samples
        return RewardOutput(sequence_scores=[1.0], metadata={})


class _XCometSpanMismatchScorer:
    def score_batch(self, samples):  # type: ignore[no-untyped-def]
        del samples
        return RewardOutput(
            sequence_scores=[0.2, 0.3],
            metadata={"error_spans": [[{"start": 0, "end": 1, "severity": "MINOR"}]]},
        )


class _MQMShortScorer:
    def score_batch(self, samples):  # type: ignore[no-untyped-def]
        del samples
        return RewardOutput(
            sequence_scores=[-1.0],
            metadata={"error_spans": [[], []]},
        )


class _ESAShortScorer:
    def score_batch(self, samples):  # type: ignore[no-untyped-def]
        del samples
        return RewardOutput(sequence_scores=[5.0], metadata={})


def _base_cfg() -> RLPostTrainConfig:
    cfg = RLPostTrainConfig()
    cfg.reward.metricx.enabled = False
    cfg.reward.xcomet.enabled = False
    cfg.reward.mqm.enabled = False
    cfg.reward.esa.enabled = False
    return cfg


def _base_examples() -> list[Example]:
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
    ]


def _base_rollouts() -> list[Rollout]:
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
            ref_text=None,
        ),
    ]


def test_evaluate_raises_on_metricx_sequence_length_mismatch(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    cfg = _base_cfg()
    cfg.reward.metricx.enabled = True
    monkeypatch.setattr(eval_mod, "generate_rollouts", lambda **_: _base_rollouts())  # type: ignore[assignment]

    try:
        eval_mod.evaluate_on_dataset(
            examples=_base_examples(),
            policy_model=object(),
            tokenizer=_DummyTokenizer(),
            cfg=cfg,
            device="cpu",
            metricx_scorer=_MetricXShortScorer(),  # type: ignore[arg-type]
            collect_outputs=True,
        )
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "MetricX scorer returned mismatched sequence_scores length" in str(exc)


def test_evaluate_raises_on_xcomet_span_length_mismatch(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    cfg = _base_cfg()
    cfg.reward.xcomet.enabled = True
    monkeypatch.setattr(eval_mod, "generate_rollouts", lambda **_: _base_rollouts())  # type: ignore[assignment]

    try:
        eval_mod.evaluate_on_dataset(
            examples=_base_examples(),
            policy_model=object(),
            tokenizer=_DummyTokenizer(),
            cfg=cfg,
            device="cpu",
            xcomet_scorer=_XCometSpanMismatchScorer(),  # type: ignore[arg-type]
            collect_outputs=True,
        )
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "xCOMET scorer returned mismatched error_spans length" in str(exc)


def test_evaluate_raises_on_mqm_sequence_length_mismatch(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    cfg = _base_cfg()
    cfg.reward.mqm.enabled = True
    monkeypatch.setattr(eval_mod, "generate_rollouts", lambda **_: _base_rollouts())  # type: ignore[assignment]

    try:
        eval_mod.evaluate_on_dataset(
            examples=_base_examples(),
            policy_model=object(),
            tokenizer=_DummyTokenizer(),
            cfg=cfg,
            device="cpu",
            mqm_scorer=_MQMShortScorer(),  # type: ignore[arg-type]
            collect_outputs=True,
        )
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "MQM scorer returned mismatched sequence_scores length" in str(exc)


def test_evaluate_raises_on_esa_sequence_length_mismatch(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    cfg = _base_cfg()
    cfg.reward.esa.enabled = True
    monkeypatch.setattr(eval_mod, "generate_rollouts", lambda **_: _base_rollouts())  # type: ignore[assignment]

    try:
        eval_mod.evaluate_on_dataset(
            examples=_base_examples(),
            policy_model=object(),
            tokenizer=_DummyTokenizer(),
            cfg=cfg,
            device="cpu",
            esa_scorer=_ESAShortScorer(),  # type: ignore[arg-type]
            collect_outputs=True,
        )
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "ESA scorer returned mismatched sequence_scores length" in str(exc)
