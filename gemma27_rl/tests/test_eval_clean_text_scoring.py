from __future__ import annotations

from gemma27_rl import eval as eval_mod
from gemma27_rl.config import RLPostTrainConfig
from gemma27_rl.prompting import sanitize_text_for_scoring
from gemma27_rl.rl_types import Example, RewardOutput, Rollout


class _DummyTokenizer:
    def __init__(self) -> None:
        self.all_special_tokens = ["<bos>", "<|assistant|>"]
        self.additional_special_tokens = ["<|assistant|>"]
        self.special_tokens_map = {
            "bos_token": "<bos>",
            "additional_special_tokens": ["<|assistant|>"],
        }


class _MetricXStub:
    def __init__(self) -> None:
        self.seen_mts: list[str] = []

    def score_batch(self, samples):  # type: ignore[no-untyped-def]
        self.seen_mts = [str(sample.mt) for sample in samples]
        return RewardOutput(sequence_scores=[1.0 for _ in samples], metadata={})


def _base_cfg() -> RLPostTrainConfig:
    cfg = RLPostTrainConfig()
    cfg.reward.metricx.enabled = True
    cfg.reward.xcomet.enabled = False
    cfg.reward.mqm.enabled = False
    cfg.reward.esa.enabled = False
    return cfg


def _base_example() -> Example:
    return Example(
        example_id="ex-1",
        src_text="hello",
        src_lang="English",
        tgt_lang="Korean",
        src_lang_code="en",
        tgt_lang_code="ko",
        ref_text=None,
    )


def _base_rollout(*, completion_text: str, completion_raw_text: str | None, completion_clean_text: str | None) -> Rollout:
    return Rollout(
        example_id="ex-1",
        prompt_text="p",
        prompt_input_ids=[1],
        completion_text=completion_text,
        completion_token_ids=[2, 3],
        old_logprobs=[0.0, 0.0],
        ref_logprobs=None,
        token_char_offsets=[(0, 1), (1, 2)],
        src_text="hello",
        ref_text=None,
        raw_completion_token_ids=[2, 3],
        completion_raw_text=completion_raw_text,
        completion_clean_text=completion_clean_text,
    )


def test_evaluate_uses_completion_clean_text_for_scorers(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    rollout = _base_rollout(
        completion_text="<bos>raw<think>hidden</think><|assistant|>",
        completion_raw_text="<bos>raw<think>hidden</think><|assistant|>",
        completion_clean_text="clean output",
    )
    monkeypatch.setattr(eval_mod, "generate_rollouts", lambda **_: [rollout])  # type: ignore[assignment]

    scorer = _MetricXStub()
    report = eval_mod.evaluate_on_dataset(
        examples=[_base_example()],
        policy_model=object(),  # unused by mocked generate_rollouts
        tokenizer=_DummyTokenizer(),
        cfg=_base_cfg(),
        device="cpu",
        metricx_scorer=scorer,
        collect_outputs=True,
    )

    assert scorer.seen_mts == ["clean output"]
    assert report["eval_rows"][0]["completion_raw_text"] == "<bos>raw<think>hidden</think><|assistant|>"
    assert report["eval_rows"][0]["completion_clean_text"] == "clean output"


def test_evaluate_sanitizes_raw_text_when_completion_clean_text_missing(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    raw_text = "<bos>raw<think>hidden</think><|assistant|>"
    rollout = _base_rollout(
        completion_text=raw_text,
        completion_raw_text=None,
        completion_clean_text=None,
    )
    monkeypatch.setattr(eval_mod, "generate_rollouts", lambda **_: [rollout])  # type: ignore[assignment]

    tokenizer = _DummyTokenizer()
    expected_clean, _ = sanitize_text_for_scoring(raw_text, special_tokens=tokenizer.all_special_tokens)
    scorer = _MetricXStub()
    report = eval_mod.evaluate_on_dataset(
        examples=[_base_example()],
        policy_model=object(),  # unused by mocked generate_rollouts
        tokenizer=tokenizer,
        cfg=_base_cfg(),
        device="cpu",
        metricx_scorer=scorer,
        collect_outputs=True,
    )

    assert scorer.seen_mts == [expected_clean]
    assert "<think>" not in scorer.seen_mts[0].lower()
    assert "</think>" not in scorer.seen_mts[0].lower()
    assert "<bos>" not in scorer.seen_mts[0]
    assert "<|assistant|>" not in scorer.seen_mts[0]
    assert report["eval_rows"][0]["completion_clean_text"] == expected_clean
