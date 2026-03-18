from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from gemma27_rl.config import XCometConfig
from gemma27_rl.rewards import XCometXLScorer, extract_error_spans
from gemma27_rl.rl_types import RewardOutput, SampleForScoring
from gemma27_rl.trainer import (
    _score_with_cache_esa,
    _score_with_cache_metricx,
    _score_with_cache_mqm,
    _score_with_cache_xcomet,
)


def _samples() -> list[SampleForScoring]:
    return [
        SampleForScoring(src="s1", mt="m1", ref="r1"),
        SampleForScoring(src="s2", mt="m2", ref="r2"),
    ]


class _MetricXShortScorer:
    cfg = SimpleNamespace(use_reference=False)

    def score_batch(self, samples):  # type: ignore[no-untyped-def]
        del samples
        return RewardOutput(sequence_scores=[0.1], metadata={})


class _ESAShortScorer:
    cfg = SimpleNamespace(use_reference=False)

    def score_batch(self, samples):  # type: ignore[no-untyped-def]
        del samples
        return RewardOutput(sequence_scores=[1.0], metadata={})


class _XCometSpanMismatchScorer:
    cfg = SimpleNamespace(use_reference=False)

    def score_batch(self, samples):  # type: ignore[no-untyped-def]
        del samples
        return RewardOutput(
            sequence_scores=[0.2, 0.3],
            metadata={"error_spans": [[{"start": 0, "end": 1, "severity": "MINOR"}]]},
        )


class _MQMScoreMismatchScorer:
    cfg = SimpleNamespace(use_reference=False)

    def score_batch(self, samples):  # type: ignore[no-untyped-def]
        del samples
        return RewardOutput(
            sequence_scores=[-1.0],
            metadata={"error_spans": [[], []]},
        )


class _MQMFailureThenSuccessScorer:
    cfg = SimpleNamespace(use_reference=False)

    def __init__(self) -> None:
        self.calls = 0

    def score_batch(self, samples):  # type: ignore[no-untyped-def]
        assert len(samples) == 1
        self.calls += 1
        failure = self.calls == 1
        return RewardOutput(
            sequence_scores=[-25.0 if failure else -5.0],
            metadata={
                "error_spans": [[]],
                "unanchored_errors": [[]],
                "skipped_rows": [False],
                "failure_rows": [failure],
            },
        )


class _MQMUnanchoredMismatchScorer:
    cfg = SimpleNamespace(use_reference=False)

    def score_batch(self, samples):  # type: ignore[no-untyped-def]
        del samples
        return RewardOutput(
            sequence_scores=[-1.0, -2.0],
            metadata={
                "error_spans": [[], []],
                "unanchored_errors": [[]],
            },
        )


def test_metricx_cache_helper_raises_on_sequence_length_mismatch() -> None:
    with pytest.raises(RuntimeError, match="MetricX scorer returned mismatched sequence_scores length"):
        _ = _score_with_cache_metricx(
            samples=_samples(),
            scorer=_MetricXShortScorer(),  # type: ignore[arg-type]
            cache={},
            use_cache=False,
        )


def test_esa_cache_helper_raises_on_sequence_length_mismatch() -> None:
    with pytest.raises(RuntimeError, match="ESA scorer returned mismatched sequence_scores length"):
        _ = _score_with_cache_esa(
            samples=_samples(),
            scorer=_ESAShortScorer(),  # type: ignore[arg-type]
            cache={},
            use_cache=False,
        )


def test_xcomet_cache_helper_raises_on_span_length_mismatch() -> None:
    with pytest.raises(RuntimeError, match="xCOMET scorer returned mismatched error_spans length"):
        _ = _score_with_cache_xcomet(
            samples=_samples(),
            scorer=_XCometSpanMismatchScorer(),  # type: ignore[arg-type]
            cache={},
            use_cache=False,
        )


def test_extract_error_spans_raises_on_metadata_list_length_mismatch() -> None:
    with pytest.raises(ValueError, match="xCOMET metadata returned mismatched error_spans length"):
        _ = extract_error_spans(
            metadata=[{"error_spans": [{"start": 0, "end": 1, "severity": "MINOR"}]}],
            expected=2,
        )


def test_extract_error_spans_raises_on_unbatched_spans_for_batched_output() -> None:
    with pytest.raises(ValueError, match="xCOMET metadata returned mismatched error_spans length"):
        _ = extract_error_spans(
            metadata={"error_spans": [{"start": 0, "end": 1, "severity": "MINOR"}]},
            expected=2,
        )


def test_xcomet_worker_raises_on_span_length_mismatch_directly() -> None:
    class _FakeWorker:
        def request(self, payload):  # type: ignore[no-untyped-def]
            assert payload["type"] == "score"
            return {
                "ok": True,
                "scores": [0.2, 0.3],
                "error_spans": [[{"start": 0, "end": 1, "severity": "MINOR"}]],
            }

        def close(self) -> None:
            return None

    scorer = XCometXLScorer(cfg=XCometConfig(enabled=True), predict_fn=lambda payload: None)
    scorer._worker = _FakeWorker()  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="xCOMET worker returned mismatched error_spans length"):
        _ = scorer.score_batch(_samples())


def test_xcomet_predict_raises_on_span_length_mismatch_directly() -> None:
    scorer = XCometXLScorer(
        cfg=XCometConfig(enabled=True),
        predict_fn=lambda payload: {
            "scores": [0.2, 0.3],
            "metadata": {"error_spans": [{"start": 0, "end": 1, "severity": "MINOR"}]},
        },
    )

    with pytest.raises(ValueError, match="xCOMET prediction returned mismatched error_spans length"):
        _ = scorer.score_batch(_samples())


def test_mqm_cache_helper_raises_on_sequence_length_mismatch() -> None:
    with pytest.raises(RuntimeError, match="MQM scorer returned mismatched sequence_scores length"):
        _ = _score_with_cache_mqm(
            samples=_samples(),
            scorer=_MQMScoreMismatchScorer(),  # type: ignore[arg-type]
            cache={},
            use_cache=False,
        )


def test_mqm_cache_helper_does_not_cache_failure_rows() -> None:
    scorer = _MQMFailureThenSuccessScorer()
    cache: dict[tuple[str, str, str, str, str], tuple[float, list[dict[str, object]], list[dict[str, object]]]] = {}
    sample = [SampleForScoring(src="s1", mt="m1", ref="r1")]

    first_scores, _, _, first_failures, first_unanchored = _score_with_cache_mqm(
        samples=sample,
        scorer=scorer,  # type: ignore[arg-type]
        cache=cache,
        use_cache=True,
    )
    second_scores, _, _, second_failures, second_unanchored = _score_with_cache_mqm(
        samples=sample,
        scorer=scorer,  # type: ignore[arg-type]
        cache=cache,
        use_cache=True,
    )

    assert first_scores == [-25.0]
    assert first_failures == [True]
    assert first_unanchored == [[]]
    assert second_scores == [-5.0]
    assert second_failures == [False]
    assert second_unanchored == [[]]
    assert scorer.calls == 2
    assert len(cache) == 1


def test_mqm_cache_helper_raises_on_unanchored_length_mismatch() -> None:
    with pytest.raises(RuntimeError, match="MQM scorer returned mismatched unanchored_errors length"):
        _ = _score_with_cache_mqm(
            samples=_samples(),
            scorer=_MQMUnanchoredMismatchScorer(),  # type: ignore[arg-type]
            cache={},
            use_cache=False,
        )
