from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

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


def test_mqm_cache_helper_raises_on_sequence_length_mismatch() -> None:
    with pytest.raises(RuntimeError, match="MQM scorer returned mismatched sequence_scores length"):
        _ = _score_with_cache_mqm(
            samples=_samples(),
            scorer=_MQMScoreMismatchScorer(),  # type: ignore[arg-type]
            cache={},
            use_cache=False,
        )
