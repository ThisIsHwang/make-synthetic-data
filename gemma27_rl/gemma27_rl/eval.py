from __future__ import annotations

from collections import Counter
from copy import deepcopy
import logging
import math
from statistics import mean
from typing import Any

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .config import GenerationConfig, RLPostTrainConfig
from .rewards import OpenAICompatibleMQMScorer, MetricXQEScorer, XCometXLScorer, metricx_score_to_reward
from .rollout import generate_rollouts
from .types import Example, SampleForScoring

logger = logging.getLogger(__name__)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    m = mean(values)
    var = sum((v - m) ** 2 for v in values) / len(values)
    return float(m), float(var**0.5)


def evaluate_on_dataset(
    examples: list[Example],
    policy_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    cfg: RLPostTrainConfig,
    device: str,
    metricx_scorer: MetricXQEScorer | None = None,
    xcomet_scorer: XCometXLScorer | None = None,
    mqm_scorer: OpenAICompatibleMQMScorer | None = None,
    collect_outputs: bool = False,
) -> dict[str, Any]:
    if not examples:
        empty = {
            "metricx_score_mean": 0.0,
            "metricx_reward_mean": 0.0,
            "xcomet_score_mean": 0.0,
            "mqm_score_mean": 0.0,
            "avg_span_count": 0.0,
            "severity_counts": {},
            "avg_completion_len": 0.0,
        }
        if collect_outputs:
            empty["eval_rows"] = []
        return empty

    gen_cfg = deepcopy(cfg.generation)
    gen_cfg.num_samples_per_prompt = 1
    gen_cfg.do_sample = False
    gen_cfg.temperature = 0.0

    rollouts = generate_rollouts(
        examples=examples,
        policy_model=policy_model,
        tokenizer=tokenizer,
        gen_cfg=gen_cfg,
        device=device,
        ref_model=None,
        prompt_template=cfg.prompt.template,
    )

    samples = [SampleForScoring(src=r.src_text, mt=r.completion_text, ref=r.ref_text) for r in rollouts]

    metricx_scores = [0.0 for _ in rollouts]
    metricx_rewards = [0.0 for _ in rollouts]
    if metricx_scorer is not None and cfg.reward.metricx.enabled:
        metricx_scores = metricx_scorer.score_batch(samples).sequence_scores
        non_finite_idx = [idx for idx, value in enumerate(metricx_scores) if not math.isfinite(float(value))]
        if non_finite_idx:
            msg = (
                f"MetricX produced {len(non_finite_idx)} non-finite eval scores "
                f"(model={cfg.reward.metricx.model_name}, device={cfg.reward.metricx.device}, "
                f"dtype={cfg.reward.metricx.dtype})."
            )
            if cfg.reward.metricx.overflow_policy == "skip":
                logger.warning(
                    "%s Replacing them with fallback offset %.4f due to overflow_policy=skip.",
                    msg,
                    cfg.reward.metricx.offset,
                )
                for idx in non_finite_idx:
                    metricx_scores[idx] = float(cfg.reward.metricx.offset)
            else:
                raise RuntimeError(
                    f"{msg} Eval aborted to avoid silently recording invalid MetricX values."
                )
        metricx_rewards = [metricx_score_to_reward(v, offset=cfg.reward.metricx.offset) for v in metricx_scores]

    xcomet_scores = [0.0 for _ in rollouts]
    spans: list[list[dict[str, Any]]] = [[] for _ in rollouts]
    if xcomet_scorer is not None and cfg.reward.xcomet.enabled:
        xcomet_out = xcomet_scorer.score_batch(samples)
        xcomet_scores = xcomet_out.sequence_scores
        non_finite_idx = [idx for idx, value in enumerate(xcomet_scores) if not math.isfinite(float(value))]
        if non_finite_idx:
            logger.warning(
                "xCOMET produced %s non-finite eval scores; replacing with 0.0.",
                len(non_finite_idx),
            )
            for idx in non_finite_idx:
                xcomet_scores[idx] = 0.0
        meta = xcomet_out.metadata or {}
        spans = meta.get("error_spans", spans)

    mqm_scores = [0.0 for _ in rollouts]
    mqm_spans: list[list[dict[str, Any]]] = [[] for _ in rollouts]
    if mqm_scorer is not None and cfg.reward.mqm.enabled:
        mqm_out = mqm_scorer.score_batch(samples)
        mqm_scores = mqm_out.sequence_scores
        mqm_meta = mqm_out.metadata or {}
        mqm_spans = mqm_meta.get("error_spans", mqm_spans)
        non_finite_idx = [idx for idx, value in enumerate(mqm_scores) if not math.isfinite(float(value))]
        if non_finite_idx:
            logger.warning(
                "MQM scorer produced %s non-finite eval scores; replacing with 0.0.",
                len(non_finite_idx),
            )
            for idx in non_finite_idx:
                mqm_scores[idx] = 0.0
    spans = [
        [
            *(spans[idx] if idx < len(spans) else []),
            *(mqm_spans[idx] if idx < len(mqm_spans) else []),
        ]
        for idx in range(len(rollouts))
    ]

    span_counts = [len(s) for s in spans]
    severity = Counter()
    for span_list in spans:
        for span in span_list:
            label = str(span.get("severity", "UNKNOWN")).upper()
            severity[label] += 1

    metricx_m, metricx_s = _mean_std(metricx_scores)
    metricx_r_m, metricx_r_s = _mean_std(metricx_rewards)
    xcomet_m, xcomet_s = _mean_std(xcomet_scores)
    mqm_m, mqm_s = _mean_std(mqm_scores)
    avg_completion_len = mean([len(r.completion_token_ids) for r in rollouts]) if rollouts else 0.0

    report = {
        "metricx_score_mean": metricx_m,
        "metricx_score_std": metricx_s,
        "metricx_reward_mean": metricx_r_m,
        "metricx_reward_std": metricx_r_s,
        "xcomet_score_mean": xcomet_m,
        "xcomet_score_std": xcomet_s,
        "mqm_score_mean": mqm_m,
        "mqm_score_std": mqm_s,
        "avg_span_count": mean(span_counts) if span_counts else 0.0,
        "severity_counts": dict(severity),
        "avg_completion_len": float(avg_completion_len),
        "num_eval_rollouts": len(rollouts),
    }

    if collect_outputs:
        rows: list[dict[str, Any]] = []
        for idx, rollout in enumerate(rollouts):
            span_row = spans[idx] if idx < len(spans) else []
            rows.append(
                {
                    "example_id": rollout.example_id,
                    "src_text": rollout.src_text,
                    "completion_text": rollout.completion_text,
                    "ref_text": rollout.ref_text,
                    "completion_len": len(rollout.completion_token_ids),
                    "metricx_score": float(metricx_scores[idx]) if idx < len(metricx_scores) else 0.0,
                    "metricx_reward": float(metricx_rewards[idx]) if idx < len(metricx_rewards) else 0.0,
                    "xcomet_score": float(xcomet_scores[idx]) if idx < len(xcomet_scores) else 0.0,
                    "mqm_score": float(mqm_scores[idx]) if idx < len(mqm_scores) else 0.0,
                    "span_count": len(span_row),
                    "error_spans": span_row,
                }
            )
        report["eval_rows"] = rows

    return report
