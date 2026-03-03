from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from copy import deepcopy
import logging
import math
import os
from statistics import mean
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover - optional for lightweight tests
    torch = None  # type: ignore[assignment]

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .config import GenerationConfig, RLPostTrainConfig
from .rewards import OpenAICompatibleMQMScorer, MetricXQEScorer, XCometXLScorer, metricx_score_to_reward
from .rollout import generate_rollouts
from .rl_types import Example, Rollout, SampleForScoring

logger = logging.getLogger(__name__)
_EVAL_PAD_PREFIX = "__eval_pad__:"
_EVAL_OBJECT_GROUP: Any | None = None
_EVAL_OBJECT_GROUP_WORLD_SIZE: int = -1


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    m = mean(values)
    var = sum((v - m) ** 2 for v in values) / len(values)
    return float(m), float(var**0.5)


def _distributed_ready(rank: int, world_size: int) -> bool:
    if torch is None:
        return False
    if world_size <= 1:
        return False
    if rank < 0:
        return False
    if not torch.distributed.is_available():
        return False
    return torch.distributed.is_initialized()


def _set_distributed_cuda_device() -> None:
    if torch is None or (not torch.cuda.is_available()):
        return
    local_rank_raw = os.environ.get("LOCAL_RANK")
    if local_rank_raw is None or (not local_rank_raw.isdigit()):
        return
    torch.cuda.set_device(int(local_rank_raw))


def _get_eval_object_collective_group(rank: int, world_size: int) -> Any | None:
    if not _distributed_ready(rank, world_size):
        return None
    if torch is None:
        return None
    backend = str(torch.distributed.get_backend()).lower()
    if backend == "gloo":
        return None

    global _EVAL_OBJECT_GROUP, _EVAL_OBJECT_GROUP_WORLD_SIZE
    if _EVAL_OBJECT_GROUP is not None and _EVAL_OBJECT_GROUP_WORLD_SIZE == int(world_size):
        return _EVAL_OBJECT_GROUP

    # Build a dedicated Gloo group for Python-object collectives in eval.
    # This avoids NCCL-specific device alignment issues/deadlocks.
    _EVAL_OBJECT_GROUP = torch.distributed.new_group(backend="gloo")
    _EVAL_OBJECT_GROUP_WORLD_SIZE = int(world_size)
    if rank == 0:
        logger.info(
            "evaluate_on_dataset: created dedicated gloo group for object collectives (world_size=%s).",
            world_size,
        )
    return _EVAL_OBJECT_GROUP


def _gather_rollouts_to_rank0(local_rollouts: list[Rollout], *, rank: int, world_size: int) -> list[Rollout]:
    if not _distributed_ready(rank, world_size):
        return local_rollouts
    object_group = _get_eval_object_collective_group(rank, world_size)
    # When falling back to default NCCL group, keep rank/device aligned.
    if object_group is None:
        _set_distributed_cuda_device()
    gathered: list[Any] = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered, local_rollouts, group=object_group)
    if rank != 0:
        return []
    merged: list[Rollout] = []
    for shard in gathered:
        if isinstance(shard, list):
            merged.extend(shard)
    return merged


def _broadcast_report_from_rank0(report: dict[str, Any] | None, *, rank: int, world_size: int) -> dict[str, Any] | None:
    if not _distributed_ready(rank, world_size):
        return report
    object_group = _get_eval_object_collective_group(rank, world_size)
    if object_group is None:
        _set_distributed_cuda_device()
    payload: list[Any] = [report if rank == 0 else None]
    torch.distributed.broadcast_object_list(payload, src=0, group=object_group)
    value = payload[0]
    return value if isinstance(value, dict) else None


def _pad_eval_shard_examples(
    examples: list[Example],
    *,
    rank: int,
    world_size: int,
) -> tuple[list[Example], int]:
    local_examples = list(examples[rank::world_size])
    if world_size <= 1:
        return local_examples, 0
    target_count = math.ceil(len(examples) / float(world_size))
    pad_count = max(0, int(target_count) - len(local_examples))
    if pad_count <= 0:
        return local_examples, 0

    if local_examples:
        anchor = local_examples[0]
    elif examples:
        anchor = examples[0]
    else:
        # Defensive fallback; evaluate_on_dataset already handles empty examples.
        anchor = Example(
            example_id="__eval_anchor__",
            src_text="",
            src_lang="English",
            tgt_lang="Korean",
            src_lang_code="en",
            tgt_lang_code="ko",
            ref_text=None,
        )

    padded = list(local_examples)
    for i in range(pad_count):
        padded.append(
            Example(
                example_id=f"{_EVAL_PAD_PREFIX}{rank}:{i}",
                src_text=anchor.src_text or ".",
                src_lang=anchor.src_lang,
                tgt_lang=anchor.tgt_lang,
                src_lang_code=anchor.src_lang_code,
                tgt_lang_code=anchor.tgt_lang_code,
                ref_text=anchor.ref_text,
            )
        )
    return padded, pad_count


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
    show_progress: bool = False,
    distributed_eval_shard: bool = False,
    distributed_rank: int = 0,
    distributed_world_size: int = 1,
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

    logger.info(
        "evaluate_on_dataset: start examples=%s collect_outputs=%s",
        len(examples),
        bool(collect_outputs),
    )
    gen_cfg = deepcopy(cfg.generation)
    gen_cfg.num_samples_per_prompt = 1
    gen_cfg.do_sample = False
    gen_cfg.temperature = 0.0
    eval_overrides = dict(getattr(cfg.eval, "generation_overrides", {}) or {})
    if eval_overrides:
        for key, value in eval_overrides.items():
            if hasattr(gen_cfg, key):
                setattr(gen_cfg, key, value)
            else:
                logger.warning("Ignoring unknown eval.generation_overrides key: %s", key)
        logger.info("evaluate_on_dataset: applied eval generation overrides: %s", eval_overrides)
    if int(getattr(gen_cfg, "num_samples_per_prompt", 1)) != 1:
        logger.warning(
            "Eval always uses one sample per prompt; overriding num_samples_per_prompt=%s -> 1.",
            getattr(gen_cfg, "num_samples_per_prompt", None),
        )
        gen_cfg.num_samples_per_prompt = 1
    if bool(getattr(gen_cfg, "do_sample", False)):
        logger.warning(
            "Eval enforces deterministic decoding; overriding do_sample=%s -> False.",
            getattr(gen_cfg, "do_sample", None),
        )
    gen_cfg.do_sample = False
    shard_eval = bool(distributed_eval_shard and distributed_world_size > 1)
    if shard_eval:
        local_examples, pad_count = _pad_eval_shard_examples(
            examples,
            rank=distributed_rank,
            world_size=distributed_world_size,
        )
        if distributed_rank == 0:
            logger.info(
                "evaluate_on_dataset: distributed shard mode enabled world_size=%s rank=%s local_examples=%s pad=%s",
                distributed_world_size,
                distributed_rank,
                len(local_examples),
                pad_count,
            )
    else:
        local_examples = examples
        pad_count = 0

    local_rollouts = generate_rollouts(
        examples=local_examples,
        policy_model=policy_model,
        tokenizer=tokenizer,
        gen_cfg=gen_cfg,
        device=device,
        ref_model=None,
        prompt_template=cfg.prompt.template,
        show_progress=bool(show_progress),
        progress_desc="eval rollout",
        compute_old_logprobs=False,
        compute_token_offsets=False,
        include_prompt_input_ids=False,
    )
    if pad_count > 0:
        local_rollouts = [r for r in local_rollouts if not str(r.example_id).startswith(_EVAL_PAD_PREFIX)]
    if shard_eval:
        logger.info(
            "evaluate_on_dataset: local rollout done rank=%s local_rollouts=%s",
            distributed_rank,
            len(local_rollouts),
        )
        logger.info("evaluate_on_dataset: gather begin rank=%s", distributed_rank)
        rollouts = _gather_rollouts_to_rank0(
            local_rollouts,
            rank=distributed_rank,
            world_size=distributed_world_size,
        )
        logger.info("evaluate_on_dataset: gather done rank=%s merged_rollouts=%s", distributed_rank, len(rollouts))
    else:
        # Non-sharded distributed eval runs generation on all ranks for ZeRO/NCCL safety,
        # but report/scoring must use only one copy of each example (rank0 local results).
        rollouts = local_rollouts if distributed_rank == 0 else []
        if distributed_world_size > 1 and distributed_rank == 0:
            logger.info(
                "evaluate_on_dataset: non-sharded distributed eval; using rank0 local rollouts only "
                "(rollouts=%s, world_size=%s).",
                len(rollouts),
                distributed_world_size,
            )
    if (not shard_eval) or distributed_rank == 0:
        logger.info("evaluate_on_dataset: rollout complete rollouts=%s", len(rollouts))

    samples = [SampleForScoring(src=r.src_text, mt=r.completion_text, ref=r.ref_text) for r in rollouts]

    metricx_scores = [0.0 for _ in rollouts]
    metricx_rewards = [0.0 for _ in rollouts]
    xcomet_scores = [0.0 for _ in rollouts]
    mqm_scores = [0.0 for _ in rollouts]
    spans: list[list[dict[str, Any]]] = [[] for _ in rollouts]
    mqm_spans: list[list[dict[str, Any]]] = [[] for _ in rollouts]

    metricx_enabled = metricx_scorer is not None and cfg.reward.metricx.enabled
    xcomet_enabled = xcomet_scorer is not None and cfg.reward.xcomet.enabled
    mqm_enabled = mqm_scorer is not None and cfg.reward.mqm.enabled

    def _score_metricx_eval() -> tuple[list[float], list[float]]:
        metricx_local_scores = metricx_scorer.score_batch(samples).sequence_scores  # type: ignore[union-attr]
        non_finite_idx = [idx for idx, value in enumerate(metricx_local_scores) if not math.isfinite(float(value))]
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
                    metricx_local_scores[idx] = float(cfg.reward.metricx.offset)
            else:
                raise RuntimeError(
                    f"{msg} Eval aborted to avoid silently recording invalid MetricX values."
                )
        metricx_local_rewards = [metricx_score_to_reward(v, offset=cfg.reward.metricx.offset) for v in metricx_local_scores]
        return metricx_local_scores, metricx_local_rewards

    def _score_xcomet_eval() -> tuple[list[float], list[list[dict[str, Any]]]]:
        xcomet_out = xcomet_scorer.score_batch(samples)  # type: ignore[union-attr]
        xcomet_local_scores = xcomet_out.sequence_scores
        non_finite_idx = [idx for idx, value in enumerate(xcomet_local_scores) if not math.isfinite(float(value))]
        if non_finite_idx:
            logger.warning(
                "xCOMET produced %s non-finite eval scores; replacing with 0.0.",
                len(non_finite_idx),
            )
            for idx in non_finite_idx:
                xcomet_local_scores[idx] = 0.0
        meta = xcomet_out.metadata or {}
        xcomet_local_spans: list[list[dict[str, Any]]] = meta.get("error_spans", [[] for _ in rollouts])
        return xcomet_local_scores, xcomet_local_spans

    def _score_mqm_eval() -> tuple[list[float], list[list[dict[str, Any]]]]:
        mqm_out = mqm_scorer.score_batch(samples)
        mqm_local_scores = mqm_out.sequence_scores
        mqm_meta = mqm_out.metadata or {}
        mqm_local_spans: list[list[dict[str, Any]]] = mqm_meta.get("error_spans", [[] for _ in rollouts])
        non_finite_idx = [idx for idx, value in enumerate(mqm_local_scores) if not math.isfinite(float(value))]
        if non_finite_idx:
            logger.warning(
                "MQM scorer produced %s non-finite eval scores; replacing with 0.0.",
                len(non_finite_idx),
            )
            for idx in non_finite_idx:
                mqm_local_scores[idx] = 0.0
        return mqm_local_scores, mqm_local_spans

    enabled_scorers = sum(int(flag) for flag in (metricx_enabled, xcomet_enabled, mqm_enabled))
    if enabled_scorers > 1:
        logger.info(
            "evaluate_on_dataset: scoring in parallel (metricx=%s xcomet=%s mqm=%s)...",
            metricx_enabled,
            xcomet_enabled,
            mqm_enabled,
        )
        with ThreadPoolExecutor(max_workers=enabled_scorers, thread_name_prefix="eval-scorer") as executor:
            futures: dict[str, Any] = {}
            if metricx_enabled:
                futures["metricx"] = executor.submit(_score_metricx_eval)
            if xcomet_enabled:
                futures["xcomet"] = executor.submit(_score_xcomet_eval)
            if mqm_enabled:
                futures["mqm"] = executor.submit(_score_mqm_eval)
            if "metricx" in futures:
                metricx_scores, metricx_rewards = futures["metricx"].result()
            if "xcomet" in futures:
                xcomet_scores, spans = futures["xcomet"].result()
            if "mqm" in futures:
                mqm_scores, mqm_spans = futures["mqm"].result()
    else:
        if metricx_enabled:
            logger.info("evaluate_on_dataset: scoring metricx...")
            metricx_scores, metricx_rewards = _score_metricx_eval()
        if xcomet_enabled:
            logger.info("evaluate_on_dataset: scoring xcomet...")
            xcomet_scores, spans = _score_xcomet_eval()
        if mqm_enabled:
            logger.info("evaluate_on_dataset: scoring mqm...")
            mqm_scores, mqm_spans = _score_mqm_eval()
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

    if shard_eval:
        report_summary = dict(report)
        report_summary.pop("eval_rows", None)
        synced = _broadcast_report_from_rank0(
            report_summary if distributed_rank == 0 else None,
            rank=distributed_rank,
            world_size=distributed_world_size,
        )
        if distributed_rank != 0 and synced is not None:
            report = synced

    logger.info(
        "evaluate_on_dataset: done metricx=%.4f xcomet=%.4f mqm=%.4f",
        float(report.get("metricx_score_mean", 0.0)),
        float(report.get("xcomet_score_mean", 0.0)),
        float(report.get("mqm_score_mean", 0.0)),
    )
    return report
