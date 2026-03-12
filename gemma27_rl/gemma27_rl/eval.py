from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import datetime
import logging
import math
import os
from statistics import mean
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover - optional for lightweight tests
    torch = None  # type: ignore[assignment]

try:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase
except Exception:  # pragma: no cover - optional during lightweight tests
    PreTrainedModel = Any  # type: ignore[assignment,misc]
    PreTrainedTokenizerBase = Any  # type: ignore[assignment,misc]

from .config import GenerationConfig, RLPostTrainConfig
from .prompting import collect_tokenizer_special_token_strings, sanitize_text_for_scoring
from .rewards import (
    OpenAICompatibleESAScorer,
    OpenAICompatibleMQMScorer,
    MetricXQEScorer,
    XCometXLScorer,
    metricx_score_to_reward,
)
from .rollout import generate_rollouts
from .rl_types import Example, Rollout, SampleForScoring

logger = logging.getLogger(__name__)
_EVAL_PAD_PREFIX = "__eval_pad__:"
_EVAL_OBJECT_GROUP: Any | None = None
_EVAL_OBJECT_GROUP_WORLD_SIZE: int = -1
_EVAL_OBJECT_GROUP_TIMEOUT_SEC: float = -1.0
_DEFAULT_DEEPSPEED_EVAL_OBJECT_TIMEOUT_SEC = 7200.0


def _empty_eval_report(*, collect_outputs: bool) -> dict[str, Any]:
    empty = {
        "metricx_score_mean": 0.0,
        "metricx_reward_mean": 0.0,
        "xcomet_score_mean": 0.0,
        "mqm_score_mean": 0.0,
        "esa_score_mean": 0.0,
        "esa_score_std": 0.0,
        "avg_span_count": 0.0,
        "severity_counts": {},
        "avg_completion_len": 0.0,
    }
    if collect_outputs:
        empty["eval_rows"] = []
    return empty


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return max(minimum, int(default))
    try:
        value = int(raw.strip())
    except Exception:
        return max(minimum, int(default))
    return max(minimum, value)


def _truncate_for_log(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"...[truncated {len(text) - max_chars} chars]"


def _safe_convert_ids_to_tokens(tokenizer: PreTrainedTokenizerBase, token_ids: list[int]) -> list[str]:
    try:
        converter = getattr(tokenizer, "convert_ids_to_tokens", None)
        if callable(converter):
            out = converter([int(v) for v in token_ids])
            if isinstance(out, list):
                return [str(v) for v in out]
            if isinstance(out, str):
                return [out]
    except Exception:
        pass
    return []


def _safe_decode_ids_with_specials(tokenizer: PreTrainedTokenizerBase, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(
            [int(v) for v in token_ids],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except Exception:
        return ""


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    m = mean(values)
    var = sum((v - m) ** 2 for v in values) / len(values)
    return float(m), float(var**0.5)


def _rollout_direction_label(rollout: Rollout) -> str:
    src_code = str(rollout.src_lang_code or "").strip().lower()
    tgt_code = str(rollout.tgt_lang_code or "").strip().lower()
    if src_code and tgt_code:
        return f"{src_code}->{tgt_code}"
    src_lang = str(rollout.src_lang or "").strip() or "unknown"
    tgt_lang = str(rollout.tgt_lang or "").strip() or "unknown"
    return f"{src_lang}->{tgt_lang}"


def _build_direction_metrics(
    *,
    rollouts: list[Rollout],
    metricx_scores: list[float],
    metricx_rewards: list[float],
    xcomet_scores: list[float],
    mqm_scores: list[float],
    mqm_skipped: list[bool],
    esa_scores: list[float],
    esa_skipped: list[bool],
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[int]] = {}
    for idx, rollout in enumerate(rollouts):
        grouped.setdefault(_rollout_direction_label(rollout), []).append(idx)

    direction_metrics: dict[str, dict[str, float]] = {}
    for direction, indices in grouped.items():
        metricx_dir = [float(metricx_scores[idx]) for idx in indices if idx < len(metricx_scores)]
        metricx_reward_dir = [float(metricx_rewards[idx]) for idx in indices if idx < len(metricx_rewards)]
        xcomet_dir = [float(xcomet_scores[idx]) for idx in indices if idx < len(xcomet_scores)]
        mqm_dir = [
            float(mqm_scores[idx])
            for idx in indices
            if idx < len(mqm_scores) and idx < len(mqm_skipped) and (not mqm_skipped[idx])
        ]
        esa_dir = [
            float(esa_scores[idx])
            for idx in indices
            if idx < len(esa_scores) and idx < len(esa_skipped) and (not esa_skipped[idx])
        ]
        metricx_m, metricx_s = _mean_std(metricx_dir)
        metricx_r_m, metricx_r_s = _mean_std(metricx_reward_dir)
        xcomet_m, xcomet_s = _mean_std(xcomet_dir)
        mqm_m, mqm_s = _mean_std(mqm_dir)
        esa_m, esa_s = _mean_std(esa_dir)
        direction_metrics[direction] = {
            "num_eval_rollouts": float(len(indices)),
            "metricx_score_mean": metricx_m,
            "metricx_score_std": metricx_s,
            "metricx_reward_mean": metricx_r_m,
            "metricx_reward_std": metricx_r_s,
            "xcomet_score_mean": xcomet_m,
            "xcomet_score_std": xcomet_s,
            "mqm_score_mean": mqm_m,
            "mqm_score_std": mqm_s,
            "mqm_skipped_count": float(sum(1 for idx in indices if idx < len(mqm_skipped) and mqm_skipped[idx])),
            "esa_score_mean": esa_m,
            "esa_score_std": esa_s,
            "esa_skipped_count": float(sum(1 for idx in indices if idx < len(esa_skipped) and esa_skipped[idx])),
        }
    return direction_metrics


def _validate_optional_bool_rows(*, scorer_name: str, requested: int, skipped_rows: Any | None) -> list[bool]:
    if skipped_rows is None:
        return [False for _ in range(int(requested))]
    if not isinstance(skipped_rows, (list, tuple)):
        raise RuntimeError(
            f"{scorer_name} scorer returned non-list skipped_rows "
            f"(type={type(skipped_rows).__name__}, requested={requested})."
        )
    rows = [bool(v) for v in list(skipped_rows)]
    if len(rows) != int(requested):
        raise RuntimeError(
            f"{scorer_name} scorer returned mismatched skipped_rows length: "
            f"requested={requested} returned={len(rows)}"
        )
    return rows


def _validate_scorer_batch_lengths(
    *,
    scorer_name: str,
    requested: int,
    sequence_scores: Any,
    error_spans: Any | None = None,
) -> tuple[list[Any], list[Any] | None]:
    if not isinstance(sequence_scores, (list, tuple)):
        raise RuntimeError(
            f"{scorer_name} scorer returned non-list sequence_scores "
            f"(type={type(sequence_scores).__name__}, requested={requested})."
        )
    score_rows = list(sequence_scores)
    if len(score_rows) != int(requested):
        raise RuntimeError(
            f"{scorer_name} scorer returned mismatched sequence_scores length: "
            f"requested={requested} returned={len(score_rows)}"
        )

    span_rows: list[Any] | None = None
    if error_spans is not None:
        if not isinstance(error_spans, (list, tuple)):
            raise RuntimeError(
                f"{scorer_name} scorer returned non-list error_spans "
                f"(type={type(error_spans).__name__}, requested={requested})."
            )
        span_rows = list(error_spans)
        if len(span_rows) != int(requested):
            raise RuntimeError(
                f"{scorer_name} scorer returned mismatched error_spans length: "
                f"requested={requested} returned={len(span_rows)}"
            )

    return score_rows, span_rows


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


def _torch_default_distributed_timeout_sec() -> float:
    if torch is None:
        return 1800.0
    default_timeout = getattr(torch.distributed.constants, "default_pg_timeout", None)
    if isinstance(default_timeout, datetime.timedelta):
        return float(default_timeout.total_seconds())
    return 1800.0


def _resolve_eval_object_group_timeout_sec(cfg: RLPostTrainConfig, world_size: int) -> float:
    env_key = "TORCH_DISTRIBUTED_TIMEOUT_SEC"
    raw_env = os.environ.get(env_key)
    if raw_env is not None and str(raw_env).strip():
        try:
            timeout_sec = float(str(raw_env).strip())
        except Exception as exc:
            raise ValueError(f"{env_key} must be a positive number of seconds.") from exc
        if (not math.isfinite(timeout_sec)) or timeout_sec <= 0:
            raise ValueError(f"{env_key} must be a positive number of seconds.")
        return timeout_sec

    configured_timeout = cfg.misc.distributed_timeout_sec
    if configured_timeout is not None:
        return float(configured_timeout)

    if str(cfg.rl.backend).strip().lower() == "deepspeed" and int(world_size) > 1:
        return _DEFAULT_DEEPSPEED_EVAL_OBJECT_TIMEOUT_SEC
    return _torch_default_distributed_timeout_sec()


def _set_distributed_cuda_device() -> None:
    if torch is None or (not torch.cuda.is_available()):
        return
    local_rank_raw = os.environ.get("LOCAL_RANK")
    if local_rank_raw is None or (not local_rank_raw.isdigit()):
        return
    torch.cuda.set_device(int(local_rank_raw))


def _get_eval_object_collective_group(cfg: RLPostTrainConfig, rank: int, world_size: int) -> Any | None:
    if not _distributed_ready(rank, world_size):
        return None
    if torch is None:
        return None
    backend = str(torch.distributed.get_backend()).lower()
    if backend == "gloo":
        return None

    timeout_sec = _resolve_eval_object_group_timeout_sec(cfg, world_size)

    global _EVAL_OBJECT_GROUP, _EVAL_OBJECT_GROUP_WORLD_SIZE, _EVAL_OBJECT_GROUP_TIMEOUT_SEC
    if (
        _EVAL_OBJECT_GROUP is not None
        and _EVAL_OBJECT_GROUP_WORLD_SIZE == int(world_size)
        and math.isclose(_EVAL_OBJECT_GROUP_TIMEOUT_SEC, float(timeout_sec), rel_tol=0.0, abs_tol=1e-9)
    ):
        return _EVAL_OBJECT_GROUP

    # Build a dedicated Gloo group for Python-object collectives in eval.
    # This avoids NCCL-specific device alignment issues/deadlocks.
    _EVAL_OBJECT_GROUP = torch.distributed.new_group(
        backend="gloo",
        timeout=datetime.timedelta(seconds=float(timeout_sec)),
    )
    _EVAL_OBJECT_GROUP_WORLD_SIZE = int(world_size)
    _EVAL_OBJECT_GROUP_TIMEOUT_SEC = float(timeout_sec)
    if rank == 0:
        logger.info(
            "evaluate_on_dataset: created dedicated gloo group for object collectives "
            "(world_size=%s timeout=%.1fs).",
            world_size,
            timeout_sec,
        )
    return _EVAL_OBJECT_GROUP


def _gather_rollouts_to_rank0(
    local_rollouts: list[Rollout],
    *,
    cfg: RLPostTrainConfig,
    rank: int,
    world_size: int,
) -> list[Rollout]:
    if not _distributed_ready(rank, world_size):
        return local_rollouts
    object_group = _get_eval_object_collective_group(cfg, rank, world_size)
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


def _broadcast_report_from_rank0(
    report: dict[str, Any] | None,
    *,
    cfg: RLPostTrainConfig,
    rank: int,
    world_size: int,
) -> dict[str, Any] | None:
    if not _distributed_ready(rank, world_size):
        return report
    object_group = _get_eval_object_collective_group(cfg, rank, world_size)
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


@dataclass
class _EvalScoringBatch:
    rollouts: list[Rollout]
    raw_completion_texts: list[str]
    clean_completion_texts: list[str]
    metricx_scores: list[float]
    metricx_rewards: list[float]
    xcomet_scores: list[float]
    mqm_scores: list[float]
    esa_scores: list[float]
    spans: list[list[dict[str, Any]]]
    mqm_skipped: list[bool]
    esa_skipped: list[bool]
    sanitized_target_rows: int
    sanitized_marker_total: int


def _should_pipeline_eval_api_scoring(
    *,
    cfg: RLPostTrainConfig,
    mqm_scorer: OpenAICompatibleMQMScorer | None,
    esa_scorer: OpenAICompatibleESAScorer | None,
) -> bool:
    if _env_flag("GEMMA27_RL_DISABLE_ROLLOUT_PIPELINE", default=False):
        return False
    return bool(
        (cfg.reward.mqm.enabled and mqm_scorer is not None)
        or (cfg.reward.esa.enabled and esa_scorer is not None)
    )


def _resolve_eval_rollout_pipeline_chunk_size(*, total_examples: int, cfg: RLPostTrainConfig) -> int:
    total = max(0, int(total_examples))
    if total <= 1:
        return total

    raw = os.environ.get("GEMMA27_RL_ROLLOUT_PIPELINE_CHUNK")
    if raw is not None:
        try:
            requested = int(raw.strip())
        except Exception:
            requested = 0
        if requested > 0:
            return min(total, requested)

    batch_hints: list[int] = []
    if cfg.reward.mqm.enabled:
        batch_hints.append(max(1, int(cfg.reward.mqm.batch_size)))
    if cfg.reward.esa.enabled:
        batch_hints.append(max(1, int(cfg.reward.esa.batch_size)))
    hint = min(batch_hints) if batch_hints else total
    default_chunk = min(hint, int(math.ceil(total / 2.0)))
    return max(1, min(total, default_chunk))


def _resolve_eval_generation_config(cfg: RLPostTrainConfig) -> GenerationConfig:
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
    return gen_cfg


def _score_eval_rollouts(
    *,
    rollouts: list[Rollout],
    tokenizer: PreTrainedTokenizerBase,
    cfg: RLPostTrainConfig,
    metricx_scorer: MetricXQEScorer | None = None,
    xcomet_scorer: XCometXLScorer | None = None,
    mqm_scorer: OpenAICompatibleMQMScorer | None = None,
    esa_scorer: OpenAICompatibleESAScorer | None = None,
) -> _EvalScoringBatch:
    special_token_strings = collect_tokenizer_special_token_strings(tokenizer)
    samples: list[SampleForScoring] = []
    raw_completion_texts: list[str] = []
    clean_completion_texts: list[str] = []
    sanitized_target_rows = 0
    sanitized_marker_total = 0
    for rollout in rollouts:
        raw_mt = str(rollout.completion_raw_text if rollout.completion_raw_text is not None else rollout.completion_text or "")
        sanitized_mt, replacement_count = sanitize_text_for_scoring(
            raw_mt,
            special_tokens=special_token_strings,
        )
        clean_mt = str(rollout.completion_clean_text if rollout.completion_clean_text is not None else sanitized_mt)
        raw_completion_texts.append(raw_mt)
        clean_completion_texts.append(clean_mt)
        if replacement_count > 0:
            sanitized_target_rows += 1
            sanitized_marker_total += int(replacement_count)
        samples.append(
            SampleForScoring(
                src=rollout.src_text,
                mt=clean_mt,
                ref=rollout.ref_text,
                source_lang=rollout.src_lang,
                target_lang=rollout.tgt_lang,
            )
        )

    metricx_scores = [0.0 for _ in rollouts]
    metricx_rewards = [0.0 for _ in rollouts]
    xcomet_scores = [0.0 for _ in rollouts]
    mqm_scores = [0.0 for _ in rollouts]
    esa_scores = [0.0 for _ in rollouts]
    spans: list[list[dict[str, Any]]] = [[] for _ in rollouts]
    mqm_spans: list[list[dict[str, Any]]] = [[] for _ in rollouts]
    mqm_skipped = [False for _ in rollouts]
    esa_skipped = [False for _ in rollouts]

    metricx_enabled = metricx_scorer is not None and cfg.reward.metricx.enabled
    xcomet_enabled = xcomet_scorer is not None and cfg.reward.xcomet.enabled
    mqm_enabled = mqm_scorer is not None and cfg.reward.mqm.enabled
    esa_enabled = esa_scorer is not None and cfg.reward.esa.enabled

    def _score_metricx_eval() -> tuple[list[float], list[float]]:
        metricx_out = metricx_scorer.score_batch(samples)  # type: ignore[union-attr]
        metricx_local_scores, _ = _validate_scorer_batch_lengths(
            scorer_name="MetricX",
            requested=len(samples),
            sequence_scores=metricx_out.sequence_scores,
        )
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
        xcomet_local_scores, span_rows = _validate_scorer_batch_lengths(
            scorer_name="xCOMET",
            requested=len(samples),
            sequence_scores=xcomet_out.sequence_scores,
            error_spans=(xcomet_out.metadata or {}).get("error_spans", [[] for _ in samples]),
        )
        non_finite_idx = [idx for idx, value in enumerate(xcomet_local_scores) if not math.isfinite(float(value))]
        if non_finite_idx:
            logger.warning(
                "xCOMET produced %s non-finite eval scores; replacing with 0.0.",
                len(non_finite_idx),
            )
            for idx in non_finite_idx:
                xcomet_local_scores[idx] = 0.0
        assert span_rows is not None
        xcomet_local_spans = [
            [item for item in span_row if isinstance(item, dict)] if isinstance(span_row, (list, tuple)) else []
            for span_row in span_rows
        ]
        return xcomet_local_scores, xcomet_local_spans

    def _score_mqm_eval() -> tuple[list[float], list[list[dict[str, Any]]], list[bool]]:
        mqm_out = mqm_scorer.score_batch(samples)
        mqm_local_scores, span_rows = _validate_scorer_batch_lengths(
            scorer_name="MQM",
            requested=len(samples),
            sequence_scores=mqm_out.sequence_scores,
            error_spans=(mqm_out.metadata or {}).get("error_spans", [[] for _ in samples]),
        )
        skipped_rows = _validate_optional_bool_rows(
            scorer_name="MQM",
            requested=len(samples),
            skipped_rows=(mqm_out.metadata or {}).get("skipped_rows", [False for _ in samples]),
        )
        non_finite_idx = [idx for idx, value in enumerate(mqm_local_scores) if not math.isfinite(float(value))]
        if non_finite_idx:
            logger.warning(
                "MQM scorer produced %s non-finite eval scores; replacing with 0.0.",
                len(non_finite_idx),
            )
            for idx in non_finite_idx:
                mqm_local_scores[idx] = 0.0
        assert span_rows is not None
        mqm_local_spans = [
            [item for item in span_row if isinstance(item, dict)] if isinstance(span_row, (list, tuple)) else []
            for span_row in span_rows
        ]
        return mqm_local_scores, mqm_local_spans, skipped_rows

    def _score_esa_eval() -> tuple[list[float], list[bool]]:
        esa_out = esa_scorer.score_batch(samples)
        esa_local_scores, _ = _validate_scorer_batch_lengths(
            scorer_name="ESA",
            requested=len(samples),
            sequence_scores=esa_out.sequence_scores,
        )
        skipped_rows = _validate_optional_bool_rows(
            scorer_name="ESA",
            requested=len(samples),
            skipped_rows=(esa_out.metadata or {}).get("skipped_rows", [False for _ in samples]),
        )
        non_finite_idx = [idx for idx, value in enumerate(esa_local_scores) if not math.isfinite(float(value))]
        if non_finite_idx:
            logger.warning(
                "ESA scorer produced %s non-finite eval scores; replacing with 0.0.",
                len(non_finite_idx),
            )
            for idx in non_finite_idx:
                esa_local_scores[idx] = 0.0
        return esa_local_scores, skipped_rows

    enabled_scorers = sum(int(flag) for flag in (metricx_enabled, xcomet_enabled, mqm_enabled, esa_enabled))
    if enabled_scorers > 1:
        logger.info(
            "evaluate_on_dataset: scoring in parallel (metricx=%s xcomet=%s mqm=%s esa=%s)...",
            metricx_enabled,
            xcomet_enabled,
            mqm_enabled,
            esa_enabled,
        )
        with ThreadPoolExecutor(max_workers=enabled_scorers, thread_name_prefix="eval-scorer") as executor:
            futures: dict[str, Any] = {}
            if metricx_enabled:
                futures["metricx"] = executor.submit(_score_metricx_eval)
            if xcomet_enabled:
                futures["xcomet"] = executor.submit(_score_xcomet_eval)
            if mqm_enabled:
                futures["mqm"] = executor.submit(_score_mqm_eval)
            if esa_enabled:
                futures["esa"] = executor.submit(_score_esa_eval)
            if "metricx" in futures:
                metricx_scores, metricx_rewards = futures["metricx"].result()
            if "xcomet" in futures:
                xcomet_scores, spans = futures["xcomet"].result()
            if "mqm" in futures:
                mqm_scores, mqm_spans, mqm_skipped = futures["mqm"].result()
            if "esa" in futures:
                esa_scores, esa_skipped = futures["esa"].result()
    else:
        if metricx_enabled:
            logger.info("evaluate_on_dataset: scoring metricx...")
            metricx_scores, metricx_rewards = _score_metricx_eval()
        if xcomet_enabled:
            logger.info("evaluate_on_dataset: scoring xcomet...")
            xcomet_scores, spans = _score_xcomet_eval()
        if mqm_enabled:
            logger.info("evaluate_on_dataset: scoring mqm...")
            mqm_scores, mqm_spans, mqm_skipped = _score_mqm_eval()
        if esa_enabled:
            logger.info("evaluate_on_dataset: scoring esa...")
            esa_scores, esa_skipped = _score_esa_eval()

    merged_spans = [
        [
            *(spans[idx] if idx < len(spans) else []),
            *(mqm_spans[idx] if idx < len(mqm_spans) else []),
        ]
        for idx in range(len(rollouts))
    ]
    return _EvalScoringBatch(
        rollouts=list(rollouts),
        raw_completion_texts=raw_completion_texts,
        clean_completion_texts=clean_completion_texts,
        metricx_scores=metricx_scores,
        metricx_rewards=metricx_rewards,
        xcomet_scores=xcomet_scores,
        mqm_scores=mqm_scores,
        esa_scores=esa_scores,
        spans=merged_spans,
        mqm_skipped=mqm_skipped,
        esa_skipped=esa_skipped,
        sanitized_target_rows=int(sanitized_target_rows),
        sanitized_marker_total=int(sanitized_marker_total),
    )


def _merge_eval_scoring_batches(scored_batches: list[_EvalScoringBatch]) -> _EvalScoringBatch:
    merged = _EvalScoringBatch(
        rollouts=[],
        raw_completion_texts=[],
        clean_completion_texts=[],
        metricx_scores=[],
        metricx_rewards=[],
        xcomet_scores=[],
        mqm_scores=[],
        esa_scores=[],
        spans=[],
        mqm_skipped=[],
        esa_skipped=[],
        sanitized_target_rows=0,
        sanitized_marker_total=0,
    )
    for batch in scored_batches:
        merged.rollouts.extend(batch.rollouts)
        merged.raw_completion_texts.extend(batch.raw_completion_texts)
        merged.clean_completion_texts.extend(batch.clean_completion_texts)
        merged.metricx_scores.extend(batch.metricx_scores)
        merged.metricx_rewards.extend(batch.metricx_rewards)
        merged.xcomet_scores.extend(batch.xcomet_scores)
        merged.mqm_scores.extend(batch.mqm_scores)
        merged.esa_scores.extend(batch.esa_scores)
        merged.spans.extend(batch.spans)
        merged.mqm_skipped.extend(batch.mqm_skipped)
        merged.esa_skipped.extend(batch.esa_skipped)
        merged.sanitized_target_rows += int(batch.sanitized_target_rows)
        merged.sanitized_marker_total += int(batch.sanitized_marker_total)
    return merged


def prepare_eval_rollouts(
    *,
    examples: list[Example],
    policy_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    cfg: RLPostTrainConfig,
    device: str,
    show_progress: bool = False,
    distributed_eval_shard: bool = False,
    distributed_rank: int = 0,
    distributed_world_size: int = 1,
) -> list[Rollout]:
    if not examples:
        return []

    gen_cfg = _resolve_eval_generation_config(cfg)
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
            cfg=cfg,
            rank=distributed_rank,
            world_size=distributed_world_size,
        )
        logger.info("evaluate_on_dataset: gather done rank=%s merged_rollouts=%s", distributed_rank, len(rollouts))
    else:
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
    return rollouts


def _build_eval_report_from_scored_rollouts(
    *,
    rollouts: list[Rollout],
    scored_rollouts: _EvalScoringBatch,
    tokenizer: PreTrainedTokenizerBase,
    collect_outputs: bool,
) -> dict[str, Any]:
    raw_completion_texts = list(scored_rollouts.raw_completion_texts)
    clean_completion_texts = list(scored_rollouts.clean_completion_texts)
    metricx_scores = list(scored_rollouts.metricx_scores)
    metricx_rewards = list(scored_rollouts.metricx_rewards)
    xcomet_scores = list(scored_rollouts.xcomet_scores)
    mqm_scores = list(scored_rollouts.mqm_scores)
    esa_scores = list(scored_rollouts.esa_scores)
    spans = list(scored_rollouts.spans)
    mqm_skipped = list(scored_rollouts.mqm_skipped)
    esa_skipped = list(scored_rollouts.esa_skipped)
    sanitized_target_rows = int(scored_rollouts.sanitized_target_rows)
    sanitized_marker_total = int(scored_rollouts.sanitized_marker_total)
    if sanitized_target_rows > 0:
        special_token_strings = collect_tokenizer_special_token_strings(tokenizer)
        logger.info(
            "evaluate_on_dataset: scorer target sanitize applied: rows=%s/%s marker_replacements=%s tokenizer_special_tokens=%s",
            int(sanitized_target_rows),
            len(rollouts),
            int(sanitized_marker_total),
            len(special_token_strings),
        )

    span_counts = [len(s) for s in spans]
    severity = Counter()
    for span_list in spans:
        for span in span_list:
            label = str(span.get("severity", "UNKNOWN")).upper()
            severity[label] += 1

    metricx_m, metricx_s = _mean_std(metricx_scores)
    metricx_r_m, metricx_r_s = _mean_std(metricx_rewards)
    xcomet_m, xcomet_s = _mean_std(xcomet_scores)
    mqm_used_scores = [float(v) for v, skipped in zip(mqm_scores, mqm_skipped) if not skipped]
    esa_used_scores = [float(v) for v, skipped in zip(esa_scores, esa_skipped) if not skipped]
    mqm_m, mqm_s = _mean_std(mqm_used_scores)
    esa_m, esa_s = _mean_std(esa_used_scores)
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
        "mqm_skipped_count": float(sum(mqm_skipped)),
        "esa_score_mean": esa_m,
        "esa_score_std": esa_s,
        "esa_skipped_count": float(sum(esa_skipped)),
        "avg_span_count": mean(span_counts) if span_counts else 0.0,
        "severity_counts": dict(severity),
        "avg_completion_len": float(avg_completion_len),
        "num_eval_rollouts": len(rollouts),
    }
    direction_metrics = _build_direction_metrics(
        rollouts=rollouts,
        metricx_scores=metricx_scores,
        metricx_rewards=metricx_rewards,
        xcomet_scores=xcomet_scores,
        mqm_scores=mqm_scores,
        mqm_skipped=mqm_skipped,
        esa_scores=esa_scores,
        esa_skipped=esa_skipped,
    )
    if direction_metrics:
        report["direction_metrics"] = direction_metrics

    raw_io_log_enabled = _env_flag("GEMMA27_RL_LOG_RAW_IO", default=False)
    raw_io_max_chars = _env_int("GEMMA27_RL_LOG_RAW_IO_MAX_CHARS", default=20000, minimum=256)
    raw_io_max_rows = _env_int("GEMMA27_RL_LOG_RAW_IO_MAX_ROWS", default=0, minimum=0)
    if raw_io_log_enabled:
        for idx, rollout in enumerate(rollouts):
            if raw_io_max_rows > 0 and idx >= raw_io_max_rows:
                break
            span_row = spans[idx] if idx < len(spans) else []
            completion_ids = [int(v) for v in list(rollout.completion_token_ids or [])]
            completion_tokens = _safe_convert_ids_to_tokens(tokenizer, completion_ids)
            completion_decoded_with_specials = _safe_decode_ids_with_specials(tokenizer, completion_ids)
            logger.info(
                "[raw-io][eval][scored] idx=%s example_id=%s src=%r ref=%r mt_raw=%r mt_clean=%r metricx=%.6f metricx_reward=%.6f "
                "xcomet=%.6f mqm=%.6f esa=%.6f completion_ids=%s completion_tokens=%s completion_decoded_with_specials=%r spans=%s",
                idx,
                rollout.example_id,
                _truncate_for_log(rollout.src_text, raw_io_max_chars),
                _truncate_for_log(rollout.ref_text, raw_io_max_chars),
                _truncate_for_log(
                    raw_completion_texts[idx] if idx < len(raw_completion_texts) else str(rollout.completion_text or ""),
                    raw_io_max_chars,
                ),
                _truncate_for_log(
                    clean_completion_texts[idx]
                    if idx < len(clean_completion_texts)
                    else str(rollout.completion_clean_text if rollout.completion_clean_text is not None else rollout.completion_text or ""),
                    raw_io_max_chars,
                ),
                float(metricx_scores[idx]) if idx < len(metricx_scores) else 0.0,
                float(metricx_rewards[idx]) if idx < len(metricx_rewards) else 0.0,
                float(xcomet_scores[idx]) if idx < len(xcomet_scores) else 0.0,
                float(mqm_scores[idx]) if idx < len(mqm_scores) else 0.0,
                float(esa_scores[idx]) if idx < len(esa_scores) else 0.0,
                _truncate_for_log(str(completion_ids), raw_io_max_chars),
                _truncate_for_log(str(completion_tokens), raw_io_max_chars),
                _truncate_for_log(completion_decoded_with_specials, raw_io_max_chars),
                _truncate_for_log(str(span_row), raw_io_max_chars),
            )

    if collect_outputs:
        rows: list[dict[str, Any]] = []
        for idx, rollout in enumerate(rollouts):
            span_row = spans[idx] if idx < len(spans) else []
            rows.append(
                {
                    "example_id": rollout.example_id,
                    "direction": _rollout_direction_label(rollout),
                    "src_text": rollout.src_text,
                    "src_lang": rollout.src_lang,
                    "tgt_lang": rollout.tgt_lang,
                    "src_lang_code": rollout.src_lang_code,
                    "tgt_lang_code": rollout.tgt_lang_code,
                    "completion_text": rollout.completion_text,
                    "completion_raw_text": (
                        raw_completion_texts[idx]
                        if idx < len(raw_completion_texts)
                        else str(rollout.completion_raw_text if rollout.completion_raw_text is not None else rollout.completion_text or "")
                    ),
                    "completion_clean_text": (
                        clean_completion_texts[idx]
                        if idx < len(clean_completion_texts)
                        else str(rollout.completion_clean_text if rollout.completion_clean_text is not None else rollout.completion_text or "")
                    ),
                    "ref_text": rollout.ref_text,
                    "completion_len": len(rollout.completion_token_ids),
                    "metricx_score": float(metricx_scores[idx]) if idx < len(metricx_scores) else 0.0,
                    "metricx_reward": float(metricx_rewards[idx]) if idx < len(metricx_rewards) else 0.0,
                    "xcomet_score": float(xcomet_scores[idx]) if idx < len(xcomet_scores) else 0.0,
                    "mqm_score": float(mqm_scores[idx]) if idx < len(mqm_scores) else 0.0,
                    "mqm_skipped": bool(mqm_skipped[idx]) if idx < len(mqm_skipped) else False,
                    "esa_score": float(esa_scores[idx]) if idx < len(esa_scores) else 0.0,
                    "esa_skipped": bool(esa_skipped[idx]) if idx < len(esa_skipped) else False,
                    "span_count": len(span_row),
                    "error_spans": span_row,
                }
            )
        report["eval_rows"] = rows

    return report


def build_eval_report_from_rollouts(
    *,
    rollouts: list[Rollout],
    tokenizer: PreTrainedTokenizerBase,
    cfg: RLPostTrainConfig,
    metricx_scorer: MetricXQEScorer | None = None,
    xcomet_scorer: XCometXLScorer | None = None,
    mqm_scorer: OpenAICompatibleMQMScorer | None = None,
    esa_scorer: OpenAICompatibleESAScorer | None = None,
    collect_outputs: bool = False,
) -> dict[str, Any]:
    if not rollouts:
        return _empty_eval_report(collect_outputs=collect_outputs)
    scored_rollouts = _score_eval_rollouts(
        rollouts=rollouts,
        tokenizer=tokenizer,
        cfg=cfg,
        metricx_scorer=metricx_scorer,
        xcomet_scorer=xcomet_scorer,
        mqm_scorer=mqm_scorer,
        esa_scorer=esa_scorer,
    )
    return _build_eval_report_from_scored_rollouts(
        rollouts=rollouts,
        scored_rollouts=scored_rollouts,
        tokenizer=tokenizer,
        collect_outputs=collect_outputs,
    )


def _generate_and_score_eval_rollouts_pipelined(
    *,
    examples: list[Example],
    policy_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    gen_cfg: GenerationConfig,
    cfg: RLPostTrainConfig,
    device: str,
    show_progress: bool,
    score_on_this_rank: bool,
    metricx_scorer: MetricXQEScorer | None = None,
    xcomet_scorer: XCometXLScorer | None = None,
    mqm_scorer: OpenAICompatibleMQMScorer | None = None,
    esa_scorer: OpenAICompatibleESAScorer | None = None,
) -> tuple[list[Rollout], _EvalScoringBatch | None]:
    if not examples:
        return [], None

    chunk_size = _resolve_eval_rollout_pipeline_chunk_size(total_examples=len(examples), cfg=cfg)
    chunk_ranges = [
        (start, min(len(examples), start + chunk_size))
        for start in range(0, len(examples), max(1, chunk_size))
    ]
    if score_on_this_rank:
        logger.info(
            "evaluate_on_dataset: pipelining rollout generation with API scoring chunk_size=%s chunks=%s examples=%s",
            chunk_size,
            len(chunk_ranges),
            len(examples),
        )

    def _generate_chunk(chunk_idx: int, chunk_examples: list[Example]) -> list[Rollout]:
        return generate_rollouts(
            examples=chunk_examples,
            policy_model=policy_model,
            tokenizer=tokenizer,
            gen_cfg=gen_cfg,
            device=device,
            ref_model=None,
            prompt_template=cfg.prompt.template,
            show_progress=bool(show_progress),
            progress_desc=f"eval rollout [{chunk_idx + 1}/{len(chunk_ranges)}]",
            compute_old_logprobs=False,
            compute_token_offsets=False,
            include_prompt_input_ids=False,
        )

    first_start, first_end = chunk_ranges[0]
    current_rollouts = _generate_chunk(0, examples[first_start:first_end])
    local_rollouts = list(current_rollouts)
    scored_batches: list[_EvalScoringBatch] = []
    executor: ThreadPoolExecutor | None = None
    try:
        if score_on_this_rank:
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="eval-pipeline")
        for chunk_idx, (start, end) in enumerate(chunk_ranges[1:], start=1):
            score_future = None
            if score_on_this_rank and executor is not None and current_rollouts:
                score_future = executor.submit(
                    _score_eval_rollouts,
                    rollouts=current_rollouts,
                    tokenizer=tokenizer,
                    cfg=cfg,
                    metricx_scorer=metricx_scorer,
                    xcomet_scorer=xcomet_scorer,
                    mqm_scorer=mqm_scorer,
                    esa_scorer=esa_scorer,
                )
            next_rollouts = _generate_chunk(chunk_idx, examples[start:end])
            local_rollouts.extend(next_rollouts)
            if score_future is not None:
                scored_batches.append(score_future.result())
            current_rollouts = next_rollouts
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    if score_on_this_rank and current_rollouts:
        scored_batches.append(
            _score_eval_rollouts(
                rollouts=current_rollouts,
                tokenizer=tokenizer,
                cfg=cfg,
                metricx_scorer=metricx_scorer,
                xcomet_scorer=xcomet_scorer,
                mqm_scorer=mqm_scorer,
                esa_scorer=esa_scorer,
            )
        )

    merged_scores = _merge_eval_scoring_batches(scored_batches) if scored_batches else None
    return local_rollouts, merged_scores


def evaluate_on_dataset(
    examples: list[Example],
    policy_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    cfg: RLPostTrainConfig,
    device: str,
    metricx_scorer: MetricXQEScorer | None = None,
    xcomet_scorer: XCometXLScorer | None = None,
    mqm_scorer: OpenAICompatibleMQMScorer | None = None,
    esa_scorer: OpenAICompatibleESAScorer | None = None,
    collect_outputs: bool = False,
    show_progress: bool = False,
    distributed_eval_shard: bool = False,
    distributed_rank: int = 0,
    distributed_world_size: int = 1,
) -> dict[str, Any]:
    if not examples:
        return _empty_eval_report(collect_outputs=collect_outputs)

    logger.info(
        "evaluate_on_dataset: start examples=%s collect_outputs=%s",
        len(examples),
        bool(collect_outputs),
    )
    gen_cfg = _resolve_eval_generation_config(cfg)
    shard_eval = bool(distributed_eval_shard and distributed_world_size > 1)
    local_examples = examples

    pipeline_enabled = (
        (not shard_eval)
        and _should_pipeline_eval_api_scoring(cfg=cfg, mqm_scorer=mqm_scorer, esa_scorer=esa_scorer)
        and _resolve_eval_rollout_pipeline_chunk_size(total_examples=len(local_examples), cfg=cfg) < len(local_examples)
    )
    local_scored_rollouts: _EvalScoringBatch | None = None
    if pipeline_enabled:
        local_rollouts, local_scored_rollouts = _generate_and_score_eval_rollouts_pipelined(
            examples=local_examples,
            policy_model=policy_model,
            tokenizer=tokenizer,
            gen_cfg=gen_cfg,
            cfg=cfg,
            device=device,
            show_progress=bool(show_progress),
            score_on_this_rank=bool(distributed_world_size <= 1 or distributed_rank == 0),
            metricx_scorer=metricx_scorer,
            xcomet_scorer=xcomet_scorer,
            mqm_scorer=mqm_scorer,
            esa_scorer=esa_scorer,
        )
        rollouts = local_rollouts if distributed_rank == 0 else []
        scored_rollouts = local_scored_rollouts if distributed_rank == 0 else None
        if distributed_world_size > 1 and distributed_rank == 0:
            logger.info(
                "evaluate_on_dataset: non-sharded distributed eval; using rank0 local rollouts only "
                "(rollouts=%s, world_size=%s).",
                len(rollouts),
                distributed_world_size,
            )
        if distributed_rank == 0:
            logger.info("evaluate_on_dataset: rollout complete rollouts=%s", len(rollouts))
    else:
        rollouts = prepare_eval_rollouts(
            examples=local_examples,
            policy_model=policy_model,
            tokenizer=tokenizer,
            cfg=cfg,
            device=device,
            show_progress=bool(show_progress),
            distributed_eval_shard=distributed_eval_shard,
            distributed_rank=distributed_rank,
            distributed_world_size=distributed_world_size,
        )
        scored_rollouts = None

    if scored_rollouts is None:
        report = build_eval_report_from_rollouts(
            rollouts=rollouts,
            tokenizer=tokenizer,
            cfg=cfg,
            metricx_scorer=metricx_scorer,
            xcomet_scorer=xcomet_scorer,
            mqm_scorer=mqm_scorer,
            esa_scorer=esa_scorer,
            collect_outputs=collect_outputs,
        )
    else:
        report = _build_eval_report_from_scored_rollouts(
            rollouts=rollouts,
            scored_rollouts=scored_rollouts,
            tokenizer=tokenizer,
            collect_outputs=collect_outputs,
        )
    direction_metrics = dict(report.get("direction_metrics", {}) or {})

    if shard_eval:
        report_summary = dict(report)
        report_summary.pop("eval_rows", None)
        synced = _broadcast_report_from_rank0(
            report_summary if distributed_rank == 0 else None,
            cfg=cfg,
            rank=distributed_rank,
            world_size=distributed_world_size,
        )
        if distributed_rank != 0 and synced is not None:
            report = synced

    logger.info(
        "evaluate_on_dataset: done metricx=%.4f xcomet=%.4f mqm=%.4f esa=%.4f",
        float(report.get("metricx_score_mean", 0.0)),
        float(report.get("xcomet_score_mean", 0.0)),
        float(report.get("mqm_score_mean", 0.0)),
        float(report.get("esa_score_mean", 0.0)),
    )
    if len(direction_metrics) > 1:
        for direction, stats in sorted(direction_metrics.items()):
            logger.info(
                "evaluate_on_dataset: direction=%s metricx=%.4f xcomet=%.4f mqm=%.4f esa=%.4f",
                direction,
                float(stats.get("metricx_score_mean", 0.0)),
                float(stats.get("xcomet_score_mean", 0.0)),
                float(stats.get("mqm_score_mean", 0.0)),
                float(stats.get("esa_score_mean", 0.0)),
            )
    return report
