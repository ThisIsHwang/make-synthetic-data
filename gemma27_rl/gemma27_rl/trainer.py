from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
import random
import shutil
from statistics import mean
from typing import Any

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import deepspeed
except Exception:  # pragma: no cover - optional dependency
    deepspeed = None  # type: ignore[assignment]

from .advantage import (
    apply_group_relative_advantage,
    broadcast_sequence_reward,
    build_sequence_rewards,
    combine_advantages,
    normalize_advantages,
)
from .config import RLPostTrainConfig, dump_config
from .data import load_examples
from .eval import evaluate_on_dataset
from .grpo import update_policy
from .rewards import (
    OpenAICompatibleMQMScorer,
    MetricXQEScorer,
    XCometXLScorer,
    metricx_score_to_reward,
    spans_to_token_rewards,
)
from .rollout import generate_rollouts
from .types import Rollout, SampleForScoring
from .utils import (
    configure_huggingface_cache,
    resolve_device,
    resolve_huggingface_token,
    resolve_torch_dtype,
    set_seed,
)


logger = logging.getLogger(__name__)


def _is_distributed_initialized() -> bool:
    return bool(torch.distributed.is_available() and torch.distributed.is_initialized())


def _distributed_rank() -> int:
    if _is_distributed_initialized():
        return int(torch.distributed.get_rank())
    raw = os.environ.get("RANK")
    if raw and raw.isdigit():
        return int(raw)
    return 0


def _distributed_world_size() -> int:
    if _is_distributed_initialized():
        return int(torch.distributed.get_world_size())
    raw = os.environ.get("WORLD_SIZE")
    if raw and raw.isdigit():
        return int(raw)
    return 1


def _is_rank0() -> bool:
    return _distributed_rank() == 0


def _dist_barrier() -> None:
    if _is_distributed_initialized():
        torch.distributed.barrier()


def _broadcast_object_list(payload: list[Any], src: int = 0) -> list[Any]:
    if _is_distributed_initialized():
        torch.distributed.broadcast_object_list(payload, src=src)
    return payload


def _local_rank_device(default_device: str) -> str:
    local_rank_text = os.environ.get("LOCAL_RANK")
    if local_rank_text and local_rank_text.isdigit() and torch.cuda.is_available():
        local_rank = int(local_rank_text)
        torch.cuda.set_device(local_rank)
        return f"cuda:{local_rank}"
    return default_device


def _build_deepspeed_config_dict(cfg: RLPostTrainConfig, world_size: int) -> dict[str, Any]:
    dtype_text = str(cfg.misc.dtype).strip().lower()
    use_bf16 = dtype_text in {"bf16", "bfloat16"}
    use_fp16 = dtype_text in {"fp16", "float16"}

    micro_batch = max(1, int(cfg.rl.batch_size))
    # GRPO update_policy already performs manual grad accumulation.
    ds_grad_accum = 1
    global_batch = micro_batch * ds_grad_accum * max(1, int(world_size))

    zero_stage = int(cfg.rl.deepspeed_zero_stage)
    zero_cfg: dict[str, Any] = {"stage": zero_stage}
    if cfg.rl.deepspeed_offload_optimizer:
        zero_cfg["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
    if cfg.rl.deepspeed_offload_param:
        zero_cfg["offload_param"] = {"device": "cpu", "pin_memory": True}

    ds_cfg: dict[str, Any] = {
        "train_micro_batch_size_per_gpu": micro_batch,
        "gradient_accumulation_steps": ds_grad_accum,
        "train_batch_size": global_batch,
        "zero_optimization": zero_cfg,
        "bf16": {"enabled": bool(use_bf16)},
        "fp16": {"enabled": bool(use_fp16 and not use_bf16)},
    }
    if cfg.rl.max_grad_norm > 0:
        ds_cfg["gradient_clipping"] = float(cfg.rl.max_grad_norm)
    ds_cfg["optimizer"] = {
        "type": "AdamW",
        "params": {
            "lr": float(cfg.rl.lr),
            "weight_decay": float(cfg.rl.weight_decay),
            "betas": [0.9, 0.999],
            "eps": float(cfg.rl.eps),
        },
    }
    return ds_cfg


def _load_deepspeed_config(cfg: RLPostTrainConfig, world_size: int) -> dict[str, Any]:
    if cfg.rl.deepspeed_config_path:
        path = Path(cfg.rl.deepspeed_config_path)
        if not path.exists():
            raise FileNotFoundError(f"rl.deepspeed_config_path not found: {path}")
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            payload = yaml.safe_load(text)
        else:
            payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid DeepSpeed config payload: {path}")
        return payload
    return _build_deepspeed_config_dict(cfg, world_size=world_size)


def _deepspeed_initialize(
    cfg: RLPostTrainConfig,
    policy_model: AutoModelForCausalLM,
) -> tuple[Any, Any]:
    if deepspeed is None:
        raise RuntimeError(
            "rl.backend=deepspeed but `deepspeed` is not installed. "
            "Install it and re-run."
        )

    world_size = _distributed_world_size()
    if world_size > 1 and not _is_distributed_initialized():
        deepspeed.init_distributed()

    ds_config = _load_deepspeed_config(cfg, world_size=world_size)
    engine, optimizer, _, _ = deepspeed.initialize(
        model=policy_model,
        model_parameters=[p for p in policy_model.parameters() if p.requires_grad],
        config=ds_config,
    )
    return engine, optimizer


def _unwrap_for_generation(model: Any) -> Any:
    module = getattr(model, "module", None)
    return module if module is not None else model


def _parse_cuda_index(device: str | None) -> int | None:
    if not device:
        return None
    text = str(device).strip().lower()
    if text == "cuda":
        return None
    if text.startswith("cuda:"):
        idx_text = text.split(":", 1)[1].strip()
        if idx_text.isdigit():
            return int(idx_text)
    return None


def _normalize_gpu_id_list(raw_ids: list[int] | None) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for idx in raw_ids or []:
        value = int(idx)
        if value < 0:
            raise ValueError(f"GPU index must be >= 0, got {value}")
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _pick_free_gpu(
    preferred: int | None,
    used: set[int],
    device_count: int,
) -> int:
    if preferred is not None and 0 <= preferred < device_count and preferred not in used:
        return preferred
    for cand in range(device_count):
        if cand not in used:
            return cand
    if preferred is not None and 0 <= preferred < device_count:
        return preferred
    return 0


def _assign_disjoint_gpu_devices(cfg: RLPostTrainConfig) -> None:
    if not torch.cuda.is_available():
        return

    device_count = int(torch.cuda.device_count())
    if device_count <= 0:
        return

    def _is_cuda_text(value: str | None) -> bool:
        return bool(value) and str(value).strip().lower().startswith("cuda")

    explicit_policy_ids = _normalize_gpu_id_list(cfg.model.policy_gpu_ids)
    explicit_reference_ids = _normalize_gpu_id_list(cfg.model.reference_gpu_ids)

    if explicit_policy_ids or explicit_reference_ids:
        for idx in explicit_policy_ids + explicit_reference_ids:
            if idx >= device_count:
                raise ValueError(
                    f"Configured GPU index out of range: {idx} (cuda_count={device_count})"
                )

        used: set[int] = set(explicit_policy_ids) | set(explicit_reference_ids)

        if _is_cuda_text(cfg.misc.device):
            if not explicit_policy_ids:
                explicit_policy_ids = [_pick_free_gpu(preferred=0, used=used, device_count=device_count)]
                used.update(explicit_policy_ids)
            cfg.misc.device = f"cuda:{explicit_policy_ids[0]}"
        cfg.model.policy_gpu_ids = explicit_policy_ids

        if explicit_reference_ids:
            cfg.model.reference_device = f"cuda:{explicit_reference_ids[0]}"
        cfg.model.reference_gpu_ids = explicit_reference_ids

        if cfg.reward.metricx.enabled and _is_cuda_text(cfg.reward.metricx.device):
            metricx_idx = _pick_free_gpu(
                preferred=_parse_cuda_index(cfg.reward.metricx.device),
                used=used,
                device_count=device_count,
            )
            used.add(metricx_idx)
            cfg.reward.metricx.device = f"cuda:{metricx_idx}"

        if cfg.reward.xcomet.enabled and _is_cuda_text(cfg.reward.xcomet.device):
            xcomet_idx = _pick_free_gpu(
                preferred=_parse_cuda_index(cfg.reward.xcomet.device),
                used=used,
                device_count=device_count,
            )
            used.add(xcomet_idx)
            cfg.reward.xcomet.device = f"cuda:{xcomet_idx}"

        logger.info(
            "Applied explicit GPU partition: policy_gpu_ids=%s reference_gpu_ids=%s metricx=%s xcomet=%s",
            cfg.model.policy_gpu_ids,
            cfg.model.reference_gpu_ids,
            cfg.reward.metricx.device,
            cfg.reward.xcomet.device,
        )
        return

    components: list[dict[str, Any]] = [
        {"name": "policy", "enabled": _is_cuda_text(cfg.misc.device), "raw": cfg.misc.device, "required": True},
        {
            "name": "metricx",
            "enabled": bool(cfg.reward.metricx.enabled and _is_cuda_text(cfg.reward.metricx.device)),
            "raw": cfg.reward.metricx.device,
            "required": False,
        },
        {
            "name": "xcomet",
            "enabled": bool(cfg.reward.xcomet.enabled and _is_cuda_text(cfg.reward.xcomet.device)),
            "raw": cfg.reward.xcomet.device,
            "required": False,
        },
    ]

    used: set[int] = set()
    assigned: dict[str, int] = {}

    for comp in components:
        if not comp["enabled"]:
            continue
        raw = str(comp["raw"]).strip().lower()
        preferred = _parse_cuda_index(raw)
        if preferred is None and comp["name"] == "policy":
            preferred = 0

        idx: int | None = None
        if preferred is not None and 0 <= preferred < device_count and preferred not in used:
            idx = preferred
        else:
            for cand in range(device_count):
                if cand not in used:
                    idx = cand
                    break
            if idx is None:
                idx = preferred if preferred is not None else 0

        used.add(idx)
        assigned[comp["name"]] = idx

    if "policy" in assigned:
        cfg.misc.device = f"cuda:{assigned['policy']}"
    if "metricx" in assigned:
        cfg.reward.metricx.device = f"cuda:{assigned['metricx']}"
    if "xcomet" in assigned:
        cfg.reward.xcomet.device = f"cuda:{assigned['xcomet']}"

    if len({assigned.get("policy"), assigned.get("metricx"), assigned.get("xcomet")} - {None}) < len(assigned):
        logger.warning(
            "Could not assign fully disjoint GPUs for all models (cuda_count=%s, assigned=%s).",
            device_count,
            assigned,
        )
    else:
        logger.info("Assigned model devices: %s", assigned)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    m = sum(values) / len(values)
    var = sum((v - m) ** 2 for v in values) / len(values)
    return float(m), float(var**0.5)


def _flatten(rows: list[list[float]]) -> list[float]:
    out: list[float] = []
    for row in rows:
        out.extend(row)
    return out


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _truncate_jsonl_by_update(path: Path, max_update: int) -> None:
    if not path.exists():
        return

    kept: list[str] = []
    dropped = 0
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                kept.append(line + "\n")
                continue
            if not isinstance(row, dict) or "update" not in row:
                kept.append(line + "\n")
                continue
            try:
                update_idx = int(row["update"])
            except (TypeError, ValueError):
                kept.append(line + "\n")
                continue
            if update_idx <= max_update:
                kept.append(line + "\n")
            else:
                dropped += 1

    if dropped > 0:
        with path.open("w", encoding="utf-8") as f:
            f.writelines(kept)
        logger.info("Truncated %s rows from %s for resume consistency.", dropped, path)


def _append_rollout_jsonl(
    path: Path,
    update_idx: int,
    rollouts: list[Rollout],
    advantages: list[list[float]],
    reward_stats: dict[str, float],
) -> None:
    for ridx, rollout in enumerate(rollouts):
        adv_row = advantages[ridx] if ridx < len(advantages) else []
        payload = {
            "type": "rollout",
            "update": update_idx,
            "rollout_idx": ridx,
            "example_id": rollout.example_id,
            "src_text": rollout.src_text,
            "completion_text": rollout.completion_text,
            "ref_text": rollout.ref_text,
            "completion_len": len(rollout.completion_token_ids),
            "adv_mean": float(sum(adv_row) / len(adv_row)) if adv_row else 0.0,
            "adv_sum": float(sum(adv_row)) if adv_row else 0.0,
            "old_logprob_mean": float(sum(rollout.old_logprobs) / len(rollout.old_logprobs))
            if rollout.old_logprobs
            else 0.0,
            "ref_logprob_mean": float(sum(rollout.ref_logprobs) / len(rollout.ref_logprobs))
            if rollout.ref_logprobs
            else None,
            "metricx_score_mean_batch": reward_stats.get("metricx_score_mean", 0.0),
            "xcomet_score_mean_batch": reward_stats.get("xcomet_score_mean", 0.0),
            "mqm_score_mean_batch": reward_stats.get("mqm_score_mean", 0.0),
            "token_rewards_non_zero_ratio_batch": reward_stats.get("token_rewards_non_zero_ratio", 0.0),
        }
        _append_jsonl(path, payload)


def _append_eval_output_jsonl(path: Path, update_idx: int, eval_rows: list[dict[str, Any]]) -> None:
    for ridx, row in enumerate(eval_rows):
        payload = {
            "type": "eval_output",
            "update": update_idx,
            "eval_row_idx": ridx,
            **row,
        }
        _append_jsonl(path, payload)


def _parse_checkpoint_update_idx(path: Path) -> int | None:
    name = path.name.strip()
    prefix = "checkpoint-"
    if not name.startswith(prefix):
        return None
    idx_text = name[len(prefix) :]
    if not idx_text.isdigit():
        return None
    return int(idx_text)


def _save_trainer_state(path: Path, payload: dict[str, Any]) -> None:
    state_path = path / "trainer_state.json"
    state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_trainer_state(path: Path) -> dict[str, Any] | None:
    state_path = path / "trainer_state.json"
    if not state_path.exists():
        return None
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read trainer_state.json from %s: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        logger.warning("Invalid trainer_state.json format at %s", path)
        return None
    return payload


def _restore_best_from_log(log_path: Path) -> tuple[float, int | None]:
    if not log_path.exists():
        return float("-inf"), None

    best_score = float("-inf")
    best_update: int | None = None
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            if row.get("type") != "eval":
                continue
            score_raw = row.get("model_select_score")
            update_raw = row.get("update")
            try:
                score = float(score_raw)
                update = int(update_raw)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(score):
                continue
            if score > best_score:
                best_score = score
                best_update = update
    return best_score, best_update


def _resolve_resume_checkpoint(
    cfg: RLPostTrainConfig,
    output_dir: Path,
) -> tuple[Path | None, int]:
    explicit = cfg.logging.resume_from_checkpoint
    if explicit:
        resume_path = Path(explicit)
        if not resume_path.exists():
            raise FileNotFoundError(f"logging.resume_from_checkpoint not found: {resume_path}")
        state = _load_trainer_state(resume_path) or {}
        update_idx = int(state.get("update_idx", _parse_checkpoint_update_idx(resume_path) or 0))
        return resume_path, update_idx

    if not cfg.logging.auto_resume:
        return None, 0

    candidates: list[tuple[int, Path]] = []
    resume_latest = output_dir / "resume_latest"
    if resume_latest.exists() and resume_latest.is_dir():
        state = _load_trainer_state(resume_latest) or {}
        update_raw = state.get("update_idx")
        try:
            update_idx = int(update_raw)
        except (TypeError, ValueError):
            update_idx = 0
        candidates.append((update_idx, resume_latest))

    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        update_idx = _parse_checkpoint_update_idx(path)
        if update_idx is None:
            continue
        candidates.append((update_idx, path))

    if not candidates:
        return None, 0

    update_idx, resume_path = max(candidates, key=lambda item: item[0])
    return resume_path, update_idx


def _resolve_model_dtype_and_attn(
    cfg: RLPostTrainConfig,
    device: str,
) -> tuple[torch.dtype | None, str | None]:
    dtype = resolve_torch_dtype(cfg.misc.dtype)
    attn_impl_raw = (cfg.model.attn_implementation or "").strip()
    attn_impl = attn_impl_raw.lower()
    is_cuda_device = str(device).strip().lower().startswith("cuda")

    if not is_cuda_device:
        if dtype in {torch.float16, torch.bfloat16}:
            dtype = torch.float32
        if attn_impl == "flash_attention_2":
            logger.warning(
                "flash_attention_2 requested on %s; falling back to sdpa for this model load.",
                device,
            )
            return dtype, "sdpa"
        if attn_impl and attn_impl != "auto":
            return dtype, attn_impl_raw
        return dtype, None

    if attn_impl == "flash_attention_2" and dtype not in {torch.float16, torch.bfloat16}:
        logger.warning(
            "flash_attention_2 requires fp16/bf16. Overriding model dtype from %s to bfloat16.",
            dtype,
        )
        dtype = torch.bfloat16

    if attn_impl and attn_impl != "auto":
        return dtype, attn_impl_raw
    return dtype, None


def _load_causal_lm(
    model_name_or_path: str,
    kwargs: dict[str, Any],
    single_device: str,
    gpu_ids: list[int],
    component_name: str,
) -> AutoModelForCausalLM:
    explicit_gpu_ids = _normalize_gpu_id_list(gpu_ids)
    if len(explicit_gpu_ids) <= 1:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        model.to(single_device)
        return model
    raise RuntimeError(
        f"{component_name} model requested on multiple GPUs {explicit_gpu_ids}, "
        "but device_map=auto path is disabled. "
        "Use rl.backend=deepspeed for multi-GPU policy training, and keep reference model on a single GPU."
    )


def _load_policy_model(
    cfg: RLPostTrainConfig,
    device: str,
    model_name_or_path: str | None = None,
) -> AutoModelForCausalLM:
    dtype, attn_impl = _resolve_model_dtype_and_attn(cfg, device)

    kwargs: dict[str, Any] = {
        "trust_remote_code": cfg.model.trust_remote_code,
    }
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl

    source = model_name_or_path or cfg.model.policy_name_or_path
    policy_gpu_ids = [] if cfg.rl.backend == "deepspeed" else cfg.model.policy_gpu_ids
    return _load_causal_lm(
        model_name_or_path=source,
        kwargs=kwargs,
        single_device=device,
        gpu_ids=policy_gpu_ids,
        component_name="policy",
    )


def _load_reference_model(cfg: RLPostTrainConfig, default_device: str) -> tuple[AutoModelForCausalLM | None, str | None]:
    if cfg.rl.kl_coef <= 0:
        return None, None

    ref_name = cfg.model.reference_name_or_path or cfg.model.policy_name_or_path
    reference_gpu_ids = _normalize_gpu_id_list(cfg.model.reference_gpu_ids)
    if reference_gpu_ids:
        if len(reference_gpu_ids) > 1:
            logger.warning(
                "reference_gpu_ids=%s requested, but multi-GPU reference loading is disabled. "
                "Using only cuda:%s",
                reference_gpu_ids,
                reference_gpu_ids[0],
            )
        ref_device = resolve_device(f"cuda:{reference_gpu_ids[0]}")
    else:
        ref_device = resolve_device(cfg.model.reference_device or default_device)

    dtype, attn_impl = _resolve_model_dtype_and_attn(cfg, ref_device)

    kwargs: dict[str, Any] = {
        "trust_remote_code": cfg.model.trust_remote_code,
    }
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl

    model = _load_causal_lm(
        model_name_or_path=ref_name,
        kwargs=kwargs,
        single_device=ref_device,
        gpu_ids=reference_gpu_ids[:1],
        component_name="reference",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, ref_device


def _sample_batch(examples: list, batch_size: int, rng: random.Random) -> list:
    if not examples:
        return []
    n = min(batch_size, len(examples))
    if n == len(examples):
        indices = list(range(len(examples)))
        rng.shuffle(indices)
        return [examples[i] for i in indices]
    indices = [rng.randrange(len(examples)) for _ in range(n)]
    return [examples[i] for i in indices]


def _score_with_cache_metricx(
    samples: list[SampleForScoring],
    scorer: MetricXQEScorer,
    cache: dict[tuple[str, str, str], float],
    use_cache: bool,
) -> list[float]:
    out = [0.0 for _ in samples]
    uncached: list[SampleForScoring] = []
    uncached_idx: list[int] = []

    for idx, sample in enumerate(samples):
        key = (sample.src, sample.mt, (sample.ref or "") if scorer.cfg.use_reference else "")
        if use_cache and key in cache:
            out[idx] = cache[key]
        else:
            uncached.append(sample)
            uncached_idx.append(idx)

    if uncached:
        scores = scorer.score_batch(uncached).sequence_scores
        for idx, score, sample in zip(uncached_idx, scores, uncached):
            out[idx] = float(score)
            if use_cache:
                cache[(sample.src, sample.mt, (sample.ref or "") if scorer.cfg.use_reference else "")] = float(score)

    return out


def _score_with_cache_xcomet(
    samples: list[SampleForScoring],
    scorer: XCometXLScorer,
    cache: dict[tuple[str, str, str], tuple[float, list[dict[str, Any]]]],
    use_cache: bool,
) -> tuple[list[float], list[list[dict[str, Any]]]]:
    scores = [0.0 for _ in samples]
    spans = [[] for _ in samples]

    uncached: list[SampleForScoring] = []
    uncached_idx: list[int] = []
    for idx, sample in enumerate(samples):
        key = (sample.src, sample.mt, sample.ref or "")
        if use_cache and key in cache:
            scores[idx], spans[idx] = cache[key]
        else:
            uncached.append(sample)
            uncached_idx.append(idx)

    if uncached:
        out = scorer.score_batch(uncached)
        span_rows = (out.metadata or {}).get("error_spans", [[] for _ in uncached])
        for idx, score, span_row, sample in zip(uncached_idx, out.sequence_scores, span_rows, uncached):
            score_f = float(score)
            span_list = [s for s in span_row if isinstance(s, dict)]
            scores[idx] = score_f
            spans[idx] = span_list
            if use_cache:
                cache[(sample.src, sample.mt, sample.ref or "")] = (score_f, span_list)

    return scores, spans


def _score_with_cache_mqm(
    samples: list[SampleForScoring],
    scorer: OpenAICompatibleMQMScorer,
    cache: dict[tuple[str, str, str], tuple[float, list[dict[str, Any]]]],
    use_cache: bool,
) -> tuple[list[float], list[list[dict[str, Any]]]]:
    scores = [0.0 for _ in samples]
    spans = [[] for _ in samples]
    uncached: list[SampleForScoring] = []
    uncached_idx: list[int] = []

    for idx, sample in enumerate(samples):
        key = (sample.src, sample.mt, (sample.ref or "") if scorer.cfg.use_reference else "")
        if use_cache and key in cache:
            scores[idx], spans[idx] = cache[key]
        else:
            uncached.append(sample)
            uncached_idx.append(idx)

    if uncached:
        out = scorer.score_batch(uncached)
        span_rows = (out.metadata or {}).get("error_spans", [[] for _ in uncached])
        for idx, score, span_row, sample in zip(uncached_idx, out.sequence_scores, span_rows, uncached):
            score_f = float(score)
            span_list = [s for s in span_row if isinstance(s, dict)]
            scores[idx] = score_f
            spans[idx] = span_list
            if use_cache:
                cache[(sample.src, sample.mt, (sample.ref or "") if scorer.cfg.use_reference else "")] = (
                    score_f,
                    span_list,
                )

    return scores, spans


def _prepare_rewards_and_advantages(
    rollouts: list[Rollout],
    cfg: RLPostTrainConfig,
    metricx_scorer: MetricXQEScorer | None,
    xcomet_scorer: XCometXLScorer | None,
    mqm_scorer: OpenAICompatibleMQMScorer | None,
    metricx_cache: dict[tuple[str, str, str], float],
    xcomet_cache: dict[tuple[str, str, str], tuple[float, list[dict[str, Any]]]],
    mqm_cache: dict[tuple[str, str, str], tuple[float, list[dict[str, Any]]]],
) -> tuple[list[list[float]], dict[str, float], dict[str, float]]:
    def _sanitize(values: list[float], fallback: float) -> tuple[list[float], int]:
        out: list[float] = []
        replaced = 0
        for value in values:
            if math.isfinite(value):
                out.append(float(value))
            else:
                out.append(float(fallback))
                replaced += 1
        return out, replaced

    samples = [SampleForScoring(src=r.src_text, mt=r.completion_text, ref=r.ref_text) for r in rollouts]

    if cfg.reward.metricx.enabled and metricx_scorer is not None:
        metricx_scores = _score_with_cache_metricx(
            samples=samples,
            scorer=metricx_scorer,
            cache=metricx_cache,
            use_cache=cfg.reward.cache_enabled,
        )
    else:
        metricx_scores = [cfg.reward.metricx.offset for _ in rollouts]
    metricx_scores, metricx_replaced = _sanitize(metricx_scores, fallback=cfg.reward.metricx.offset)
    if metricx_replaced > 0:
        msg = (
            f"MetricX produced {metricx_replaced} non-finite scores "
            f"(model={cfg.reward.metricx.model_name}, device={cfg.reward.metricx.device}, "
            f"dtype={cfg.reward.metricx.dtype})."
        )
        if cfg.reward.metricx.overflow_policy == "skip":
            logger.warning("%s Replacing with fallback offset %.4f due to overflow_policy=skip.", msg, cfg.reward.metricx.offset)
        else:
            raise RuntimeError(
                f"{msg} Training aborted to avoid silent fallback-to-{cfg.reward.metricx.offset:.4f}. "
                "Check MetricX model load/inference and dtype/device settings."
            )

    metricx_rewards = [metricx_score_to_reward(v, offset=cfg.reward.metricx.offset) for v in metricx_scores]

    if cfg.reward.xcomet.enabled and xcomet_scorer is not None:
        xcomet_scores, span_rows = _score_with_cache_xcomet(
            samples=samples,
            scorer=xcomet_scorer,
            cache=xcomet_cache,
            use_cache=cfg.reward.cache_enabled,
        )
    else:
        xcomet_scores = [0.0 for _ in rollouts]
        span_rows = [[] for _ in rollouts]
    xcomet_scores, xcomet_replaced = _sanitize(xcomet_scores, fallback=0.0)
    if xcomet_replaced > 0:
        logger.warning(
            "xCOMET produced %s non-finite scores; replaced with fallback 0.0.",
            xcomet_replaced,
        )

    if cfg.reward.mqm.enabled and mqm_scorer is not None:
        mqm_scores, mqm_span_rows = _score_with_cache_mqm(
            samples=samples,
            scorer=mqm_scorer,
            cache=mqm_cache,
            use_cache=cfg.reward.cache_enabled and cfg.reward.mqm.enabled,
        )
    else:
        mqm_scores = [0.0 for _ in rollouts]
        mqm_span_rows = [[] for _ in rollouts]
    mqm_scores, mqm_replaced = _sanitize(mqm_scores, fallback=0.0)
    if mqm_replaced > 0:
        logger.warning(
            "MQM scorer produced %s non-finite scores; replaced with fallback 0.0.",
            mqm_replaced,
        )
    span_rows = [
        [
            *(span_rows[idx] if idx < len(span_rows) else []),
            *(mqm_span_rows[idx] if idx < len(mqm_span_rows) else []),
        ]
        for idx in range(len(rollouts))
    ]

    seq_rewards = build_sequence_rewards(
        metricx_scores=metricx_scores,
        xcomet_scores=xcomet_scores,
        metricx_offset=cfg.reward.metricx.offset,
        w_metricx=cfg.reward.w_metricx,
        w_xcomet_seq=cfg.reward.w_xcomet_seq,
        xcomet_seq_scale=cfg.reward.xcomet_seq_scale,
        mqm_scores=mqm_scores,
        w_mqm_seq=cfg.reward.w_mqm_seq,
        mqm_seq_scale=cfg.reward.mqm_seq_scale,
    )

    token_reward_rows: list[list[float]] = []
    raw_adv_rows: list[list[float]] = []
    severity_counts: dict[str, list[float]] = {"MINOR": [], "MAJOR": [], "CRITICAL": []}

    for rollout, span_row, seq_reward in zip(rollouts, span_rows, seq_rewards):
        token_rewards = spans_to_token_rewards(
            mt_text=rollout.completion_text,
            token_char_offsets=rollout.token_char_offsets,
            error_spans=span_row,
            severity_weights=cfg.reward.severity_weights,
            overlap_policy=cfg.reward.overlap_policy,
            majority_threshold=cfg.reward.majority_threshold,
            use_confidence=cfg.reward.use_confidence,
            combine_policy=cfg.reward.span_combine_policy,
        )
        seq_row = broadcast_sequence_reward(seq_reward, token_count=len(token_rewards))
        raw_adv = combine_advantages(seq_row, token_rewards)

        token_reward_rows.append(token_rewards)
        raw_adv_rows.append(raw_adv)

        span_counter = {"MINOR": 0, "MAJOR": 0, "CRITICAL": 0}
        for span in span_row:
            sev = str(span.get("severity", "")).upper()
            if sev in span_counter:
                span_counter[sev] += 1
        for key in severity_counts:
            severity_counts[key].append(float(span_counter[key]))

    if cfg.rl.group_normalize:
        raw_adv_rows, _ = apply_group_relative_advantage(
            raw_advantages=raw_adv_rows,
            group_ids=[r.example_id for r in rollouts],
            coef=cfg.rl.group_advantage_coef,
            eps=cfg.rl.eps,
        )

    if cfg.rl.normalize_advantage:
        norm_adv_rows, norm_stats = normalize_advantages(raw_adv_rows, eps=cfg.rl.eps)
    else:
        norm_adv_rows = raw_adv_rows
        flat_raw = _flatten(raw_adv_rows)
        raw_m, raw_s = _mean_std(flat_raw)
        norm_stats = {
            "raw_mean": raw_m,
            "raw_std": raw_s,
            "norm_mean": raw_m,
            "norm_std": raw_s,
        }

    flat_token_rewards = _flatten(token_reward_rows)
    token_reward_m, token_reward_s = _mean_std(flat_token_rewards)
    non_zero_token_ratio = (
        sum(1 for v in flat_token_rewards if abs(v) > 0) / max(1, len(flat_token_rewards))
        if flat_token_rewards
        else 0.0
    )

    metricx_m, metricx_s = _mean_std(metricx_scores)
    metricx_r_m, metricx_r_s = _mean_std(metricx_rewards)
    xcomet_m, xcomet_s = _mean_std(xcomet_scores)
    mqm_m, mqm_s = _mean_std(mqm_scores)

    reward_stats = {
        "metricx_score_mean": metricx_m,
        "metricx_score_std": metricx_s,
        "metricx_reward_mean": metricx_r_m,
        "metricx_reward_std": metricx_r_s,
        "xcomet_score_mean": xcomet_m,
        "xcomet_score_std": xcomet_s,
        "mqm_score_mean": mqm_m,
        "mqm_score_std": mqm_s,
        "token_rewards_mean": token_reward_m,
        "token_rewards_std": token_reward_s,
        "token_rewards_non_zero_ratio": float(non_zero_token_ratio),
        "span_minor_mean": float(mean(severity_counts["MINOR"]) if severity_counts["MINOR"] else 0.0),
        "span_major_mean": float(mean(severity_counts["MAJOR"]) if severity_counts["MAJOR"] else 0.0),
        "span_critical_mean": float(mean(severity_counts["CRITICAL"]) if severity_counts["CRITICAL"] else 0.0),
    }

    return norm_adv_rows, reward_stats, norm_stats


def _save_checkpoint_to_dir(
    ckpt_dir: Path,
    model: AutoModelForCausalLM,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    trainer_state: dict[str, Any] | None = None,
) -> Path:
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    if trainer_state:
        _save_trainer_state(ckpt_dir, trainer_state)
    return ckpt_dir


def _save_deepspeed_checkpoint_to_dir(
    ckpt_dir: Path,
    engine: Any,
    tokenizer,
    trainer_state: dict[str, Any] | None = None,
) -> Path:
    if _is_rank0():
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    _dist_barrier()
    engine.save_checkpoint(str(ckpt_dir), tag="state")
    _dist_barrier()
    if _is_rank0():
        tokenizer.save_pretrained(ckpt_dir)
        if trainer_state:
            _save_trainer_state(ckpt_dir, trainer_state)
    _dist_barrier()
    return ckpt_dir


def _save_checkpoint(
    output_dir: Path,
    update_idx: int,
    model: AutoModelForCausalLM,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    trainer_state: dict[str, Any] | None = None,
) -> Path:
    ckpt_dir = output_dir / f"checkpoint-{update_idx}"
    return _save_checkpoint_to_dir(
        ckpt_dir=ckpt_dir,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        trainer_state=trainer_state,
    )


def _save_deepspeed_checkpoint(
    output_dir: Path,
    update_idx: int,
    engine: Any,
    tokenizer,
    trainer_state: dict[str, Any] | None = None,
) -> Path:
    ckpt_dir = output_dir / f"checkpoint-{update_idx}"
    return _save_deepspeed_checkpoint_to_dir(
        ckpt_dir=ckpt_dir,
        engine=engine,
        tokenizer=tokenizer,
        trainer_state=trainer_state,
    )


def _save_resume_checkpoint(
    output_dir: Path,
    update_idx: int,
    model: AutoModelForCausalLM,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    trainer_state: dict[str, Any] | None = None,
) -> Path:
    ckpt_dir = output_dir / "resume_latest"
    state = dict(trainer_state or {})
    state["update_idx"] = int(update_idx)
    return _save_checkpoint_to_dir(
        ckpt_dir=ckpt_dir,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        trainer_state=state,
    )


def _save_deepspeed_resume_checkpoint(
    output_dir: Path,
    update_idx: int,
    engine: Any,
    tokenizer,
    trainer_state: dict[str, Any] | None = None,
) -> Path:
    ckpt_dir = output_dir / "resume_latest"
    state = dict(trainer_state or {})
    state["update_idx"] = int(update_idx)
    return _save_deepspeed_checkpoint_to_dir(
        ckpt_dir=ckpt_dir,
        engine=engine,
        tokenizer=tokenizer,
        trainer_state=state,
    )


def _save_model_only(
    save_dir: Path,
    model: AutoModelForCausalLM,
    tokenizer,
) -> Path:
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    return save_dir


def _compute_eval_selection_score(report: dict[str, Any], cfg: RLPostTrainConfig) -> float:
    metricx_term = float(report.get("metricx_reward_mean", 0.0)) * float(cfg.reward.w_metricx)
    xcomet_term = (
        float(report.get("xcomet_score_mean", 0.0))
        * float(cfg.reward.w_xcomet_seq)
        * float(cfg.reward.xcomet_seq_scale)
    )
    mqm_term = (
        float(report.get("mqm_score_mean", 0.0))
        * float(cfg.reward.w_mqm_seq)
        * float(cfg.reward.mqm_seq_scale)
    )
    return float(metricx_term + xcomet_term + mqm_term)


def run_metric_only_eval(cfg: RLPostTrainConfig) -> dict[str, Any]:
    set_seed(cfg.misc.seed)
    hf_token = resolve_huggingface_token(
        explicit_token=cfg.misc.huggingface_token,
        token_env_name=cfg.misc.huggingface_token_env,
    )
    configure_huggingface_cache(cfg.misc.huggingface_cache_dir, token=hf_token)
    _assign_disjoint_gpu_devices(cfg)
    device = resolve_device(cfg.misc.device)

    tokenizer_name = cfg.model.tokenizer_name_or_path or cfg.model.policy_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=cfg.model.use_fast_tokenizer)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_policy_model(cfg, device=device)

    eval_limit = cfg.eval.eval_limit if cfg.eval.eval_limit is not None else cfg.data.eval_limit
    eval_examples = load_examples(cfg.data, split="eval", limit=eval_limit)
    if (
        cfg.data.hf_dataset_name
        and not cfg.data.eval_file
        and (cfg.data.hf_eval_split or cfg.data.hf_train_split) == cfg.data.hf_train_split
    ):
        logger.warning(
            "Eval is currently read from the same HF split as train (%s). "
            "For SFT eval-set selection, set data.eval_file (recommended) or data.hf_eval_split to a distinct split.",
            cfg.data.hf_train_split,
        )

    if not (cfg.reward.mqm.source_lang or "").strip():
        cfg.reward.mqm.source_lang = cfg.data.default_src_lang
    if not (cfg.reward.mqm.target_lang or "").strip():
        cfg.reward.mqm.target_lang = cfg.data.default_tgt_lang

    metricx_scorer = MetricXQEScorer(cfg.reward.metricx) if cfg.reward.metricx.enabled else None
    xcomet_scorer = XCometXLScorer(cfg.reward.xcomet) if cfg.reward.xcomet.enabled else None
    mqm_scorer = OpenAICompatibleMQMScorer(cfg.reward.mqm) if cfg.reward.mqm.enabled else None

    report = evaluate_on_dataset(
        examples=eval_examples,
        policy_model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        device=device,
        metricx_scorer=metricx_scorer,
        xcomet_scorer=xcomet_scorer,
        mqm_scorer=mqm_scorer,
    )
    logger.info("Eval report: %s", report)
    return report


def run_toy_rl(cfg: RLPostTrainConfig) -> dict[str, Any]:
    set_seed(cfg.misc.seed)
    hf_token = resolve_huggingface_token(
        explicit_token=cfg.misc.huggingface_token,
        token_env_name=cfg.misc.huggingface_token_env,
    )
    configure_huggingface_cache(cfg.misc.huggingface_cache_dir, token=hf_token)
    _assign_disjoint_gpu_devices(cfg)
    use_deepspeed = cfg.rl.backend == "deepspeed"
    if use_deepspeed and _distributed_world_size() > 1 and not _is_distributed_initialized():
        if deepspeed is None:
            raise RuntimeError(
                "rl.backend=deepspeed but deepspeed is not installed. Install it first."
            )
        deepspeed.init_distributed()

    base_device = resolve_device(cfg.misc.device)
    if use_deepspeed:
        base_device = _local_rank_device(base_device)
    device = base_device
    rank0 = _is_rank0()

    output_dir = Path(cfg.logging.output_dir)
    if rank0:
        output_dir.mkdir(parents=True, exist_ok=True)
        dump_config(cfg, output_dir / "resolved_config.yaml")
    _dist_barrier()

    log_path = output_dir / cfg.logging.jsonl_name
    rollout_log_path = output_dir / cfg.logging.rollout_jsonl_name
    eval_output_path = output_dir / cfg.logging.eval_output_jsonl_name

    resume_ckpt, resume_update_idx = _resolve_resume_checkpoint(cfg, output_dir)
    resume_state = _load_trainer_state(resume_ckpt) if resume_ckpt is not None else None
    if resume_state and "update_idx" in resume_state:
        try:
            resume_update_idx = int(resume_state["update_idx"])
        except (TypeError, ValueError):
            pass
    is_resuming = resume_ckpt is not None
    start_update = resume_update_idx + 1 if is_resuming else 1

    if rank0:
        if log_path.exists() and not is_resuming:
            log_path.unlink()
        if cfg.logging.save_rollouts and rollout_log_path.exists() and not is_resuming:
            rollout_log_path.unlink()
        if cfg.logging.save_eval_outputs and eval_output_path.exists() and not is_resuming:
            eval_output_path.unlink()
        if is_resuming:
            _truncate_jsonl_by_update(log_path, resume_update_idx)
            if cfg.logging.save_rollouts:
                _truncate_jsonl_by_update(rollout_log_path, resume_update_idx)
            if cfg.logging.save_eval_outputs:
                _truncate_jsonl_by_update(eval_output_path, resume_update_idx)
    _dist_barrier()

    tokenizer_name = cfg.model.tokenizer_name_or_path or cfg.model.policy_name_or_path
    tokenizer_source = (
        str(resume_ckpt)
        if resume_ckpt is not None and (resume_ckpt / "tokenizer_config.json").exists()
        else tokenizer_name
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=cfg.model.use_fast_tokenizer)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_deepspeed:
        # DeepSpeed checkpoints are restored via engine.load_checkpoint(), not HF from_pretrained().
        policy_source = cfg.model.policy_name_or_path
    else:
        policy_source = str(resume_ckpt) if resume_ckpt is not None else cfg.model.policy_name_or_path
    policy_model = _load_policy_model(cfg, device=device, model_name_or_path=policy_source)
    train_model: Any = policy_model
    optimizer: torch.optim.Optimizer | None = None
    if use_deepspeed:
        train_model, _ = _deepspeed_initialize(cfg, policy_model)
    policy_eval_model = _unwrap_for_generation(train_model)

    ref_model: AutoModelForCausalLM | None = None
    ref_device: str | None = None
    if (not use_deepspeed) or rank0:
        ref_model, ref_device = _load_reference_model(cfg, default_device=device)

    train_examples = load_examples(cfg.data, split="train", limit=cfg.data.limit)
    if not train_examples:
        raise ValueError(
            "No train examples loaded. "
            f"Check data fields and filters: "
            f"src_text_field={cfg.data.src_text_field!r}, "
            f"id_field={cfg.data.id_field!r}, "
            f"skip_bad_source={cfg.data.skip_bad_source}, "
            f"is_bad_source_field={cfg.data.is_bad_source_field!r}, "
            f"limit={cfg.data.limit}, "
            f"hf_dataset_name={cfg.data.hf_dataset_name!r}, "
            f"hf_train_split={cfg.data.hf_train_split!r}, "
            f"train_file={cfg.data.train_file!r}."
        )

    eval_limit = cfg.eval.eval_limit if cfg.eval.eval_limit is not None else cfg.data.eval_limit
    eval_examples = load_examples(cfg.data, split="eval", limit=eval_limit)
    if (
        cfg.data.hf_dataset_name
        and not cfg.data.eval_file
        and (cfg.data.hf_eval_split or cfg.data.hf_train_split) == cfg.data.hf_train_split
    ):
        logger.warning(
            "Eval is currently read from the same HF split as train (%s). "
            "For SFT eval-set selection, set data.eval_file (recommended) or data.hf_eval_split to a distinct split.",
            cfg.data.hf_train_split,
        )

    if not (cfg.reward.mqm.source_lang or "").strip():
        cfg.reward.mqm.source_lang = cfg.data.default_src_lang
    if not (cfg.reward.mqm.target_lang or "").strip():
        cfg.reward.mqm.target_lang = cfg.data.default_tgt_lang

    metricx_scorer = (
        MetricXQEScorer(cfg.reward.metricx) if cfg.reward.metricx.enabled and ((not use_deepspeed) or rank0) else None
    )
    xcomet_scorer = (
        XCometXLScorer(cfg.reward.xcomet) if cfg.reward.xcomet.enabled and ((not use_deepspeed) or rank0) else None
    )
    mqm_scorer = (
        OpenAICompatibleMQMScorer(cfg.reward.mqm)
        if cfg.reward.mqm.enabled and ((not use_deepspeed) or rank0)
        else None
    )

    if not use_deepspeed:
        optimizer = torch.optim.AdamW(
            [p for p in policy_model.parameters() if p.requires_grad],
            lr=cfg.rl.lr,
            weight_decay=cfg.rl.weight_decay,
        )
        if is_resuming:
            optimizer_path = resume_ckpt / "optimizer.pt"
            if optimizer_path.exists():
                optimizer_state = torch.load(optimizer_path, map_location="cpu")
                optimizer.load_state_dict(optimizer_state)
            else:
                logger.warning("Resume checkpoint has no optimizer.pt: %s", resume_ckpt)
    elif is_resuming and resume_ckpt is not None:
        try:
            load_path, _ = train_model.load_checkpoint(
                str(resume_ckpt),
                tag="state",
                load_optimizer_states=True,
                load_lr_scheduler_states=False,
            )
            if load_path is None and rank0:
                logger.warning("DeepSpeed resume checkpoint could not be loaded from %s", resume_ckpt)
        except Exception as exc:
            raise RuntimeError(f"Failed to load DeepSpeed checkpoint from {resume_ckpt}") from exc

    artifacts: dict[str, Any] = {
        "output_dir": str(output_dir),
        "checkpoints": [],
    }
    if is_resuming and resume_ckpt is not None:
        artifacts["resumed_from"] = str(resume_ckpt)
        artifacts["resume_update"] = resume_update_idx

    best_dir = output_dir / "best"
    best_eval_score = float("-inf")
    best_eval_update: int | None = None

    if is_resuming:
        if resume_state:
            score_raw = resume_state.get("best_eval_score")
            update_raw = resume_state.get("best_eval_update")
            try:
                score = float(score_raw)
            except (TypeError, ValueError):
                score = float("-inf")
            try:
                update = int(update_raw) if update_raw is not None else None
            except (TypeError, ValueError):
                update = None
            if math.isfinite(score):
                best_eval_score = score
                best_eval_update = update

        if log_path.exists():
            log_best_score, log_best_update = _restore_best_from_log(log_path)
            if log_best_update is not None and (
                best_eval_update is None or log_best_score > best_eval_score
            ):
                best_eval_score = log_best_score
                best_eval_update = log_best_update

        logger.info(
            "Resuming training from %s (resume_update=%s, start_update=%s, best_eval_update=%s, best_eval_score=%s)",
            resume_ckpt,
            resume_update_idx,
            start_update,
            best_eval_update,
            best_eval_score if math.isfinite(best_eval_score) else None,
        )

    if cfg.logging.save_only_best and cfg.logging.save_every_n_updates <= 0:
        logger.warning(
            "logging.save_only_best=true with logging.save_every_n_updates<=0: "
            "resume checkpoints will not be written during training."
        )

    if cfg.eval.run_before_train and eval_examples and start_update <= 1:
        if (not use_deepspeed) or rank0:
            report = evaluate_on_dataset(
                examples=eval_examples,
                policy_model=policy_eval_model,
                tokenizer=tokenizer,
                cfg=cfg,
                device=device,
                metricx_scorer=metricx_scorer,
                xcomet_scorer=xcomet_scorer,
                mqm_scorer=mqm_scorer,
                collect_outputs=cfg.logging.save_eval_outputs,
            )
            eval_select_score = _compute_eval_selection_score(report, cfg)
            report["model_select_score"] = eval_select_score
            eval_rows = report.pop("eval_rows", [])
            _append_jsonl(log_path, {"type": "eval", "update": 0, **report})
            if cfg.logging.save_eval_outputs:
                _append_eval_output_jsonl(eval_output_path, update_idx=0, eval_rows=eval_rows)
            if math.isfinite(eval_select_score) and eval_select_score > best_eval_score:
                _save_model_only(best_dir, policy_eval_model, tokenizer)
                best_eval_score = eval_select_score
                best_eval_update = 0
                logger.info("new best eval at update=%s score=%.6f", 0, eval_select_score)
        _dist_barrier()
    elif cfg.eval.run_before_train and eval_examples and start_update > 1:
        logger.info(
            "Skipping run_before_train eval because training is resumed from update=%s.",
            start_update - 1,
        )

    metricx_cache: dict[tuple[str, str, str], float] = {}
    xcomet_cache: dict[tuple[str, str, str], tuple[float, list[dict[str, Any]]]] = {}
    mqm_cache: dict[tuple[str, str, str], tuple[float, list[dict[str, Any]]]] = {}
    rng = random.Random(cfg.misc.seed)
    train_indices = list(range(len(train_examples)))
    rng.shuffle(train_indices)
    train_cursor = 0
    updates_per_epoch = math.ceil(len(train_examples) / max(1, cfg.rl.batch_size))
    logger.info(
        "train_examples=%s batch_size=%s updates_per_epoch=%s configured_updates=%s",
        len(train_examples),
        cfg.rl.batch_size,
        updates_per_epoch,
        cfg.rl.updates,
    )

    if start_update > cfg.rl.updates:
        logger.info(
            "Nothing to train: start_update=%s exceeds configured updates=%s.",
            start_update,
            cfg.rl.updates,
        )

    for update_idx in range(start_update, cfg.rl.updates + 1):
        rollouts: list[Rollout] = []
        advantages: list[list[float]] = []
        reward_stats: dict[str, float] = {}
        adv_stats: dict[str, float] = {}
        if (not use_deepspeed) or rank0:
            if train_cursor >= len(train_indices):
                rng.shuffle(train_indices)
                train_cursor = 0
            batch_end = min(train_cursor + max(1, cfg.rl.batch_size), len(train_indices))
            batch_indices = train_indices[train_cursor:batch_end]
            train_cursor = batch_end
            batch_examples = [train_examples[i] for i in batch_indices]
            rollouts = generate_rollouts(
                examples=batch_examples,
                policy_model=policy_eval_model,
                tokenizer=tokenizer,
                gen_cfg=cfg.generation,
                device=device,
                ref_model=ref_model,
                ref_device=ref_device,
                prompt_template=cfg.prompt.template,
            )
            if rollouts:
                advantages, reward_stats, adv_stats = _prepare_rewards_and_advantages(
                    rollouts=rollouts,
                    cfg=cfg,
                    metricx_scorer=metricx_scorer,
                    xcomet_scorer=xcomet_scorer,
                    mqm_scorer=mqm_scorer,
                    metricx_cache=metricx_cache,
                    xcomet_cache=xcomet_cache,
                    mqm_cache=mqm_cache,
                )
            else:
                logger.warning("No rollouts generated at update=%s; skipping step.", update_idx)

        if use_deepspeed and _distributed_world_size() > 1:
            shared: list[Any] = [rollouts, advantages, reward_stats, adv_stats]
            _broadcast_object_list(shared, src=0)
            rollouts = shared[0] or []
            advantages = shared[1] or []
            reward_stats = shared[2] or {}
            adv_stats = shared[3] or {}

        if not rollouts:
            _dist_barrier()
            continue

        step_stats = []
        for _ in range(max(1, cfg.rl.ppo_epochs)):
            step_stats.append(
                update_policy(
                    rollouts=rollouts,
                    advantages=advantages,
                    policy_model=train_model,
                    optimizer=optimizer,
                    rl_cfg=cfg.rl,
                    device=device,
                )
            )
        train_stats = step_stats[-1]

        completion_lens = [len(r.completion_token_ids) for r in rollouts]
        payload = {
            "type": "train",
            "update": update_idx,
            "rollout_avg_completion_len": float(mean(completion_lens) if completion_lens else 0.0),
            "adv_raw_mean": adv_stats["raw_mean"],
            "adv_raw_std": adv_stats["raw_std"],
            "adv_norm_mean": adv_stats["norm_mean"],
            "adv_norm_std": adv_stats["norm_std"],
            "policy_loss": train_stats.policy_loss,
            "approx_kl": train_stats.approx_kl,
            "clip_fraction": train_stats.clip_fraction,
            "entropy": train_stats.entropy,
            "kl_to_reference": train_stats.kl_to_reference,
            "token_count": train_stats.token_count,
            **reward_stats,
        }
        if (not use_deepspeed) or rank0:
            _append_jsonl(log_path, payload)
            if cfg.logging.save_rollouts:
                _append_rollout_jsonl(
                    path=rollout_log_path,
                    update_idx=update_idx,
                    rollouts=rollouts,
                    advantages=advantages,
                    reward_stats=reward_stats,
                )

            logger.info(
                "update=%s loss=%.6f len=%.2f metricx=%.4f±%.4f xcomet=%.4f±%.4f mqm=%.4f±%.4f token_nonzero=%.4f A(raw)=%.4f/%.4f A(norm)=%.4f/%.4f",
                update_idx,
                train_stats.policy_loss,
                payload["rollout_avg_completion_len"],
                payload["metricx_score_mean"],
                payload["metricx_score_std"],
                payload["xcomet_score_mean"],
                payload["xcomet_score_std"],
                payload["mqm_score_mean"],
                payload["mqm_score_std"],
                payload["token_rewards_non_zero_ratio"],
                payload["adv_raw_mean"],
                payload["adv_raw_std"],
                payload["adv_norm_mean"],
                payload["adv_norm_std"],
            )

        trainer_state_payload = {
            "update_idx": int(update_idx),
            "best_eval_update": int(best_eval_update) if best_eval_update is not None else None,
            "best_eval_score": float(best_eval_score) if math.isfinite(best_eval_score) else None,
        }

        if cfg.logging.save_every_n_updates > 0 and update_idx % cfg.logging.save_every_n_updates == 0:
            if use_deepspeed:
                if cfg.logging.save_only_best:
                    resume_path = _save_deepspeed_resume_checkpoint(
                        output_dir=output_dir,
                        update_idx=update_idx,
                        engine=train_model,
                        tokenizer=tokenizer,
                        trainer_state=trainer_state_payload,
                    )
                    if rank0:
                        artifacts["resume_checkpoint"] = str(resume_path)
                else:
                    ckpt = _save_deepspeed_checkpoint(
                        output_dir=output_dir,
                        update_idx=update_idx,
                        engine=train_model,
                        tokenizer=tokenizer,
                        trainer_state=trainer_state_payload,
                    )
                    if rank0:
                        artifacts["checkpoints"].append(str(ckpt))
            elif rank0:
                if cfg.logging.save_only_best:
                    assert optimizer is not None
                    resume_path = _save_resume_checkpoint(
                        output_dir=output_dir,
                        update_idx=update_idx,
                        model=policy_eval_model,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        trainer_state=trainer_state_payload,
                    )
                    artifacts["resume_checkpoint"] = str(resume_path)
                else:
                    assert optimizer is not None
                    ckpt = _save_checkpoint(
                        output_dir=output_dir,
                        update_idx=update_idx,
                        model=policy_eval_model,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        trainer_state=trainer_state_payload,
                    )
                    artifacts["checkpoints"].append(str(ckpt))

        if (
            cfg.eval.eval_every_n_updates > 0
            and eval_examples
            and update_idx % cfg.eval.eval_every_n_updates == 0
        ):
            if (not use_deepspeed) or rank0:
                report = evaluate_on_dataset(
                    examples=eval_examples,
                    policy_model=policy_eval_model,
                    tokenizer=tokenizer,
                    cfg=cfg,
                    device=device,
                    metricx_scorer=metricx_scorer,
                    xcomet_scorer=xcomet_scorer,
                    mqm_scorer=mqm_scorer,
                    collect_outputs=cfg.logging.save_eval_outputs,
                )
                eval_select_score = _compute_eval_selection_score(report, cfg)
                report["model_select_score"] = eval_select_score
                eval_rows = report.pop("eval_rows", [])
                _append_jsonl(log_path, {"type": "eval", "update": update_idx, **report})
                if cfg.logging.save_eval_outputs:
                    _append_eval_output_jsonl(eval_output_path, update_idx=update_idx, eval_rows=eval_rows)
                if math.isfinite(eval_select_score) and eval_select_score > best_eval_score:
                    _save_model_only(best_dir, policy_eval_model, tokenizer)
                    best_eval_score = eval_select_score
                    best_eval_update = update_idx
                    logger.info("new best eval at update=%s score=%.6f", update_idx, eval_select_score)

        # Early-stop guard for divergence in toy runs.
        if not math.isfinite(train_stats.policy_loss):
            raise RuntimeError(f"Non-finite loss at update {update_idx}")
        _dist_barrier()

    if (not use_deepspeed) or rank0:
        final_dir = output_dir / "final"
        if final_dir.exists():
            shutil.rmtree(final_dir)
        if best_eval_update is not None and best_dir.exists():
            shutil.copytree(best_dir, final_dir)
        else:
            final_dir.mkdir(parents=True, exist_ok=True)
            policy_eval_model.save_pretrained(final_dir)
            tokenizer.save_pretrained(final_dir)
        artifacts["final_model_dir"] = str(final_dir)
        artifacts["best_model_dir"] = str(best_dir) if best_eval_update is not None and best_dir.exists() else None
        artifacts["best_eval_update"] = best_eval_update
        artifacts["best_eval_score"] = best_eval_score if best_eval_update is not None else None
        artifacts["log_path"] = str(log_path)
        if cfg.logging.save_rollouts:
            artifacts["rollout_log_path"] = str(rollout_log_path)
        if cfg.logging.save_eval_outputs:
            artifacts["eval_output_path"] = str(eval_output_path)
    _dist_barrier()

    if use_deepspeed and (not rank0):
        return {
            "output_dir": str(output_dir),
            "worker_rank": _distributed_rank(),
            "status": "ok",
        }

    return artifacts
