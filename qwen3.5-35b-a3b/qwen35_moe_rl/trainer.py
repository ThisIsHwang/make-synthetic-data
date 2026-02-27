from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import random
import shutil
from statistics import mean
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .advantage import (
    apply_group_relative_advantage,
    broadcast_sequence_reward,
    build_sequence_level_advantages,
    build_sequence_rewards,
    combine_advantages,
    normalize_advantages,
)
from .config import RLPostTrainConfig, dump_config
from .data import load_examples
from .eval import evaluate_on_dataset
from .grpo import update_policy
from .rewards import (
    MetricXQEScorer,
    XCometXLScorer,
    metricx_score_to_reward,
    spans_to_token_rewards,
)
from .rollout import generate_rollouts
from .types import DistributedContext, Rollout, SampleForScoring
from .utils import (
    all_reduce_scalar,
    barrier,
    cleanup_distributed,
    configure_huggingface_cache,
    get_local_rank,
    get_rank,
    get_world_size,
    init_distributed,
    is_distributed,
    is_main_process,
    resolve_device,
    resolve_huggingface_token,
    resolve_torch_dtype,
    set_seed,
)


logger = logging.getLogger(__name__)


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


def _append_rollout_jsonl(
    path: Path,
    update_idx: int,
    rollouts: list[Rollout],
    advantages: list[list[float]] | list[float],
    reward_stats: dict[str, float],
) -> None:
    is_seq_level = bool(advantages and not isinstance(advantages[0], list))
    for ridx, rollout in enumerate(rollouts):
        if is_seq_level:
            adv_scalar = float(advantages[ridx]) if ridx < len(advantages) else 0.0
            adv_mean = adv_scalar
            adv_sum = adv_scalar
        else:
            adv_row = advantages[ridx] if ridx < len(advantages) else []
            adv_mean = float(sum(adv_row) / len(adv_row)) if adv_row else 0.0
            adv_sum = float(sum(adv_row)) if adv_row else 0.0

        payload = {
            "type": "rollout",
            "update": update_idx,
            "rollout_idx": ridx,
            "example_id": rollout.example_id,
            "src_text": rollout.src_text,
            "completion_text": rollout.completion_text,
            "ref_text": rollout.ref_text,
            "completion_len": len(rollout.completion_token_ids),
            "adv_mean": adv_mean,
            "adv_sum": adv_sum,
            "old_logprob_mean": float(sum(rollout.old_logprobs) / len(rollout.old_logprobs))
            if rollout.old_logprobs
            else 0.0,
            "ref_logprob_mean": float(sum(rollout.ref_logprobs) / len(rollout.ref_logprobs))
            if rollout.ref_logprobs
            else None,
            "metricx_score_mean_batch": reward_stats.get("metricx_score_mean", 0.0),
            "xcomet_score_mean_batch": reward_stats.get("xcomet_score_mean", 0.0),
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


def _build_max_memory_map(device_ids: list[int]) -> dict[int | str, str]:
    if not device_ids or torch is None or not torch.cuda.is_available():
        return {}
    max_memory: dict[int | str, str] = {}
    for idx in device_ids:
        total_bytes = int(torch.cuda.get_device_properties(idx).total_memory)
        usable_bytes = int(total_bytes * 0.92)
        gib = max(1, usable_bytes // (1024**3))
        max_memory[idx] = f"{gib}GiB"
    max_memory["cpu"] = "256GiB"
    return max_memory


# ---------------------------------------------------------------------------
# Distributed training helpers
# ---------------------------------------------------------------------------


def _build_deepspeed_config(cfg: RLPostTrainConfig) -> dict[str, Any]:
    """Build a DeepSpeed config dict from our training configuration.

    If ``cfg.distributed.deepspeed_config_path`` is set, load and return
    that JSON instead (allowing full user override).
    """
    if cfg.distributed.deepspeed_config_path:
        path = Path(cfg.distributed.deepspeed_config_path)
        if not path.exists():
            raise FileNotFoundError(f"DeepSpeed config not found: {path}")
        ds_cfg: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        logger.info("Loaded external DeepSpeed config from %s", path)
        return ds_cfg

    use_bf16 = cfg.misc.dtype in ("bfloat16", "bf16")
    use_fp16 = cfg.misc.dtype in ("float16", "fp16")

    zero_opt: dict[str, Any] = {
        "stage": cfg.distributed.zero_stage,
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "overlap_comm": True,
        "contiguous_gradients": True,
    }
    if cfg.distributed.offload_optimizer:
        zero_opt["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
    if cfg.distributed.offload_param and cfg.distributed.zero_stage >= 3:
        zero_opt["offload_param"] = {"device": "cpu", "pin_memory": True}

    ds_config: dict[str, Any] = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": max(1, cfg.rl.grad_accum),
        "gradient_clipping": cfg.rl.max_grad_norm,
        "zero_optimization": zero_opt,
        "zero_allow_untested_optimizer": True,
        "steps_per_print": 9999999,
    }
    if use_bf16:
        ds_config["bf16"] = {"enabled": True}
    elif use_fp16:
        ds_config["fp16"] = {"enabled": True, "loss_scale": 0, "initial_scale_power": 16}
    else:
        ds_config["fp16"] = {"enabled": False}

    return ds_config


def _init_deepspeed(
    model: AutoModelForCausalLM,
    cfg: RLPostTrainConfig,
) -> tuple[Any, torch.optim.Optimizer, DistributedContext]:
    """Initialize DeepSpeed engine, wrapping model + optimizer.

    Returns ``(engine, optimizer_ref, dist_ctx)``.
    The ``engine`` acts as both model (for forward) and optimizer controller.
    """
    try:
        import deepspeed
    except ImportError as exc:
        raise ImportError(
            "DeepSpeed is required for distributed.backend='deepspeed'. "
            "Install with: pip install deepspeed"
        ) from exc

    ds_config = _build_deepspeed_config(cfg)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.rl.lr,
        weight_decay=cfg.rl.weight_decay,
    )

    engine, optimizer_ref, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config,
    )
    logger.info(
        "DeepSpeed engine initialized: zero_stage=%s world_size=%s",
        cfg.distributed.zero_stage,
        get_world_size(),
    )

    dist_ctx = DistributedContext(
        backend="deepspeed",
        deepspeed_engine=engine,
    )
    return engine, optimizer_ref, dist_ctx


def _init_fsdp(
    model: AutoModelForCausalLM,
    cfg: RLPostTrainConfig,
    device: str,
) -> tuple[AutoModelForCausalLM, DistributedContext]:
    """Wrap model with PyTorch FSDP.

    Returns ``(wrapped_model, dist_ctx)``.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

    strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
    }
    sharding = strategy_map.get(
        cfg.distributed.fsdp_sharding_strategy.upper(),
        ShardingStrategy.FULL_SHARD,
    )

    dtype = resolve_torch_dtype(cfg.misc.dtype)
    mp_policy = None
    if dtype in (torch.bfloat16, torch.float16):
        mp_policy = MixedPrecision(
            param_dtype=dtype,
            reduce_dtype=dtype,
            buffer_dtype=dtype,
        )

    wrapped = FSDP(
        model,
        sharding_strategy=sharding,
        mixed_precision=mp_policy,
        device_id=torch.device(device),
    )
    logger.info(
        "FSDP initialized: sharding=%s world_size=%s",
        cfg.distributed.fsdp_sharding_strategy,
        get_world_size(),
    )

    dist_ctx = DistributedContext(backend="fsdp")
    return wrapped, dist_ctx


def _init_ddp(
    model: AutoModelForCausalLM,
    device: str,
) -> tuple[AutoModelForCausalLM, DistributedContext]:
    """Wrap model with PyTorch DDP."""
    from torch.nn.parallel import DistributedDataParallel as DDP

    wrapped = DDP(model, device_ids=[torch.device(device).index])
    logger.info("DDP initialized: world_size=%s", get_world_size())
    dist_ctx = DistributedContext(backend="ddp")
    return wrapped, dist_ctx


def _save_checkpoint_distributed(
    output_dir: Path,
    update_idx: int,
    model: AutoModelForCausalLM,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    dist_ctx: DistributedContext | None,
) -> Path:
    """Save checkpoint with distributed awareness."""
    ckpt_dir = output_dir / f"checkpoint-{update_idx}"

    if dist_ctx is not None and dist_ctx.deepspeed_engine is not None:
        dist_ctx.deepspeed_engine.save_checkpoint(str(output_dir), tag=f"checkpoint-{update_idx}")
        if is_main_process():
            tokenizer.save_pretrained(ckpt_dir)
    elif is_main_process():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = model.module if hasattr(model, "module") else model
        unwrapped.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")

    barrier()
    return ckpt_dir


def _save_model_only_distributed(
    save_dir: Path,
    model: AutoModelForCausalLM,
    tokenizer: Any,
    dist_ctx: DistributedContext | None,
) -> Path:
    """Save model with distributed awareness (rank 0 only for DDP/FSDP)."""
    if dist_ctx is not None and dist_ctx.deepspeed_engine is not None:
        if is_main_process():
            if save_dir.exists():
                shutil.rmtree(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        barrier()
        dist_ctx.deepspeed_engine.save_checkpoint(str(save_dir), tag="model")
        if is_main_process():
            tokenizer.save_pretrained(save_dir)
    elif is_main_process():
        if save_dir.exists():
            shutil.rmtree(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = model.module if hasattr(model, "module") else model
        unwrapped.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

    barrier()
    return save_dir


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

    max_memory = _build_max_memory_map(explicit_gpu_ids)
    mp_kwargs = dict(kwargs)
    mp_kwargs["device_map"] = "auto"
    if max_memory:
        mp_kwargs["max_memory"] = max_memory

    logger.info(
        "Loading %s model with model-parallel device_map=auto on GPUs=%s",
        component_name,
        explicit_gpu_ids,
    )
    try:
        return AutoModelForCausalLM.from_pretrained(model_name_or_path, **mp_kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load {component_name} model with multi-GPU partition {explicit_gpu_ids}. "
            "Check accelerate/transformers versions and model size."
        ) from exc


def _load_policy_model(cfg: RLPostTrainConfig, device: str) -> AutoModelForCausalLM:
    dtype, attn_impl = _resolve_model_dtype_and_attn(cfg, device)

    kwargs: dict[str, Any] = {
        "trust_remote_code": cfg.model.trust_remote_code,
    }
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl

    model = _load_causal_lm(
        model_name_or_path=cfg.model.policy_name_or_path,
        kwargs=kwargs,
        single_device=device,
        gpu_ids=cfg.model.policy_gpu_ids,
        component_name="policy",
    )

    # Enable output_router_logits in model config for MoE aux loss / monitoring.
    if cfg.rl.output_router_logits:
        if hasattr(model.config, "output_router_logits"):
            model.config.output_router_logits = True
            logger.info("Enabled output_router_logits in model config.")
        else:
            logger.warning(
                "Model config does not have output_router_logits attribute. "
                "Router logits may not be available. Is this a MoE model?"
            )

    # Enable gradient checkpointing for MoE memory optimization.
    if cfg.model.gradient_checkpointing:
        gc_kwargs = cfg.model.gradient_checkpointing_kwargs or {}
        model.gradient_checkpointing_enable(gc_kwargs)
        logger.info(
            "Enabled gradient checkpointing with kwargs=%s",
            gc_kwargs,
        )

    return model


def _load_reference_model(cfg: RLPostTrainConfig, default_device: str) -> tuple[AutoModelForCausalLM | None, str | None]:
    if cfg.rl.kl_coef <= 0:
        return None, None

    ref_name = cfg.model.reference_name_or_path or cfg.model.policy_name_or_path
    reference_gpu_ids = _normalize_gpu_id_list(cfg.model.reference_gpu_ids)
    if reference_gpu_ids:
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
        gpu_ids=reference_gpu_ids,
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


def _prepare_rewards_and_advantages(
    rollouts: list[Rollout],
    cfg: RLPostTrainConfig,
    metricx_scorer: MetricXQEScorer | None,
    xcomet_scorer: XCometXLScorer | None,
    metricx_cache: dict[tuple[str, str, str], float],
    xcomet_cache: dict[tuple[str, str, str], tuple[float, list[dict[str, Any]]]],
) -> tuple[list[list[float]] | list[float], dict[str, float], dict[str, float]]:
    """Compute rewards and advantages for the batch of rollouts.

    Returns token-level advantages (list[list[float]]) for GRPO/REINFORCE,
    or sequence-level scalar advantages (list[float]) for GSPO.
    """
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

    seq_rewards = build_sequence_rewards(
        metricx_scores=metricx_scores,
        xcomet_scores=xcomet_scores,
        metricx_offset=cfg.reward.metricx.offset,
        w_metricx=cfg.reward.w_metricx,
        w_xcomet_seq=cfg.reward.w_xcomet_seq,
        xcomet_seq_scale=cfg.reward.xcomet_seq_scale,
    )

    # ---- Common reward stats ----
    metricx_m, metricx_s = _mean_std(metricx_scores)
    metricx_r_m, metricx_r_s = _mean_std(metricx_rewards)
    xcomet_m, xcomet_s = _mean_std(xcomet_scores)

    # ---- GSPO branch: sequence-level advantages ----
    if cfg.rl.algorithm == "gspo":
        seq_advantages, adv_stats = build_sequence_level_advantages(
            sequence_rewards=seq_rewards,
            group_ids=[r.example_id for r in rollouts],
            normalize=cfg.rl.normalize_advantage,
            group_normalize=cfg.rl.group_normalize,
            group_coef=cfg.rl.group_advantage_coef,
            eps=cfg.rl.eps,
        )

        reward_stats = {
            "metricx_score_mean": metricx_m,
            "metricx_score_std": metricx_s,
            "metricx_reward_mean": metricx_r_m,
            "metricx_reward_std": metricx_r_s,
            "xcomet_score_mean": xcomet_m,
            "xcomet_score_std": xcomet_s,
            "token_rewards_mean": 0.0,
            "token_rewards_std": 0.0,
            "token_rewards_non_zero_ratio": 0.0,
            "span_minor_mean": 0.0,
            "span_major_mean": 0.0,
            "span_critical_mean": 0.0,
        }
        return seq_advantages, reward_stats, adv_stats

    # ---- GRPO / REINFORCE branch: token-level advantages ----
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

    reward_stats = {
        "metricx_score_mean": metricx_m,
        "metricx_score_std": metricx_s,
        "metricx_reward_mean": metricx_r_m,
        "metricx_reward_std": metricx_r_s,
        "xcomet_score_mean": xcomet_m,
        "xcomet_score_std": xcomet_s,
        "token_rewards_mean": token_reward_m,
        "token_rewards_std": token_reward_s,
        "token_rewards_non_zero_ratio": float(non_zero_token_ratio),
        "span_minor_mean": float(mean(severity_counts["MINOR"]) if severity_counts["MINOR"] else 0.0),
        "span_major_mean": float(mean(severity_counts["MAJOR"]) if severity_counts["MAJOR"] else 0.0),
        "span_critical_mean": float(mean(severity_counts["CRITICAL"]) if severity_counts["CRITICAL"] else 0.0),
    }

    return norm_adv_rows, reward_stats, norm_stats


def _save_checkpoint(
    output_dir: Path,
    update_idx: int,
    model: AutoModelForCausalLM,
    tokenizer,
    optimizer: torch.optim.Optimizer,
) -> Path:
    ckpt_dir = output_dir / f"checkpoint-{update_idx}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    return ckpt_dir


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
    return float(metricx_term + xcomet_term)


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

    metricx_scorer = MetricXQEScorer(cfg.reward.metricx) if cfg.reward.metricx.enabled else None
    xcomet_scorer = XCometXLScorer(cfg.reward.xcomet) if cfg.reward.xcomet.enabled else None

    report = evaluate_on_dataset(
        examples=eval_examples,
        policy_model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        device=device,
        metricx_scorer=metricx_scorer,
        xcomet_scorer=xcomet_scorer,
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

    # ---- Distributed initialization (opt-in) ----
    dist_ctx: DistributedContext | None = None
    use_distributed = cfg.distributed.enabled

    if use_distributed:
        init_distributed(backend="nccl")
        device = f"cuda:{get_local_rank()}"
        cfg.misc.device = device
        logger.info(
            "Distributed mode: rank=%s/%s local_rank=%s device=%s backend=%s",
            get_rank(), get_world_size(), get_local_rank(), device, cfg.distributed.backend,
        )
    else:
        _assign_disjoint_gpu_devices(cfg)
        device = resolve_device(cfg.misc.device)

    output_dir = Path(cfg.logging.output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        dump_config(cfg, output_dir / "resolved_config.yaml")
    if use_distributed:
        barrier()

    tokenizer_name = cfg.model.tokenizer_name_or_path or cfg.model.policy_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=cfg.model.use_fast_tokenizer)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    policy_model = _load_policy_model(cfg, device=device)
    ref_model, ref_device = _load_reference_model(cfg, default_device=device)

    # ---- Distributed wrapping (DeepSpeed / FSDP / DDP) ----
    optimizer: torch.optim.Optimizer
    if use_distributed:
        backend = cfg.distributed.backend.lower()
        if backend == "deepspeed":
            policy_model, optimizer, dist_ctx = _init_deepspeed(policy_model, cfg)
        elif backend == "fsdp":
            policy_model, dist_ctx = _init_fsdp(policy_model, cfg, device)
            optimizer = torch.optim.AdamW(
                [p for p in policy_model.parameters() if p.requires_grad],
                lr=cfg.rl.lr,
                weight_decay=cfg.rl.weight_decay,
            )
        elif backend == "ddp":
            policy_model, dist_ctx = _init_ddp(policy_model, device)
            optimizer = torch.optim.AdamW(
                [p for p in policy_model.parameters() if p.requires_grad],
                lr=cfg.rl.lr,
                weight_decay=cfg.rl.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported distributed backend: {backend}")
    else:
        optimizer = torch.optim.AdamW(
            [p for p in policy_model.parameters() if p.requires_grad],
            lr=cfg.rl.lr,
            weight_decay=cfg.rl.weight_decay,
        )

    try:
        return _run_training_loop(
            cfg=cfg,
            device=device,
            policy_model=policy_model,
            ref_model=ref_model,
            ref_device=ref_device,
            tokenizer=tokenizer,
            optimizer=optimizer,
            dist_ctx=dist_ctx,
            output_dir=output_dir,
        )
    finally:
        if use_distributed:
            cleanup_distributed()


def _run_training_loop(
    cfg: RLPostTrainConfig,
    device: str,
    policy_model: Any,
    ref_model: Any,
    ref_device: str | None,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    dist_ctx: DistributedContext | None,
    output_dir: Path,
) -> dict[str, Any]:
    """Core training loop, extracted to keep run_toy_rl() focused on setup/teardown."""
    use_distributed = cfg.distributed.enabled

    train_examples = load_examples(cfg.data, split="train", limit=cfg.data.limit)
    if not train_examples:
        raise ValueError("No train examples loaded.")

    eval_limit = cfg.eval.eval_limit if cfg.eval.eval_limit is not None else cfg.data.eval_limit
    eval_examples = load_examples(cfg.data, split="eval", limit=eval_limit)

    metricx_scorer = MetricXQEScorer(cfg.reward.metricx) if cfg.reward.metricx.enabled else None
    xcomet_scorer = XCometXLScorer(cfg.reward.xcomet) if cfg.reward.xcomet.enabled else None

    log_path = output_dir / cfg.logging.jsonl_name
    rollout_log_path = output_dir / cfg.logging.rollout_jsonl_name
    eval_output_path = output_dir / cfg.logging.eval_output_jsonl_name

    if is_main_process():
        if log_path.exists():
            log_path.unlink()
        if cfg.logging.save_rollouts and rollout_log_path.exists():
            rollout_log_path.unlink()
        if cfg.logging.save_eval_outputs and eval_output_path.exists():
            eval_output_path.unlink()

    artifacts: dict[str, Any] = {
        "output_dir": str(output_dir),
        "checkpoints": [],
    }
    best_dir = output_dir / "best"
    best_eval_score = float("-inf")
    best_eval_update: int | None = None

    # ---- Pre-training evaluation ----
    if cfg.eval.run_before_train and eval_examples and is_main_process():
        report = evaluate_on_dataset(
            examples=eval_examples,
            policy_model=policy_model,
            tokenizer=tokenizer,
            cfg=cfg,
            device=device,
            metricx_scorer=metricx_scorer,
            xcomet_scorer=xcomet_scorer,
            collect_outputs=cfg.logging.save_eval_outputs,
        )
        eval_select_score = _compute_eval_selection_score(report, cfg)
        report["model_select_score"] = eval_select_score
        eval_rows = report.pop("eval_rows", [])
        _append_jsonl(log_path, {"type": "eval", "update": 0, **report})
        if cfg.logging.save_eval_outputs:
            _append_eval_output_jsonl(eval_output_path, update_idx=0, eval_rows=eval_rows)
        if math.isfinite(eval_select_score) and eval_select_score > best_eval_score:
            if use_distributed:
                _save_model_only_distributed(best_dir, policy_model, tokenizer, dist_ctx)
            else:
                _save_model_only(best_dir, policy_model, tokenizer)
            best_eval_score = eval_select_score
            best_eval_update = 0
            logger.info("new best eval at update=%s score=%.6f", 0, eval_select_score)
    if use_distributed:
        barrier()

    metricx_cache: dict[tuple[str, str, str], float] = {}
    xcomet_cache: dict[tuple[str, str, str], tuple[float, list[dict[str, Any]]]] = {}
    rng = random.Random(cfg.misc.seed + get_rank())
    train_indices = list(range(len(train_examples)))
    rng.shuffle(train_indices)

    # ---- Data sharding for distributed ----
    if use_distributed:
        ws = get_world_size()
        rank = get_rank()
        shard_size = len(train_indices) // ws
        shard_start = rank * shard_size
        shard_end = shard_start + shard_size if rank < ws - 1 else len(train_indices)
        train_indices = train_indices[shard_start:shard_end]
        logger.info(
            "Rank %s data shard: %s examples (indices %s..%s of %s total)",
            rank, len(train_indices), shard_start, shard_end, len(train_examples),
        )

    train_cursor = 0
    updates_per_epoch = math.ceil(len(train_indices) / max(1, cfg.rl.batch_size))
    if is_main_process():
        logger.info(
            "train_examples=%s batch_size=%s updates_per_epoch=%s configured_updates=%s algorithm=%s distributed=%s",
            len(train_examples),
            cfg.rl.batch_size,
            updates_per_epoch,
            cfg.rl.updates,
            cfg.rl.algorithm,
            use_distributed,
        )

    for update_idx in range(1, cfg.rl.updates + 1):
        if train_cursor >= len(train_indices):
            rng.shuffle(train_indices)
            train_cursor = 0
        batch_end = min(train_cursor + max(1, cfg.rl.batch_size), len(train_indices))
        batch_indices = train_indices[train_cursor:batch_end]
        train_cursor = batch_end
        batch_examples = [train_examples[i] for i in batch_indices]

        # Use the underlying model for rollout generation (not the DeepSpeed engine).
        rollout_model = policy_model
        if dist_ctx is not None and dist_ctx.deepspeed_engine is not None:
            rollout_model = dist_ctx.deepspeed_engine.module

        rollouts = generate_rollouts(
            examples=batch_examples,
            policy_model=rollout_model,
            tokenizer=tokenizer,
            gen_cfg=cfg.generation,
            device=device,
            ref_model=ref_model,
            ref_device=ref_device,
            prompt_template=cfg.prompt.template,
        )
        if not rollouts:
            logger.warning("No rollouts generated at update=%s; skipping step.", update_idx)
            continue

        advantages, reward_stats, adv_stats = _prepare_rewards_and_advantages(
            rollouts=rollouts,
            cfg=cfg,
            metricx_scorer=metricx_scorer,
            xcomet_scorer=xcomet_scorer,
            metricx_cache=metricx_cache,
            xcomet_cache=xcomet_cache,
        )

        step_stats = []
        for _ in range(max(1, cfg.rl.ppo_epochs)):
            step_stats.append(
                update_policy(
                    rollouts=rollouts,
                    advantages=advantages,
                    policy_model=policy_model,
                    optimizer=optimizer,
                    rl_cfg=cfg.rl,
                    device=device,
                    dist_ctx=dist_ctx,
                )
            )
        train_stats = step_stats[-1]

        # ---- Logging (rank 0 only in distributed mode) ----
        if is_main_process():
            completion_lens = [len(r.completion_token_ids) for r in rollouts]
            payload = {
                "type": "train",
                "update": update_idx,
                "algorithm": cfg.rl.algorithm,
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
                # MoE monitoring metrics.
                "aux_loss": train_stats.aux_loss,
                "expert_utilization_entropy": train_stats.expert_utilization_entropy,
                "max_expert_load": train_stats.max_expert_load,
                "min_expert_load": train_stats.min_expert_load,
                **reward_stats,
            }
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
                "update=%s loss=%.6f len=%.2f metricx=%.4f±%.4f xcomet=%.4f±%.4f "
                "aux_loss=%.6f expert_ent=%.4f A(raw)=%.4f/%.4f A(norm)=%.4f/%.4f",
                update_idx,
                train_stats.policy_loss,
                payload["rollout_avg_completion_len"],
                payload["metricx_score_mean"],
                payload["metricx_score_std"],
                payload["xcomet_score_mean"],
                payload["xcomet_score_std"],
                train_stats.aux_loss,
                train_stats.expert_utilization_entropy,
                payload["adv_raw_mean"],
                payload["adv_raw_std"],
                payload["adv_norm_mean"],
                payload["adv_norm_std"],
            )

        # ---- Checkpoint saving ----
        if cfg.logging.save_every_n_updates > 0 and update_idx % cfg.logging.save_every_n_updates == 0:
            if use_distributed:
                ckpt = _save_checkpoint_distributed(
                    output_dir, update_idx, policy_model, tokenizer, optimizer, dist_ctx,
                )
            else:
                ckpt = _save_checkpoint(output_dir, update_idx, policy_model, tokenizer, optimizer)
            artifacts["checkpoints"].append(str(ckpt))

        # ---- Periodic evaluation ----
        if (
            cfg.eval.eval_every_n_updates > 0
            and eval_examples
            and update_idx % cfg.eval.eval_every_n_updates == 0
        ):
            eval_model = policy_model
            if dist_ctx is not None and dist_ctx.deepspeed_engine is not None:
                eval_model = dist_ctx.deepspeed_engine.module

            if is_main_process():
                report = evaluate_on_dataset(
                    examples=eval_examples,
                    policy_model=eval_model,
                    tokenizer=tokenizer,
                    cfg=cfg,
                    device=device,
                    metricx_scorer=metricx_scorer,
                    xcomet_scorer=xcomet_scorer,
                    collect_outputs=cfg.logging.save_eval_outputs,
                )
                eval_select_score = _compute_eval_selection_score(report, cfg)
                report["model_select_score"] = eval_select_score
                eval_rows = report.pop("eval_rows", [])
                _append_jsonl(log_path, {"type": "eval", "update": update_idx, **report})
                if cfg.logging.save_eval_outputs:
                    _append_eval_output_jsonl(eval_output_path, update_idx=update_idx, eval_rows=eval_rows)
                if math.isfinite(eval_select_score) and eval_select_score > best_eval_score:
                    if use_distributed:
                        _save_model_only_distributed(best_dir, policy_model, tokenizer, dist_ctx)
                    else:
                        _save_model_only(best_dir, policy_model, tokenizer)
                    best_eval_score = eval_select_score
                    best_eval_update = update_idx
                    logger.info("new best eval at update=%s score=%.6f", update_idx, eval_select_score)
            if use_distributed:
                barrier()

        # Early-stop guard for divergence.
        if not math.isfinite(train_stats.policy_loss):
            raise RuntimeError(f"Non-finite loss at update {update_idx}")

    # ---- Save final model ----
    final_dir = output_dir / "final"
    if is_main_process():
        if final_dir.exists():
            shutil.rmtree(final_dir)
        if best_eval_update is not None and best_dir.exists():
            shutil.copytree(best_dir, final_dir)
        else:
            final_dir.mkdir(parents=True, exist_ok=True)
            unwrapped = policy_model.module if hasattr(policy_model, "module") else policy_model
            unwrapped.save_pretrained(final_dir)
            tokenizer.save_pretrained(final_dir)
    if use_distributed:
        barrier()

    artifacts["final_model_dir"] = str(final_dir)
    artifacts["best_model_dir"] = str(best_dir) if best_eval_update is not None and best_dir.exists() else None
    artifacts["best_eval_update"] = best_eval_update
    artifacts["best_eval_score"] = best_eval_score if best_eval_update is not None else None
    artifacts["log_path"] = str(log_path)
    if cfg.logging.save_rollouts:
        artifacts["rollout_log_path"] = str(rollout_log_path)
    if cfg.logging.save_eval_outputs:
        artifacts["eval_output_path"] = str(eval_output_path)

    return artifacts
