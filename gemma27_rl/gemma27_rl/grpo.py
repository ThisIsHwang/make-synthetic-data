from __future__ import annotations

import logging
import math
import os

import torch
import torch.nn.functional as F
from torch import nn

from .config import RLConfig
from .rl_types import Rollout, TrainStats

logger = logging.getLogger(__name__)


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


def _is_rank0_process() -> bool:
    rank_raw = os.environ.get("RANK")
    if rank_raw is None:
        return True
    if not rank_raw.isdigit():
        return True
    return int(rank_raw) == 0


def _resolve_model_vocab_size(model: nn.Module) -> int | None:
    getter = getattr(model, "get_input_embeddings", None)
    if callable(getter):
        try:
            emb = getter()
            size = int(getattr(emb, "num_embeddings", 0) or 0)
            if size > 0:
                return size
        except Exception:
            pass
    cfg_obj = getattr(model, "config", None)
    try:
        size = int(getattr(cfg_obj, "vocab_size", 0) or 0)
    except Exception:
        size = 0
    return size if size > 0 else None


def _validate_rollout_token_ids(
    *,
    rollout: Rollout,
    vocab_size: int | None,
) -> None:
    if vocab_size is None or vocab_size <= 0:
        return
    for field_name, ids in (
        ("prompt_input_ids", rollout.prompt_input_ids),
        ("completion_token_ids", rollout.completion_token_ids),
    ):
        if not ids:
            continue
        low = min(int(v) for v in ids)
        high = max(int(v) for v in ids)
        if low < 0 or high >= vocab_size:
            raise ValueError(
                "Token id out of vocab range before policy forward: "
                f"example_id={rollout.example_id} field={field_name} min={low} max={high} vocab_size={vocab_size}. "
                "Tokenizer/model mismatch is likely."
            )


def _token_logprobs_and_entropy(
    logits: torch.Tensor,
    prompt_len: int,
    completion_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(completion_ids) == 0:
        empty = torch.empty(0, device=logits.device, dtype=logits.dtype)
        return empty, empty

    start = prompt_len - 1
    end = start + len(completion_ids)
    selected = logits[0, start:end, :]
    log_probs = F.log_softmax(selected, dim=-1)
    labels = torch.tensor(completion_ids, device=logits.device, dtype=torch.long)
    token_logprobs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    return token_logprobs, entropy


def _align_tensors(
    new_logprobs: torch.Tensor,
    old_logprobs: list[float],
    advantages: list[float],
    ref_logprobs: list[float] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    target_device = new_logprobs.device
    length = min(len(new_logprobs), len(old_logprobs), len(advantages))
    if ref_logprobs is not None:
        length = min(length, len(ref_logprobs))

    if length <= 0:
        empty = torch.empty(0, device=target_device, dtype=new_logprobs.dtype)
        return empty, empty, empty, None

    new_lp = new_logprobs[:length]
    old_lp = torch.tensor(old_logprobs[:length], device=target_device, dtype=new_logprobs.dtype)
    adv = torch.tensor(advantages[:length], device=target_device, dtype=new_logprobs.dtype)
    ref_lp = None
    if ref_logprobs is not None:
        ref_lp = torch.tensor(ref_logprobs[:length], device=target_device, dtype=new_logprobs.dtype)
    return new_lp, old_lp, adv, ref_lp


def update_policy(
    rollouts: list[Rollout],
    advantages: list[list[float]],
    policy_model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    rl_cfg: RLConfig,
    device: str,
) -> TrainStats:
    if len(rollouts) != len(advantages):
        raise ValueError("rollouts and advantages length mismatch")
    if not rollouts:
        raise ValueError("rollouts are empty")

    policy_model.train()

    # DeepSpeed engines expose backward/step on the model object.
    use_engine_step = bool(
        optimizer is None
        and callable(getattr(policy_model, "backward", None))
        and callable(getattr(policy_model, "step", None))
    )
    if optimizer is None and not use_engine_step:
        raise ValueError("optimizer is required unless policy_model is a DeepSpeed-like engine.")

    vocab_size = _resolve_model_vocab_size(policy_model)

    if use_engine_step:
        zero_grad_fn = getattr(policy_model, "zero_grad", None)
        if callable(zero_grad_fn):
            try:
                zero_grad_fn(set_to_none=True)
            except TypeError:
                zero_grad_fn()
    else:
        assert optimizer is not None  # for static type checkers
        optimizer.zero_grad(set_to_none=True)

    total_tokens = 0
    total_loss_value = 0.0
    pending_backward = 0

    total_approx_kl = 0.0
    total_clip = 0.0
    total_entropy = 0.0
    total_ref_kl = 0.0
    debug_loss_trace = _is_rank0_process() and (
        _env_flag("GEMMA27_RL_DEBUG_LOSS_TRACE", default=False)
        or _env_flag("GEMMA27_RL_DEBUG_SPAN_LOSS", default=False)
    )
    debug_loss_max_rollouts = _env_int("GEMMA27_RL_DEBUG_LOSS_MAX_ROLLOUTS", default=1, minimum=1)
    debug_loss_max_tokens = _env_int("GEMMA27_RL_DEBUG_LOSS_MAX_TOKENS", default=256, minimum=1)
    debug_loss_only_nonzero_adv = _env_flag("GEMMA27_RL_DEBUG_LOSS_ONLY_NONZERO_ADV", default=False)
    debug_loss_logged = 0

    for rollout, adv_row in zip(rollouts, advantages):
        _validate_rollout_token_ids(rollout=rollout, vocab_size=vocab_size)
        input_ids = torch.tensor(
            [rollout.prompt_input_ids + rollout.completion_token_ids],
            device=device,
            dtype=torch.long,
        )
        attention_mask = torch.ones_like(input_ids)
        try:
            # Keep training activation memory bounded; KV cache is only useful for autoregressive generation.
            outputs = policy_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        except TypeError:
            outputs = policy_model(input_ids=input_ids, attention_mask=attention_mask)
        new_logprobs, entropy = _token_logprobs_and_entropy(
            outputs.logits,
            prompt_len=len(rollout.prompt_input_ids),
            completion_ids=rollout.completion_token_ids,
        )

        new_lp, old_lp, adv, ref_lp = _align_tensors(
            new_logprobs,
            old_logprobs=rollout.old_logprobs,
            advantages=adv_row,
            ref_logprobs=rollout.ref_logprobs,
        )
        if new_lp.numel() == 0:
            continue

        if rl_cfg.algorithm == "grpo":
            ratio = torch.exp(new_lp - old_lp)
            clipped = torch.clamp(ratio, 1.0 - rl_cfg.clip_eps, 1.0 + rl_cfg.clip_eps)
            policy_term = -torch.minimum(ratio * adv, clipped * adv)
            clip_fraction = ((ratio > (1.0 + rl_cfg.clip_eps)) | (ratio < (1.0 - rl_cfg.clip_eps))).float()
            approx_kl = 0.5 * ((new_lp - old_lp) ** 2)
        elif rl_cfg.algorithm == "reinforce":
            ratio = torch.ones_like(new_lp)
            clipped = ratio
            policy_term = -(new_lp * adv)
            clip_fraction = torch.zeros_like(new_lp)
            approx_kl = 0.5 * ((new_lp - old_lp) ** 2)
        else:
            raise ValueError(f"Unsupported algorithm: {rl_cfg.algorithm}")

        per_token_loss = policy_term
        kl_term = torch.zeros_like(per_token_loss)
        if rl_cfg.kl_coef > 0 and ref_lp is not None:
            ref_kl = new_lp - ref_lp
            kl_term = rl_cfg.kl_coef * ref_kl
            per_token_loss = per_token_loss + kl_term
            total_ref_kl += float(ref_kl.detach().sum().item())

        entropy_term = torch.zeros_like(per_token_loss)
        if rl_cfg.entropy_coef > 0:
            ent = entropy[: per_token_loss.numel()]
            entropy_term = -(rl_cfg.entropy_coef * ent)
            per_token_loss = per_token_loss + entropy_term
            total_entropy += float(ent.detach().sum().item())

        if debug_loss_trace and debug_loss_logged < debug_loss_max_rollouts:
            logger.info(
                "[loss-debug] example_id=%s tokens=%s algorithm=%s",
                rollout.example_id,
                int(per_token_loss.numel()),
                rl_cfg.algorithm,
            )
            printed = 0
            non_zero_adv = 0
            for tok_idx in range(int(per_token_loss.numel())):
                adv_i = float(adv[tok_idx].detach().item())
                if abs(adv_i) > 0:
                    non_zero_adv += 1
                if debug_loss_only_nonzero_adv and abs(adv_i) <= 0:
                    continue
                if printed >= debug_loss_max_tokens:
                    break
                tok_id = int(rollout.completion_token_ids[tok_idx]) if tok_idx < len(rollout.completion_token_ids) else -1
                if ref_lp is not None and tok_idx < int(ref_lp.numel()):
                    logger.info(
                        "[loss-debug] tok[%03d] id=%s old_lp=%.6f new_lp=%.6f ref_lp=%.6f adv=%.6f ratio=%.6f clipped=%.6f policy=%.6f kl=%.6f entropy=%.6f total=%.6f",
                        tok_idx,
                        tok_id,
                        float(old_lp[tok_idx].detach().item()),
                        float(new_lp[tok_idx].detach().item()),
                        float(ref_lp[tok_idx].detach().item()),
                        adv_i,
                        float(ratio[tok_idx].detach().item()),
                        float(clipped[tok_idx].detach().item()),
                        float(policy_term[tok_idx].detach().item()),
                        float(kl_term[tok_idx].detach().item()),
                        float(entropy_term[tok_idx].detach().item()),
                        float(per_token_loss[tok_idx].detach().item()),
                    )
                else:
                    logger.info(
                        "[loss-debug] tok[%03d] id=%s old_lp=%.6f new_lp=%.6f adv=%.6f ratio=%.6f clipped=%.6f policy=%.6f kl=%.6f entropy=%.6f total=%.6f",
                        tok_idx,
                        tok_id,
                        float(old_lp[tok_idx].detach().item()),
                        float(new_lp[tok_idx].detach().item()),
                        adv_i,
                        float(ratio[tok_idx].detach().item()),
                        float(clipped[tok_idx].detach().item()),
                        float(policy_term[tok_idx].detach().item()),
                        float(kl_term[tok_idx].detach().item()),
                        float(entropy_term[tok_idx].detach().item()),
                        float(per_token_loss[tok_idx].detach().item()),
                    )
                printed += 1
            if int(per_token_loss.numel()) > printed:
                logger.info(
                    "[loss-debug] token rows truncated: printed=%s total=%s non_zero_adv=%s",
                    printed,
                    int(per_token_loss.numel()),
                    non_zero_adv,
                )
            else:
                logger.info(
                    "[loss-debug] token rows complete: printed=%s total=%s non_zero_adv=%s",
                    printed,
                    int(per_token_loss.numel()),
                    non_zero_adv,
                )
            debug_loss_logged += 1

        loss_sum = per_token_loss.sum()
        token_count = int(per_token_loss.numel())
        total_tokens += token_count
        total_loss_value += float(loss_sum.detach().item())

        total_clip += float(clip_fraction.detach().sum().item())
        total_approx_kl += float(approx_kl.detach().sum().item())

        micro_loss = per_token_loss.mean() / max(1, rl_cfg.grad_accum)
        if use_engine_step:
            policy_model.backward(micro_loss)  # type: ignore[attr-defined]
        else:
            micro_loss.backward()
        pending_backward += 1
        if pending_backward % max(1, rl_cfg.grad_accum) == 0:
            if (not use_engine_step) and rl_cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), rl_cfg.max_grad_norm)
            if use_engine_step:
                policy_model.step()  # type: ignore[attr-defined]
            else:
                assert optimizer is not None
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

    if total_tokens == 0:
        raise RuntimeError("No valid tokens found for update.")
    if pending_backward % max(1, rl_cfg.grad_accum) != 0:
        if (not use_engine_step) and rl_cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), rl_cfg.max_grad_norm)
        if use_engine_step:
            policy_model.step()  # type: ignore[attr-defined]
        else:
            assert optimizer is not None
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    mean_loss = float(total_loss_value / total_tokens)
    if not math.isfinite(mean_loss):
        raise RuntimeError(f"Non-finite loss detected: {mean_loss}")

    mean_clip = total_clip / max(1, total_tokens)
    mean_approx_kl = total_approx_kl / max(1, total_tokens)
    mean_entropy = total_entropy / max(1, total_tokens)
    mean_ref_kl = total_ref_kl / max(1, total_tokens)

    stats = TrainStats(
        policy_loss=mean_loss,
        approx_kl=float(mean_approx_kl),
        clip_fraction=float(mean_clip),
        entropy=float(mean_entropy),
        kl_to_reference=float(mean_ref_kl),
        token_count=total_tokens,
    )

    for value in [stats.policy_loss, stats.approx_kl, stats.clip_fraction, stats.entropy, stats.kl_to_reference]:
        if not math.isfinite(value):
            raise RuntimeError(f"Non-finite training stat detected: {value}")
    return stats
