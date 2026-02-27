from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import RLConfig
from .types import DistributedContext, Rollout, TrainStats


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# MoE expert utilization monitoring
# ---------------------------------------------------------------------------


def _compute_expert_utilization_stats(
    router_logits: tuple[torch.Tensor, ...] | None,
) -> dict[str, float]:
    """Compute expert utilization statistics from router logits.

    Args:
        router_logits: tuple of tensors from MoE layers, each of shape
                       (batch_size * seq_len, num_experts) or (batch_size, seq_len, num_experts).

    Returns:
        Dict with expert_utilization_entropy, max_expert_load, min_expert_load.
    """
    empty_stats = {
        "expert_utilization_entropy": 0.0,
        "max_expert_load": 0.0,
        "min_expert_load": 0.0,
    }
    if not router_logits:
        return empty_stats

    valid = [rl for rl in router_logits if rl is not None]
    if not valid:
        return empty_stats

    # Flatten to (total_tokens_across_layers, num_experts).
    flat_parts: list[torch.Tensor] = []
    for rl in valid:
        if rl.dim() == 3:
            flat_parts.append(rl.reshape(-1, rl.shape[-1]))
        elif rl.dim() == 2:
            flat_parts.append(rl)
        else:
            continue
    if not flat_parts:
        return empty_stats

    all_logits = torch.cat(flat_parts, dim=0)
    probs = F.softmax(all_logits.float(), dim=-1)
    mean_utilization = probs.mean(dim=0)

    entropy = -(mean_utilization * (mean_utilization + 1e-10).log()).sum()

    return {
        "expert_utilization_entropy": float(entropy.detach().item()),
        "max_expert_load": float(mean_utilization.max().detach().item()),
        "min_expert_load": float(mean_utilization.min().detach().item()),
    }


# ---------------------------------------------------------------------------
# Main policy update (GRPO / REINFORCE / GSPO)
# ---------------------------------------------------------------------------


def update_policy(
    rollouts: list[Rollout],
    advantages: list[list[float]] | list[float],
    policy_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    rl_cfg: RLConfig,
    device: str,
    dist_ctx: DistributedContext | None = None,
) -> TrainStats:
    """Update policy model with GRPO, REINFORCE, or GSPO algorithm.

    For GSPO, ``advantages`` is a flat ``list[float]`` with one scalar per
    rollout (sequence-level advantage).  For GRPO / REINFORCE it remains
    ``list[list[float]]`` (per-token advantages).
    """
    if len(rollouts) != len(advantages):
        raise ValueError("rollouts and advantages length mismatch")
    if not rollouts:
        raise ValueError("rollouts are empty")

    is_gspo = rl_cfg.algorithm == "gspo"

    policy_model.train()
    if dist_ctx is not None:
        dist_ctx.zero_grad(optimizer)
    else:
        optimizer.zero_grad(set_to_none=True)

    total_tokens = 0
    total_loss_value = 0.0
    pending_backward = 0

    total_approx_kl = 0.0
    total_clip = 0.0
    total_entropy = 0.0
    total_ref_kl = 0.0

    # MoE accumulators.
    total_aux_loss = 0.0
    expert_entropy_accum = 0.0
    max_load_accum = 0.0
    min_load_accum = 1.0
    rollout_count = 0

    for r_idx, rollout in enumerate(rollouts):
        adv_item = advantages[r_idx]  # list[float] for GRPO/REINFORCE, float for GSPO

        input_ids = torch.tensor(
            [rollout.prompt_input_ids + rollout.completion_token_ids],
            device=device,
            dtype=torch.long,
        )
        attention_mask = torch.ones_like(input_ids)

        # Forward pass — request router logits when configured.
        forward_kwargs: dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if rl_cfg.output_router_logits:
            forward_kwargs["output_router_logits"] = True

        outputs = policy_model(**forward_kwargs)

        new_logprobs, entropy = _token_logprobs_and_entropy(
            outputs.logits,
            prompt_len=len(rollout.prompt_input_ids),
            completion_ids=rollout.completion_token_ids,
        )
        if new_logprobs.numel() == 0:
            continue

        rollout_count += 1

        # ---------------------------------------------------------------
        # Algorithm dispatch
        # ---------------------------------------------------------------
        if is_gspo:
            # GSPO: sequence-level importance sampling ratio.
            adv_scalar = float(adv_item)

            comp_len = new_logprobs.numel()
            new_lp_mean = new_logprobs.mean()
            old_lp_slice = rollout.old_logprobs[:comp_len]
            old_lp_mean = torch.tensor(
                sum(old_lp_slice) / max(1, len(old_lp_slice)),
                device=device,
                dtype=new_logprobs.dtype,
            )

            seq_ratio = torch.exp(new_lp_mean - old_lp_mean)
            seq_clipped = torch.clamp(seq_ratio, 1.0 - rl_cfg.clip_eps, 1.0 + rl_cfg.clip_eps)

            adv_tensor = torch.tensor(adv_scalar, device=device, dtype=new_logprobs.dtype)
            per_seq_loss = -torch.minimum(seq_ratio * adv_tensor, seq_clipped * adv_tensor)

            # KL penalty (sequence-level mean).
            if rl_cfg.kl_coef > 0 and rollout.ref_logprobs is not None:
                ref_lp = torch.tensor(
                    rollout.ref_logprobs[:comp_len],
                    device=device,
                    dtype=new_logprobs.dtype,
                )
                ref_kl_mean = (new_logprobs - ref_lp).mean()
                per_seq_loss = per_seq_loss + rl_cfg.kl_coef * ref_kl_mean
                total_ref_kl += float(ref_kl_mean.detach().item())

            # Entropy bonus (sequence-level mean).
            if rl_cfg.entropy_coef > 0:
                ent_mean = entropy[:comp_len].mean()
                per_seq_loss = per_seq_loss - rl_cfg.entropy_coef * ent_mean
                total_entropy += float(ent_mean.detach().item())

            loss_sum = per_seq_loss  # scalar
            token_count = comp_len
            total_tokens += token_count
            total_loss_value += float(loss_sum.detach().item())

            is_clipped = (seq_ratio > (1.0 + rl_cfg.clip_eps)) | (seq_ratio < (1.0 - rl_cfg.clip_eps))
            total_clip += float(is_clipped.float().item())
            total_approx_kl += float(0.5 * ((new_lp_mean - old_lp_mean) ** 2).detach().item())

        elif rl_cfg.algorithm == "grpo":
            # Token-level GRPO (original implementation).
            new_lp, old_lp, adv, ref_lp = _align_tensors(
                new_logprobs,
                old_logprobs=rollout.old_logprobs,
                advantages=adv_item,
                ref_logprobs=rollout.ref_logprobs,
            )
            if new_lp.numel() == 0:
                continue

            ratio = torch.exp(new_lp - old_lp)
            clipped = torch.clamp(ratio, 1.0 - rl_cfg.clip_eps, 1.0 + rl_cfg.clip_eps)
            per_token_loss = -torch.minimum(ratio * adv, clipped * adv)
            clip_fraction = ((ratio > (1.0 + rl_cfg.clip_eps)) | (ratio < (1.0 - rl_cfg.clip_eps))).float()
            approx_kl = 0.5 * ((new_lp - old_lp) ** 2)

            if rl_cfg.kl_coef > 0 and ref_lp is not None:
                ref_kl = new_lp - ref_lp
                per_token_loss = per_token_loss + (rl_cfg.kl_coef * ref_kl)
                total_ref_kl += float(ref_kl.detach().sum().item())

            if rl_cfg.entropy_coef > 0:
                ent = entropy[: per_token_loss.numel()]
                per_token_loss = per_token_loss - (rl_cfg.entropy_coef * ent)
                total_entropy += float(ent.detach().sum().item())

            loss_sum = per_token_loss.sum()
            token_count = int(per_token_loss.numel())
            total_tokens += token_count
            total_loss_value += float(loss_sum.detach().item())

            total_clip += float(clip_fraction.detach().sum().item())
            total_approx_kl += float(approx_kl.detach().sum().item())

        elif rl_cfg.algorithm == "reinforce":
            new_lp, old_lp, adv, ref_lp = _align_tensors(
                new_logprobs,
                old_logprobs=rollout.old_logprobs,
                advantages=adv_item,
                ref_logprobs=rollout.ref_logprobs,
            )
            if new_lp.numel() == 0:
                continue

            per_token_loss = -(new_lp * adv)
            clip_fraction = torch.zeros_like(new_lp)
            approx_kl = 0.5 * ((new_lp - old_lp) ** 2)

            if rl_cfg.kl_coef > 0 and ref_lp is not None:
                ref_kl = new_lp - ref_lp
                per_token_loss = per_token_loss + (rl_cfg.kl_coef * ref_kl)
                total_ref_kl += float(ref_kl.detach().sum().item())

            if rl_cfg.entropy_coef > 0:
                ent = entropy[: per_token_loss.numel()]
                per_token_loss = per_token_loss - (rl_cfg.entropy_coef * ent)
                total_entropy += float(ent.detach().sum().item())

            loss_sum = per_token_loss.sum()
            token_count = int(per_token_loss.numel())
            total_tokens += token_count
            total_loss_value += float(loss_sum.detach().item())

            total_clip += float(clip_fraction.detach().sum().item())
            total_approx_kl += float(approx_kl.detach().sum().item())

        else:
            raise ValueError(f"Unsupported algorithm: {rl_cfg.algorithm}")

        # ---------------------------------------------------------------
        # Auxiliary load-balancing loss for MoE
        # ---------------------------------------------------------------
        if rl_cfg.output_router_logits and rl_cfg.router_aux_loss_coef > 0:
            aux_loss_tensor = getattr(outputs, "aux_loss", None)
            if aux_loss_tensor is not None:
                total_aux_loss += float(aux_loss_tensor.detach().item())
                loss_sum = loss_sum + rl_cfg.router_aux_loss_coef * aux_loss_tensor

        # ---------------------------------------------------------------
        # Expert utilization monitoring (no grad graph)
        # ---------------------------------------------------------------
        if rl_cfg.monitor_expert_utilization:
            router_logits = getattr(outputs, "router_logits", None)
            if router_logits:
                with torch.no_grad():
                    expert_stats = _compute_expert_utilization_stats(router_logits)
                expert_entropy_accum += expert_stats["expert_utilization_entropy"]
                max_load_accum = max(max_load_accum, expert_stats["max_expert_load"])
                min_load_accum = min(min_load_accum, expert_stats["min_expert_load"])

        # ---------------------------------------------------------------
        # Backward + optional gradient accumulation step
        # ---------------------------------------------------------------
        if is_gspo:
            micro_loss = loss_sum / max(1, rl_cfg.grad_accum)
        else:
            micro_loss = per_token_loss.mean() / max(1, rl_cfg.grad_accum)

        if dist_ctx is not None:
            dist_ctx.backward(micro_loss)
        else:
            micro_loss.backward()
        pending_backward += 1

        if pending_backward % max(1, rl_cfg.grad_accum) == 0:
            if dist_ctx is not None:
                dist_ctx.step(optimizer, policy_model, rl_cfg.max_grad_norm)
                dist_ctx.zero_grad(optimizer)
            else:
                if rl_cfg.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), rl_cfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

    if total_tokens == 0:
        raise RuntimeError("No valid tokens found for update.")
    if pending_backward % max(1, rl_cfg.grad_accum) != 0:
        if dist_ctx is not None:
            dist_ctx.step(optimizer, policy_model, rl_cfg.max_grad_norm)
            dist_ctx.zero_grad(optimizer)
        else:
            if rl_cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), rl_cfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    # ---------------------------------------------------------------
    # Aggregate statistics
    # ---------------------------------------------------------------
    if is_gspo:
        # For GSPO: clip_fraction and approx_kl are per-sequence.
        seq_count = max(1, rollout_count)
        mean_loss = float(total_loss_value / seq_count)
        mean_clip = total_clip / seq_count
        mean_approx_kl = total_approx_kl / seq_count
        mean_entropy = total_entropy / seq_count
        mean_ref_kl = total_ref_kl / seq_count
    else:
        mean_loss = float(total_loss_value / total_tokens)
        mean_clip = total_clip / max(1, total_tokens)
        mean_approx_kl = total_approx_kl / max(1, total_tokens)
        mean_entropy = total_entropy / max(1, total_tokens)
        mean_ref_kl = total_ref_kl / max(1, total_tokens)

    if not math.isfinite(mean_loss):
        raise RuntimeError(f"Non-finite loss detected: {mean_loss}")

    rc = max(1, rollout_count)
    stats = TrainStats(
        policy_loss=mean_loss,
        approx_kl=float(mean_approx_kl),
        clip_fraction=float(mean_clip),
        entropy=float(mean_entropy),
        kl_to_reference=float(mean_ref_kl),
        token_count=total_tokens,
        aux_loss=float(total_aux_loss / rc),
        expert_utilization_entropy=float(expert_entropy_accum / rc),
        max_expert_load=float(max_load_accum),
        min_expert_load=float(min_load_accum) if rollout_count > 0 else 0.0,
    )

    for value in [stats.policy_loss, stats.approx_kl, stats.clip_fraction, stats.entropy, stats.kl_to_reference]:
        if not math.isfinite(value):
            raise RuntimeError(f"Non-finite training stat detected: {value}")
    return stats
