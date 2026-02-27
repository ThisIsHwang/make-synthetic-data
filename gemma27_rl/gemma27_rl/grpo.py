from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import RLConfig
from .types import Rollout, TrainStats


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

    for rollout, adv_row in zip(rollouts, advantages):
        input_ids = torch.tensor(
            [rollout.prompt_input_ids + rollout.completion_token_ids],
            device=device,
            dtype=torch.long,
        )
        attention_mask = torch.ones_like(input_ids)
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
            per_token_loss = -torch.minimum(ratio * adv, clipped * adv)
            clip_fraction = ((ratio > (1.0 + rl_cfg.clip_eps)) | (ratio < (1.0 - rl_cfg.clip_eps))).float()
            approx_kl = 0.5 * ((new_lp - old_lp) ** 2)
        elif rl_cfg.algorithm == "reinforce":
            per_token_loss = -(new_lp * adv)
            clip_fraction = torch.zeros_like(new_lp)
            approx_kl = 0.5 * ((new_lp - old_lp) ** 2)
        else:
            raise ValueError(f"Unsupported algorithm: {rl_cfg.algorithm}")

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
