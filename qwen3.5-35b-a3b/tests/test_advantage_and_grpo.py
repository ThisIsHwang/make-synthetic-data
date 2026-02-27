from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from qwen35_moe_rl.advantage import (
    build_sequence_level_advantages,
    normalize_advantages,
)
from qwen35_moe_rl.config import RLConfig
from qwen35_moe_rl.grpo import update_policy
from qwen35_moe_rl.rollout import compute_completion_logprobs
from qwen35_moe_rl.types import Rollout


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 16):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        del attention_mask
        x = self.emb(input_ids)
        logits = self.proj(x)
        return SimpleNamespace(logits=logits)


class TinyCausalLMWithRouter(nn.Module):
    """Tiny model that returns aux_loss and router_logits like a MoE model."""

    def __init__(self, vocab_size: int = 32, hidden_size: int = 16, num_experts: int = 4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size)
        self.num_experts = num_experts

    def forward(self, input_ids, attention_mask=None, output_router_logits=False, **kwargs):
        del attention_mask
        x = self.emb(input_ids)
        logits = self.proj(x)
        ns = SimpleNamespace(logits=logits)
        if output_router_logits:
            batch_seq = input_ids.shape[0] * input_ids.shape[1]
            ns.aux_loss = torch.tensor(0.5, requires_grad=True)
            ns.router_logits = (
                torch.randn(batch_seq, self.num_experts),
            )
        return ns


def _make_rollout(model, device="cpu") -> Rollout:
    prompt_ids = [1, 2, 3]
    completion_ids = [4, 5, 6]
    old_lp = compute_completion_logprobs(model, prompt_ids, completion_ids, device=device).tolist()
    ref_lp = [x - 0.05 for x in old_lp]
    old_lp_mean = sum(old_lp) / max(1, len(old_lp))
    ref_lp_mean = sum(ref_lp) / max(1, len(ref_lp))
    return Rollout(
        example_id="ex-1",
        prompt_text="p",
        prompt_input_ids=prompt_ids,
        completion_text="c",
        completion_token_ids=completion_ids,
        old_logprobs=old_lp,
        ref_logprobs=ref_lp,
        token_char_offsets=[(0, 1), (1, 2), (2, 3)],
        src_text="src",
        ref_text=None,
        old_logprob_mean=old_lp_mean,
        ref_logprob_mean=ref_lp_mean,
    )


# ---------------------------------------------------------------------------
# Advantage tests
# ---------------------------------------------------------------------------


def test_advantage_normalization_mean_std() -> None:
    normalized, stats = normalize_advantages([[1.0, 2.0], [3.0]], eps=1e-8)
    flat = [v for row in normalized for v in row]
    mean = sum(flat) / len(flat)
    std = (sum((v - mean) ** 2 for v in flat) / len(flat)) ** 0.5
    assert abs(mean) < 1e-6
    assert std == pytest.approx(1.0, rel=1e-5)
    assert abs(stats["norm_mean"]) < 1e-6


def test_sequence_level_advantages_basic() -> None:
    rewards = [1.0, 3.0, 2.0, 4.0]
    group_ids = ["g1", "g1", "g2", "g2"]
    advantages, stats = build_sequence_level_advantages(
        rewards,
        group_ids,
        normalize=True,
        group_normalize=True,
    )
    assert len(advantages) == 4
    assert "raw_mean" in stats
    assert "norm_mean" in stats


# ---------------------------------------------------------------------------
# Policy update smoke tests
# ---------------------------------------------------------------------------


def test_grpo_update_smoke_step_no_nan() -> None:
    torch.manual_seed(0)
    model = TinyCausalLM(vocab_size=64, hidden_size=32)
    rollout = _make_rollout(model)
    advantages = [[0.5, -0.1, 0.2]]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    before = [p.detach().clone() for p in model.parameters()]
    stats = update_policy(
        rollouts=[rollout],
        advantages=advantages,
        policy_model=model,
        optimizer=optimizer,
        rl_cfg=RLConfig(algorithm="grpo", clip_eps=0.2, kl_coef=0.01, entropy_coef=0.0),
        device="cpu",
    )
    after = [p.detach().clone() for p in model.parameters()]

    assert math.isfinite(stats.policy_loss)
    assert math.isfinite(stats.approx_kl)
    assert math.isfinite(stats.clip_fraction)
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))


def test_gspo_update_smoke_step_no_nan() -> None:
    torch.manual_seed(0)
    model = TinyCausalLM(vocab_size=64, hidden_size=32)
    rollout = _make_rollout(model)
    # GSPO: scalar advantages (one per rollout).
    advantages = [0.5]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    before = [p.detach().clone() for p in model.parameters()]
    stats = update_policy(
        rollouts=[rollout],
        advantages=advantages,
        policy_model=model,
        optimizer=optimizer,
        rl_cfg=RLConfig(algorithm="gspo", clip_eps=0.2, kl_coef=0.01, entropy_coef=0.0),
        device="cpu",
    )
    after = [p.detach().clone() for p in model.parameters()]

    assert math.isfinite(stats.policy_loss)
    assert math.isfinite(stats.approx_kl)
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))


def test_reinforce_update_smoke_no_nan() -> None:
    torch.manual_seed(0)
    model = TinyCausalLM(vocab_size=64, hidden_size=32)
    rollout = _make_rollout(model)
    advantages = [[0.5, -0.1, 0.2]]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    stats = update_policy(
        rollouts=[rollout],
        advantages=advantages,
        policy_model=model,
        optimizer=optimizer,
        rl_cfg=RLConfig(algorithm="reinforce", clip_eps=0.2, kl_coef=0.01, entropy_coef=0.0),
        device="cpu",
    )
    assert math.isfinite(stats.policy_loss)
    assert math.isfinite(stats.approx_kl)


def test_gspo_with_dist_ctx_none() -> None:
    """dist_ctx=None should follow the original non-distributed code path."""
    torch.manual_seed(0)
    model = TinyCausalLM(vocab_size=64, hidden_size=32)
    rollout = _make_rollout(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    stats = update_policy(
        rollouts=[rollout],
        advantages=[0.5],
        policy_model=model,
        optimizer=optimizer,
        rl_cfg=RLConfig(algorithm="gspo"),
        device="cpu",
        dist_ctx=None,
    )
    assert math.isfinite(stats.policy_loss)


def test_grpo_with_moe_router_model() -> None:
    """MoE model with aux_loss and router_logits should not cause errors."""
    torch.manual_seed(0)
    model = TinyCausalLMWithRouter(vocab_size=64, hidden_size=32, num_experts=4)
    rollout = _make_rollout(model)
    advantages = [[0.5, -0.1, 0.2]]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    stats = update_policy(
        rollouts=[rollout],
        advantages=advantages,
        policy_model=model,
        optimizer=optimizer,
        rl_cfg=RLConfig(
            algorithm="grpo",
            output_router_logits=True,
            router_aux_loss_coef=0.001,
            monitor_expert_utilization=True,
        ),
        device="cpu",
    )
    assert math.isfinite(stats.policy_loss)
    assert math.isfinite(stats.aux_loss)
    assert math.isfinite(stats.expert_utilization_entropy)
