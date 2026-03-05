from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
torch = pytest.importorskip("torch")
from torch import nn

from gemma27_rl.advantage import normalize_advantages
from gemma27_rl.config import RLConfig
from gemma27_rl.grpo import update_policy
from gemma27_rl.rollout import compute_completion_logprobs, compute_completion_logprobs_batch
from gemma27_rl.rl_types import Rollout


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 16):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        x = self.emb(input_ids)
        logits = self.proj(x)
        return SimpleNamespace(logits=logits)


def test_advantage_normalization_mean_std() -> None:
    normalized, stats = normalize_advantages([[1.0, 2.0], [3.0]], eps=1e-8)
    flat = [v for row in normalized for v in row]
    mean = sum(flat) / len(flat)
    std = (sum((v - mean) ** 2 for v in flat) / len(flat)) ** 0.5
    assert abs(mean) < 1e-6
    assert std == pytest.approx(1.0, rel=1e-5)
    assert abs(stats["norm_mean"]) < 1e-6


def test_grpo_update_smoke_step_no_nan() -> None:
    torch.manual_seed(0)
    model = TinyCausalLM(vocab_size=64, hidden_size=32)
    device = "cpu"

    prompt_ids = [1, 2, 3]
    completion_ids = [4, 5, 6]
    old_lp = compute_completion_logprobs(model, prompt_ids, completion_ids, device=device).tolist()
    ref_lp = [x - 0.05 for x in old_lp]

    rollout = Rollout(
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
    )

    advantages = [[0.5, -0.1, 0.2]]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    before = [p.detach().clone() for p in model.parameters()]
    stats = update_policy(
        rollouts=[rollout],
        advantages=advantages,
        policy_model=model,
        optimizer=optimizer,
        rl_cfg=RLConfig(algorithm="grpo", clip_eps=0.2, kl_coef=0.01, entropy_coef=0.0),
        device=device,
    )
    after = [p.detach().clone() for p in model.parameters()]

    assert math.isfinite(stats.policy_loss)
    assert math.isfinite(stats.approx_kl)
    assert math.isfinite(stats.clip_fraction)
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))


def test_compute_completion_logprobs_rejects_out_of_vocab_token_ids() -> None:
    model = TinyCausalLM(vocab_size=8, hidden_size=8)
    with pytest.raises(ValueError, match="token id out of range"):
        _ = compute_completion_logprobs(
            model=model,
            prompt_input_ids=[1, 2, 9],
            completion_token_ids=[3],
            device="cpu",
        )


def test_compute_completion_logprobs_batch_rejects_out_of_vocab_token_ids() -> None:
    model = TinyCausalLM(vocab_size=8, hidden_size=8)
    with pytest.raises(ValueError, match="token id out of range"):
        _ = compute_completion_logprobs_batch(
            model=model,
            items=[([1, 2], [3]), ([4], [10])],
            device="cpu",
        )
