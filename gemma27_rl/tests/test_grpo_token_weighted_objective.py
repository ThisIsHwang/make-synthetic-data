from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F
from torch import nn

from gemma27_rl.config import RLConfig
from gemma27_rl.grpo import update_policy
from gemma27_rl.rl_types import Rollout


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 16):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, use_cache=None):
        del attention_mask, token_type_ids, use_cache
        x = self.emb(input_ids)
        logits = self.proj(x)
        return SimpleNamespace(logits=logits)


def _make_rollout(
    *,
    example_id: str,
    prompt_ids: list[int],
    completion_ids: list[int],
) -> Rollout:
    return Rollout(
        example_id=example_id,
        prompt_text="p",
        prompt_input_ids=list(prompt_ids),
        completion_text="c",
        completion_token_ids=list(completion_ids),
        old_logprobs=[0.0 for _ in completion_ids],
        ref_logprobs=None,
        token_char_offsets=[(0, 1) for _ in completion_ids],
        src_text="src",
        ref_text=None,
    )


def _reinforce_loss_sum(
    *,
    model: nn.Module,
    rollout: Rollout,
    advantages: list[float],
) -> tuple[torch.Tensor, int]:
    input_ids = torch.tensor(
        [rollout.prompt_input_ids + rollout.completion_token_ids],
        dtype=torch.long,
    )
    outputs = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
    prompt_len = len(rollout.prompt_input_ids)
    completion_ids = list(rollout.completion_token_ids)
    start = prompt_len - 1
    end = start + len(completion_ids)
    selected = outputs.logits[0, start:end, :]
    log_probs = F.log_softmax(selected, dim=-1)
    labels = torch.tensor(completion_ids, dtype=torch.long)
    new_lp = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    adv = torch.tensor(advantages, dtype=new_lp.dtype)
    loss_sum = (-(new_lp * adv)).sum()
    return loss_sum, int(new_lp.numel())


def _manual_token_weighted_update(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    rollouts: list[Rollout],
    advantages: list[list[float]],
    grad_accum: int,
) -> None:
    valid_rows = [idx for idx, rollout in enumerate(rollouts) if len(rollout.completion_token_ids) > 0]
    cursor = 0
    while cursor < len(valid_rows):
        batch_rows = valid_rows[cursor : cursor + max(1, int(grad_accum))]
        optimizer.zero_grad(set_to_none=True)
        loss_sum_total: torch.Tensor | None = None
        token_total = 0
        for row_idx in batch_rows:
            loss_sum, token_count = _reinforce_loss_sum(
                model=model,
                rollout=rollouts[row_idx],
                advantages=advantages[row_idx],
            )
            if loss_sum_total is None:
                loss_sum_total = loss_sum
            else:
                loss_sum_total = loss_sum_total + loss_sum
            token_total += int(token_count)
        assert loss_sum_total is not None
        assert token_total > 0
        (loss_sum_total / float(token_total)).backward()
        optimizer.step()
        cursor += max(1, int(grad_accum))


def _assert_models_close(lhs: nn.Module, rhs: nn.Module, *, atol: float = 1e-6, rtol: float = 1e-6) -> None:
    lhs_params = dict(lhs.named_parameters())
    rhs_params = dict(rhs.named_parameters())
    assert lhs_params.keys() == rhs_params.keys()
    for name in lhs_params:
        assert torch.allclose(lhs_params[name], rhs_params[name], atol=atol, rtol=rtol), name


def test_update_policy_matches_token_weighted_objective_for_variable_lengths() -> None:
    torch.manual_seed(7)
    model = TinyCausalLM(vocab_size=64, hidden_size=24)
    model_manual = TinyCausalLM(vocab_size=64, hidden_size=24)
    model_manual.load_state_dict(model.state_dict())

    rollouts = [
        _make_rollout(example_id="ex-1", prompt_ids=[1, 2], completion_ids=[3]),
        _make_rollout(example_id="ex-2", prompt_ids=[1, 2], completion_ids=[4, 5, 6, 7]),
    ]
    advantages = [
        [1.0],
        [1.0, 1.0, 1.0, 1.0],
    ]

    cfg = RLConfig(
        algorithm="reinforce",
        grad_accum=2,
        kl_coef=0.0,
        entropy_coef=0.0,
        max_grad_norm=0.0,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    optimizer_manual = torch.optim.SGD(model_manual.parameters(), lr=0.05)

    _ = update_policy(
        rollouts=rollouts,
        advantages=advantages,
        policy_model=model,
        optimizer=optimizer,
        rl_cfg=cfg,
        device="cpu",
    )
    _manual_token_weighted_update(
        model=model_manual,
        optimizer=optimizer_manual,
        rollouts=rollouts,
        advantages=advantages,
        grad_accum=2,
    )

    _assert_models_close(model, model_manual)


def test_update_policy_handles_partial_grad_accum_with_same_token_weighted_rule() -> None:
    torch.manual_seed(11)
    model = TinyCausalLM(vocab_size=64, hidden_size=24)
    model_manual = TinyCausalLM(vocab_size=64, hidden_size=24)
    model_manual.load_state_dict(model.state_dict())

    rollouts = [
        _make_rollout(example_id="ex-1", prompt_ids=[1, 2], completion_ids=[3, 4]),
        _make_rollout(example_id="ex-2", prompt_ids=[1, 2], completion_ids=[5]),
        _make_rollout(example_id="ex-3", prompt_ids=[1, 2], completion_ids=[6, 7, 8]),
    ]
    advantages = [
        [0.5, 0.25],
        [1.0],
        [0.1, -0.2, 0.3],
    ]

    cfg = RLConfig(
        algorithm="reinforce",
        grad_accum=2,
        kl_coef=0.0,
        entropy_coef=0.0,
        max_grad_norm=0.0,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    optimizer_manual = torch.optim.SGD(model_manual.parameters(), lr=0.03)

    _ = update_policy(
        rollouts=rollouts,
        advantages=advantages,
        policy_model=model,
        optimizer=optimizer,
        rl_cfg=cfg,
        device="cpu",
    )
    _manual_token_weighted_update(
        model=model_manual,
        optimizer=optimizer_manual,
        rollouts=rollouts,
        advantages=advantages,
        grad_accum=2,
    )

    _assert_models_close(model, model_manual)
