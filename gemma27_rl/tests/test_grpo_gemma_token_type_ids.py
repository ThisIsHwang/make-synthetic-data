from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from gemma27_rl.config import RLConfig
from gemma27_rl.grpo import update_policy
from gemma27_rl.rollout import compute_completion_logprobs
from gemma27_rl.rl_types import Rollout


class TinyGemmaLikeLM(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 16):
        super().__init__()
        self.config = SimpleNamespace(model_type="gemma2", vocab_size=vocab_size)
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size)
        self.last_token_type_ids = None

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, use_cache=None):
        del attention_mask, use_cache
        self.last_token_type_ids = None if token_type_ids is None else token_type_ids.detach().clone()
        x = self.emb(input_ids)
        logits = self.proj(x)
        return SimpleNamespace(logits=logits)


def test_update_policy_includes_token_type_ids_for_gemma_models() -> None:
    torch.manual_seed(0)
    model = TinyGemmaLikeLM(vocab_size=64, hidden_size=32)
    device = "cpu"

    prompt_ids = [1, 2, 3]
    completion_ids = [4, 5, 6]
    old_lp = compute_completion_logprobs(model, prompt_ids, completion_ids, device=device).tolist()

    rollout = Rollout(
        example_id="ex-gemma",
        prompt_text="p",
        prompt_input_ids=prompt_ids,
        completion_text="c",
        completion_token_ids=completion_ids,
        old_logprobs=old_lp,
        ref_logprobs=None,
        token_char_offsets=[(0, 1), (1, 2), (2, 3)],
        src_text="src",
        ref_text=None,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    _ = update_policy(
        rollouts=[rollout],
        advantages=[[0.1, -0.1, 0.2]],
        policy_model=model,
        optimizer=optimizer,
        rl_cfg=RLConfig(algorithm="grpo", clip_eps=0.2, kl_coef=0.0, entropy_coef=0.0),
        device=device,
    )

    assert model.last_token_type_ids is not None
    expected_shape = (1, len(prompt_ids) + len(completion_ids))
    assert tuple(model.last_token_type_ids.shape) == expected_shape
    assert torch.count_nonzero(model.last_token_type_ids).item() == 0
