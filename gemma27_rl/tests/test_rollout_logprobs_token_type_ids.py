from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from gemma27_rl.rollout import compute_completion_logprobs, compute_completion_logprobs_batch


class _TinyTokenTypeRequiredLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 16) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, hidden_size)
        self.proj = torch.nn.Linear(hidden_size, vocab_size)
        self.config = SimpleNamespace(vocab_size=vocab_size, model_type="gemma")
        self.last_token_type_ids = None

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        del attention_mask
        if token_type_ids is None:
            raise ValueError("token_type_ids is required as a model input when training")
        self.last_token_type_ids = token_type_ids.detach().clone()
        hidden = self.emb(input_ids)
        logits = self.proj(hidden)
        return SimpleNamespace(logits=logits)


def test_compute_completion_logprobs_supports_models_requiring_token_type_ids() -> None:
    model = _TinyTokenTypeRequiredLM().eval()

    row = compute_completion_logprobs(
        model=model,
        prompt_input_ids=[1, 2, 3],
        completion_token_ids=[4, 5],
        device="cpu",
    )

    assert tuple(row.shape) == (2,)
    assert model.last_token_type_ids is not None
    assert tuple(model.last_token_type_ids.shape) == (1, 5)


def test_compute_completion_logprobs_batch_supports_models_requiring_token_type_ids() -> None:
    model = _TinyTokenTypeRequiredLM().eval()

    rows = compute_completion_logprobs_batch(
        model=model,
        items=[([1, 2, 3], [4, 5]), ([2, 3], [6])],
        device="cpu",
    )

    assert [tuple(row.shape) for row in rows] == [(2,), (1,)]
    assert model.last_token_type_ids is not None
