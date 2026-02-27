from __future__ import annotations

from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")

from qwen35_moe_rl.types import DistributedContext


def test_distributed_context_default_backward() -> None:
    """Without a deepspeed engine, backward() should call loss.backward()."""
    ctx = DistributedContext(backend="none")
    loss = torch.tensor(1.0, requires_grad=True) * 2.0
    ctx.backward(loss)
    # No error means loss.backward() was called successfully.
    assert loss.grad_fn is not None


def test_distributed_context_default_step() -> None:
    """Without a deepspeed engine, step() clips gradients and steps the optimizer."""
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(1, 4)
    loss = model(x).sum()
    loss.backward()

    ctx = DistributedContext(backend="fsdp")
    before = [p.detach().clone() for p in model.parameters()]
    ctx.step(optimizer, model, max_grad_norm=1.0)
    after = [p.detach().clone() for p in model.parameters()]
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))


def test_distributed_context_default_zero_grad() -> None:
    """Without a deepspeed engine, zero_grad() calls optimizer.zero_grad(set_to_none=True)."""
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(1, 4)
    loss = model(x).sum()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())

    ctx = DistributedContext(backend="ddp")
    ctx.zero_grad(optimizer)
    assert all(p.grad is None for p in model.parameters())


def test_distributed_context_with_mock_engine() -> None:
    """When deepspeed_engine is set, all operations delegate to the engine."""
    engine = MagicMock()
    ctx = DistributedContext(backend="deepspeed", deepspeed_engine=engine)

    fake_loss = MagicMock()
    ctx.backward(fake_loss)
    engine.backward.assert_called_once_with(fake_loss)

    ctx.step(optimizer=MagicMock(), model=MagicMock(), max_grad_norm=1.0)
    engine.step.assert_called_once()

    ctx.zero_grad(optimizer=MagicMock())
    engine.zero_grad.assert_called_once()
