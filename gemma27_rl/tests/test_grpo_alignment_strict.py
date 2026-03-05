from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gemma27_rl.grpo import _align_tensors


def test_align_tensors_raises_on_old_or_adv_length_mismatch() -> None:
    new_lp = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    with pytest.raises(ValueError, match="token-alignment length mismatch"):
        _ = _align_tensors(
            new_lp,
            old_logprobs=[-0.1, -0.2],
            advantages=[1.0, 2.0, 3.0],
            ref_logprobs=None,
            example_id="ex-old-mismatch",
        )


def test_align_tensors_raises_on_reference_length_mismatch() -> None:
    new_lp = torch.tensor([0.1, 0.2], dtype=torch.float32)
    with pytest.raises(ValueError, match="token-alignment length mismatch"):
        _ = _align_tensors(
            new_lp,
            old_logprobs=[-0.1, -0.2],
            advantages=[1.0, 2.0],
            ref_logprobs=[-0.1],
            example_id="ex-ref-mismatch",
        )
