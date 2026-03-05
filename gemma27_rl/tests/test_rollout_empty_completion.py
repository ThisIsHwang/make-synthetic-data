from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from gemma27_rl.config import GenerationConfig
from gemma27_rl.rollout import generate_rollouts
from gemma27_rl.rl_types import Example


class _FakeTokenizer:
    is_fast = False
    pad_token_id = 0
    eos_token_id = 2
    all_special_tokens = ["<pad>", "</s>"]
    additional_special_tokens: list[str] = []
    special_tokens_map = {"pad_token": "<pad>", "eos_token": "</s>"}

    def __call__(self, texts, return_tensors="pt", add_special_tokens=True, padding=True):  # type: ignore[no-untyped-def]
        del return_tensors, add_special_tokens, padding
        if isinstance(texts, str):
            texts = [texts]
        rows = [[11, 12] for _ in texts]
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.ones((len(rows), len(rows[0])), dtype=torch.long),
        }

    def decode(self, ids, *, skip_special_tokens=False, clean_up_tokenization_spaces=False):  # type: ignore[no-untyped-def]
        del clean_up_tokenization_spaces
        if isinstance(ids, int):
            ids = [ids]
        pieces: list[str] = []
        for raw in ids:
            tok = int(raw)
            if skip_special_tokens and tok in {0, 2}:
                continue
            if tok == 0:
                pieces.append("<pad>")
            elif tok == 2:
                pieces.append("</s>")
            elif tok == 11:
                pieces.append("A")
            elif tok == 12:
                pieces.append("B")
            else:
                pieces.append(f"t{tok}")
        return "".join(pieces)

    def convert_tokens_to_ids(self, token):  # type: ignore[no-untyped-def]
        if token == "</s>":
            return 2
        if token == "<pad>":
            return 0
        return -1

    def get_added_vocab(self):  # type: ignore[no-untyped-def]
        return {}


class _FakePolicyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(64, 8)
        self.config = SimpleNamespace(vocab_size=64)
        self.generation_config = SimpleNamespace(eos_token_id=2)

    def get_input_embeddings(self):
        return self.emb

    def generate(self, input_ids, attention_mask=None, **kwargs):  # type: ignore[no-untyped-def]
        del attention_mask, kwargs
        eos = torch.full((int(input_ids.shape[0]), 1), 2, dtype=torch.long, device=input_ids.device)
        return torch.cat([input_ids, eos], dim=1)


def test_generate_rollouts_keeps_empty_completion_without_fallback_token() -> None:
    tokenizer = _FakeTokenizer()
    model = _FakePolicyModel()
    example = Example(
        example_id="ex-1",
        src_text="hello",
        src_lang="English",
        tgt_lang="Korean",
        src_lang_code="en",
        tgt_lang_code="ko",
        ref_text=None,
    )
    gen_cfg = GenerationConfig(max_new_tokens=1, do_sample=False, temperature=0.0, num_samples_per_prompt=1)

    rollouts = generate_rollouts(
        examples=[example],
        policy_model=model,  # type: ignore[arg-type]
        tokenizer=tokenizer,  # type: ignore[arg-type]
        gen_cfg=gen_cfg,
        device="cpu",
        compute_old_logprobs=False,
    )

    assert len(rollouts) == 1
    rollout = rollouts[0]
    assert rollout.completion_token_ids == []
    assert rollout.raw_completion_token_ids == []
    assert rollout.completion_text == ""
    assert rollout.completion_raw_text == ""
    assert rollout.completion_clean_text == ""
    assert rollout.old_logprobs == []
    assert rollout.ref_logprobs is None
    assert rollout.token_char_offsets == []
