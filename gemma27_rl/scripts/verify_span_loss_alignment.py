#!/usr/bin/env python3
from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from gemma27_rl.advantage import broadcast_sequence_reward, combine_advantages
from gemma27_rl.config import RLPostTrainConfig
from gemma27_rl.grpo import update_policy
from gemma27_rl.rewards import spans_to_token_rewards
from gemma27_rl.rollout import compute_completion_logprobs, generate_rollouts
from gemma27_rl.rl_types import Example


class FakeChatTokenizer:
    is_fast = False
    chat_template = "{fake_chat_template}"
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self) -> None:
        self.padding_side = "right"
        self._char_to_id: dict[str, int] = {}
        self._id_to_char: dict[int, str] = {self.pad_token_id: "", self.eos_token_id: ""}
        self._next_id = 10

    def _id_for_char(self, ch: str) -> int:
        if ch in self._char_to_id:
            return self._char_to_id[ch]
        token_id = self._next_id
        self._next_id += 1
        self._char_to_id[ch] = token_id
        self._id_to_char[token_id] = ch
        return token_id

    def encode_text(self, text: str) -> list[int]:
        return [self._id_for_char(ch) for ch in text]

    def decode(self, ids: Any, clean_up_tokenization_spaces: bool = False, skip_special_tokens: bool = False) -> str:
        del clean_up_tokenization_spaces
        del skip_special_tokens
        if isinstance(ids, int):
            ids = [ids]
        chars = [self._id_to_char.get(int(tok), "") for tok in ids]
        return "".join(chars)

    def __call__(
        self,
        texts: str | list[str],
        return_tensors: str | None = None,
        add_special_tokens: bool = True,
        padding: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        del add_special_tokens
        if isinstance(texts, str):
            return {"input_ids": self.encode_text(texts)}

        rows = [self.encode_text(text) for text in texts]
        if not padding:
            if return_tensors == "pt":
                return {"input_ids": torch.tensor(rows, dtype=torch.long)}
            return {"input_ids": rows}

        max_len = max((len(row) for row in rows), default=0)
        padded: list[list[int]] = []
        attention: list[list[int]] = []
        for row in rows:
            pad_len = max_len - len(row)
            if self.padding_side == "left":
                padded.append(([self.pad_token_id] * pad_len) + row)
                attention.append(([0] * pad_len) + ([1] * len(row)))
            else:
                padded.append(row + ([self.pad_token_id] * pad_len))
                attention.append(([1] * len(row)) + ([0] * pad_len))

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(attention, dtype=torch.long),
            }
        return {"input_ids": padded, "attention_mask": attention}

    def apply_chat_template(
        self,
        chats: list[list[dict[str, str]]],
        tokenize: bool = True,
        add_generation_prompt: bool = True,
        return_tensors: str | None = None,
        padding: bool = False,
        **_: Any,
    ) -> Any:
        if not tokenize:
            raise ValueError("FakeChatTokenizer only supports tokenize=True in this verifier.")

        texts: list[str] = []
        for convo in chats:
            user_text = ""
            for turn in convo:
                if str(turn.get("role", "")).lower() == "user":
                    user_text = str(turn.get("content", ""))
                    break
            prompt = "<|user|>\n" + user_text
            if add_generation_prompt:
                prompt += "\n<|assistant|>\n"
            texts.append(prompt)

        encoded = self(texts, return_tensors=return_tensors, add_special_tokens=False, padding=padding)
        if return_tensors == "pt":
            return encoded["input_ids"]
        return encoded


class TinyCausalLM(nn.Module):
    def __init__(self, forced_completion_ids: list[int], vocab_size: int = 8192, hidden_size: int = 32) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size)
        self.generation_config = SimpleNamespace(eos_token_id=1)
        self._forced_completion_ids = [int(t) for t in forced_completion_ids]

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        x = self.emb(input_ids)
        logits = self.proj(x)
        return SimpleNamespace(logits=logits)

    def generate(self, input_ids, attention_mask=None, num_return_sequences: int = 1, **_: Any):
        del attention_mask
        rows: list[list[int]] = []
        batch = int(input_ids.shape[0])
        for idx in range(batch):
            prefix = [int(t) for t in input_ids[idx].tolist()]
            for _sample_idx in range(max(1, int(num_return_sequences))):
                rows.append(prefix + self._forced_completion_ids)
        return torch.tensor(rows, dtype=input_ids.dtype, device=input_ids.device)


def _overlaps(span_start: int, span_end: int, tok_s: int, tok_e: int) -> bool:
    return max(0, min(tok_e, span_end) - max(tok_s, span_start)) > 0


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    tokenizer = FakeChatTokenizer()
    forced_completion_raw = "  번역 오류 단어 테스트  "
    forced_completion_ids = tokenizer.encode_text(forced_completion_raw)
    model = TinyCausalLM(forced_completion_ids=forced_completion_ids)

    cfg = RLPostTrainConfig()
    cfg.generation.num_samples_per_prompt = 1
    cfg.generation.max_new_tokens = 64
    cfg.generation.do_sample = False
    cfg.generation.chat_template_kwargs = {}

    # Keep reward path focused on span->token reward only.
    cfg.reward.metricx.enabled = False
    cfg.reward.xcomet.enabled = True
    cfg.reward.mqm.enabled = False
    cfg.reward.w_metricx = 0.0
    cfg.reward.w_xcomet_seq = 0.0
    cfg.reward.w_mqm_seq = 0.0

    cfg.rl.normalize_advantage = False
    cfg.rl.group_normalize = False
    cfg.rl.grad_accum = 1
    cfg.rl.ppo_epochs = 1

    example = Example(
        example_id="debug-1",
        src_text="This is a source sentence with an intentional term error.",
        src_lang="English",
        tgt_lang="Korean",
        src_lang_code="en",
        tgt_lang_code="ko",
        ref_text="참고 번역",
    )

    rollouts = generate_rollouts(
        examples=[example],
        policy_model=model,
        tokenizer=tokenizer,
        gen_cfg=cfg.generation,
        device="cpu",
        ref_model=None,
        ref_device=None,
        ref_logprob_fn=None,
        prompt_template=cfg.prompt.template,
        show_progress=False,
    )
    if len(rollouts) != 1:
        raise RuntimeError(f"Expected 1 rollout, got {len(rollouts)}")
    rollout = rollouts[0]

    target_text = "오류 단어"
    span_start = rollout.completion_text.index(target_text)
    span_end = span_start + len(target_text)
    span = {
        "start": span_start,
        "end": span_end,
        "severity": "MAJOR",
        "confidence": 1.0,
    }

    token_rewards = spans_to_token_rewards(
        mt_text=rollout.completion_text,
        token_char_offsets=rollout.token_char_offsets,
        error_spans=[span],
        severity_weights=cfg.reward.severity_weights,
        overlap_policy=cfg.reward.overlap_policy,
        majority_threshold=cfg.reward.majority_threshold,
        use_confidence=cfg.reward.use_confidence,
        combine_policy=cfg.reward.span_combine_policy,
    )
    seq_reward = 0.0
    advantages = combine_advantages(
        broadcast_sequence_reward(seq_reward, token_count=len(token_rewards)),
        token_rewards,
    )
    non_zero_ratio = sum(1 for v in token_rewards if abs(v) > 0) / max(1, len(token_rewards))
    expected_nonzero = [
        idx
        for idx, (tok_s, tok_e) in enumerate(rollout.token_char_offsets)
        if _overlaps(span_start, span_end, tok_s, tok_e)
    ]
    actual_nonzero = [idx for idx, value in enumerate(token_rewards) if abs(value) > 0]

    print("=== Span/Loss Alignment Verifier ===")
    print(f"prompt_text: {rollout.prompt_text!r}")
    print(f"completion_text: {rollout.completion_text!r}")
    print(
        f"span: start={span_start}, end={span_end}, severity={span['severity']}, text={rollout.completion_text[span_start:span_end]!r}"
    )
    print(f"expected_nonzero_token_indices: {expected_nonzero}")
    print(f"actual_nonzero_token_indices:   {actual_nonzero}")
    print(f"token_rewards_non_zero_ratio: {non_zero_ratio:.6f}")

    if actual_nonzero != expected_nonzero:
        raise RuntimeError(
            f"Mismatch: expected non-zero token indices {expected_nonzero}, got {actual_nonzero}"
        )

    new_lp = compute_completion_logprobs(
        model,
        rollout.prompt_input_ids,
        rollout.completion_token_ids,
        device="cpu",
    )
    old_lp = torch.tensor(rollout.old_logprobs[: len(new_lp)], dtype=new_lp.dtype)
    adv = torch.tensor(advantages[: len(new_lp)], dtype=new_lp.dtype)

    ratio = torch.exp(new_lp - old_lp)
    clipped = torch.clamp(ratio, 1.0 - cfg.rl.clip_eps, 1.0 + cfg.rl.clip_eps)
    per_token_loss = -torch.minimum(ratio * adv, clipped * adv)

    print("idx | range    | token | in_span | tok_reward | adv      | loss_contrib")
    for idx, (tok_s, tok_e) in enumerate(rollout.token_char_offsets):
        piece = rollout.completion_text[tok_s:tok_e]
        in_span = idx in expected_nonzero
        tok_reward = token_rewards[idx]
        adv_val = float(adv[idx].item())
        loss_val = float(per_token_loss[idx].item())
        print(
            f"{idx:03d} | [{tok_s:02d},{tok_e:02d}) | {piece!r:8} | {str(in_span):7} | "
            f"{tok_reward:+9.4f} | {adv_val:+8.4f} | {loss_val:+11.6f}"
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    stats = update_policy(
        rollouts=rollouts,
        advantages=[advantages],
        policy_model=model,
        optimizer=optimizer,
        rl_cfg=cfg.rl,
        device="cpu",
    )
    print(
        "update_policy stats: "
        f"loss={stats.policy_loss:.6f}, approx_kl={stats.approx_kl:.6f}, "
        f"clip_fraction={stats.clip_fraction:.6f}, token_count={stats.token_count}"
    )
    print("Verifier finished successfully: span-aligned token rewards flowed into token-level policy loss.")


if __name__ == "__main__":
    main()
