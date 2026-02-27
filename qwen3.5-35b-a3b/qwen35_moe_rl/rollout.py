from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - optional during lightweight tests
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

try:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase
except Exception:  # pragma: no cover - optional during lightweight tests
    PreTrainedModel = Any  # type: ignore[assignment,misc]
    PreTrainedTokenizerBase = Any  # type: ignore[assignment,misc]

from .config import GenerationConfig
from .prompting import (
    DEFAULT_TRANSLATION_PROMPT_TEMPLATE,
    format_translation_prompt,
    postprocess_translation,
)
from .types import Example, Rollout


logger = logging.getLogger(__name__)


@dataclass
class TokenDecodeConfig:
    clean_up_tokenization_spaces: bool = False
    skip_special_tokens: bool = False


def _require_torch() -> None:
    if torch is None or F is None:
        raise RuntimeError("torch is required for rollout generation/logprob computation.")


def _decode_single_token(tokenizer: PreTrainedTokenizerBase, token_id: int, cfg: TokenDecodeConfig) -> str:
    return tokenizer.decode(
        [token_id],
        clean_up_tokenization_spaces=cfg.clean_up_tokenization_spaces,
        skip_special_tokens=cfg.skip_special_tokens,
    )


def _resolve_eos_token_ids(
    tokenizer_eos_token_id: int | list[int] | None,
    model_eos_token_id: int | list[int] | None,
) -> list[int]:
    eos_ids: list[int] = []
    for raw in (model_eos_token_id, tokenizer_eos_token_id):
        if raw is None:
            continue
        if isinstance(raw, int):
            eos_ids.append(int(raw))
            continue
        if isinstance(raw, (list, tuple)):
            for item in raw:
                try:
                    eos_ids.append(int(item))
                except Exception:
                    continue

    uniq: list[int] = []
    seen: set[int] = set()
    for tok in eos_ids:
        if tok in seen:
            continue
        seen.add(tok)
        uniq.append(tok)
    return uniq


def compute_token_char_offsets(
    tokenizer: PreTrainedTokenizerBase,
    completion_token_ids: list[int],
    decode_cfg: TokenDecodeConfig | None = None,
    completion_text: str | None = None,
) -> list[tuple[int, int]]:
    cfg = decode_cfg or TokenDecodeConfig()
    offsets: list[tuple[int, int]] = []
    chunks: list[str] = []
    cursor = 0

    for token_id in completion_token_ids:
        piece = _decode_single_token(tokenizer, token_id, cfg)
        start = cursor
        cursor += len(piece)
        offsets.append((start, cursor))
        chunks.append(piece)

    reconstructed = "".join(chunks)
    if completion_text is None:
        completion_text = tokenizer.decode(
            completion_token_ids,
            clean_up_tokenization_spaces=cfg.clean_up_tokenization_spaces,
            skip_special_tokens=cfg.skip_special_tokens,
        )

    if reconstructed != completion_text:
        is_fast = bool(getattr(tokenizer, "is_fast", False))
        if is_fast:
            try:
                encoded = tokenizer(
                    completion_text,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
                ids = encoded.get("input_ids", [])
                mapping = encoded.get("offset_mapping", [])
                if list(ids) == list(completion_token_ids) and len(mapping) == len(completion_token_ids):
                    return [(int(s), int(e)) for s, e in mapping]
            except Exception as exc:  # pragma: no cover - tokenizer dependent
                logger.warning("offset fallback failed: %s", exc)

        logger.warning(
            "Token offset reconstruction mismatch. reconstructed_len=%s completion_len=%s",
            len(reconstructed),
            len(completion_text),
        )
    return offsets


def _trim_completion_ids(ids: list[int], eos_token_ids: list[int], pad_token_id: int | None) -> list[int]:
    out: list[int] = []
    eos_set = set(int(t) for t in eos_token_ids)
    for token_id in ids:
        if eos_set and token_id in eos_set:
            break
        if pad_token_id is not None and token_id == pad_token_id:
            break
        out.append(int(token_id))
    return out


def compute_completion_logprobs(
    model: PreTrainedModel,
    prompt_input_ids: list[int],
    completion_token_ids: list[int],
    device: str,
) -> torch.Tensor:
    _require_torch()
    if not completion_token_ids:
        return torch.empty(0, dtype=torch.float32)

    if len(prompt_input_ids) == 0:
        raise ValueError("prompt_input_ids must be non-empty")

    model_device = next(model.parameters()).device
    target_device = str(model_device)
    if device != target_device:
        logger.debug(
            "compute_completion_logprobs device override: requested=%s actual_model_device=%s",
            device,
            target_device,
        )

    input_ids = torch.tensor([prompt_input_ids + completion_token_ids], device=model_device, dtype=torch.long)
    attn = torch.ones_like(input_ids)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attn)
    logits = outputs.logits[0]

    prompt_len = len(prompt_input_ids)
    comp_len = len(completion_token_ids)
    start = prompt_len - 1
    end = start + comp_len
    target_logits = logits[start:end, :]

    log_probs = F.log_softmax(target_logits, dim=-1)
    labels = torch.tensor(completion_token_ids, device=model_device, dtype=torch.long)
    token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.detach().cpu()


def generate_rollouts(
    examples: list[Example],
    policy_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    gen_cfg: GenerationConfig,
    device: str,
    ref_model: PreTrainedModel | None = None,
    ref_device: str | None = None,
    prompt_template: str | None = None,
) -> list[Rollout]:
    _require_torch()
    if not examples:
        return []

    pad_token_id = tokenizer.pad_token_id
    model_eos = getattr(getattr(policy_model, "generation_config", None), "eos_token_id", None)
    eos_token_ids = _resolve_eos_token_ids(tokenizer.eos_token_id, model_eos)
    eos_for_generate: int | list[int] | None
    if not eos_token_ids:
        eos_for_generate = None
    elif len(eos_token_ids) == 1:
        eos_for_generate = eos_token_ids[0]
    else:
        eos_for_generate = eos_token_ids

    if pad_token_id is None and eos_token_ids:
        pad_token_id = eos_token_ids[0]

    policy_model.eval()
    if ref_model is not None:
        ref_model.eval()

    rollouts: list[Rollout] = []
    decode_cfg = TokenDecodeConfig()
    ref_dev = ref_device or device

    prompt_texts: list[str] = [
        format_translation_prompt(
            ex,
            template=prompt_template or DEFAULT_TRANSLATION_PROMPT_TEMPLATE,
        )
        for ex in examples
    ]

    original_padding_side = getattr(tokenizer, "padding_side", "right")
    if original_padding_side != "left":
        tokenizer.padding_side = "left"
    try:
        tokenized = None
        use_chat_template = bool(getattr(tokenizer, "chat_template", None)) and hasattr(
            tokenizer,
            "apply_chat_template",
        )
        if use_chat_template:
            try:
                chats = [[{"role": "user", "content": prompt}] for prompt in prompt_texts]
                tokenized = tokenizer.apply_chat_template(
                    chats,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    padding=True,
                )
            except Exception as exc:
                logger.warning("Chat template encode failed; falling back to plain prompt encode: %s", exc)
                tokenized = None

        if tokenized is None:
            tokenized = tokenizer(
                prompt_texts,
                return_tensors="pt",
                add_special_tokens=True,
                padding=True,
            )
    finally:
        tokenizer.padding_side = original_padding_side

    if isinstance(tokenized, dict):
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = attention_mask.to(device)
    else:
        input_ids = tokenized.to(device)
        if pad_token_id is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = (input_ids != int(pad_token_id)).long()

    input_width = int(input_ids.shape[1])
    input_ids_cpu = input_ids.detach().cpu()
    attention_cpu = attention_mask.detach().cpu()
    prompt_id_rows: list[list[int]] = []
    for i in range(input_ids_cpu.shape[0]):
        keep = attention_cpu[i].bool()
        prompt_id_rows.append([int(tok) for tok in input_ids_cpu[i][keep].tolist()])

    do_sample = bool(gen_cfg.do_sample and gen_cfg.temperature > 0)
    with torch.no_grad():
        generated = policy_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=gen_cfg.max_new_tokens,
            do_sample=do_sample,
            temperature=gen_cfg.temperature if do_sample else None,
            top_p=gen_cfg.top_p if do_sample else None,
            top_k=gen_cfg.top_k if do_sample else None,
            repetition_penalty=gen_cfg.repetition_penalty,
            num_return_sequences=gen_cfg.num_samples_per_prompt,
            pad_token_id=pad_token_id,
            eos_token_id=eos_for_generate,
        )

    sequences = generated if isinstance(generated, torch.Tensor) else generated.sequences
    num_return = max(1, int(gen_cfg.num_samples_per_prompt))
    for seq_idx, seq in enumerate(sequences):
        ex_idx = seq_idx // num_return
        if ex_idx >= len(examples):
            break
        ex = examples[ex_idx]
        prompt_text = prompt_texts[ex_idx]
        prompt_ids = prompt_id_rows[ex_idx]

        full_ids = seq.detach().cpu().tolist()
        completion_raw_ids = full_ids[input_width:]
        completion_raw_ids = _trim_completion_ids(
            completion_raw_ids,
            eos_token_ids=eos_token_ids,
            pad_token_id=pad_token_id,
        )
        raw_text = tokenizer.decode(completion_raw_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        completion_text = postprocess_translation(raw_text)
        completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
        completion_ids = [int(x) for x in completion_ids]
        if not completion_ids:
            continue

        old_lp = compute_completion_logprobs(policy_model, prompt_ids, completion_ids, device=device).tolist()
        ref_lp = None
        if ref_model is not None:
            ref_lp = compute_completion_logprobs(ref_model, prompt_ids, completion_ids, device=ref_dev).tolist()

        offsets = compute_token_char_offsets(
            tokenizer=tokenizer,
            completion_token_ids=completion_ids,
            decode_cfg=decode_cfg,
            completion_text=completion_text,
        )

        # Precompute sequence-level logprob means for GSPO.
        old_lp_mean = sum(old_lp) / max(1, len(old_lp))
        ref_lp_mean = None
        if ref_lp is not None:
            ref_lp_mean = sum(ref_lp) / max(1, len(ref_lp))

        rollouts.append(
            Rollout(
                example_id=ex.example_id,
                prompt_text=prompt_text,
                prompt_input_ids=prompt_ids,
                completion_text=completion_text,
                completion_token_ids=completion_ids,
                old_logprobs=old_lp,
                ref_logprobs=ref_lp,
                token_char_offsets=offsets,
                src_text=ex.src_text,
                ref_text=ex.ref_text,
                old_logprob_mean=old_lp_mean,
                ref_logprob_mean=ref_lp_mean,
            )
        )

    return rollouts
