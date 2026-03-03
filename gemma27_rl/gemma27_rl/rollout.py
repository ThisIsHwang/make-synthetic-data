from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
import logging
import os
from typing import Any, Callable

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - optional during lightweight tests
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

try:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase
    from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
except Exception:  # pragma: no cover - optional during lightweight tests
    PreTrainedModel = Any  # type: ignore[assignment,misc]
    PreTrainedTokenizerBase = Any  # type: ignore[assignment,misc]

    class LogitsProcessor:  # type: ignore[no-redef]
        pass

    class LogitsProcessorList(list):  # type: ignore[no-redef]
        pass

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None  # type: ignore[assignment]

from .config import GenerationConfig
from .prompting import (
    DEFAULT_TRANSLATION_PROMPT_TEMPLATE,
    format_translation_prompt,
    postprocess_translation,
)
from .rl_types import Example, Rollout


logger = logging.getLogger(__name__)
_PROMPT_ENCODING_CACHE: OrderedDict[tuple[int, bool, str, str], tuple[int, ...]] = OrderedDict()


@dataclass
class TokenDecodeConfig:
    clean_up_tokenization_spaces: bool = False
    skip_special_tokens: bool = False


def _require_torch() -> None:
    if torch is None or F is None:
        raise RuntimeError("torch is required for rollout generation/logprob computation.")


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return max(minimum, int(default))
    try:
        value = int(raw.strip())
    except Exception:
        return max(minimum, int(default))
    return max(minimum, value)


def _serialize_chat_template_kwargs(chat_template_kwargs: dict[str, Any] | None) -> str:
    if not chat_template_kwargs:
        return "{}"
    try:
        return json.dumps(chat_template_kwargs, sort_keys=True, ensure_ascii=True, default=str)
    except Exception:
        return repr(chat_template_kwargs)


def _prompt_cache_limit() -> int:
    return _env_int("GEMMA27_RL_PROMPT_CACHE_SIZE", default=8192, minimum=0)


def _prompt_cache_get(key: tuple[int, bool, str, str]) -> list[int] | None:
    hit = _PROMPT_ENCODING_CACHE.get(key)
    if hit is None:
        return None
    _PROMPT_ENCODING_CACHE.move_to_end(key)
    return list(hit)


def _prompt_cache_put(key: tuple[int, bool, str, str], prompt_ids: list[int]) -> None:
    limit = _prompt_cache_limit()
    if limit <= 0:
        return
    _PROMPT_ENCODING_CACHE[key] = tuple(int(tok) for tok in prompt_ids)
    _PROMPT_ENCODING_CACHE.move_to_end(key)
    while len(_PROMPT_ENCODING_CACHE) > limit:
        _PROMPT_ENCODING_CACHE.popitem(last=False)


def _looks_like_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return ("out of memory" in text) or ("cuda oom" in text)


def _resolve_model_vocab_size(model: PreTrainedModel) -> int | None:
    try:
        emb = model.get_input_embeddings()
    except Exception:
        emb = None
    if emb is not None and hasattr(emb, "num_embeddings"):
        try:
            return int(emb.num_embeddings)
        except Exception:
            pass
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "vocab_size"):
        try:
            return int(cfg.vocab_size)
        except Exception:
            pass
    return None


def _validate_token_ids_in_vocab(
    token_ids: list[int],
    *,
    vocab_size: int | None,
    context: str,
) -> None:
    if vocab_size is None or vocab_size <= 0 or not token_ids:
        return
    low = min(token_ids)
    high = max(token_ids)
    if low < 0 or high >= vocab_size:
        raise ValueError(
            f"{context} token id out of vocab range: min={low} max={high} vocab_size={vocab_size}. "
            "Policy/reference tokenizer or model vocab may be mismatched."
        )


def _get_model_vocab_size(model: PreTrainedModel) -> int | None:
    getter = getattr(model, "get_input_embeddings", None)
    if callable(getter):
        try:
            embeddings = getter()
            size = int(getattr(embeddings, "num_embeddings", 0) or 0)
            if size > 0:
                return size
        except Exception:
            pass
    config_obj = getattr(model, "config", None)
    try:
        size = int(getattr(config_obj, "vocab_size", 0) or 0)
    except Exception:
        size = 0
    return size if size > 0 else None


def _validate_item_token_ids(
    *,
    items: list[tuple[list[int], list[int]]],
    vocab_size: int | None,
    tag: str,
) -> None:
    if not items or vocab_size is None or vocab_size <= 0:
        return

    for item_idx, (prompt_ids, completion_ids) in enumerate(items):
        for field_name, ids in (("prompt_input_ids", prompt_ids), ("completion_token_ids", completion_ids)):
            for pos, tok in enumerate(ids):
                token_id = int(tok)
                if 0 <= token_id < vocab_size:
                    continue
                raise ValueError(
                    f"{tag}: token id out of range for model vocab_size={vocab_size} "
                    f"(item={item_idx}, field={field_name}, position={pos}, token_id={token_id}). "
                    "Tokenizer/model mismatch is likely."
                )


def _compute_logprobs_batch_with_backoff(
    model: PreTrainedModel,
    items: list[tuple[list[int], list[int]]],
    *,
    device: str,
    tag: str,
) -> list[torch.Tensor]:
    if not items:
        return []
    micro_batch = min(
        len(items),
        _env_int("GEMMA27_RL_LOGPROB_MICRO_BATCH", default=32, minimum=1),
    )
    while True:
        try:
            return compute_completion_logprobs_batch(
                model=model,
                items=items,
                device=device,
                micro_batch_size=micro_batch,
            )
        except Exception as exc:
            if (not _looks_like_cuda_oom(exc)) or micro_batch <= 1:
                raise
            micro_batch = max(1, micro_batch // 2)
            logger.warning(
                "logprob batch OOM in %s; retrying with smaller micro_batch=%s.",
                tag,
                micro_batch,
            )
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()


def _distributed_world_size() -> int:
    if torch is None:
        return 1
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_world_size())
    except Exception:  # pragma: no cover - defensive
        return 1
    return 1


def _should_enable_synced_gpus() -> bool:
    # For ZeRO-3 style distributed generation, ranks can otherwise diverge in
    # decode-step counts and deadlock in collectives.
    world_size = _distributed_world_size()
    if world_size <= 1:
        return False
    return _env_flag("GEMMA27_RL_SYNCED_GENERATION", default=True)


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


def _extract_prompt_rows_from_tokenized(
    tokenized: Any,
    *,
    pad_token_id: int | None,
) -> list[list[int]]:
    _require_torch()
    if hasattr(tokenized, "keys") and ("input_ids" in tokenized):
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized.get("attention_mask")
    elif torch.is_tensor(tokenized):
        input_ids = tokenized
        attention_mask = None
    else:
        raise TypeError(
            f"Unsupported tokenized output type: {type(tokenized)!r}. "
            "Expected mapping with `input_ids` or torch.Tensor."
        )

    if not torch.is_tensor(input_ids):
        input_ids = torch.as_tensor(input_ids, dtype=torch.long)
    if attention_mask is None:
        if pad_token_id is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = (input_ids != int(pad_token_id)).long()
    else:
        if not torch.is_tensor(attention_mask):
            attention_mask = torch.as_tensor(attention_mask, dtype=torch.long)

    input_ids_cpu = input_ids.detach().cpu()
    attention_cpu = attention_mask.detach().cpu()
    rows: list[list[int]] = []
    for i in range(input_ids_cpu.shape[0]):
        keep = attention_cpu[i].bool()
        rows.append([int(tok) for tok in input_ids_cpu[i][keep].tolist()])
    return rows


def _encode_prompt_rows(
    *,
    tokenizer: PreTrainedTokenizerBase,
    prompt_texts: list[str],
    gen_cfg: GenerationConfig,
    pad_token_id: int | None,
) -> list[list[int]]:
    _require_torch()
    if not prompt_texts:
        return []

    use_chat_template = bool(getattr(tokenizer, "chat_template", None)) and hasattr(
        tokenizer,
        "apply_chat_template",
    )
    chat_kwargs = (
        dict(gen_cfg.chat_template_kwargs)
        if getattr(gen_cfg, "chat_template_kwargs", None) is not None
        else {}
    )
    kwargs_key = _serialize_chat_template_kwargs(chat_kwargs if use_chat_template else None)
    tok_id = id(tokenizer)

    rows: list[list[int] | None] = [None] * len(prompt_texts)
    missing_indices: list[int] = []
    missing_texts: list[str] = []
    missing_keys: list[tuple[int, bool, str, str]] = []

    for idx, prompt in enumerate(prompt_texts):
        key = (tok_id, use_chat_template, kwargs_key, prompt)
        cached = _prompt_cache_get(key)
        if cached is not None:
            rows[idx] = cached
        else:
            missing_indices.append(idx)
            missing_texts.append(prompt)
            missing_keys.append(key)

    if missing_texts:
        tokenized = None
        cache_missing_rows = True
        if use_chat_template:
            try:
                chats = [[{"role": "user", "content": prompt}] for prompt in missing_texts]
                tokenized = tokenizer.apply_chat_template(
                    chats,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    padding=True,
                    **chat_kwargs,
                )
            except Exception as exc:
                logger.warning("Chat template encode failed; falling back to plain prompt encode: %s", exc)
                tokenized = None
                cache_missing_rows = False

        if tokenized is None:
            tokenized = tokenizer(
                missing_texts,
                return_tensors="pt",
                add_special_tokens=True,
                padding=True,
            )

        missing_rows = _extract_prompt_rows_from_tokenized(tokenized, pad_token_id=pad_token_id)
        if len(missing_rows) != len(missing_indices):
            raise RuntimeError(
                "prompt encoding row mismatch: "
                f"missing={len(missing_indices)} encoded={len(missing_rows)}"
            )
        for idx, key, row in zip(missing_indices, missing_keys, missing_rows):
            clean_row = [int(tok) for tok in row]
            rows[idx] = clean_row
            if cache_missing_rows:
                _prompt_cache_put(key, clean_row)

    out_rows: list[list[int]] = []
    for idx, row in enumerate(rows):
        if row is None:
            raise RuntimeError(f"prompt row missing after encoding at index={idx}")
        out_rows.append(row)
    return out_rows


def _build_left_padded_prompt_tensors(
    *,
    prompt_id_rows: list[list[int]],
    device: str,
    pad_token_id: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_torch()
    if not prompt_id_rows:
        empty = torch.empty((0, 0), dtype=torch.long, device=device)
        return empty, empty

    width = max(1, max(len(row) for row in prompt_id_rows))
    fill_value = int(pad_token_id) if pad_token_id is not None else 0
    input_ids = torch.full((len(prompt_id_rows), width), fill_value=fill_value, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(prompt_id_rows), width), dtype=torch.long, device=device)
    for row_idx, row in enumerate(prompt_id_rows):
        if not row:
            continue
        values = torch.tensor(row, dtype=torch.long, device=device)
        take = len(row)
        input_ids[row_idx, width - take:width] = values
        attention_mask[row_idx, width - take:width] = 1
    return input_ids, attention_mask


def compute_token_char_offsets(
    tokenizer: PreTrainedTokenizerBase,
    completion_token_ids: list[int],
    decode_cfg: TokenDecodeConfig | None = None,
    completion_text: str | None = None,
) -> list[tuple[int, int]]:
    cfg = decode_cfg or TokenDecodeConfig()
    if not completion_token_ids:
        return []

    # Fast tokenizers can return native offset mapping without per-token decode.
    is_fast = bool(getattr(tokenizer, "is_fast", False))
    if is_fast:
        if completion_text is None:
            completion_text = tokenizer.decode(
                completion_token_ids,
                clean_up_tokenization_spaces=cfg.clean_up_tokenization_spaces,
                skip_special_tokens=cfg.skip_special_tokens,
            )
        try:
            encoded = tokenizer(
                completion_text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            ids = encoded.get("input_ids", [])
            mapping = encoded.get("offset_mapping", [])
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            if mapping and isinstance(mapping[0], list):
                mapping = mapping[0]
            if list(ids) == list(completion_token_ids) and len(mapping) == len(completion_token_ids):
                return [(int(s), int(e)) for s, e in mapping]
        except Exception as exc:  # pragma: no cover - tokenizer dependent
            logger.warning("offset fast-path failed: %s", exc)

    # Fallback for tokenizer/text normalization mismatch: decode each token piece.
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


class _PresenceFrequencyPenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, *, start_index: int, presence_penalty: float, frequency_penalty: float) -> None:
        self.start_index = max(0, int(start_index))
        self.presence_penalty = float(presence_penalty)
        self.frequency_penalty = float(frequency_penalty)

    def __call__(self, input_ids: Any, scores: Any) -> Any:  # transformers runtime signature
        if self.presence_penalty == 0.0 and self.frequency_penalty == 0.0:
            return scores
        if not torch.is_tensor(input_ids) or not torch.is_tensor(scores):  # pragma: no cover - defensive
            return scores
        if input_ids.dim() != 2 or scores.dim() != 2:  # pragma: no cover - defensive
            return scores

        seq_len = int(input_ids.shape[1])
        if seq_len <= self.start_index:
            return scores

        for row_idx in range(int(input_ids.shape[0])):
            generated_ids = input_ids[row_idx, self.start_index:]
            if generated_ids.numel() == 0:
                continue
            unique_ids, counts = torch.unique(generated_ids, return_counts=True)
            if unique_ids.numel() == 0:
                continue
            if self.presence_penalty != 0.0:
                scores[row_idx, unique_ids] = scores[row_idx, unique_ids] - self.presence_penalty
            if self.frequency_penalty != 0.0:
                scores[row_idx, unique_ids] = (
                    scores[row_idx, unique_ids] - self.frequency_penalty * counts.to(dtype=scores.dtype)
                )
        return scores


def _build_custom_logits_processors(gen_cfg: GenerationConfig, *, start_index: int) -> LogitsProcessorList | None:
    presence_penalty = float(getattr(gen_cfg, "presence_penalty", 0.0) or 0.0)
    frequency_penalty = float(getattr(gen_cfg, "frequency_penalty", 0.0) or 0.0)
    if presence_penalty == 0.0 and frequency_penalty == 0.0:
        return None
    processors = LogitsProcessorList()
    processors.append(
        _PresenceFrequencyPenaltyLogitsProcessor(
            start_index=start_index,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
    )
    return processors


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
    vocab_size = _resolve_model_vocab_size(model)
    _validate_token_ids_in_vocab(prompt_input_ids, vocab_size=vocab_size, context="prompt_input_ids")
    _validate_token_ids_in_vocab(completion_token_ids, vocab_size=vocab_size, context="completion_token_ids")

    _validate_item_token_ids(
        items=[([int(v) for v in prompt_input_ids], [int(v) for v in completion_token_ids])],
        vocab_size=_get_model_vocab_size(model),
        tag="compute_completion_logprobs",
    )

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


def compute_completion_logprobs_batch(
    model: PreTrainedModel,
    items: list[tuple[list[int], list[int]]],
    device: str,
    *,
    micro_batch_size: int | None = None,
) -> list[torch.Tensor]:
    _require_torch()
    if not items:
        return []

    model_device = next(model.parameters()).device
    target_device = str(model_device)
    if device != target_device:
        logger.debug(
            "compute_completion_logprobs_batch device override: requested=%s actual_model_device=%s",
            device,
            target_device,
        )

    parsed_items: list[tuple[list[int], list[int]]] = []
    vocab_size = _resolve_model_vocab_size(model)
    for prompt_input_ids, completion_token_ids in items:
        if len(prompt_input_ids) == 0:
            raise ValueError("prompt_input_ids must be non-empty")
        _validate_token_ids_in_vocab(prompt_input_ids, vocab_size=vocab_size, context="prompt_input_ids")
        _validate_token_ids_in_vocab(completion_token_ids, vocab_size=vocab_size, context="completion_token_ids")
        parsed_items.append(
            (
                [int(v) for v in prompt_input_ids],
                [int(v) for v in completion_token_ids],
            )
        )

    _validate_item_token_ids(
        items=parsed_items,
        vocab_size=_get_model_vocab_size(model),
        tag="compute_completion_logprobs_batch",
    )

    step = int(micro_batch_size or len(parsed_items))
    step = max(1, min(step, len(parsed_items)))
    all_rows: list[torch.Tensor] = []

    for start_idx in range(0, len(parsed_items), step):
        chunk = parsed_items[start_idx:start_idx + step]
        if not chunk:
            continue

        prompt_lens = [len(prompt_ids) for prompt_ids, _ in chunk]
        comp_lens = [len(completion_ids) for _, completion_ids in chunk]
        seq_lens = [p + c for p, c in zip(prompt_lens, comp_lens)]
        max_seq_len = max(seq_lens)

        batch_size = len(chunk)
        input_ids = torch.zeros((batch_size, max_seq_len), device=model_device, dtype=torch.long)
        attn = torch.zeros((batch_size, max_seq_len), device=model_device, dtype=torch.long)

        for row_idx, (prompt_ids, completion_ids) in enumerate(chunk):
            full_ids = prompt_ids + completion_ids
            seq_len = len(full_ids)
            if seq_len <= 0:
                continue
            row_tensor = torch.tensor(full_ids, device=model_device, dtype=torch.long)
            input_ids[row_idx, :seq_len] = row_tensor
            attn[row_idx, :seq_len] = 1

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attn)
        logits = outputs.logits

        for row_idx, (_, completion_ids) in enumerate(chunk):
            comp_len = len(completion_ids)
            if comp_len <= 0:
                all_rows.append(torch.empty(0, dtype=torch.float32))
                continue

            start = prompt_lens[row_idx] - 1
            end = start + comp_len
            target_logits = logits[row_idx, start:end, :]
            log_probs = F.log_softmax(target_logits, dim=-1)
            labels = torch.tensor(completion_ids, device=model_device, dtype=torch.long)
            token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            all_rows.append(token_log_probs.detach().cpu())

    return all_rows


def generate_rollouts(
    examples: list[Example],
    policy_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    gen_cfg: GenerationConfig,
    device: str,
    ref_model: PreTrainedModel | None = None,
    ref_device: str | None = None,
    ref_logprob_fn: Callable[[list[int], list[int]], list[float]] | None = None,
    prompt_template: str | None = None,
    show_progress: bool = False,
    progress_desc: str | None = None,
    compute_old_logprobs: bool = True,
    compute_token_offsets: bool = True,
    include_prompt_input_ids: bool = True,
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
    empty_completion_fallbacks = 0
    pending_policy_logprob_items: list[tuple[int, list[int], list[int]]] = []
    pending_ref_model_logprob_items: list[tuple[int, list[int], list[int]]] = []

    prompt_texts: list[str] = [
        format_translation_prompt(
            ex,
            template=prompt_template or DEFAULT_TRANSLATION_PROMPT_TEMPLATE,
        )
        for ex in examples
    ]

    prompt_id_rows = _encode_prompt_rows(
        tokenizer=tokenizer,
        prompt_texts=prompt_texts,
        gen_cfg=gen_cfg,
        pad_token_id=pad_token_id,
    )
    input_ids, attention_mask = _build_left_padded_prompt_tensors(
        prompt_id_rows=prompt_id_rows,
        device=device,
        pad_token_id=pad_token_id,
    )
    input_width = int(input_ids.shape[1])

    do_sample = bool(gen_cfg.do_sample and gen_cfg.temperature > 0)
    logits_processor = _build_custom_logits_processors(gen_cfg, start_index=input_width)
    synced_gpus = _should_enable_synced_gpus()
    if synced_gpus:
        logger.info(
            "generate_rollouts: enabling synced_gpus=True (world_size=%s).",
            _distributed_world_size(),
        )
    with torch.no_grad():
        generated = policy_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=gen_cfg.max_new_tokens,
            use_cache=True,
            do_sample=do_sample,
            temperature=gen_cfg.temperature if do_sample else None,
            top_p=gen_cfg.top_p if do_sample else None,
            top_k=gen_cfg.top_k if do_sample else None,
            repetition_penalty=gen_cfg.repetition_penalty,
            num_return_sequences=gen_cfg.num_samples_per_prompt,
            pad_token_id=pad_token_id,
            eos_token_id=eos_for_generate,
            logits_processor=logits_processor,
            synced_gpus=synced_gpus,
        )

    sequences = generated if isinstance(generated, torch.Tensor) else generated.sequences
    num_return = max(1, int(gen_cfg.num_samples_per_prompt))
    progress_total = int(sequences.shape[0]) if hasattr(sequences, "shape") else None
    iterable = enumerate(sequences)
    bar = None
    if show_progress and tqdm is not None:
        bar = tqdm(
            iterable,
            total=progress_total,
            desc=progress_desc or "rollout",
            leave=False,
            mininterval=2.0,
        )
        iterable = bar
    try:
        for seq_idx, seq in iterable:
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
                empty_completion_fallbacks += 1
                fallback_ids = tokenizer(" ", add_special_tokens=False)["input_ids"]
                fallback_ids = [int(x) for x in fallback_ids]
                if not fallback_ids:
                    if completion_raw_ids:
                        fallback_ids = [int(completion_raw_ids[0])]
                    elif eos_token_ids:
                        fallback_ids = [int(eos_token_ids[0])]
                    elif pad_token_id is not None:
                        fallback_ids = [int(pad_token_id)]
                    else:
                        fallback_ids = [0]
                completion_ids = fallback_ids[:1]
                completion_text = tokenizer.decode(
                    completion_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

            old_lp: list[float] = []
            rollout_idx = len(rollouts)
            if compute_old_logprobs:
                pending_policy_logprob_items.append((rollout_idx, list(prompt_ids), list(completion_ids)))
            ref_lp = None
            if compute_old_logprobs and ref_logprob_fn is not None:
                ref_lp = [float(v) for v in ref_logprob_fn(prompt_ids, completion_ids)]
            elif compute_old_logprobs and ref_model is not None:
                pending_ref_model_logprob_items.append((rollout_idx, list(prompt_ids), list(completion_ids)))

            offsets: list[tuple[int, int]] = []
            if compute_token_offsets:
                offsets = compute_token_char_offsets(
                    tokenizer=tokenizer,
                    completion_token_ids=completion_ids,
                    decode_cfg=decode_cfg,
                    completion_text=completion_text,
                )

            rollouts.append(
                Rollout(
                    example_id=ex.example_id,
                    prompt_text=prompt_text,
                    prompt_input_ids=(prompt_ids if include_prompt_input_ids else []),
                    completion_text=completion_text,
                    completion_token_ids=completion_ids,
                    old_logprobs=old_lp,
                    ref_logprobs=ref_lp,
                    token_char_offsets=offsets,
                    src_text=ex.src_text,
                    ref_text=ex.ref_text,
                )
            )
    finally:
        if bar is not None:
            bar.close()

    if compute_old_logprobs and pending_policy_logprob_items:
        policy_rows = _compute_logprobs_batch_with_backoff(
            model=policy_model,
            items=[(prompt_ids, completion_ids) for _, prompt_ids, completion_ids in pending_policy_logprob_items],
            device=device,
            tag="policy_old_logprobs",
        )
        if len(policy_rows) != len(pending_policy_logprob_items):
            raise RuntimeError(
                "policy old_logprobs batch size mismatch: "
                f"requested={len(pending_policy_logprob_items)} returned={len(policy_rows)}"
            )
        for (rollout_idx, _, _), row in zip(pending_policy_logprob_items, policy_rows):
            if rollout_idx < len(rollouts):
                rollouts[rollout_idx].old_logprobs = [float(v) for v in row.tolist()]

    if compute_old_logprobs and pending_ref_model_logprob_items and ref_model is not None:
        ref_rows = _compute_logprobs_batch_with_backoff(
            model=ref_model,
            items=[(prompt_ids, completion_ids) for _, prompt_ids, completion_ids in pending_ref_model_logprob_items],
            device=ref_dev,
            tag="reference_model_logprobs",
        )
        if len(ref_rows) != len(pending_ref_model_logprob_items):
            raise RuntimeError(
                "reference logprobs batch size mismatch: "
                f"requested={len(pending_ref_model_logprob_items)} returned={len(ref_rows)}"
            )
        for (rollout_idx, _, _), row in zip(pending_ref_model_logprob_items, ref_rows):
            if rollout_idx < len(rollouts):
                rollouts[rollout_idx].ref_logprobs = [float(v) for v in row.tolist()]

    if empty_completion_fallbacks > 0:
        logger.info(
            "generate_rollouts: replaced %s empty completions with fallback token to keep rollout shape stable.",
            empty_completion_fallbacks,
        )

    return rollouts
