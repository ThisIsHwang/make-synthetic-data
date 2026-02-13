from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase

from .config import DataConfig, SFTConfig


def _messages(source_lang_name: str, target_lang_name: str, source_text: str, target_text: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    system = "You are a professional translator."
    user = (
        f"Translate the following text from {source_lang_name} to {target_lang_name}.\n"
        "Output only the translation without additional commentary.\n\n"
        f"Source:\n{source_text}"
    )
    prompt_messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    full_messages = prompt_messages + [{"role": "assistant", "content": target_text}]
    return prompt_messages, full_messages


def _apply_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    add_generation_prompt: bool,
    max_seq_length: int,
) -> list[int]:
    if getattr(tokenizer, "chat_template", None):
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            truncation=True,
            max_length=max_seq_length,
        )
        return list(ids)

    parts = []
    for msg in messages:
        role = msg["role"].upper()
        parts.append(f"{role}: {msg['content']}")
    if add_generation_prompt:
        parts.append("ASSISTANT:")
    text = "\n\n".join(parts)
    return list(
        tokenizer(
            text,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=True,
        )["input_ids"]
    )


def _build_tokenize_fn(cfg: SFTConfig, tokenizer: PreTrainedTokenizerBase):
    data_cfg = cfg.data
    max_len = cfg.train.max_seq_length

    def _tokenize(example: dict[str, Any]) -> dict[str, Any]:
        source_text = str(example[data_cfg.source_field])
        target_text = str(example[data_cfg.target_field])
        prompt_messages, full_messages = _messages(
            data_cfg.source_lang_name,
            data_cfg.target_lang_name,
            source_text,
            target_text,
        )
        prompt_ids = _apply_chat_template(
            tokenizer=tokenizer,
            messages=prompt_messages,
            add_generation_prompt=True,
            max_seq_length=max_len,
        )
        full_ids = _apply_chat_template(
            tokenizer=tokenizer,
            messages=full_messages,
            add_generation_prompt=False,
            max_seq_length=max_len,
        )
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        non_ignored = sum(1 for x in labels if x != -100)
        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
            "num_target_tokens": non_ignored,
        }

    return _tokenize


@dataclass
class CompletionDataCollator:
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int | None = 8

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch_inputs = [
            {"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]}
            for f in features
        ]
        padded = self.tokenizer.pad(
            batch_inputs,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        seq_len = int(padded["input_ids"].shape[1])
        labels = []
        for feature in features:
            raw = feature["labels"]
            labels.append(raw + [-100] * (seq_len - len(raw)))
        padded["labels"] = torch.tensor(labels, dtype=torch.long)
        return padded


def _load_json_dataset(path: str) -> Dataset:
    ds = load_dataset("json", data_files=path, split="train")
    assert isinstance(ds, Dataset)
    return ds


def build_datasets(cfg: SFTConfig, tokenizer: PreTrainedTokenizerBase) -> tuple[Dataset, Dataset | None]:
    data_cfg: DataConfig = cfg.data
    train_ds = _load_json_dataset(data_cfg.train_file)
    if data_cfg.max_train_samples is not None:
        train_ds = train_ds.select(range(min(len(train_ds), data_cfg.max_train_samples)))

    eval_ds = None
    if data_cfg.eval_file:
        eval_ds = _load_json_dataset(data_cfg.eval_file)
        if data_cfg.max_eval_samples is not None:
            eval_ds = eval_ds.select(range(min(len(eval_ds), data_cfg.max_eval_samples)))

    tokenize_fn = _build_tokenize_fn(cfg, tokenizer)
    train_ds = train_ds.map(
        tokenize_fn,
        num_proc=data_cfg.preprocessing_num_workers or None,
        desc="Tokenizing train dataset",
    )
    train_ds = train_ds.filter(
        lambda ex: ex["num_target_tokens"] > 0,
        num_proc=data_cfg.preprocessing_num_workers or None,
        desc="Filtering empty-train labels",
    )
    train_ds = train_ds.remove_columns(
        [c for c in train_ds.column_names if c not in {"input_ids", "attention_mask", "labels"}]
    )

    if eval_ds is not None:
        eval_ds = eval_ds.map(
            tokenize_fn,
            num_proc=data_cfg.preprocessing_num_workers or None,
            desc="Tokenizing eval dataset",
        )
        eval_ds = eval_ds.filter(
            lambda ex: ex["num_target_tokens"] > 0,
            num_proc=data_cfg.preprocessing_num_workers or None,
            desc="Filtering empty-eval labels",
        )
        eval_ds = eval_ds.remove_columns(
            [c for c in eval_ds.column_names if c not in {"input_ids", "attention_mask", "labels"}]
        )

    return train_ds, eval_ds
