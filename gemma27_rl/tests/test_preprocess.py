from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch

from gemma27_rl.config import RLPostTrainConfig
from gemma27_rl.preprocess import (
    filter_examples_by_max_prompt_tokens,
    prepare_dataset_artifacts,
    prepare_prompt_token_lengths,
)
from gemma27_rl.rl_types import Example
from gemma27_rl.rollout import compute_prompt_token_lengths


class _TokenizerStub:
    def __init__(self) -> None:
        self.is_fast = True
        self.vocab_size = 1234
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.special_tokens_map = {"pad_token": "<pad>", "eos_token": "</s>"}
        self.chat_template = None

    def __len__(self) -> int:
        return 1234


class _ChunkingTokenizer(_TokenizerStub):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def __call__(self, texts, return_tensors="pt", add_special_tokens=True, padding=True):  # type: ignore[no-untyped-def]
        del return_tensors, add_special_tokens, padding
        text_list = [str(text) for text in texts]
        self.batch_sizes.append(len(text_list))

        rows: list[list[int]] = []
        for idx, text in enumerate(text_list, start=1):
            token_count = max(1, len(text) % 11 + 1)
            rows.append([idx] * token_count)

        width = max(len(row) for row in rows)
        input_ids = []
        attention_mask = []
        for row in rows:
            pad = width - len(row)
            input_ids.append(row + ([0] * pad))
            attention_mask.append(([1] * len(row)) + ([0] * pad))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def _examples() -> list[Example]:
    return [
        Example(
            example_id="ex-0",
            src_text="hello",
            src_lang="English",
            tgt_lang="Korean",
            src_lang_code="en",
            tgt_lang_code="ko",
            ref_text="안녕하세요",
            domain="casual",
            input_file_path="/tmp/source-a.jsonl",
        ),
        Example(
            example_id="ex-1",
            src_text="goodbye",
            src_lang="English",
            tgt_lang="Korean",
            src_lang_code="en",
            tgt_lang_code="ko",
            ref_text="안녕",
            domain="casual",
            input_file_path="/tmp/source-a.jsonl",
        ),
    ]


def test_prepare_prompt_token_lengths_reuses_cache(tmp_path: Path, monkeypatch) -> None:
    cfg = RLPostTrainConfig()
    cfg.data.preprocess_cache_dir = str(tmp_path / "preprocess_cache")
    cfg.data.prompt_length_batch_size = 3
    tokenizer = _TokenizerStub()
    examples = _examples()
    seen_batch_sizes: list[int] = []

    monkeypatch.setattr(
        "gemma27_rl.preprocess.compute_prompt_token_lengths",
        lambda **kwargs: seen_batch_sizes.append(int(kwargs["batch_size"])) or [11, 17],
    )

    lengths_first, cache_info_first = prepare_prompt_token_lengths(
        cfg=cfg,
        split="train",
        examples=examples,
        tokenizer=tokenizer,
        limit=None,
    )

    assert lengths_first == [11, 17]
    assert seen_batch_sizes == [3]
    assert cache_info_first.cache_hit is False
    assert cache_info_first.path.exists()

    monkeypatch.setattr(
        "gemma27_rl.preprocess.compute_prompt_token_lengths",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("cache should be reused")),
    )

    lengths_second, cache_info_second = prepare_prompt_token_lengths(
        cfg=cfg,
        split="train",
        examples=examples,
        tokenizer=tokenizer,
        limit=None,
    )

    assert lengths_second == [11, 17]
    assert cache_info_second.cache_hit is True
    assert cache_info_second.path == cache_info_first.path


def test_filter_examples_by_max_prompt_tokens_filters_long_examples() -> None:
    examples = _examples()

    filtered_examples, filtered_lengths, dropped = filter_examples_by_max_prompt_tokens(
        examples=examples,
        prompt_lengths=[100, 5000],
        max_prompt_tokens=4096,
    )

    assert [example.example_id for example in filtered_examples] == ["ex-0"]
    assert filtered_lengths == [100]
    assert dropped == 1


def test_compute_prompt_token_lengths_chunks_batches() -> None:
    tokenizer = _ChunkingTokenizer()
    examples = [
        Example(
            example_id=f"ex-{idx}",
            src_text=("x" * (idx + 1)),
            src_lang="English",
            tgt_lang="Korean",
            src_lang_code="en",
            tgt_lang_code="ko",
        )
        for idx in range(5)
    ]

    chunked_lengths = compute_prompt_token_lengths(
        examples=examples,
        tokenizer=tokenizer,
        batch_size=2,
    )
    unchunked_lengths = compute_prompt_token_lengths(
        examples=examples,
        tokenizer=_ChunkingTokenizer(),
        batch_size=32,
    )

    assert tokenizer.batch_sizes == [2, 2, 1]
    assert chunked_lengths == unchunked_lengths


def test_prepare_dataset_artifacts_warms_split_and_prompt_length_caches(tmp_path: Path, monkeypatch) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    split_cache_dir = tmp_path / "split_cache"
    preprocess_cache_dir = tmp_path / "preprocess_cache"
    tokenizer = _TokenizerStub()

    casual_rows = [
        {
            "id": f"casual-{idx}",
            "source": f"casual-src-{idx}",
            "target": f"casual-tgt-{idx}",
            "teacher": {"path": "/root/raw/translation/aihub.en-ko-casual.71265/train.json"},
        }
        for idx in range(4)
    ]
    expert_rows = [
        {
            "id": f"expert-{idx}",
            "source": f"expert-src-{idx}",
            "target": f"expert-tgt-{idx}",
            "teacher": {"path": "/root/raw/translation/aihub.en-ko-expert.111/train.json"},
        }
        for idx in range(4)
    ]

    for name, rows in (("casual.jsonl", casual_rows), ("expert.jsonl", expert_rows)):
        path = train_dir / name
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    cfg = RLPostTrainConfig()
    cfg.data.train_file = None
    cfg.data.eval_file = None
    cfg.data.train_dir = str(train_dir)
    cfg.data.train_glob = "*.jsonl"
    cfg.data.id_field = "id"
    cfg.data.src_text_field = "source"
    cfg.data.ref_text_field = "target"
    cfg.data.split_cache_dir = str(split_cache_dir)
    cfg.data.preprocess_cache_dir = str(preprocess_cache_dir)
    cfg.data.eval_sampling_ratio = 0.25
    cfg.data.eval_sampling_seed = 17
    cfg.data.eval_sampling_min_samples = 1
    cfg.data.prompt_length_batch_size = 4
    cfg.data.batching_strategy = "direction_domain_length"

    monkeypatch.setattr(
        "gemma27_rl.preprocess.compute_prompt_token_lengths",
        lambda **kwargs: [100 + idx for idx, _ in enumerate(kwargs["examples"])],
    )

    first_summary = prepare_dataset_artifacts(cfg, tokenizer=tokenizer)

    split_cache_children = list(split_cache_dir.iterdir())
    assert len(split_cache_children) == 1
    assert (split_cache_children[0] / "train.jsonl").exists()
    assert (split_cache_children[0] / "eval.jsonl").exists()
    assert first_summary["train_count"] == 6
    assert first_summary["eval_count"] == 2
    assert first_summary["prepared_prompt_lengths"] is True
    assert first_summary["prompt_length_count"] == 6
    assert first_summary["prompt_length_batch_size"] == 4
    assert first_summary["prompt_length_cache_hit"] is False
    assert Path(first_summary["prompt_length_cache_path"]).exists()

    monkeypatch.setattr(
        "gemma27_rl.data._load_records_from_dir",
        lambda directory, pattern: (_ for _ in ()).throw(AssertionError("split cache should be reused")),
    )
    monkeypatch.setattr(
        "gemma27_rl.preprocess.compute_prompt_token_lengths",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("prompt-length cache should be reused")),
    )

    second_summary = prepare_dataset_artifacts(cfg, tokenizer=tokenizer)

    assert second_summary["train_count"] == 6
    assert second_summary["eval_count"] == 2
    assert second_summary["prepared_prompt_lengths"] is True
    assert second_summary["prompt_length_cache_hit"] is True
    assert second_summary["prompt_length_cache_path"] == first_summary["prompt_length_cache_path"]


def test_prepare_dataset_artifacts_reports_max_prompt_token_filtering(tmp_path: Path, monkeypatch) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    split_cache_dir = tmp_path / "split_cache"
    preprocess_cache_dir = tmp_path / "preprocess_cache"
    tokenizer = _TokenizerStub()

    rows = [
        {"id": "a", "source": "short-a", "target": "tgt-a"},
        {"id": "b", "source": "long-b", "target": "tgt-b"},
        {"id": "c", "source": "short-c", "target": "tgt-c"},
        {"id": "d", "source": "long-d", "target": "tgt-d"},
    ]
    path = train_dir / "train.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    cfg = RLPostTrainConfig()
    cfg.data.train_file = None
    cfg.data.eval_file = None
    cfg.data.train_dir = str(train_dir)
    cfg.data.train_glob = "*.jsonl"
    cfg.data.id_field = "id"
    cfg.data.src_text_field = "source"
    cfg.data.ref_text_field = "target"
    cfg.data.split_cache_dir = str(split_cache_dir)
    cfg.data.preprocess_cache_dir = str(preprocess_cache_dir)
    cfg.data.eval_sampling_ratio = 0.25
    cfg.data.eval_sampling_seed = 17
    cfg.data.eval_sampling_min_samples = 1
    cfg.data.max_prompt_tokens = 150

    def _fake_lengths(**kwargs):
        split = kwargs["split"]
        examples = kwargs["examples"]
        if split == "train":
            return [100, 200, 110]
        if split == "eval":
            assert len(examples) == 1
            return [250]
        raise AssertionError(f"unexpected split {split}")

    monkeypatch.setattr("gemma27_rl.preprocess.compute_prompt_token_lengths", _fake_lengths)

    summary = prepare_dataset_artifacts(cfg, tokenizer=tokenizer)

    assert summary["max_prompt_tokens"] == 150
    assert summary["train_count"] == 3
    assert summary["filtered_train_count"] == 2
    assert summary["filtered_train_dropped_count"] == 1
    assert summary["eval_count"] == 1
    assert summary["filtered_eval_count"] == 0
    assert summary["filtered_eval_dropped_count"] == 1
    assert Path(summary["eval_prompt_length_cache_path"]).exists()
