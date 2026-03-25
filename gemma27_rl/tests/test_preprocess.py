from __future__ import annotations

import json
from pathlib import Path

from gemma27_rl.config import RLPostTrainConfig
from gemma27_rl.preprocess import prepare_dataset_artifacts, prepare_prompt_token_lengths
from gemma27_rl.rl_types import Example


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
    tokenizer = _TokenizerStub()
    examples = _examples()

    monkeypatch.setattr(
        "gemma27_rl.preprocess.compute_prompt_token_lengths",
        lambda **kwargs: [11, 17],
    )

    lengths_first, cache_info_first = prepare_prompt_token_lengths(
        cfg=cfg,
        split="train",
        examples=examples,
        tokenizer=tokenizer,
        limit=None,
    )

    assert lengths_first == [11, 17]
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
