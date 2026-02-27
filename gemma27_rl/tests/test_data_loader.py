from __future__ import annotations

import json

from gemma27_rl.config import DataConfig
from gemma27_rl.data import load_examples


def test_load_examples_from_hf_path(monkeypatch) -> None:
    rows = [
        {
            "segment_id": 1,
            "source": "hello",
            "target": "안녕하세요",
            "is_bad_source": False,
        },
        {
            "segment_id": 2,
            "source": "bad",
            "target": "나쁨",
            "is_bad_source": True,
        },
    ]

    def fake_loader(**kwargs):
        del kwargs
        return rows

    monkeypatch.setattr("gemma27_rl.data._load_records_from_hf_dataset", fake_loader)

    cfg = DataConfig(
        train_file=None,
        hf_dataset_name="google/wmt24pp",
        hf_dataset_config_name="en-ko_KR",
        hf_train_split="train",
        id_field="segment_id",
        src_text_field="source",
        ref_text_field="target",
        skip_bad_source=True,
        default_src_lang="English",
        default_tgt_lang="Korean",
        default_src_lang_code="en",
        default_tgt_lang_code="ko",
    )

    examples = load_examples(cfg, split="train", limit=16)
    assert len(examples) == 1
    assert examples[0].example_id == "1"
    assert examples[0].src_text == "hello"
    assert examples[0].ref_text == "안녕하세요"


def test_eval_file_override_with_hf_train(monkeypatch, tmp_path) -> None:
    def fake_loader(**kwargs):
        raise AssertionError("HF loader must not be used when eval_file is set.")

    monkeypatch.setattr("gemma27_rl.data._load_records_from_hf_dataset", fake_loader)

    eval_path = tmp_path / "eval.jsonl"
    rows = [
        {"segment_id": 10, "source": "eval-a", "target": "평가-a"},
        {"segment_id": 11, "source": "eval-b", "target": "평가-b"},
    ]
    with eval_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    cfg = DataConfig(
        train_file=None,
        eval_file=str(eval_path),
        hf_dataset_name="google/wmt24pp",
        hf_dataset_config_name="en-ko_KR",
        hf_train_split="train",
        hf_eval_split="train",
        id_field="segment_id",
        src_text_field="source",
        ref_text_field="target",
    )

    examples = load_examples(cfg, split="eval", limit=16)
    assert [x.example_id for x in examples] == ["10", "11"]
    assert [x.src_text for x in examples] == ["eval-a", "eval-b"]


def test_eval_sampling_split_from_single_train_file(tmp_path) -> None:
    data_path = tmp_path / "runs.jsonl"
    rows = []
    for i in range(10):
        rows.append({"id": i, "source": f"src-{i}", "target": f"tgt-{i}"})
    with data_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    cfg = DataConfig(
        train_file=str(data_path),
        eval_file=None,
        hf_dataset_name=None,
        id_field="id",
        src_text_field="source",
        ref_text_field="target",
        eval_sampling_ratio=0.2,
        eval_sampling_seed=7,
        eval_sampling_min_samples=1,
    )

    train_examples = load_examples(cfg, split="train", limit=None)
    eval_examples = load_examples(cfg, split="eval", limit=None)

    assert len(train_examples) == 8
    assert len(eval_examples) == 2
    train_ids = {ex.example_id for ex in train_examples}
    eval_ids = {ex.example_id for ex in eval_examples}
    assert train_ids.isdisjoint(eval_ids)
    assert train_ids.union(eval_ids) == {str(i) for i in range(10)}


def test_eval_sampling_split_is_order_invariant_by_id(tmp_path) -> None:
    rows = [{"id": i, "source": f"src-{i}", "target": f"tgt-{i}"} for i in range(30)]
    p1 = tmp_path / "runs_a.jsonl"
    p2 = tmp_path / "runs_b.jsonl"

    with p1.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with p2.open("w", encoding="utf-8") as f:
        for row in reversed(rows):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    base = dict(
        eval_file=None,
        hf_dataset_name=None,
        id_field="id",
        src_text_field="source",
        ref_text_field="target",
        eval_sampling_ratio=0.2,
        eval_sampling_seed=123,
        eval_sampling_min_samples=1,
    )

    cfg1 = DataConfig(train_file=str(p1), **base)
    cfg2 = DataConfig(train_file=str(p2), **base)

    eval_ids_1 = {ex.example_id for ex in load_examples(cfg1, split="eval", limit=None)}
    eval_ids_2 = {ex.example_id for ex in load_examples(cfg2, split="eval", limit=None)}
    assert eval_ids_1 == eval_ids_2


def test_eval_sampling_count_uses_absolute_size(tmp_path) -> None:
    data_path = tmp_path / "runs_count.jsonl"
    rows = [{"id": i, "source": f"src-{i}", "target": f"tgt-{i}"} for i in range(12)]
    with data_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    cfg = DataConfig(
        train_file=str(data_path),
        eval_file=None,
        hf_dataset_name=None,
        id_field="id",
        src_text_field="source",
        ref_text_field="target",
        eval_sampling_count=3,
        eval_sampling_ratio=0.5,
        eval_sampling_seed=999,
        eval_sampling_min_samples=1,
    )

    train_examples = load_examples(cfg, split="train", limit=None)
    eval_examples = load_examples(cfg, split="eval", limit=None)
    assert len(eval_examples) == 3
    assert len(train_examples) == 9
