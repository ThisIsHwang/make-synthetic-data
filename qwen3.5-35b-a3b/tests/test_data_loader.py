from __future__ import annotations

from qwen35_moe_rl.config import DataConfig
from qwen35_moe_rl.data import load_examples


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

    monkeypatch.setattr("qwen35_moe_rl.data._load_records_from_hf_dataset", fake_loader)

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
