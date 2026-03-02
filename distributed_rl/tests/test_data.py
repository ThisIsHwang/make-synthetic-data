from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

datasets = pytest.importorskip("datasets", reason="datasets package required")

from distributed_rl.config import DataConfig, PromptConfig
from distributed_rl.data import load_dataset_for_trl


class TestLoadDatasetForTrl:
    def test_jsonl_loading(self, tmp_path: Path):
        records = [
            {"id": "1", "src_text": "Hello world", "ref_text": "안녕하세요",
             "src_lang": "English", "tgt_lang": "Korean"},
            {"id": "2", "src_text": "Good morning", "ref_text": "좋은 아침",
             "src_lang": "English", "tgt_lang": "Korean"},
        ]
        data_file = tmp_path / "data.jsonl"
        data_file.write_text("\n".join(json.dumps(r) for r in records))

        data_cfg = DataConfig(train_file=str(data_file))
        prompt_cfg = PromptConfig()

        ds = load_dataset_for_trl(data_cfg, prompt_cfg, split="train")
        assert len(ds) == 2
        assert "prompt" in ds.column_names
        assert "src_text" in ds.column_names
        assert "ref_text" in ds.column_names

        # Prompt should be a chat message list
        prompt = ds[0]["prompt"]
        assert isinstance(prompt, list)
        assert prompt[0]["role"] == "user"
        assert "Hello world" in prompt[0]["content"]

    def test_skips_empty_source(self, tmp_path: Path):
        records = [
            {"id": "1", "src_text": "Hello", "src_lang": "English", "tgt_lang": "Korean"},
            {"id": "2", "src_text": "", "src_lang": "English", "tgt_lang": "Korean"},
            {"id": "3", "src_text": "World", "src_lang": "English", "tgt_lang": "Korean"},
        ]
        data_file = tmp_path / "data.jsonl"
        data_file.write_text("\n".join(json.dumps(r) for r in records))

        data_cfg = DataConfig(train_file=str(data_file))
        prompt_cfg = PromptConfig()

        ds = load_dataset_for_trl(data_cfg, prompt_cfg, split="train")
        assert len(ds) == 2

    def test_limit_parameter(self, tmp_path: Path):
        records = [
            {"id": str(i), "src_text": f"Text {i}", "src_lang": "English", "tgt_lang": "Korean"}
            for i in range(10)
        ]
        data_file = tmp_path / "data.jsonl"
        data_file.write_text("\n".join(json.dumps(r) for r in records))

        data_cfg = DataConfig(train_file=str(data_file), limit=3)
        prompt_cfg = PromptConfig()

        ds = load_dataset_for_trl(data_cfg, prompt_cfg, split="train")
        assert len(ds) == 3

    def test_uses_default_language(self, tmp_path: Path):
        records = [{"id": "1", "src_text": "Hello"}]
        data_file = tmp_path / "data.jsonl"
        data_file.write_text(json.dumps(records[0]))

        data_cfg = DataConfig(
            train_file=str(data_file),
            default_src_lang="English",
            default_tgt_lang="Korean",
            default_src_lang_code="en",
            default_tgt_lang_code="ko",
        )
        prompt_cfg = PromptConfig()

        ds = load_dataset_for_trl(data_cfg, prompt_cfg, split="train")
        assert len(ds) == 1
        prompt_text = ds[0]["prompt"][0]["content"]
        assert "English" in prompt_text
        assert "Korean" in prompt_text
