from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

from datasets import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gemma27b_sft.config import SFTConfig
from gemma27b_sft.data import _build_tokenize_fn, _load_json_dataset_resilient, _summarize_tokenization, build_datasets


class StubTokenizer:
    eos_token_id = 3
    unk_token_id = 999999
    model_input_names = ["input_ids", "attention_mask"]
    chat_template = None

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.unk_token_id

    def __call__(
        self,
        text: str,
        truncation: bool = False,
        add_special_tokens: bool = True,
        max_length: int | None = None,
        **_: object,
    ) -> dict[str, list[int]]:
        ids = [ord(ch) for ch in text]
        if add_special_tokens:
            ids = [2] + ids + [3]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}


class NoGenerationPromptTokenizer(StubTokenizer):
    chat_template = "dummy"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        **_: object,
    ):
        text = "\n".join(f"{message['role']}:{message['content']}" for message in messages)
        if tokenize:
            return [ord(ch) for ch in text]
        return text


def _decode(ids: list[int]) -> str:
    return "".join(chr(i) if 32 <= i < 127 else f"<{i}>" for i in ids)


class DataTokenizationTests(unittest.TestCase):
    def _make_cfg(self, max_seq_length: int) -> SFTConfig:
        cfg = SFTConfig()
        cfg.train.max_seq_length = max_seq_length
        cfg.data.prompt_template = "{text}"
        return cfg

    def test_truncation_keeps_generation_prefix_in_prompt_tail(self) -> None:
        cfg = self._make_cfg(max_seq_length=26)
        tokenize_fn = _build_tokenize_fn(cfg, StubTokenizer())

        out = tokenize_fn({"source_text": "P" * 16, "target_text": "ZZ"})
        decoded_input = _decode(out["input_ids"])

        self.assertGreater(int(out["num_target_tokens"]), 0)
        self.assertGreater(int(out["source_shrink_steps"]), 0)
        self.assertIn("ASSISTANT:", decoded_input)

    def test_row_is_dropped_when_generation_prefix_cannot_fit(self) -> None:
        cfg = self._make_cfg(max_seq_length=8)
        tokenize_fn = _build_tokenize_fn(cfg, StubTokenizer())

        out = tokenize_fn({"source_text": "P" * 4, "target_text": "ZZ"})

        self.assertEqual(int(out["num_target_tokens"]), 0)

    def test_row_is_dropped_when_no_prompt_tokens_fit(self) -> None:
        cfg = self._make_cfg(max_seq_length=20)
        tokenize_fn = _build_tokenize_fn(cfg, NoGenerationPromptTokenizer())

        out = tokenize_fn({"source_text": "P", "target_text": "Z" * 40})

        self.assertEqual(int(out["prompt_tokens"]), 0)
        self.assertEqual(int(out["num_target_tokens"]), 0)

    def test_target_truncation_also_triggers_source_shrink(self) -> None:
        cfg = self._make_cfg(max_seq_length=60)
        tokenize_fn = _build_tokenize_fn(cfg, StubTokenizer())

        out = tokenize_fn({"source_text": "PPPP", "target_text": "Z" * 40})

        self.assertGreater(int(out["num_target_tokens"]), 0)
        self.assertGreater(int(out["source_shrink_steps"]), 0)

    def test_build_datasets_drops_rows_with_empty_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_file = tmp_path / "train.jsonl"
            train_file.write_text(
                '{"source_text":"hello","target_text":"world"}\n'
                '{"wrong_source":"ignored","target_text":"still-there"}\n',
                encoding="utf-8",
            )

            cfg = self._make_cfg(max_seq_length=64)
            cfg.data.train_file = str(train_file)
            train_ds, eval_ds = build_datasets(cfg, StubTokenizer())

            self.assertIsNone(eval_ds)
            self.assertEqual(len(train_ds), 1)

    def test_build_datasets_raises_when_all_source_rows_are_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_file = tmp_path / "train.jsonl"
            train_file.write_text(
                '{"wrong_source":"hello","target_text":"world"}\n',
                encoding="utf-8",
            )

            cfg = self._make_cfg(max_seq_length=64)
            cfg.data.train_file = str(train_file)

            with self.assertRaisesRegex(ValueError, "data.source_field points to empty/wrong column."):
                build_datasets(cfg, StubTokenizer())

    def test_json_loader_restores_escaped_newlines_in_source_and_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_file = tmp_path / "train.jsonl"
            train_file.write_text(
                '{"source_text":"line1\\\\nline2","target_text":"ref1\\\\nref2"}\n',
                encoding="utf-8",
            )

            cfg = self._make_cfg(max_seq_length=64)
            ds = _load_json_dataset_resilient(str(train_file), cfg.data)

            row = ds[0]
            self.assertEqual(row["source_text"], "line1\nline2")
            self.assertEqual(row["target_text"], "ref1\nref2")

    def test_tokenization_summary_tracks_raw_and_final_truncation(self) -> None:
        cfg = self._make_cfg(max_seq_length=60)
        tokenize_fn = _build_tokenize_fn(cfg, StubTokenizer())
        rows = [
            tokenize_fn({"source_text": "P" * 80, "target_text": "ZZ"}),
            tokenize_fn({"source_text": "P", "target_text": "Z" * 200}),
        ]

        stats = _summarize_tokenization(Dataset.from_list(rows), cfg.train.max_seq_length)

        self.assertGreater(int(stats["prompt_ge_max_count"]), 0)
        self.assertGreater(int(stats["truncation_seen_count"]), 0)
        self.assertGreater(int(stats["target_truncated_count"]), 0)
        self.assertGreater(int(stats["source_shrunk_count"]), 0)


if __name__ == "__main__":
    unittest.main()
