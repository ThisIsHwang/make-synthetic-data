from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gemma27b_sft.config import load_config
from scripts.eval_xcomet import _generation_stop_ids, _load_partial_config


class ConfigValidationTests(unittest.TestCase):
    def _write_train_file(self, directory: Path) -> Path:
        train_file = directory / "train.jsonl"
        train_file.write_text('{"source_text":"a","target_text":"b"}\n', encoding="utf-8")
        return train_file

    def _write_config(self, directory: Path, body: str) -> Path:
        config_path = directory / "config.yaml"
        config_path.write_text(body, encoding="utf-8")
        return config_path

    def test_load_config_rejects_unknown_train_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_file = self._write_train_file(tmp_path)
            config_path = self._write_config(
                tmp_path,
                f"data:\n  train_file: {train_file}\ntrain:\n  global_batch_szie: 999\n",
            )

            with self.assertRaisesRegex(ValueError, "Unknown config keys in train: global_batch_szie"):
                load_config(config_path)

    def test_load_config_rejects_non_mapping_train_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_file = self._write_train_file(tmp_path)
            config_path = self._write_config(
                tmp_path,
                f"data:\n  train_file: {train_file}\ntrain: []\n",
            )

            with self.assertRaisesRegex(ValueError, "Config section train must be a mapping/object."):
                load_config(config_path)

    def test_load_config_accepts_text_only_prompt_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_file = self._write_train_file(tmp_path)
            config_path = self._write_config(
                tmp_path,
                (
                    f"data:\n  train_file: {train_file}\n  prompt_template: |\n"
                    "    Translate to Korean:\n\n"
                    "    {text}\n"
                ),
            )

            cfg = load_config(config_path)
            self.assertEqual(cfg.data.prompt_template.strip(), "Translate to Korean:\n\n{text}")

    def test_load_config_rejects_unknown_prompt_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_file = self._write_train_file(tmp_path)
            config_path = self._write_config(
                tmp_path,
                (
                    f"data:\n  train_file: {train_file}\n"
                    '  prompt_template: "Translate: {text} {bogus}"\n'
                ),
            )

            with self.assertRaisesRegex(ValueError, "data.prompt_template has unknown placeholders: bogus"):
                load_config(config_path)

    def test_load_config_requires_text_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_file = self._write_train_file(tmp_path)
            config_path = self._write_config(
                tmp_path,
                (
                    f"data:\n  train_file: {train_file}\n"
                    '  prompt_template: "Translate to Korean."\n'
                ),
            )

            with self.assertRaisesRegex(ValueError, r"data\.prompt_template must include \{text\}\."):
                load_config(config_path)

    def test_eval_xcomet_partial_config_rejects_unknown_data_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = self._write_config(tmp_path, "data:\n  target_fieldd: hypothesis\n")

            with self.assertRaisesRegex(ValueError, "Unknown config keys in data: target_fieldd"):
                _load_partial_config(config_path)

    def test_eval_xcomet_stop_ids_skip_unk_end_of_turn(self) -> None:
        class DummyTokenizer:
            eos_token_id = 1
            unk_token_id = 7

            def convert_tokens_to_ids(self, token: str) -> int:
                self.last_token = token
                return 7

        tokenizer = DummyTokenizer()
        self.assertEqual(_generation_stop_ids(tokenizer), [1])


if __name__ == "__main__":
    unittest.main()
