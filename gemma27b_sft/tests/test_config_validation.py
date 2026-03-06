from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gemma27b_sft.config import load_config
from scripts import eval_xcomet
from scripts.eval_xcomet import (
    _effective_runtime_settings,
    _find_run_config_path,
    _generation_model_kwargs,
    _generation_stop_ids,
    _load_partial_config,
    _load_eval_rows,
    _normalize_source_text,
    _normalize_target_text,
    _requires_target_text,
    _score_xcomet,
)


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

    def test_load_config_rejects_blank_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_file = self._write_train_file(tmp_path)
            config_path = self._write_config(
                tmp_path,
                f"data:\n  train_file: {train_file}\ntrain:\n  output_dir: \"\"\n",
            )

            with self.assertRaisesRegex(ValueError, "train.output_dir must not be empty."):
                load_config(config_path)

    def test_load_config_rejects_non_positive_num_train_epochs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_file = self._write_train_file(tmp_path)
            config_path = self._write_config(
                tmp_path,
                f"data:\n  train_file: {train_file}\ntrain:\n  num_train_epochs: -1\n",
            )

            with self.assertRaisesRegex(ValueError, "train.num_train_epochs must be > 0."):
                load_config(config_path)

    def test_load_config_rejects_non_positive_logging_save_eval_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_file = self._write_train_file(tmp_path)
            cases = [
                ("logging_steps", "train.logging_steps must be > 0."),
                ("save_steps", "train.save_steps must be > 0."),
                ("eval_steps", "train.eval_steps must be > 0."),
            ]

            for field_name, error_msg in cases:
                config_path = self._write_config(
                    tmp_path,
                    f"data:\n  train_file: {train_file}\ntrain:\n  {field_name}: 0\n",
                )
                with self.assertRaisesRegex(ValueError, error_msg):
                    load_config(config_path)

    def test_load_config_rejects_negative_dataloader_num_workers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_file = self._write_train_file(tmp_path)
            config_path = self._write_config(
                tmp_path,
                f"data:\n  train_file: {train_file}\ntrain:\n  dataloader_num_workers: -1\n",
            )

            with self.assertRaisesRegex(ValueError, "train.dataloader_num_workers must be >= 0."):
                load_config(config_path)

    def test_load_config_rejects_negative_weight_decay(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_file = self._write_train_file(tmp_path)
            config_path = self._write_config(
                tmp_path,
                f"data:\n  train_file: {train_file}\ntrain:\n  weight_decay: -0.1\n",
            )

            with self.assertRaisesRegex(ValueError, "train.weight_decay must be >= 0."):
                load_config(config_path)

    def test_load_config_rejects_invalid_max_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_file = self._write_train_file(tmp_path)
            config_path = self._write_config(
                tmp_path,
                f"data:\n  train_file: {train_file}\ntrain:\n  max_steps: -2\n",
            )

            with self.assertRaisesRegex(ValueError, "train.max_steps must be -1 or > 0."):
                load_config(config_path)

    def test_load_config_normalizes_resume_from_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_file = self._write_train_file(tmp_path)
            checkpoint_dir = tmp_path / "checkpoints" / "checkpoint-1"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            config_path = self._write_config(
                tmp_path,
                (
                    f"data:\n  train_file: {train_file}\n"
                    "train:\n"
                    "  resume_from_checkpoint: ./checkpoints/checkpoint-1\n"
                ),
            )

            cfg = load_config(config_path)

            self.assertEqual(cfg.train.resume_from_checkpoint, str(checkpoint_dir.resolve()))

    def test_load_config_rejects_missing_resume_from_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_file = self._write_train_file(tmp_path)
            config_path = self._write_config(
                tmp_path,
                f"data:\n  train_file: {train_file}\ntrain:\n  resume_from_checkpoint: /definitely/not/here\n",
            )

            with self.assertRaisesRegex(FileNotFoundError, "train.resume_from_checkpoint not found: /definitely/not/here"):
                load_config(config_path)

    def test_load_config_treats_blank_resume_from_checkpoint_as_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_file = self._write_train_file(tmp_path)
            config_path = self._write_config(
                tmp_path,
                f"data:\n  train_file: {train_file}\ntrain:\n  resume_from_checkpoint: \"\"\n",
            )

            cfg = load_config(config_path)

            self.assertIsNone(cfg.train.resume_from_checkpoint)

    def test_load_config_expands_user_home_in_config_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_file = self._write_train_file(tmp_path)
            config_path = self._write_config(
                tmp_path,
                f"data:\n  train_file: {train_file}\n",
            )

            with mock.patch.dict(os.environ, {"HOME": tmpdir}):
                cfg = load_config("~/config.yaml")

            self.assertEqual(cfg.data.train_file, str(train_file))

    def test_load_config_rejects_non_jsonl_train_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            bogus_file = tmp_path / "README.md"
            bogus_file.write_text("# not jsonl\n", encoding="utf-8")
            config_path = self._write_config(
                tmp_path,
                f"data:\n  train_file: {bogus_file}\n",
            )

            with self.assertRaisesRegex(ValueError, r"data\.train_file must point to a \.jsonl file:"):
                load_config(config_path)

    def test_load_config_rejects_directory_train_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_dir = tmp_path / "dataset_dir"
            train_dir.mkdir(parents=True, exist_ok=True)
            config_path = self._write_config(
                tmp_path,
                f"data:\n  train_file: {train_dir}\n",
            )

            with self.assertRaisesRegex(ValueError, "data.train_file must be a file:"):
                load_config(config_path)

    def test_load_config_rejects_invalid_jsonl_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_file = tmp_path / "train.jsonl"
            train_file.write_text("not-json\n", encoding="utf-8")
            config_path = self._write_config(
                tmp_path,
                f"data:\n  train_file: {train_file}\n",
            )

            with self.assertRaisesRegex(
                ValueError,
                "data.train_file must be newline-delimited JSON objects. First non-empty line is invalid JSON",
            ):
                load_config(config_path)

    def test_load_config_rejects_blank_source_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_file = self._write_train_file(tmp_path)
            config_path = self._write_config(
                tmp_path,
                f"data:\n  train_file: {train_file}\n  source_field: \"\"\n",
            )

            with self.assertRaisesRegex(ValueError, "data.source_field must not be empty."):
                load_config(config_path)

    def test_load_config_rejects_blank_target_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            train_file = self._write_train_file(tmp_path)
            config_path = self._write_config(
                tmp_path,
                f"data:\n  train_file: {train_file}\n  target_field: \"\"\n",
            )

            with self.assertRaisesRegex(ValueError, "data.target_field must not be empty."):
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

    def test_eval_xcomet_partial_config_resolves_tokenizer_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = self._write_config(
                tmp_path,
                (
                    "model:\n"
                    "  tokenizer_name_or_path: ./tok\n"
                    "train:\n"
                    "  output_dir: ./out\n"
                ),
            )

            _, model_cfg, train_output_dir, _ = _load_partial_config(config_path)

            self.assertEqual(model_cfg.tokenizer_name_or_path, str((tmp_path / "tok").resolve()))
            self.assertEqual(train_output_dir, (tmp_path / "out").resolve())

    def test_eval_xcomet_prefers_resolved_config_for_checkpoint_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            checkpoint_dir = tmp_path / "output" / "checkpoint-10"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            resolved_config = checkpoint_dir.parent / "resolved_config.yaml"
            resolved_config.write_text("{}", encoding="utf-8")
            fallback_config = tmp_path / "config.yaml"
            fallback_config.write_text("{}", encoding="utf-8")

            selected = _find_run_config_path(None, fallback_config, checkpoint_dir)

            self.assertEqual(selected, resolved_config.resolve())

    def test_eval_xcomet_allows_source_only_rows_without_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            eval_file = tmp_path / "eval.jsonl"
            eval_file.write_text(
                '{"source_text":"hello"}\n'
                '{"source_text":"bye","target_text":"annyeong"}\n',
                encoding="utf-8",
            )

            rows = _load_eval_rows(
                eval_file=eval_file,
                source_field="source_text",
                target_field="target_text",
                max_samples=10,
                require_target=False,
            )

            self.assertEqual(len(rows), 2)

    def test_eval_xcomet_normalizes_source_like_training(self) -> None:
        self.assertEqual(_normalize_source_text("  line1\\nline2  "), "  line1\nline2  ")
        self.assertEqual(_normalize_target_text("  ref\\nmore  "), "  ref\nmore  ")

    def test_eval_xcomet_requires_target_for_reference_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            eval_file = tmp_path / "eval.jsonl"
            eval_file.write_text(
                '{"source_text":"hello"}\n'
                '{"source_text":"bye","target_text":"annyeong"}\n',
                encoding="utf-8",
            )

            rows = _load_eval_rows(
                eval_file=eval_file,
                source_field="source_text",
                target_field="target_text",
                max_samples=10,
                require_target=True,
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["target_text"], "annyeong")

    def test_eval_xcomet_skips_invalid_json_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            eval_file = tmp_path / "eval.jsonl"
            eval_file.write_text(
                '{"source_text":"ok","target_text":"ref"}\n'
                'not-json\n'
                '{"source_text":"ok2","target_text":"ref2"}\n',
                encoding="utf-8",
            )

            rows = _load_eval_rows(
                eval_file=eval_file,
                source_field="source_text",
                target_field="target_text",
                max_samples=10,
                require_target=True,
            )

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["source_text"], "ok")
            self.assertEqual(rows[1]["source_text"], "ok2")

    def test_eval_xcomet_target_requirement_depends_on_active_reference_scoring(self) -> None:
        self.assertEqual(_requires_target_text(skip_xcomet=True, use_reference=True), False)
        self.assertEqual(_requires_target_text(skip_xcomet=False, use_reference=False), False)
        self.assertEqual(_requires_target_text(skip_xcomet=False, use_reference=True), True)

    def test_eval_xcomet_generation_uses_normalized_source_and_target(self) -> None:
        captured: list[tuple[str, str]] = []

        class DummyTokenizer:
            eos_token_id = 1
            pad_token_id = 0
            unk_token_id = 999

            def convert_tokens_to_ids(self, token: str) -> int:
                return self.unk_token_id

            def __call__(
                self,
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=None,
                add_special_tokens=False,
            ):
                import torch

                input_ids = torch.tensor([[11, 12]] * len(prompts), dtype=torch.long)
                return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

            def decode(self, ids, skip_special_tokens=True) -> str:
                return "decoded"

        class DummyModel:
            def generate(self, **encoded):
                import torch

                input_ids = encoded["input_ids"]
                suffix = torch.full((input_ids.shape[0], 1), 13, dtype=input_ids.dtype, device=input_ids.device)
                return torch.cat([input_ids, suffix], dim=1)

        def fake_messages(data_cfg, row, src, tgt):
            captured.append((src, tgt))
            return [{"role": "user", "content": src}], []

        old_messages = eval_xcomet._messages
        try:
            eval_xcomet._messages = fake_messages
            hyps = eval_xcomet._generate_translations(
                model=DummyModel(),
                tokenizer=DummyTokenizer(),
                data_cfg=eval_xcomet.DataConfig(),
                rows=[{"source_text": "  line1\\nline2  ", "target_text": "  ref\\nmore  "}],
                generation_batch_size=1,
                max_input_tokens=32,
                max_new_tokens=8,
                device="cpu",
            )
        finally:
            eval_xcomet._messages = old_messages

        self.assertEqual(hyps, ["decoded"])
        self.assertEqual(captured, [("  line1\nline2  ", "  ref\nmore  ")])

    def test_eval_xcomet_scoring_payload_uses_normalized_source_and_reference(self) -> None:
        class DummyScorer:
            def __init__(self) -> None:
                self.payload = None

            def prepare_for_inference(self, payload):
                self.payload = payload
                return payload

            def predict_step(self, batch_inputs):
                return {"scores": [0.5]}

        scorer = DummyScorer()
        scores = _score_xcomet(
            scorer=scorer,
            rows=[{"source_text": "  line1\\nline2  ", "target_text": "  ref\\nmore  "}],
            hypotheses=["hyp"],
            source_field="source_text",
            target_field="target_text",
            use_reference=True,
            batch_size=1,
            device="cpu",
        )

        self.assertEqual(scores, [0.5])
        self.assertEqual(
            scorer.payload,
            [{"src": "  line1\nline2  ", "mt": "hyp", "ref": "  ref\nmore  "}],
        )

    def test_eval_xcomet_stop_ids_skip_unk_end_of_turn(self) -> None:
        class DummyTokenizer:
            eos_token_id = 1
            unk_token_id = 7

            def convert_tokens_to_ids(self, token: str) -> int:
                self.last_token = token
                return 7

        tokenizer = DummyTokenizer()
        self.assertEqual(_generation_stop_ids(tokenizer), [1])

    def test_eval_xcomet_load_tokenizer_passes_trust_remote_code(self) -> None:
        calls: list[tuple[str, bool, bool]] = []

        class DummyTokenizer:
            pad_token = None
            eos_token = "</s>"
            padding_side = "right"

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(name_or_path: str, use_fast: bool, trust_remote_code: bool):
                calls.append((name_or_path, use_fast, trust_remote_code))
                if use_fast:
                    raise RuntimeError("fast tokenizer unavailable")
                return DummyTokenizer()

        old_auto_tokenizer = eval_xcomet.AutoTokenizer
        try:
            eval_xcomet.AutoTokenizer = FakeAutoTokenizer
            tokenizer = eval_xcomet._load_tokenizer("repo-or-path", trust_remote_code=True)
        finally:
            eval_xcomet.AutoTokenizer = old_auto_tokenizer

        self.assertEqual(calls, [("repo-or-path", True, True), ("repo-or-path", False, True)])
        self.assertEqual(tokenizer.pad_token, "</s>")
        self.assertEqual(tokenizer.padding_side, "left")

    def test_eval_xcomet_generation_model_kwargs_include_config_flags(self) -> None:
        model_cfg = eval_xcomet.ModelConfig(
            name_or_path="dummy",
            tokenizer_name_or_path=None,
            trust_remote_code=True,
            attn_implementation="eager",
        )

        kwargs = _generation_model_kwargs(model_cfg, dtype=eval_xcomet.torch.float16)

        self.assertEqual(kwargs["trust_remote_code"], True)
        self.assertEqual(kwargs["attn_implementation"], "eager")
        self.assertEqual(kwargs["torch_dtype"], eval_xcomet.torch.float16)

    def test_eval_xcomet_effective_runtime_settings_clamp_summary_values(self) -> None:
        args = eval_xcomet.argparse.Namespace(
            generation_batch_size=0,
            max_input_tokens=8,
            max_new_tokens=1,
            gen_device="cuda",
            xcomet_batch_size=0,
            xcomet_device="cuda",
        )

        with mock.patch.object(eval_xcomet.torch.cuda, "is_available", return_value=False):
            settings = _effective_runtime_settings(args)

        self.assertEqual(settings["generation_batch_size"], 1)
        self.assertEqual(settings["max_input_tokens"], 32)
        self.assertEqual(settings["max_new_tokens"], 8)
        self.assertEqual(settings["xcomet_batch_size"], 1)
        self.assertEqual(settings["gen_device"], "cpu")
        self.assertEqual(settings["xcomet_device"], "cpu")


if __name__ == "__main__":
    unittest.main()
