from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "sample_infer.sh"


class SampleInferScriptTests(unittest.TestCase):
    def _write_fake_transformers(self, directory: Path) -> Path:
        package_dir = directory / "transformers"
        package_dir.mkdir(parents=True, exist_ok=True)
        (package_dir / "__init__.py").write_text(
            textwrap.dedent(
                """
                import torch


                class _FakeBatch(dict):
                    def to(self, device):
                        return self


                class _FakeTokenizer:
                    def __init__(self, name_or_path):
                        self.name_or_path = name_or_path
                        self.pad_token = None
                        self.eos_token = "</s>"
                        self.eos_token_id = 1
                        self.unk_token_id = 999
                        self.chat_template = None

                    def __call__(self, text, return_tensors=None, add_special_tokens=True, **kwargs):
                        ids = torch.tensor([[11, 12]], dtype=torch.long)
                        return _FakeBatch(
                            {
                                "input_ids": ids,
                                "attention_mask": torch.ones_like(ids),
                            }
                        )

                    def decode(self, ids, skip_special_tokens=True):
                        return "decoded"


                class AutoTokenizer:
                    @classmethod
                    def from_pretrained(cls, name_or_path, use_fast=True, trust_remote_code=False, **kwargs):
                        print(
                            f"FAKE_TOKENIZER_LOAD path={name_or_path} "
                            f"use_fast={use_fast} trust_remote_code={trust_remote_code}"
                        )
                        return _FakeTokenizer(name_or_path)


                class _FakeModel:
                    def __init__(self):
                        self.device = "cpu"

                    def eval(self):
                        return self

                    def generate(self, input_ids=None, **kwargs):
                        prefix = input_ids if input_ids is not None else torch.tensor([[11, 12]], dtype=torch.long)
                        suffix = torch.tensor([[13, 14]], dtype=torch.long)
                        return torch.cat([prefix, suffix], dim=1)


                class AutoModelForCausalLM:
                    @classmethod
                    def from_pretrained(cls, name_or_path, **kwargs):
                        print(
                            f"FAKE_MODEL_LOAD path={name_or_path} "
                            f"trust_remote_code={kwargs.get('trust_remote_code')}"
                        )
                        return _FakeModel()
                """
            ),
            encoding="utf-8",
        )
        return directory

    def _write_model_dir(self, path: Path, with_tokenizer: bool = False) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text("{}", encoding="utf-8")
        (path / "model.safetensors").write_text("", encoding="utf-8")
        if with_tokenizer:
            (path / "tokenizer.json").write_text("{}", encoding="utf-8")

    def _run_script(self, extra_env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update(extra_env)
        env["PYTHON_BIN"] = sys.executable
        return subprocess.run(
            ["bash", str(SCRIPT_PATH)],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_prefers_resolved_config_from_checkpoint_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            fake_pkg_root = self._write_fake_transformers(tmp_path / "fake_pkgs")
            run_root = tmp_path / "run"
            output_dir = run_root / "output"
            checkpoint_dir = output_dir / "checkpoint-10"
            tokenizer_dir = run_root / "tokenizer"

            self._write_model_dir(checkpoint_dir)
            tokenizer_dir.mkdir(parents=True, exist_ok=True)
            (tokenizer_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
            (output_dir / "resolved_config.yaml").write_text(
                textwrap.dedent(
                    f"""
                    model:
                      tokenizer_name_or_path: ../tokenizer
                      trust_remote_code: true
                    data:
                      prompt_template: "RUN {{text}}"
                    train:
                      output_dir: {output_dir}
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            fallback_config = tmp_path / "fallback.yaml"
            fallback_config.write_text(
                textwrap.dedent(
                    """
                    model:
                      tokenizer_name_or_path: ./wrong-tokenizer
                      trust_remote_code: false
                    data:
                      prompt_template: "FALLBACK {text}"
                    train:
                      output_dir: ./wrong-output
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            result = self._run_script(
                {
                    "MODEL_DIR": str(checkpoint_dir),
                    "CONFIG_PATH": str(fallback_config),
                    "PYTHONPATH": f"{fake_pkg_root}:{os.environ.get('PYTHONPATH', '')}",
                }
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
            self.assertIn(f"USING_CONFIG_PATH={output_dir / 'resolved_config.yaml'}", result.stdout)
            self.assertIn(f"USING_MODEL_DIR={checkpoint_dir}", result.stdout)
            self.assertIn(f"USING_TOKENIZER_NAME_OR_PATH={tokenizer_dir}", result.stdout)
            self.assertIn("TRUST_REMOTE_CODE=True", result.stdout)

    def test_uses_train_output_dir_from_config_when_model_dir_is_unset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            fake_pkg_root = self._write_fake_transformers(tmp_path / "fake_pkgs")
            output_dir = tmp_path / "run" / "output"
            self._write_model_dir(output_dir, with_tokenizer=True)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                textwrap.dedent(
                    f"""
                    data:
                      prompt_template: "{{text}}"
                    train:
                      output_dir: {output_dir}
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            result = self._run_script(
                {
                    "CONFIG_PATH": str(config_path),
                    "PYTHONPATH": f"{fake_pkg_root}:{os.environ.get('PYTHONPATH', '')}",
                }
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
            self.assertIn(f"USING_MODEL_DIR={output_dir}", result.stdout)
            self.assertIn(f"USING_CONFIG_PATH={config_path}", result.stdout)

    def test_explicit_missing_config_path_fails_instead_of_falling_back(self) -> None:
        result = self._run_script(
            {
                "CONFIG_PATH": "definitely_missing.yaml",
            }
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Config file not found: definitely_missing.yaml", result.stderr)


if __name__ == "__main__":
    unittest.main()
