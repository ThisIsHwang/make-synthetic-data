from __future__ import annotations

import ast
from pathlib import Path
import unittest


class CliOptimizerTests(unittest.TestCase):
    def _load_tree(self, relative_path: str) -> ast.Module:
        path = Path(__file__).resolve().parents[1] / "gemma27b_sft" / relative_path
        return ast.parse(path.read_text(encoding="utf-8"))

    def _find_function(self, tree: ast.Module, name: str) -> ast.FunctionDef:
        return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)

    def _cfg_attr_assign_linenos(self, fn: ast.FunctionDef, section: str, attr_names: set[str]) -> list[int]:
        lines: list[int] = []
        for node in ast.walk(fn):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Attribute)
                    and isinstance(target.value.value, ast.Name)
                    and target.value.value.id == "cfg"
                    and target.value.attr == section
                    and target.attr in attr_names
                ):
                    lines.append(node.lineno)
        return lines

    def _call_linenos(self, fn: ast.FunctionDef, call_names: set[str]) -> list[int]:
        lines: list[int] = []
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in call_names
            ):
                lines.append(node.lineno)
        return lines

    def test_fixed_adafactor_trainer_passes_weight_decay(self) -> None:
        tree = self._load_tree("cli.py")

        adafactor_call: ast.Call | None = None
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or node.name != "FixedAdafactorTrainer":
                continue
            for item in node.body:
                if not isinstance(item, ast.FunctionDef) or item.name != "create_optimizer":
                    continue
                for inner in ast.walk(item):
                    if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name) and inner.func.id == "Adafactor":
                        adafactor_call = inner
                        break

        self.assertIsNotNone(adafactor_call)
        keyword_names = {kw.arg for kw in adafactor_call.keywords if kw.arg is not None}
        self.assertIn("weight_decay", keyword_names)

    def test_cli_load_model_updates_cfg_attn_implementation(self) -> None:
        tree = self._load_tree("cli.py")
        fn = self._find_function(tree, "_load_model")

        mutation_linenos = self._cfg_attr_assign_linenos(fn, "model", {"attn_implementation"})

        self.assertTrue(mutation_linenos)

    def test_cli_build_training_arguments_updates_cfg_save_steps(self) -> None:
        tree = self._load_tree("cli.py")
        fn = self._find_function(tree, "_build_training_arguments")

        mutation_linenos = self._cfg_attr_assign_linenos(fn, "train", {"save_steps"})

        self.assertTrue(mutation_linenos)

    def test_cli_dump_config_happens_after_runtime_resolution(self) -> None:
        tree = self._load_tree("cli.py")
        run_fn = self._find_function(tree, "run")

        dump_lineno = min(self._call_linenos(run_fn, {"dump_config"}))
        prerequisite_calls = self._call_linenos(run_fn, {"_load_model", "_build_training_arguments"})
        fsdp_mutations = self._cfg_attr_assign_linenos(run_fn, "train", {"fsdp", "fsdp_transformer_layer_cls_to_wrap"})

        self.assertTrue(prerequisite_calls)
        self.assertTrue(fsdp_mutations)
        self.assertGreater(dump_lineno, max(prerequisite_calls + fsdp_mutations))

    def test_cli_trl_dump_config_happens_after_runtime_resolution(self) -> None:
        tree = self._load_tree("cli_trl.py")
        run_fn = self._find_function(tree, "run")

        dump_lineno = min(self._call_linenos(run_fn, {"dump_config"}))
        prerequisite_calls = self._call_linenos(run_fn, {"_load_model", "_build_training_arguments"})
        fsdp_mutations = self._cfg_attr_assign_linenos(run_fn, "train", {"fsdp", "fsdp_transformer_layer_cls_to_wrap"})

        self.assertTrue(prerequisite_calls)
        self.assertTrue(fsdp_mutations)
        self.assertGreater(dump_lineno, max(prerequisite_calls + fsdp_mutations))


if __name__ == "__main__":
    unittest.main()
