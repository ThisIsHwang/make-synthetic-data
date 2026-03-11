from __future__ import annotations

import ast
from pathlib import Path
import unittest


class CliLoggingStructureTests(unittest.TestCase):
    def _load_tree(self, relative_path: str) -> ast.Module:
        path = Path(__file__).resolve().parents[1] / "gemma27b_sft" / relative_path
        return ast.parse(path.read_text(encoding="utf-8"))

    def _find_function(self, tree: ast.Module, name: str) -> ast.FunctionDef:
        return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)

    def _find_class(self, tree: ast.Module, name: str) -> ast.ClassDef:
        return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)

    def _string_constants(self, node: ast.AST) -> set[str]:
        values: set[str] = set()
        for inner in ast.walk(node):
            if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                values.add(inner.value)
        return values

    def _call_linenos(self, fn: ast.FunctionDef, call_names: set[str]) -> list[int]:
        lines: list[int] = []
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in call_names:
                lines.append(node.lineno)
        return lines

    def _attr_call_linenos(self, fn: ast.FunctionDef, attr_name: str) -> list[int]:
        lines: list[int] = []
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == attr_name
            ):
                lines.append(node.lineno)
        return lines

    def test_cli_defines_jsonl_metrics_callback(self) -> None:
        tree = self._load_tree("cli.py")
        callback_cls = self._find_class(tree, "JsonlMetricsCallback")

        method_names = {node.name for node in callback_cls.body if isinstance(node, ast.FunctionDef)}

        self.assertIn("on_log", method_names)
        self.assertIn("on_train_begin", method_names)
        self.assertIn("on_train_end", method_names)

    def test_build_training_arguments_sets_tracking_related_fields(self) -> None:
        tree = self._load_tree("cli.py")
        fn = self._find_function(tree, "_build_training_arguments")

        call_linenos = self._call_linenos(fn, {"_resolve_report_to"})
        string_constants = self._string_constants(fn)

        self.assertTrue(call_linenos)
        self.assertIn("report_to", string_constants)
        self.assertIn("logging_dir", string_constants)
        self.assertIn("run_name", string_constants)
        self.assertIn("logging_first_step", string_constants)

    def test_cli_run_attaches_jsonl_callback_and_saves_training_artifacts(self) -> None:
        tree = self._load_tree("cli.py")
        run_fn = self._find_function(tree, "run")

        add_callback_lines = self._attr_call_linenos(run_fn, "add_callback")
        metrics_path_lines = self._call_linenos(run_fn, {"_metrics_log_path"})
        save_artifact_lines = self._call_linenos(run_fn, {"_save_training_artifacts"})

        self.assertTrue(add_callback_lines)
        self.assertTrue(metrics_path_lines)
        self.assertTrue(save_artifact_lines)

    def test_cli_trl_run_attaches_jsonl_callback_and_saves_training_artifacts(self) -> None:
        tree = self._load_tree("cli_trl.py")
        run_fn = self._find_function(tree, "run")

        add_callback_lines = self._attr_call_linenos(run_fn, "add_callback")
        save_artifact_lines = self._call_linenos(run_fn, {"_save_training_artifacts"})

        self.assertTrue(add_callback_lines)
        self.assertTrue(save_artifact_lines)

    def test_readme_mentions_new_tracking_artifacts(self) -> None:
        readme_path = Path(__file__).resolve().parents[1] / "README.md"
        readme = readme_path.read_text(encoding="utf-8")

        self.assertIn("training_metrics.jsonl", readme)
        self.assertIn("trainer_state.json", readme)


if __name__ == "__main__":
    unittest.main()
