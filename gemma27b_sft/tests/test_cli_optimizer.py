from __future__ import annotations

import ast
from pathlib import Path
import unittest


class CliOptimizerTests(unittest.TestCase):
    def test_fixed_adafactor_trainer_passes_weight_decay(self) -> None:
        cli_path = Path(__file__).resolve().parents[1] / "gemma27b_sft" / "cli.py"
        tree = ast.parse(cli_path.read_text(encoding="utf-8"))

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


if __name__ == "__main__":
    unittest.main()
