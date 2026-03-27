from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_imports_succeed_when_transformers_is_unavailable() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(repo_root)
    )
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class _BlockTransformers(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                del path, target
                if fullname == "transformers" or fullname.startswith("transformers."):
                    raise ModuleNotFoundError("blocked transformers for lightweight import test")
                return None

        sys.meta_path.insert(0, _BlockTransformers())

        import gemma27_rl.preprocess as preprocess_mod
        import gemma27_rl.trainer as trainer_mod

        assert isinstance(preprocess_mod._TRANSFORMERS_IMPORT_ERROR, ModuleNotFoundError)
        assert isinstance(trainer_mod._TRANSFORMERS_IMPORT_ERROR, ModuleNotFoundError)
        print("ok")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        cwd=repo_root,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")
