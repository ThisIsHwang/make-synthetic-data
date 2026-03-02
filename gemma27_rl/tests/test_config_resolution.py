from __future__ import annotations

from pathlib import Path

from gemma27_rl.config import load_config


def test_python_executable_resolution_preserves_venv_symlink(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    config_dir = project_root / "configs" / "exp"
    config_dir.mkdir(parents=True)

    fake_base_python = tmp_path / "python3.10"
    fake_base_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

    venv_python = project_root / ".venv_metricx" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(fake_base_python)

    cfg_path = config_dir / "train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  hf_dataset_name: dummy/dataset",
                "reward:",
                "  metricx:",
                "    python_executable: ../../.venv_metricx/bin/python",
                "  xcomet:",
                "    enabled: false",
                "  mqm:",
                "    enabled: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)
    assert cfg.reward.metricx.python_executable == str(venv_python)
