from __future__ import annotations

from pathlib import Path

import pytest

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


def test_remote_worker_paths_preserve_remote_home_expressions(tmp_path: Path) -> None:
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  hf_dataset_name: dummy/dataset",
                "misc:",
                "  aux_worker_host: aux-node-1",
                "  aux_worker_remote_workdir: ~/rl_project",
                "reward:",
                "  metricx:",
                "    python_executable: ~/venv_metricx/bin/python",
                "  xcomet:",
                "    enabled: false",
                "  mqm:",
                "    enabled: false",
                "  esa:",
                "    enabled: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)
    assert cfg.misc.aux_worker_remote_workdir == "~/rl_project"
    assert cfg.reward.metricx.python_executable == "~/venv_metricx/bin/python"


def test_remote_python_executable_resolution_becomes_project_relative(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    config_dir = project_root / "configs" / "exp"
    config_dir.mkdir(parents=True)

    cfg_path = config_dir / "train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  hf_dataset_name: dummy/dataset",
                "misc:",
                "  aux_worker_host: aux-node-1",
                "  aux_worker_remote_workdir: /remote/repo",
                "reward:",
                "  metricx:",
                "    python_executable: ../../.venv_metricx/bin/python",
                "  xcomet:",
                "    enabled: false",
                "  mqm:",
                "    enabled: false",
                "  esa:",
                "    enabled: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)
    assert cfg.reward.metricx.python_executable == ".venv_metricx/bin/python"


def test_keep_last_n_checkpoints_must_be_non_negative(tmp_path: Path) -> None:
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  hf_dataset_name: dummy/dataset",
                "reward:",
                "  xcomet:",
                "    enabled: false",
                "  mqm:",
                "    enabled: false",
                "logging:",
                "  keep_last_n_checkpoints: -1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="logging.keep_last_n_checkpoints must be >= 0"):
        _ = load_config(cfg_path)


def test_disable_reference_model_allows_reference_gpu_settings_without_deepspeed(tmp_path: Path) -> None:
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  hf_dataset_name: dummy/dataset",
                "model:",
                "  use_reference_model: false",
                "  reference_gpu_ids: [0, 1]",
                "reward:",
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
    assert cfg.model.use_reference_model is False
    assert cfg.model.reference_gpu_ids == [0, 1]


def test_colocated_reference_with_lora_allows_overlapping_gpu_ids(tmp_path: Path) -> None:
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  hf_dataset_name: dummy/dataset",
                "model:",
                "  policy_name_or_path: /tmp/base-model",
                "  reference_runtime: colocate",
                "  policy_runtime_mode: colocate",
                "  policy_gpu_ids: [0, 1, 2, 3, 4, 5, 6, 7]",
                "  reference_gpu_ids: [0, 1, 2, 3, 4, 5, 6, 7]",
                "  lora:",
                "    enabled: true",
                "rl:",
                "  backend: deepspeed",
                "reward:",
                "  xcomet:",
                "    enabled: false",
                "  mqm:",
                "    enabled: false",
                "  esa:",
                "    enabled: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)
    assert cfg.model.reference_runtime == "colocate"
    assert cfg.model.lora.enabled is True
    assert cfg.model.policy_gpu_ids == [0, 1, 2, 3, 4, 5, 6, 7]
    assert cfg.model.reference_gpu_ids == [0, 1, 2, 3, 4, 5, 6, 7]


def test_colocated_reference_requires_lora(tmp_path: Path) -> None:
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  hf_dataset_name: dummy/dataset",
                "model:",
                "  reference_runtime: colocate",
                "  policy_runtime_mode: colocate",
                "reward:",
                "  xcomet:",
                "    enabled: false",
                "  mqm:",
                "    enabled: false",
                "  esa:",
                "    enabled: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model.reference_runtime=colocate requires model.lora.enabled=true"):
        _ = load_config(cfg_path)
