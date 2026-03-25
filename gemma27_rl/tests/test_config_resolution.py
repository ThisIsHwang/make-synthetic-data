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


def test_reset_best_eval_on_resume_loads_from_yaml(tmp_path: Path) -> None:
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
                "  esa:",
                "    enabled: false",
                "logging:",
                "  reset_best_eval_on_resume: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)
    assert cfg.logging.reset_best_eval_on_resume is True


def test_distributed_timeout_sec_loads_from_yaml(tmp_path: Path) -> None:
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
                "  esa:",
                "    enabled: false",
                "misc:",
                "  distributed_timeout_sec: 5400",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)
    assert cfg.misc.distributed_timeout_sec == 5400


def test_distributed_timeout_sec_must_be_positive_when_set(tmp_path: Path) -> None:
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
                "  esa:",
                "    enabled: false",
                "misc:",
                "  distributed_timeout_sec: 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="misc.distributed_timeout_sec must be > 0 when set"):
        _ = load_config(cfg_path)


def test_unknown_top_level_config_key_raises(tmp_path: Path) -> None:
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
                "loggging:",
                "  output_dir: /tmp/out",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"Unknown config key\(s\): loggging"):
        _ = load_config(cfg_path)


def test_unknown_nested_config_key_raises_with_full_path(tmp_path: Path) -> None:
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  hf_dataset_name: dummy/dataset",
                "  hf_eval_splt: validation",
                "reward:",
                "  xcomet:",
                "    enabled: false",
                "  mqm:",
                "    enabled: false",
                "rl:",
                "  deepspeeed_zero_stage: 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"Unknown config key\(s\): data\.hf_eval_splt, rl\.deepspeeed_zero_stage"):
        _ = load_config(cfg_path)


def test_freeform_dict_keys_remain_allowed(tmp_path: Path) -> None:
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  hf_dataset_name: dummy/dataset",
                "generation:",
                "  chat_template_kwargs:",
                "    enable_thinking: true",
                "    custom_server_flag: yes",
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
    assert cfg.generation.chat_template_kwargs == {
        "enable_thinking": True,
        "custom_server_flag": True,
    }


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


def test_data_dir_and_bucketing_fields_load_from_yaml(tmp_path: Path) -> None:
    data_dir = tmp_path / "datasets"
    data_dir.mkdir()
    cache_dir = tmp_path / "cache"
    preprocess_cache_dir = tmp_path / "preprocess_cache"

    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  train_dir: ./datasets",
                "  train_glob: '*.jsonl'",
                "  eval_dir: ./datasets",
                "  eval_glob: '**/*.jsonl'",
                "  split_cache_dir: ./cache",
                "  preprocess_cache_dir: ./preprocess_cache",
                "  split_cache_enabled: false",
                "  prompt_length_batch_size: 32",
                "  batching_strategy: direction_domain_length",
                "  domain_field_path: metadata.teacher_path",
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

    assert cfg.data.train_dir == str(data_dir)
    assert cfg.data.eval_dir == str(data_dir)
    assert cfg.data.split_cache_dir == str(cache_dir)
    assert cfg.data.preprocess_cache_dir == str(preprocess_cache_dir)
    assert cfg.data.split_cache_enabled is False
    assert cfg.data.prompt_length_batch_size == 32
    assert cfg.data.train_glob == "*.jsonl"
    assert cfg.data.eval_glob == "**/*.jsonl"
    assert cfg.data.batching_strategy == "direction_domain_length"
    assert cfg.data.domain_field_path == "metadata.teacher_path"


def test_train_dir_must_exist_when_configured(tmp_path: Path) -> None:
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  train_dir: ./missing",
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

    with pytest.raises(FileNotFoundError, match="data.train_dir not found"):
        _ = load_config(cfg_path)


def test_batching_strategy_validation_rejects_unknown_value(tmp_path: Path) -> None:
    data_dir = tmp_path / "datasets"
    data_dir.mkdir()

    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  train_dir: ./datasets",
                "  batching_strategy: unknown_mode",
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

    with pytest.raises(ValueError, match="data.batching_strategy must be direction\\|direction_domain_length"):
        _ = load_config(cfg_path)


def test_split_cache_dir_must_not_point_to_file(tmp_path: Path) -> None:
    cache_file = tmp_path / "cache_file"
    cache_file.write_text("x", encoding="utf-8")
    data_dir = tmp_path / "datasets"
    data_dir.mkdir()

    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  train_dir: ./datasets",
                "  split_cache_dir: ./cache_file",
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

    with pytest.raises(ValueError, match="data.split_cache_dir must be a directory path"):
        _ = load_config(cfg_path)


def test_preprocess_cache_dir_must_not_point_to_file(tmp_path: Path) -> None:
    cache_file = tmp_path / "preprocess_cache_file"
    cache_file.write_text("x", encoding="utf-8")
    data_dir = tmp_path / "datasets"
    data_dir.mkdir()

    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  train_dir: ./datasets",
                "  preprocess_cache_dir: ./preprocess_cache_file",
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

    with pytest.raises(ValueError, match="data.preprocess_cache_dir must be a directory path"):
        _ = load_config(cfg_path)


def test_prompt_length_batch_size_must_be_positive(tmp_path: Path) -> None:
    data_dir = tmp_path / "datasets"
    data_dir.mkdir()

    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  train_dir: ./datasets",
                "  prompt_length_batch_size: 0",
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

    with pytest.raises(ValueError, match="data.prompt_length_batch_size must be > 0"):
        _ = load_config(cfg_path)


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


def test_mqm_failure_policy_must_be_supported(tmp_path: Path) -> None:
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
                "    failure_policy: nope",
                "  esa:",
                "    enabled: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reward.mqm.failure_policy must be neutral_zero\\|worst_score\\|raise"):
        _ = load_config(cfg_path)


def test_mqm_failure_seq_penalty_must_be_finite(tmp_path: Path) -> None:
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
                "    failure_seq_penalty: .inf",
                "  esa:",
                "    enabled: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reward.mqm.failure_seq_penalty must be finite"):
        _ = load_config(cfg_path)


def test_mqm_token_type_weight_keys_must_be_non_empty(tmp_path: Path) -> None:
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  hf_dataset_name: dummy/dataset",
                "reward:",
                "  mqm_token_type_weights:",
                '    "": 1.5',
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

    with pytest.raises(ValueError, match="reward.mqm_token_type_weights keys must be non-empty strings"):
        _ = load_config(cfg_path)


def test_mqm_token_type_weight_values_must_be_non_negative(tmp_path: Path) -> None:
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  hf_dataset_name: dummy/dataset",
                "reward:",
                "  mqm_token_type_weights:",
                "    accuracy/omission: -1.0",
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

    with pytest.raises(ValueError, match=r"reward\.mqm_token_type_weights\['accuracy/omission'\] must be >= 0"):
        _ = load_config(cfg_path)


def test_mqm_unanchored_seq_scale_must_be_finite(tmp_path: Path) -> None:
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  hf_dataset_name: dummy/dataset",
                "reward:",
                "  mqm_unanchored_seq_scale: .inf",
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

    with pytest.raises(ValueError, match="reward.mqm_unanchored_seq_scale must be finite"):
        _ = load_config(cfg_path)


def test_mqm_unanchored_allowed_types_must_not_be_empty(tmp_path: Path) -> None:
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  hf_dataset_name: dummy/dataset",
                "reward:",
                "  mqm_unanchored_allowed_types:",
                '    - ""',
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

    with pytest.raises(ValueError, match="reward.mqm_unanchored_allowed_types\\[0\\] must be a non-empty string"):
        _ = load_config(cfg_path)


def test_load_config_allows_group_rank_only_reward(tmp_path: Path) -> None:
    cfg_path = tmp_path / "group_rank_only.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  hf_dataset_name: dummy/dataset",
                "reward:",
                "  metricx:",
                "    enabled: false",
                "  xcomet:",
                "    enabled: false",
                "  mqm:",
                "    enabled: false",
                "  esa:",
                "    enabled: false",
                "  group_rank:",
                "    enabled: true",
                "    base_url: http://localhost:8000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)
    assert cfg.reward.group_rank.enabled is True
    assert cfg.reward.group_rank.base_url == "http://localhost:8000"


def test_group_rank_candidate_max_must_cover_num_samples_per_prompt(tmp_path: Path) -> None:
    cfg_path = tmp_path / "group_rank_invalid.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  hf_dataset_name: dummy/dataset",
                "generation:",
                "  num_samples_per_prompt: 5",
                "reward:",
                "  metricx:",
                "    enabled: false",
                "  xcomet:",
                "    enabled: false",
                "  mqm:",
                "    enabled: false",
                "  esa:",
                "    enabled: false",
                "  group_rank:",
                "    enabled: true",
                "    base_url: http://localhost:8000",
                "    candidate_max: 4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="generation.num_samples_per_prompt must be <= reward.group_rank.candidate_max"):
        _ = load_config(cfg_path)


def test_eval_use_esa_requires_esa_base_url(tmp_path: Path) -> None:
    cfg_path = tmp_path / "eval_use_esa.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data:",
                "  hf_dataset_name: dummy/dataset",
                "reward:",
                "  metricx:",
                "    enabled: false",
                "  xcomet:",
                "    enabled: false",
                "  mqm:",
                "    enabled: false",
                "  group_rank:",
                "    enabled: true",
                "    base_url: http://localhost:8000",
                "eval:",
                "  use_esa: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reward.esa.base_url must be set when reward.esa.enabled=true or eval.use_esa=true"):
        _ = load_config(cfg_path)
