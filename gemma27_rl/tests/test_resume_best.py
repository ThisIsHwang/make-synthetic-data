from __future__ import annotations

import pytest

pytest.importorskip("torch")

from gemma27_rl.config import RLPostTrainConfig
from gemma27_rl.trainer import (
    _is_deepspeed_checkpoint_dir,
    _prune_old_checkpoints,
    _restore_best_eval_state_for_resume,
    _resolve_run_before_train_eval_update_idx,
    _resolve_resume_checkpoint,
    _save_trainer_state,
    _should_run_eval_before_train,
)


def test_resolve_resume_checkpoint_explicit_best_uses_trainer_state(tmp_path) -> None:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    best_dir = output_dir / "best"
    best_dir.mkdir()
    _save_trainer_state(
        best_dir,
        {
            "update_idx": 12,
            "best_eval_update": 12,
            "best_eval_score": 1.23,
        },
    )

    cfg = RLPostTrainConfig()
    cfg.logging.output_dir = str(output_dir)
    cfg.logging.auto_resume = False
    cfg.logging.resume_from_checkpoint = str(best_dir)

    resume_path, resume_update = _resolve_resume_checkpoint(cfg, output_dir)
    assert resume_path == best_dir
    assert resume_update == 12


def test_is_deepspeed_checkpoint_dir_detects_shard_files(tmp_path) -> None:
    ckpt_dir = tmp_path / "best"
    shard_dir = ckpt_dir / "state"
    shard_dir.mkdir(parents=True)
    (shard_dir / "mp_rank_00_model_states.pt").write_bytes(b"")

    assert _is_deepspeed_checkpoint_dir(ckpt_dir)


def test_is_deepspeed_checkpoint_dir_false_for_model_only_dir(tmp_path) -> None:
    model_dir = tmp_path / "best"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}\n", encoding="utf-8")

    assert not _is_deepspeed_checkpoint_dir(model_dir)


def test_prune_old_checkpoints_keeps_latest_n(tmp_path) -> None:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    (output_dir / "best").mkdir()
    (output_dir / "checkpoint-1").mkdir()
    (output_dir / "checkpoint-2").mkdir()
    (output_dir / "checkpoint-3").mkdir()
    (output_dir / "checkpoint-latest").mkdir()

    removed = _prune_old_checkpoints(output_dir, keep_last_n=2)

    removed_names = {path.name for path in removed}
    assert removed_names == {"checkpoint-1"}
    assert (output_dir / "checkpoint-2").exists()
    assert (output_dir / "checkpoint-3").exists()
    assert (output_dir / "best").exists()
    assert (output_dir / "checkpoint-latest").exists()


def test_restore_best_eval_state_for_resume_uses_saved_and_logged_best(tmp_path) -> None:
    log_path = tmp_path / "train_log.jsonl"
    log_path.write_text(
        "\n".join(
            [
                '{"type":"eval","update":3,"model_select_score":0.4}',
                '{"type":"eval","update":7,"model_select_score":0.8}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    best_score, best_update = _restore_best_eval_state_for_resume(
        resume_state={"best_eval_score": 0.6, "best_eval_update": 5},
        log_path=log_path,
        reset_best_eval_on_resume=False,
    )

    assert best_score == pytest.approx(0.8)
    assert best_update == 7


def test_restore_best_eval_state_for_resume_can_reset_previous_best(tmp_path) -> None:
    log_path = tmp_path / "train_log.jsonl"
    log_path.write_text(
        '{"type":"eval","update":7,"model_select_score":0.8}\n',
        encoding="utf-8",
    )

    best_score, best_update = _restore_best_eval_state_for_resume(
        resume_state={"best_eval_score": 0.9, "best_eval_update": 6},
        log_path=log_path,
        reset_best_eval_on_resume=True,
    )

    assert best_score == float("-inf")
    assert best_update is None


def test_should_run_eval_before_train_for_fresh_run() -> None:
    assert _should_run_eval_before_train(
        eval_enabled=True,
        has_eval_examples=True,
        start_update=1,
        reset_best_eval_on_resume=False,
    )


def test_should_run_eval_before_train_for_resume_only_when_resetting_best() -> None:
    assert not _should_run_eval_before_train(
        eval_enabled=True,
        has_eval_examples=True,
        start_update=8,
        reset_best_eval_on_resume=False,
    )
    assert _should_run_eval_before_train(
        eval_enabled=True,
        has_eval_examples=True,
        start_update=8,
        reset_best_eval_on_resume=True,
    )


def test_resolve_run_before_train_eval_update_idx_uses_resume_update_when_resetting_best() -> None:
    assert _resolve_run_before_train_eval_update_idx(
        is_resuming=False,
        resume_update_idx=0,
        reset_best_eval_on_resume=False,
    ) == 0
    assert _resolve_run_before_train_eval_update_idx(
        is_resuming=True,
        resume_update_idx=12,
        reset_best_eval_on_resume=False,
    ) == 0
    assert _resolve_run_before_train_eval_update_idx(
        is_resuming=True,
        resume_update_idx=12,
        reset_best_eval_on_resume=True,
    ) == 12
