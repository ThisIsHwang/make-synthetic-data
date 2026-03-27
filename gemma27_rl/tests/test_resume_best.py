from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch

from gemma27_rl import trainer as trainer_mod
from gemma27_rl.config import RLPostTrainConfig
from gemma27_rl.trainer import (
    _is_deepspeed_checkpoint_dir,
    _prune_old_checkpoints,
    _restore_best_eval_state_for_resume,
    _resolve_run_before_train_eval_update_idx,
    _resolve_resume_checkpoint,
    _save_checkpoint_to_dir,
    _save_trainer_state,
    _should_run_eval_before_train,
    _write_directory_atomically,
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


def test_save_checkpoint_to_dir_preserves_previous_checkpoint_on_write_failure(tmp_path: Path) -> None:
    class _Model:
        def save_pretrained(self, path: Path) -> None:
            path.mkdir(parents=True, exist_ok=True)
            (path / "weights.bin").write_bytes(b"new")

    class _Tokenizer:
        def save_pretrained(self, path: Path) -> None:
            raise RuntimeError("tokenizer failed")

    ckpt_dir = tmp_path / "checkpoint-1"
    ckpt_dir.mkdir()
    (ckpt_dir / "sentinel.txt").write_text("old\n", encoding="utf-8")
    param = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([param], lr=0.1)

    with pytest.raises(RuntimeError, match="tokenizer failed"):
        _save_checkpoint_to_dir(
            ckpt_dir=ckpt_dir,
            model=_Model(),  # type: ignore[arg-type]
            tokenizer=_Tokenizer(),
            optimizer=optimizer,
        )

    assert (ckpt_dir / "sentinel.txt").read_text(encoding="utf-8") == "old\n"
    assert not (ckpt_dir / "weights.bin").exists()


def test_write_directory_atomically_restores_previous_dir_when_promote_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target_dir = tmp_path / "final"
    target_dir.mkdir()
    (target_dir / "sentinel.txt").write_text("old\n", encoding="utf-8")

    original_replace = trainer_mod.os.replace

    def _fake_replace(src, dst):  # type: ignore[no-untyped-def]
        src_path = Path(src)
        dst_path = Path(dst)
        if src_path.name.endswith(".staging") and dst_path == target_dir:
            raise OSError("rename failed")
        return original_replace(src, dst)

    monkeypatch.setattr(trainer_mod.os, "replace", _fake_replace)

    with pytest.raises(RuntimeError, match="Failed to promote staged directory"):
        _write_directory_atomically(
            target_dir,
            lambda staged_dir: (
                staged_dir.mkdir(parents=True, exist_ok=True),
                (staged_dir / "sentinel.txt").write_text("new\n", encoding="utf-8"),
            ),
        )

    assert (target_dir / "sentinel.txt").read_text(encoding="utf-8") == "old\n"
