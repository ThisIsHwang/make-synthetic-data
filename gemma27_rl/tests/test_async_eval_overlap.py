from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from gemma27_rl import trainer as trainer_mod
from gemma27_rl.config import RLPostTrainConfig
from gemma27_rl.rl_types import Example, Rollout


class _DummyTokenizer:
    pad_token = "<pad>"
    eos_token = "</s>"
    all_special_tokens = ["<pad>", "</s>"]
    additional_special_tokens: list[str] = []
    special_tokens_map = {"pad_token": "<pad>", "eos_token": "</s>"}

    def save_pretrained(self, path: str | Path) -> None:
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "tokenizer.json").write_text("{}\n", encoding="utf-8")


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def save_pretrained(self, path: str | Path) -> None:
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "weights.bin").write_bytes(b"ok")


class _ManualFuture:
    def __init__(self) -> None:
        self._done = False
        self._result = None

    def done(self) -> bool:
        return bool(self._done)

    def set_result(self, result) -> None:  # type: ignore[no-untyped-def]
        self._result = result
        self._done = True

    def result(self):  # type: ignore[no-untyped-def]
        if not self._done:
            raise AssertionError("result() was requested before the async eval future completed.")
        return self._result


class _FakeAsyncExecutor:
    def __init__(self, *, events: list[str], reports: list[dict[str, float]]) -> None:
        self._events = events
        self._reports = reports
        self._submit_count = 0
        self.latest_future: _ManualFuture | None = None

    def submit(self, fn, *args, **kwargs):  # type: ignore[no-untyped-def]
        del fn, args, kwargs
        self._submit_count += 1
        self._events.append(f"async_eval_submit_{self._submit_count}")
        future = _ManualFuture()
        if self._submit_count > 1:
            future.set_result(dict(self._reports[min(self._submit_count - 1, len(self._reports) - 1)]))
        self.latest_future = future
        return future

    def shutdown(self, wait: bool = True) -> None:
        return


class _DummyAsyncJsonWriter:
    def append_json(self, path, payload):  # type: ignore[no-untyped-def]
        return

    def append_rollouts(self, path, *, update_idx, rollouts, advantages, reward_stats):  # type: ignore[no-untyped-def]
        return

    def append_eval_rows(self, path, *, update_idx, eval_rows):  # type: ignore[no-untyped-def]
        return

    def flush(self) -> None:
        return

    def close(self) -> None:
        return


def _example(example_id: str) -> Example:
    return Example(
        example_id=example_id,
        src_text=f"src::{example_id}",
        src_lang="English",
        tgt_lang="Korean",
        src_lang_code="en",
        tgt_lang_code="ko",
        ref_text=f"ref::{example_id}",
    )


def _rollout(example_id: str) -> Rollout:
    return Rollout(
        example_id=example_id,
        prompt_text=f"prompt::{example_id}",
        prompt_input_ids=[1, 2],
        completion_text=f"completion::{example_id}",
        completion_token_ids=[3],
        old_logprobs=[-0.1],
        ref_logprobs=None,
        token_char_offsets=[(0, 1)],
        src_text=f"src::{example_id}",
        src_lang="English",
        tgt_lang="Korean",
        src_lang_code="en",
        tgt_lang_code="ko",
        ref_text=f"ref::{example_id}",
        raw_completion_token_ids=[3],
        completion_raw_text=f"completion::{example_id}",
        completion_clean_text=f"completion::{example_id}",
    )


def test_async_eval_scoring_overlaps_next_train_rollout(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    cfg = RLPostTrainConfig()
    cfg.logging.output_dir = str(tmp_path / "outputs")
    cfg.logging.tensorboard_enabled = False
    cfg.logging.wandb_enabled = False
    cfg.logging.save_every_n_updates = 0
    cfg.logging.save_rollouts = False
    cfg.logging.save_eval_outputs = False
    cfg.eval.run_before_train = False
    cfg.eval.eval_every_n_updates = 1
    cfg.rl.updates = 2
    cfg.rl.batch_size = 1
    cfg.rl.ppo_epochs = 1
    cfg.rl.kl_coef = 0.0
    cfg.model.use_reference_model = False
    cfg.reward.metricx.enabled = False
    cfg.reward.xcomet.enabled = False
    cfg.reward.mqm.enabled = True
    cfg.reward.mqm.base_url = "http://localhost:8000/v1"
    cfg.reward.esa.enabled = False
    cfg.reward.w_mqm_seq = 1.0
    cfg.reward.mqm_seq_scale = 1.0

    train_examples = [_example("train-0"), _example("train-1")]
    eval_examples = [_example("eval-0")]
    events: list[str] = []
    async_reports = [
        {
            "metricx_score_mean": 0.0,
            "metricx_score_std": 0.0,
            "metricx_reward_mean": 0.0,
            "metricx_reward_std": 0.0,
            "xcomet_score_mean": 0.0,
            "xcomet_score_std": 0.0,
            "mqm_score_mean": 1.0,
            "mqm_score_std": 0.0,
            "mqm_skipped_count": 0.0,
            "esa_score_mean": 0.0,
            "esa_score_std": 0.0,
            "esa_skipped_count": 0.0,
            "avg_span_count": 0.0,
            "severity_counts": {},
            "avg_completion_len": 1.0,
            "num_eval_rollouts": 1,
        },
        {
            "metricx_score_mean": 0.0,
            "metricx_score_std": 0.0,
            "metricx_reward_mean": 0.0,
            "metricx_reward_std": 0.0,
            "xcomet_score_mean": 0.0,
            "xcomet_score_std": 0.0,
            "mqm_score_mean": 0.5,
            "mqm_score_std": 0.0,
            "mqm_skipped_count": 0.0,
            "esa_score_mean": 0.0,
            "esa_score_std": 0.0,
            "esa_skipped_count": 0.0,
            "avg_span_count": 0.0,
            "severity_counts": {},
            "avg_completion_len": 1.0,
            "num_eval_rollouts": 1,
        },
    ]
    fake_executor = _FakeAsyncExecutor(events=events, reports=async_reports)
    update_calls = {"n": 0}

    monkeypatch.setattr(trainer_mod, "set_seed", lambda seed: None)
    monkeypatch.setattr(trainer_mod, "_configure_nccl_heartbeat_timeout", lambda cfg: None)
    monkeypatch.setattr(trainer_mod, "_configure_cuda_allocator", lambda: None)
    monkeypatch.setattr(trainer_mod, "resolve_huggingface_token", lambda explicit_token=None, token_env_name=None: None)
    monkeypatch.setattr(trainer_mod, "configure_huggingface_cache", lambda cache_dir, token=None: None)
    monkeypatch.setattr(trainer_mod, "_apply_aux_worker_defaults", lambda cfg: None)
    monkeypatch.setattr(trainer_mod, "_assign_disjoint_gpu_devices", lambda cfg: None)
    monkeypatch.setattr(trainer_mod, "resolve_device", lambda device: "cpu")
    monkeypatch.setattr(trainer_mod, "_configure_policy_train_memory", lambda model: None)
    monkeypatch.setattr(trainer_mod, "_unwrap_for_generation", lambda model: model)
    monkeypatch.setattr(trainer_mod, "_dist_barrier", lambda: None)
    monkeypatch.setattr(trainer_mod, "_distributed_rank", lambda: 0)
    monkeypatch.setattr(trainer_mod, "_distributed_world_size", lambda: 1)
    monkeypatch.setattr(trainer_mod, "_is_rank0", lambda: True)
    monkeypatch.setattr(trainer_mod, "_AsyncJsonlWriter", _DummyAsyncJsonWriter)
    monkeypatch.setattr(
        trainer_mod,
        "ThreadPoolExecutor",
        lambda max_workers=1, thread_name_prefix="": fake_executor,
    )
    monkeypatch.setattr(
        trainer_mod,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: _DummyTokenizer()),
    )
    monkeypatch.setattr(trainer_mod, "_load_policy_model", lambda cfg, device=None, model_name_or_path=None: _DummyModel())

    def _fake_load_examples(data_cfg, *, split: str, limit=None):  # type: ignore[no-untyped-def]
        return list(train_examples if split == "train" else eval_examples)

    monkeypatch.setattr(trainer_mod, "load_examples", _fake_load_examples)

    def _fake_prepare_training_batch_rollouts_and_advantages(**kwargs):  # type: ignore[no-untyped-def]
        update_idx = int(kwargs["update_idx"])
        future = fake_executor.latest_future
        future_done = bool(future.done()) if future is not None else False
        events.append(f"train_prep_{update_idx}_future_done_{future_done}")
        if update_idx == 2 and future is not None and (not future.done()):
            future.set_result(dict(async_reports[0]))
            events.append("async_eval_completed_during_train_prep_2")
        reward_stats = {
            "metricx_score_mean": 0.0,
            "metricx_score_std": 0.0,
            "xcomet_score_mean": 0.0,
            "xcomet_score_std": 0.0,
            "mqm_score_mean": 0.0,
            "mqm_score_std": 0.0,
            "esa_score_mean": 0.0,
            "esa_score_std": 0.0,
            "token_rewards_non_zero_ratio": 0.0,
        }
        adv_stats = {
            "raw_mean": 0.0,
            "raw_std": 1.0,
            "norm_mean": 0.0,
            "norm_std": 1.0,
        }
        return [_rollout(f"train-u{update_idx}")], [[0.1]], reward_stats, adv_stats

    monkeypatch.setattr(
        trainer_mod,
        "_prepare_training_batch_rollouts_and_advantages",
        _fake_prepare_training_batch_rollouts_and_advantages,
    )

    def _fake_prepare_eval_rollouts(**kwargs):  # type: ignore[no-untyped-def]
        events.append("prepare_async_eval_rollouts")
        return [_rollout("eval-rollout")]

    monkeypatch.setattr(trainer_mod, "prepare_eval_rollouts", _fake_prepare_eval_rollouts)
    monkeypatch.setattr(
        trainer_mod,
        "build_eval_report_from_rollouts",
        lambda **kwargs: dict(async_reports[0]),
    )

    def _fake_update_policy(**kwargs):  # type: ignore[no-untyped-def]
        update_calls["n"] += 1
        events.append(f"update_policy_{update_calls['n']}")
        return SimpleNamespace(
            policy_loss=0.1,
            approx_kl=0.0,
            clip_fraction=0.0,
            entropy=0.0,
            kl_to_reference=0.0,
            token_count=1,
        )

    monkeypatch.setattr(trainer_mod, "update_policy", _fake_update_policy)

    def _fake_save_checkpoint_to_dir(*, ckpt_dir, model, tokenizer, optimizer, trainer_state=None):  # type: ignore[no-untyped-def]
        ckpt_path = Path(ckpt_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        (ckpt_path / "checkpoint.txt").write_text("ok\n", encoding="utf-8")
        events.append(f"save_ckpt_{ckpt_path.name}")
        return ckpt_path

    monkeypatch.setattr(trainer_mod, "_save_checkpoint_to_dir", _fake_save_checkpoint_to_dir)

    artifacts = trainer_mod.run_toy_rl(cfg)

    assert "final_model_dir" in artifacts
    assert "best_model_dir" in artifacts
    assert "prepare_async_eval_rollouts" in events
    assert "async_eval_submit_1" in events
    assert "train_prep_2_future_done_False" in events
    assert "async_eval_completed_during_train_prep_2" in events
    assert "save_ckpt_best" in events
    assert events.index("train_prep_2_future_done_False") < events.index("save_ckpt_best")
    assert events.index("save_ckpt_best") < events.index("update_policy_2")


def test_should_async_eval_scoring_only_allows_api_only_eval() -> None:
    cfg = RLPostTrainConfig()
    cfg.reward.metricx.enabled = False
    cfg.reward.xcomet.enabled = False
    cfg.reward.mqm.enabled = True
    cfg.reward.esa.enabled = False

    assert trainer_mod._should_async_eval_scoring(
        cfg=cfg,
        distributed_eval_shard=False,
        metricx_scorer=None,
        xcomet_scorer=None,
        mqm_scorer=object(),  # type: ignore[arg-type]
        esa_scorer=None,
    )

    cfg.reward.metricx.enabled = True
    assert not trainer_mod._should_async_eval_scoring(
        cfg=cfg,
        distributed_eval_shard=False,
        metricx_scorer=object(),  # type: ignore[arg-type]
        xcomet_scorer=None,
        mqm_scorer=object(),  # type: ignore[arg-type]
        esa_scorer=None,
    )

    cfg.reward.metricx.enabled = False
    assert not trainer_mod._should_async_eval_scoring(
        cfg=cfg,
        distributed_eval_shard=True,
        metricx_scorer=None,
        xcomet_scorer=None,
        mqm_scorer=object(),  # type: ignore[arg-type]
        esa_scorer=None,
    )
