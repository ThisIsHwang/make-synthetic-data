from __future__ import annotations

import argparse
import inspect
import logging
from pathlib import Path

from .cli import (
    DataCollatorCausalLM,
    _apply_runtime_compat_overrides,
    _build_training_arguments,
    _ensure_gemma3_tokenizer_attrs,
    _freeze_embeddings,
    _freeze_vision_encoder,
    _is_gemma3_model_name,
    _load_model,
    _load_tokenizer,
    _log_training_sanity,
    _resolve_fsdp_layer_cls_to_wrap,
    _save_processor_artifacts,
    _setup_logging,
    _validate_launch,
)
from .config import SFTConfig, compute_gradient_accumulation_steps, dump_config, load_config
from .data import build_datasets


logger = logging.getLogger(__name__)


def _import_sft_trainer():
    try:
        from trl import SFTTrainer
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "TRL SFTTrainer is required for this entrypoint. "
            "Install with: pip install 'trl>=0.10.0,<1.0.0'"
        ) from exc
    return SFTTrainer


def _build_sft_trainer(
    sft_trainer_cls,
    model,
    args,
    train_ds,
    eval_ds,
    tokenizer,
    collator,
):
    trainer_params = set(inspect.signature(sft_trainer_cls.__init__).parameters)
    kwargs: dict[str, object] = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "data_collator": collator,
    }
    if "tokenizer" in trainer_params:
        kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_params:
        kwargs["processing_class"] = tokenizer
    if "dataset_kwargs" in trainer_params:
        kwargs["dataset_kwargs"] = {"skip_prepare_dataset": True}

    supported_kwargs = {k: v for k, v in kwargs.items() if k in trainer_params}
    dropped_kwargs = sorted(set(kwargs) - set(supported_kwargs))
    if dropped_kwargs:
        logger.info("Skipping unsupported SFTTrainer kwargs: %s", ", ".join(dropped_kwargs))
    return sft_trainer_cls(**supported_kwargs)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gemma 3 27B IT SFT (TRL SFTTrainer)")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser


def run(cfg: SFTConfig) -> None:
    sft_trainer_cls = _import_sft_trainer()

    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _apply_runtime_compat_overrides(cfg)
    _validate_launch(cfg)

    tokenizer = _load_tokenizer(cfg)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.warning(
            "Tokenizer has no pad_token; falling back to eos_token as pad_token. "
            "This may reduce EOS supervision quality."
        )
    if _is_gemma3_model_name(cfg.model.name_or_path) and not hasattr(tokenizer, "image_token_id"):
        if _ensure_gemma3_tokenizer_attrs(tokenizer):
            logger.warning(
                "Gemma3 tokenizer loaded without image_token_id, but it was recovered from special tokens. "
                "Serving env should use transformers>=4.50.0."
            )
        else:
            logger.warning(
                "Gemma3 tokenizer loaded without image_token_id and recovery failed. "
                "Upgrade transformers in this environment (recommended >=4.50.0)."
            )

    train_ds, eval_ds = build_datasets(cfg, tokenizer)
    if len(train_ds) == 0:
        raise ValueError("Prepared train dataset is empty. Check data fields/prompt length/max_seq_length.")
    logger.info("Dataset ready train=%s eval=%s", len(train_ds), len(eval_ds) if eval_ds is not None else 0)
    _log_training_sanity(cfg, len(train_ds))

    model = _load_model(cfg)
    if cfg.model.freeze_vision_encoder:
        _freeze_vision_encoder(model)
    model.config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id
    use_fsdp_activation_ckpt = bool(cfg.train.fsdp and cfg.train.fsdp_activation_checkpointing)
    use_hf_gradient_ckpt = bool(cfg.train.gradient_checkpointing)
    if use_hf_gradient_ckpt and use_fsdp_activation_ckpt:
        logger.warning(
            "Both train.gradient_checkpointing and train.fsdp_activation_checkpointing are true. "
            "Disabling HF gradient_checkpointing and using FSDP activation_checkpointing only."
        )
        use_hf_gradient_ckpt = False
    if use_hf_gradient_ckpt:
        model.gradient_checkpointing_enable()
    if use_hf_gradient_ckpt or use_fsdp_activation_ckpt:
        model.config.use_cache = False

    _freeze_embeddings(
        model,
        freeze_input_embeddings=cfg.model.freeze_input_embeddings,
        freeze_output_embeddings=cfg.model.freeze_output_embeddings,
    )

    resolved_layer = _resolve_fsdp_layer_cls_to_wrap(model, cfg)
    if cfg.train.fsdp and "auto_wrap" in str(cfg.train.fsdp) and not resolved_layer:
        logger.warning(
            "Could not resolve FSDP transformer layer class; disabling auto_wrap and keeping full_shard."
        )
        cfg.train.fsdp = " ".join(part for part in str(cfg.train.fsdp).split() if part != "auto_wrap")
    elif resolved_layer:
        cfg.train.fsdp_transformer_layer_cls_to_wrap = resolved_layer

    grad_accum = compute_gradient_accumulation_steps(cfg)
    logger.info(
        "Effective batch config global_batch_size=%s per_device_train_batch_size=%s gradient_accumulation_steps=%s",
        cfg.train.global_batch_size,
        cfg.train.per_device_train_batch_size,
        grad_accum,
    )

    args = _build_training_arguments(
        cfg,
        grad_accum,
        has_eval=eval_ds is not None,
        hf_gradient_checkpointing=use_hf_gradient_ckpt,
    )
    dump_config(cfg, output_dir / "resolved_config.yaml")
    collator = DataCollatorCausalLM(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        force_token_type_ids=_is_gemma3_model_name(cfg.model.name_or_path),
    )
    trainer = _build_sft_trainer(
        sft_trainer_cls=sft_trainer_cls,
        model=model,
        args=args,
        train_ds=train_ds,
        eval_ds=eval_ds,
        tokenizer=tokenizer,
        collator=collator,
    )

    trainer.train(resume_from_checkpoint=cfg.train.resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(cfg.train.output_dir)
    try:
        _save_processor_artifacts(cfg, tokenizer)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(
            "Failed to save processor artifacts to output_dir=%s: %s",
            cfg.train.output_dir,
            exc,
        )


def main(argv: list[str] | None = None) -> int:
    _setup_logging()
    logger.info("gemma27b_sft trl cli path=%s", Path(__file__).resolve())
    parser = _build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
