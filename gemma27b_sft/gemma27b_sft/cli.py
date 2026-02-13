from __future__ import annotations

import argparse
import importlib.util
import inspect
import logging
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Adafactor, Trainer, TrainingArguments

from .config import SFTConfig, compute_gradient_accumulation_steps, dump_config, load_config
from .data import CompletionDataCollator, build_datasets


logger = logging.getLogger(__name__)


class FixedAdafactorTrainer(Trainer):
    def create_optimizer(self):
        if self.optimizer is None:
            params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = Adafactor(
                params,
                lr=self.args.learning_rate,
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
            )
        return self.optimizer


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _freeze_embeddings(model: AutoModelForCausalLM, freeze_output_embeddings: bool) -> tuple[int, int]:
    frozen_params = 0
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is not None:
        for param in input_embeddings.parameters():
            param.requires_grad = False
            frozen_params += param.numel()

    output_embeddings = model.get_output_embeddings()
    if freeze_output_embeddings and output_embeddings is not None and output_embeddings is not input_embeddings:
        for param in output_embeddings.parameters():
            param.requires_grad = False
            frozen_params += param.numel()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Parameter summary total=%s trainable=%s frozen=%s",
        total,
        trainable,
        frozen_params,
    )
    return trainable, frozen_params


def _is_flash_attn_available() -> bool:
    return torch.cuda.is_available() and importlib.util.find_spec("flash_attn") is not None


def _resolve_attn_implementation(cfg: SFTConfig) -> str | None:
    requested = (cfg.model.attn_implementation or "auto").strip().lower()
    if requested in {"", "auto"}:
        if _is_flash_attn_available():
            logger.info("Attention implementation resolved to flash_attention_2 (auto mode).")
            return "flash_attention_2"
        logger.info("flash_attn unavailable; using sdpa attention implementation.")
        return "sdpa"
    return cfg.model.attn_implementation


def _load_model(cfg: SFTConfig):
    model_kwargs: dict[str, object] = {
        "torch_dtype": torch.bfloat16 if cfg.train.bf16 else torch.float16,
        "trust_remote_code": cfg.model.trust_remote_code,
    }
    resolved_attn = _resolve_attn_implementation(cfg)
    if resolved_attn:
        model_kwargs["attn_implementation"] = resolved_attn
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg.model.name_or_path, **model_kwargs)
        return model
    except Exception as exc:
        # Allow training to continue on environments without flash-attn.
        err = str(exc)
        if resolved_attn == "flash_attention_2" and ("flash_attn" in err or "FlashAttention2" in err):
            logger.warning(
                "flash_attention_2 is unavailable at runtime (%s). Falling back to sdpa.",
                err,
            )
            model_kwargs["attn_implementation"] = "sdpa"
            model = AutoModelForCausalLM.from_pretrained(cfg.model.name_or_path, **model_kwargs)
            return model
        raise


def _build_training_arguments(cfg: SFTConfig, grad_accum: int, has_eval: bool) -> TrainingArguments:
    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ta_params = set(inspect.signature(TrainingArguments.__init__).parameters)

    kwargs: dict[str, object] = {
        "output_dir": str(output_dir),
        "seed": cfg.train.seed,
        "num_train_epochs": cfg.train.num_train_epochs,
        "max_steps": cfg.train.max_steps,
        "per_device_train_batch_size": cfg.train.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.train.per_device_eval_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": cfg.train.learning_rate,
        "lr_scheduler_type": cfg.train.lr_scheduler_type,
        "warmup_ratio": cfg.train.warmup_ratio,
        "weight_decay": cfg.train.weight_decay,
        "logging_steps": cfg.train.logging_steps,
        "save_steps": cfg.train.save_steps,
        "eval_steps": cfg.train.eval_steps,
        "save_total_limit": cfg.train.save_total_limit,
        "bf16": cfg.train.bf16,
        "tf32": cfg.train.tf32,
        "gradient_checkpointing": cfg.train.gradient_checkpointing,
        "dataloader_num_workers": cfg.train.dataloader_num_workers,
        "report_to": cfg.train.report_to,
        "remove_unused_columns": False,
        "ddp_find_unused_parameters": cfg.train.ddp_find_unused_parameters,
    }
    eval_mode = "steps" if has_eval else "no"
    if "evaluation_strategy" in ta_params:
        kwargs["evaluation_strategy"] = eval_mode
    elif "eval_strategy" in ta_params:
        kwargs["eval_strategy"] = eval_mode
    elif "do_eval" in ta_params:
        kwargs["do_eval"] = has_eval

    if "save_strategy" in ta_params:
        kwargs["save_strategy"] = "steps"
    if "logging_strategy" in ta_params:
        kwargs["logging_strategy"] = "steps"
    if "optim" in ta_params:
        kwargs["optim"] = "adafactor"
    elif "adafactor" in ta_params:
        kwargs["adafactor"] = True

    supported_kwargs = {k: v for k, v in kwargs.items() if k in ta_params}
    dropped_kwargs = sorted(set(kwargs) - set(supported_kwargs))
    if dropped_kwargs:
        logger.info("Skipping unsupported TrainingArguments kwargs: %s", ", ".join(dropped_kwargs))

    return TrainingArguments(**supported_kwargs)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gemma 27B IT SFT")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser


def _build_trainer(
    model: AutoModelForCausalLM,
    args: TrainingArguments,
    train_ds,
    eval_ds,
    tokenizer,
    collator: CompletionDataCollator,
) -> FixedAdafactorTrainer:
    trainer_params = set(inspect.signature(FixedAdafactorTrainer.__init__).parameters)
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

    supported_kwargs = {k: v for k, v in kwargs.items() if k in trainer_params}
    dropped_kwargs = sorted(set(kwargs) - set(supported_kwargs))
    if dropped_kwargs:
        logger.info("Skipping unsupported Trainer kwargs: %s", ", ".join(dropped_kwargs))
    return FixedAdafactorTrainer(**supported_kwargs)


def run(cfg: SFTConfig) -> None:
    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, output_dir / "resolved_config.yaml")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds, eval_ds = build_datasets(cfg, tokenizer)
    logger.info("Dataset ready train=%s eval=%s", len(train_ds), len(eval_ds) if eval_ds is not None else 0)

    model = _load_model(cfg)
    model.config.pad_token_id = tokenizer.pad_token_id
    if cfg.train.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    _freeze_embeddings(model, cfg.model.freeze_output_embeddings)

    grad_accum = compute_gradient_accumulation_steps(cfg)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    logger.info(
        "Effective batch config global_batch_size=%s per_device_train_batch_size=%s world_size=%s gradient_accumulation_steps=%s",
        cfg.train.global_batch_size,
        cfg.train.per_device_train_batch_size,
        world_size,
        grad_accum,
    )

    args = _build_training_arguments(cfg, grad_accum, has_eval=eval_ds is not None)
    collator = CompletionDataCollator(tokenizer=tokenizer)

    trainer = _build_trainer(
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


def main(argv: list[str] | None = None) -> int:
    _setup_logging()
    parser = _build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
