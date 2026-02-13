from __future__ import annotations

import argparse
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


def _load_model(cfg: SFTConfig):
    model_kwargs: dict[str, object] = {
        "torch_dtype": torch.bfloat16 if cfg.train.bf16 else torch.float16,
        "trust_remote_code": cfg.model.trust_remote_code,
    }
    if cfg.model.attn_implementation:
        model_kwargs["attn_implementation"] = cfg.model.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name_or_path, **model_kwargs)
    return model


def _build_training_arguments(cfg: SFTConfig, grad_accum: int, has_eval: bool) -> TrainingArguments:
    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return TrainingArguments(
        output_dir=str(output_dir),
        seed=cfg.train.seed,
        num_train_epochs=cfg.train.num_train_epochs,
        max_steps=cfg.train.max_steps,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.train.per_device_eval_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=cfg.train.learning_rate,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        warmup_ratio=cfg.train.warmup_ratio,
        weight_decay=cfg.train.weight_decay,
        logging_steps=cfg.train.logging_steps,
        save_steps=cfg.train.save_steps,
        eval_steps=cfg.train.eval_steps,
        save_total_limit=cfg.train.save_total_limit,
        bf16=cfg.train.bf16,
        tf32=cfg.train.tf32,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        dataloader_num_workers=cfg.train.dataloader_num_workers,
        report_to=cfg.train.report_to,
        remove_unused_columns=False,
        ddp_find_unused_parameters=cfg.train.ddp_find_unused_parameters,
        evaluation_strategy="steps" if has_eval else "no",
        save_strategy="steps",
        logging_strategy="steps",
        optim="adafactor",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gemma 27B IT SFT")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser


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

    trainer = FixedAdafactorTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=collator,
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
