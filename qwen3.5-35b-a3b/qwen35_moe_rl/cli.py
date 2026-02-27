from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from .config import load_config
from .utils import configure_huggingface_cache, resolve_huggingface_token


logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3.5-35B-A3B GSPO/GRPO post-training")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--eval-only", action="store_true", help="Run metric-only evaluation without training")
    parser.add_argument(
        "--local_rank", type=int, default=-1,
        help="Local rank for distributed training (auto-set by torchrun/deepspeed launcher)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _setup_logging()
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.local_rank >= 0:
        os.environ.setdefault("LOCAL_RANK", str(args.local_rank))

    cfg = load_config(args.config)
    hf_token = resolve_huggingface_token(
        explicit_token=cfg.misc.huggingface_token,
        token_env_name=cfg.misc.huggingface_token_env,
    )
    # Set HF cache/token env vars before importing trainer (which imports transformers/datasets).
    configure_huggingface_cache(cfg.misc.huggingface_cache_dir, token=hf_token)

    from .trainer import run_metric_only_eval, run_toy_rl

    if args.eval_only:
        report = run_metric_only_eval(cfg)
        logger.info("evaluation report=%s", report)
    else:
        artifacts = run_toy_rl(cfg)
        logger.info("training artifacts=%s", artifacts)
    return 0


if __name__ == "__main__":
    logger.info("qwen35_moe_rl cli path=%s", Path(__file__).resolve())
    raise SystemExit(main())
