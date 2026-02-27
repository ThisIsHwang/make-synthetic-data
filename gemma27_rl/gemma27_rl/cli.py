from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
import threading

from .config import load_config
from .utils import configure_huggingface_cache, resolve_huggingface_token


logger = logging.getLogger(__name__)


def _setup_logging(log_file: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def _install_exception_logging() -> None:
    def _log_unhandled_exception(exc_type, exc_value, exc_tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            # Keep default interrupt behavior without noisy traceback spam.
            return
        logger.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_tb))

    sys.excepthook = _log_unhandled_exception

    def _thread_exception_handler(args: threading.ExceptHookArgs) -> None:
        if args.exc_type and issubclass(args.exc_type, KeyboardInterrupt):
            return
        logger.critical(
            "Unhandled thread exception in %s",
            getattr(args.thread, "name", "unknown"),
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    threading.excepthook = _thread_exception_handler


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gemma 27B GRPO post-training")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--eval-only", action="store_true", help="Run metric-only evaluation without training")
    # Distributed launchers (deepspeed/torchrun) may inject this argument.
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    _setup_logging()
    parser = _build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    rank_text = os.environ.get("RANK")
    rank = int(rank_text) if rank_text and rank_text.isdigit() else 0
    output_dir = Path(cfg.logging.output_dir)
    # Keep a stable single-file log for the main process and per-rank logs for distributed workers.
    # This avoids all ranks racing on one file while still providing a top-level `log.txt`.
    rank_log_file = output_dir / ("log.txt" if rank == 0 else f"log_rank{rank}.txt")
    _setup_logging(rank_log_file)
    _install_exception_logging()
    logger.info("logging to %s", rank_log_file)

    try:
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
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except Exception:
        logger.exception("Fatal error in gemma27_rl main loop")
        return 1


if __name__ == "__main__":
    logger.info("gemma27_rl cli path=%s", Path(__file__).resolve())
    raise SystemExit(main())
