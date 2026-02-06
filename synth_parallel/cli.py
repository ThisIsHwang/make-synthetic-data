from __future__ import annotations

import argparse
import logging
import sys

from .config import load_config, validate_stage
from .logging_utils import setup_logging
from .pipeline import PipelineRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="synth_parallel", description="Synthetic parallel data pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run pipeline stage")
    run.add_argument("--config", required=True, help="Path to YAML config")
    run.add_argument(
        "--stage",
        required=True,
        choices=[
            "sample_sources",
            "prefilter_score",
            "select_sources",
            "generate_128",
            "score_select_best",
            "score_128_select_best",
            "format_filter",
            "export",
            "all",
        ],
    )
    run.add_argument("--dry-run", action="store_true", help="Run small subset")
    run.add_argument("--resume", action="store_true", help="Resume from progress DB")
    run.add_argument("--overwrite", action="store_true", help="Overwrite stage output")
    run.add_argument("--limit", type=int, default=None, help="Debug limit")
    run.add_argument("--shard-id", type=int, default=None, help="Shard id (0-based)")
    run.add_argument("--num-shards", type=int, default=None, help="Total shards")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command != "run":
        parser.print_help()
        return 2

    cfg = load_config(args.config)
    validate_stage(args.stage)

    log_path = setup_logging(cfg.run.out_dir, cfg.run.log_level)
    logging.getLogger(__name__).info("Logging to %s", log_path)

    runner = PipelineRunner(
        cfg=cfg,
        dry_run=args.dry_run,
        resume=args.resume,
        overwrite=args.overwrite,
        limit=args.limit,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )

    try:
        runner.run(args.stage)
    except Exception:  # pylint: disable=broad-except
        logging.getLogger(__name__).exception("Pipeline execution failed")
        return 1
    finally:
        runner.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
