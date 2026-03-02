"""Command-line interface for distributed RL training.

This is the single entry point that orchestrates the full training pipeline:
  1. Parse CLI arguments (--config, --eval-only, --local_rank)
  2. Load YAML config → DistributedRLConfig dataclass hierarchy
  3. Set up HuggingFace environment (cache dir, auth token)
  4. Configure NCCL for distributed communication (timeout, error handling)
  5. Delegate to ``run_training()`` which creates TRL's GRPOTrainer and starts training

Launch examples:
  # Single GPU toy test
  python3 -m distributed_rl --config configs/train_toy.yaml

  # Multi-GPU with torchrun (6 GPUs for training, GPUs 6-7 reserved for reward models)
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --nproc_per_node=6 \\
      -m distributed_rl --config configs/qwen35_moe_8gpu.yaml

Note on --local_rank:
  ``torchrun`` sets LOCAL_RANK via environment variable (modern approach).
  ``deepspeed`` launcher passes it as ``--local_rank`` CLI arg (legacy approach).
  We support both: if --local_rank is passed, we also set the env var so that
  HuggingFace Accelerate and TRL can find it.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys


def _setup_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.INFO,
        stream=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    _setup_logging()

    parser = argparse.ArgumentParser(
        description="Distributed RL Post-Training (TRL GRPOTrainer + DeepSpeed)",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only")
    # --local_rank is injected by the ``deepspeed`` launcher (not torchrun).
    # torchrun uses the LOCAL_RANK environment variable directly.
    parser.add_argument(
        "--local_rank", type=int, default=-1,
        help="Local rank (auto-set by torchrun/deepspeed launcher)",
    )
    args = parser.parse_args(argv)

    # Bridge legacy --local_rank to env var for HF Accelerate/TRL compatibility.
    if args.local_rank >= 0:
        os.environ.setdefault("LOCAL_RANK", str(args.local_rank))

    # --- Step 1: Load config ---
    # Delayed imports so that heavy libraries (torch, transformers) are only loaded
    # after the environment is properly configured.
    from .config import load_config
    cfg = load_config(args.config)

    # --- Step 2: Configure HuggingFace environment ---
    # Must happen before any HF model downloads (sets HF_HOME, HF_TOKEN, etc.)
    from .trainer import _setup_hf_environment
    _setup_hf_environment(cfg)

    # --- Step 3: Configure NCCL ---
    # Sets timeout and async error handling for multi-node collective operations.
    # Long timeout is important because reward model scoring can take many minutes
    # and other ranks must wait for rank 0 to finish scoring before proceeding.
    from .distributed import setup_nccl_environment
    setup_nccl_environment(cfg.distributed.nccl_timeout_minutes)

    if args.eval_only:
        logging.getLogger(__name__).info("Eval-only mode not yet implemented for TRL trainer.")
        return 1

    # --- Step 4: Start training ---
    # This creates TRL GRPOTrainer and runs the full RL training loop.
    from .trainer import run_training
    run_training(cfg)
    return 0
