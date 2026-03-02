"""DeepSpeed config generation and NCCL distributed training utilities.

=== How Distributed Training Works in This System ===

When you run:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --nproc_per_node=6 -m distributed_rl ...

``torchrun`` spawns 6 processes, each assigned a different GPU:
  - Process 0: RANK=0, LOCAL_RANK=0 → GPU 0
  - Process 1: RANK=1, LOCAL_RANK=1 → GPU 1
  - ...
  - Process 5: RANK=5, LOCAL_RANK=5 → GPU 5

For multi-node (2 nodes):
  Node 0: RANK=0-5 (6 GPUs for training)
  Node 1: RANK=6-13 (8 GPUs for training)
  Total: 14 ranks

Each process holds a copy of the model (or a shard in DeepSpeed ZeRO),
runs forward/backward independently, and synchronizes gradients via
NCCL all-reduce operations.

=== DeepSpeed ZeRO ===

DeepSpeed ZeRO partitions model state (optimizer, gradients, parameters)
across ranks to reduce per-GPU memory:

  Standard DDP:    Each GPU has full copy of model + optimizer + gradients
  ZeRO Stage 1:    Optimizer states partitioned (Adam has 2× model size)
  ZeRO Stage 2:    + Gradients partitioned
  ZeRO Stage 3:    + Parameters partitioned (gathered on-demand for compute)

The ``build_deepspeed_config()`` function auto-generates the JSON config
that DeepSpeed expects, with performance optimizations like:
  - ``overlap_comm=True``: overlap communication with computation
  - ``reduce_scatter=True``: fuse gradient reduction and scattering
  - ``contiguous_gradients=True``: pack gradients contiguously for faster collectives

=== NCCL Timeout ===

NCCL (NVIDIA Collective Communications Library) handles GPU-to-GPU communication.
The default timeout is 30 minutes, but our system needs much more because:
  1. Rank 0 scores all completions with reward models (can take 10-30 min)
  2. Other ranks call a collective (e.g. broadcast) and WAIT for rank 0
  3. If the wait exceeds the NCCL timeout → deadlock → crash

So we set the timeout to 60-120 minutes in the config.

Ref: DeepSpeed ZeRO tutorial — https://www.deepspeed.ai/tutorials/zero/
Ref: NCCL env vars — https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
"""

from __future__ import annotations

import os
from typing import Any


# ---------------------------------------------------------------------------
# Rank utilities
# ---------------------------------------------------------------------------
# These read environment variables set by torchrun/DeepSpeed launchers.
# They are used to determine which process is rank 0 (for reward scoring)
# and to configure device placement.
# ---------------------------------------------------------------------------

def get_rank() -> int:
    """Global rank of this process across all nodes (0-indexed)."""
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    """Rank within this node (0-indexed).  Determines which GPU to use."""
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size() -> int:
    """Total number of training processes across all nodes."""
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process() -> bool:
    """True only for the global rank 0 process.

    Rank 0 is special in our system:
      - It's the only rank that runs actual reward models (MetricX, XComet)
      - It saves model checkpoints
      - It logs training metrics

    Other ranks use dummy reward functions and wait for rank 0's results
    to be broadcast by TRL internally.
    """
    return get_rank() == 0


# ---------------------------------------------------------------------------
# NCCL environment setup
# ---------------------------------------------------------------------------

def setup_nccl_environment(nccl_timeout_minutes: int = 60) -> None:
    """Configure NCCL for reliable multi-node communication.

    Sets two critical environment variables:
      NCCL_TIMEOUT: How long (seconds) to wait for a collective operation.
                    Must cover the worst-case reward scoring time.
      NCCL_ASYNC_ERROR_HANDLING: If "1", failed ranks are detected and
                    reported instead of hanging indefinitely.
    """
    timeout_sec = str(nccl_timeout_minutes * 60)
    os.environ.setdefault("NCCL_TIMEOUT", timeout_sec)
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")


# ---------------------------------------------------------------------------
# DeepSpeed config auto-generation
# ---------------------------------------------------------------------------

def build_deepspeed_config(
    *,
    zero_stage: int = 2,
    offload_optimizer: bool = False,
    offload_param: bool = False,
    max_grad_norm: float = 1.0,
    use_bf16: bool = True,
) -> dict[str, Any]:
    """Auto-generate a DeepSpeed JSON config dict from training parameters.

    This is used when the user doesn't provide an explicit DeepSpeed JSON file
    in the config (``distributed.deepspeed: null``).  It creates a config
    equivalent to the static JSON files in ``configs/deepspeed_zero*.json``
    but with values derived from the training config.

    Args:
        zero_stage: ZeRO optimization stage (0, 1, 2, or 3).
        offload_optimizer: If True, offload optimizer states to CPU RAM.
            Saves ~2× model size of GPU memory but slows training ~20%.
        offload_param: If True, offload model parameters to CPU (Stage 3 only).
            Enables training models larger than total GPU memory.
        max_grad_norm: Gradient clipping threshold.
        use_bf16: Use bfloat16 mixed precision (True) or float16 (False).

    Returns:
        A dict that can be passed as ``deepspeed=`` to HuggingFace TrainingArguments.
        Keys like "auto" tell DeepSpeed to read the value from TrainingArguments.
    """

    # --- ZeRO optimization config ---
    zero_opt: dict[str, Any] = {
        "stage": zero_stage,
        # allgather_partitions: gather parameter partitions in one collective
        # (vs. multiple smaller ones).  More bandwidth-efficient.
        "allgather_partitions": True,
        # Buffer size for allgather operations (500MB).
        # Larger = fewer operations but more memory.
        "allgather_bucket_size": 5e8,
        # reduce_scatter: fuse gradient reduce + scatter into one NCCL op.
        # Halves the communication volume vs. separate all-reduce + scatter.
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        # overlap_comm: overlap gradient communication with backward pass.
        # Hides communication latency behind compute (pipelining).
        "overlap_comm": True,
        # contiguous_gradients: allocate gradient memory contiguously.
        # Avoids memory fragmentation and enables faster NCCL collectives.
        "contiguous_gradients": True,
    }

    # Stage 3 specific: parameter gathering and persistence settings.
    if zero_stage >= 3:
        # How many parameters to prefetch for the next layer while current
        # layer is computing.  "auto" lets DeepSpeed decide.
        zero_opt["stage3_prefetch_bucket_size"] = "auto"
        # Parameters smaller than this threshold are kept on all GPUs
        # (not partitioned) to avoid gather overhead for tiny tensors.
        zero_opt["stage3_param_persistence_threshold"] = "auto"
        # When saving the model, gather all parameter shards to rank 0
        # and save as full-precision (fp16/bf16) weights.
        zero_opt["stage3_gather_16bit_weights_on_model_save"] = True

    # CPU offloading reduces GPU memory at the cost of slower training.
    if offload_optimizer:
        zero_opt["offload_optimizer"] = {"device": "cpu", "pin_memory": True}

    if offload_param and zero_stage >= 3:
        zero_opt["offload_param"] = {"device": "cpu", "pin_memory": True}

    # --- Top-level DeepSpeed config ---
    ds_config: dict[str, Any] = {
        # "auto" means DeepSpeed reads these from HuggingFace TrainingArguments.
        # This avoids specifying the same values in two places.
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        # Gradient clipping: same as max_grad_norm in training config.
        "gradient_clipping": max_grad_norm,
        "zero_optimization": zero_opt,
        # Allow optimizers that DeepSpeed hasn't officially tested.
        "zero_allow_untested_optimizer": True,
        # Suppress DeepSpeed's per-step throughput logs (very noisy).
        "steps_per_print": 9999999,
    }

    # --- Mixed precision ---
    if use_bf16:
        # bfloat16: preferred for modern GPUs (A100, H100).
        # Wider dynamic range than fp16, no loss scaling needed.
        ds_config["bf16"] = {"enabled": True}
    else:
        # float16: requires loss scaling to avoid underflow during training.
        # initial_scale_power=16 means starting scale = 2^16 = 65536.
        # DeepSpeed dynamically adjusts the scale during training.
        ds_config["fp16"] = {
            "enabled": True,
            "loss_scale": 0,  # 0 = dynamic loss scaling
            "initial_scale_power": 16,
        }

    return ds_config
