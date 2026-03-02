"""TRL GRPOTrainer integration — the core training orchestration.

=== GRPO Training Loop (What TRL Does Internally) ===

Group Relative Policy Optimization (GRPO) is an RL algorithm for LLM training.
Unlike standard PPO which requires a separate value function (critic) network,
GRPO computes advantages by comparing completions within a group.

Each training step in TRL's GRPOTrainer does:

  1. SAMPLE PROMPTS: Pick a mini-batch of prompts from the dataset.

  2. GENERATE ROLLOUTS: For each prompt, generate ``num_generations`` completions
     using the current policy (the LLM being trained).  This is the "rollout"
     phase — like letting the agent act in RL.

     Example (num_generations=4 for one prompt "Translate Hello"):
       Completion 0: "안녕하세요"
       Completion 1: "안녕"
       Completion 2: "안녕하세요!"
       Completion 3: "헬로"

  3. SCORE WITH REWARD MODEL: Pass all completions to the reward function(s).
     Each function returns a scalar reward per completion.

     Example scores from MetricX (lower=better, inverted to higher=better):
       Completion 0: reward = 4.8
       Completion 1: reward = 4.2
       Completion 2: reward = 4.7
       Completion 3: reward = 1.5

  4. COMPUTE ADVANTAGE (group-relative normalization):
     For scale_rewards="group":
       mean = (4.8 + 4.2 + 4.7 + 1.5) / 4 = 3.8
       std  = 1.33
       advantages = [(4.8-3.8)/1.33, (4.2-3.8)/1.33, (4.7-3.8)/1.33, (1.5-3.8)/1.33]
                  = [0.75, 0.30, 0.68, -1.73]

     This is the key insight of GRPO: we don't need a critic network because
     we use the group statistics as the baseline.

  5. COMPUTE CLIPPED SURROGATE LOSS (PPO-style):
     For each token in each completion:
       ratio = π_θ(token) / π_old(token)    # probability ratio new/old policy
       L = min(ratio * advantage, clip(ratio, 1-ε, 1+ε) * advantage)

     The clipping prevents the policy from changing too drastically in one step.

  6. ADD KL PENALTY (if beta > 0):
     L_total = L - β * KL(π_θ || π_ref)

     This keeps the trained policy close to the original (reference) model,
     preventing reward hacking (generating text that fools the reward model
     but is actually nonsensical).

  7. BACKPROPAGATE AND UPDATE:
     Standard gradient descent with the RL loss + optional router aux loss (MoE).

=== What This Module Does ===

This module is the "glue" between our config system and TRL's GRPOTrainer:

  - ``build_grpo_config()``: Maps our DistributedRLConfig → TRL's GRPOConfig
  - ``build_reward_funcs()``: Creates reward functions for rank 0, dummies for others
  - ``run_training()``: Loads model, tokenizer, datasets, reward funcs → trains

=== Reward Function Architecture ===

Only rank 0 actually runs reward models (MetricX/XComet).  Other ranks
return dummy zeros.  TRL handles broadcasting the real scores from rank 0
to all other ranks internally.

Why rank 0 only?
  - Reward models need dedicated GPUs (GPU 6, 7) separate from training GPUs
  - The training process's CUDA_VISIBLE_DEVICES is restricted to GPUs 0-5
  - Reward subprocesses bypass this restriction (see subprocess_client.py)
  - Running reward models on every rank would be wasteful and require N copies

Ref: TRL GRPOTrainer — https://huggingface.co/docs/trl/grpo_trainer
Ref: GRPO paper — https://arxiv.org/abs/2402.03300 (DeepSeek-R1)
Ref: GSPO — https://huggingface.co/docs/trl/gspo_token
Ref: Original training loop — qwen3.5-35b-a3b/qwen35_moe_rl/trainer.py
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

from .config import DistributedRLConfig, load_config
from .data import load_dataset_for_trl
from .distributed import (
    build_deepspeed_config,
    is_main_process,
    setup_nccl_environment,
)
from .rewards.base import make_trl_reward_func
from .rewards.metricx import MetricXReward
from .rewards.xcomet import XCometReward

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HuggingFace environment setup
# ---------------------------------------------------------------------------

def _setup_hf_environment(cfg: DistributedRLConfig) -> None:
    """Configure HuggingFace cache and authentication before heavy imports.

    Must be called BEFORE importing transformers or loading any models, because
    HuggingFace reads these env vars at import time / first model download.

    Sets:
      HF_HOME: Root cache directory (e.g. /media/sdd3 for a large SSD)
      HF_HUB_CACHE: Subdirectory for model files
      HF_TOKEN: Authentication token for gated models (e.g. Llama, Gemma)
    """
    if cfg.misc.huggingface_cache_dir:
        os.environ.setdefault("HF_HOME", cfg.misc.huggingface_cache_dir)
        os.environ.setdefault("HF_HUB_CACHE", os.path.join(cfg.misc.huggingface_cache_dir, "hub"))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(cfg.misc.huggingface_cache_dir, "hub"))

    # Token can be set directly in config or read from an environment variable.
    token = cfg.misc.huggingface_token
    if not token and cfg.misc.huggingface_token_env:
        token = os.environ.get(cfg.misc.huggingface_token_env)
    if token:
        os.environ.setdefault("HF_TOKEN", token)


# ---------------------------------------------------------------------------
# GRPOConfig construction
# ---------------------------------------------------------------------------

def build_grpo_config(cfg: DistributedRLConfig) -> Any:
    """Convert our DistributedRLConfig into TRL's GRPOConfig.

    TRL's GRPOConfig extends HuggingFace TrainingArguments with GRPO-specific
    fields (loss_type, beta, epsilon, num_generations, etc.).  This function
    maps our more readable YAML config to TRL's expected parameter names.

    Key mappings:
      Our config                    → TRL's GRPOConfig
      ─────────────────────────────────────────────────────
      training.lr                   → learning_rate
      training.loss_type            → loss_type
      training.beta                 → beta (KL coefficient)
      training.epsilon              → epsilon (PPO clip range)
      training.scale_rewards        → scale_rewards
      generation.num_generations    → num_generations
      generation.max_new_tokens     → max_completion_length
      generation.temperature        → temperature
      distributed.deepspeed         → deepspeed (JSON path or dict)
      model.gradient_checkpointing  → gradient_checkpointing

    Special case — GSPO:
      When loss_type="gspo", we use the standard "grpo" loss_type in GRPOConfig
      but import from ``trl.experimental.gspo_token`` which is a modified
      GRPOTrainer that supports token-level importance sampling via
      ``importance_sampling_level="sequence_token"``.
      Ref: https://huggingface.co/docs/trl/gspo_token

    Returns:
        A TRL GRPOConfig instance ready to be passed to GRPOTrainer.
    """
    from trl import GRPOConfig

    use_bf16 = cfg.misc.dtype in ("bfloat16", "bf16")
    use_fp16 = cfg.misc.dtype in ("float16", "fp16")

    # --- DeepSpeed config ---
    # Option 1: User provides a JSON file path (e.g. configs/deepspeed_zero2.json)
    # Option 2: Auto-generate from distributed config settings
    # Option 3: None (no DeepSpeed, standard DDP or single GPU)
    ds_config: str | dict[str, Any] | None = None
    if cfg.distributed.deepspeed:
        # Use the explicit JSON file path.
        ds_config = cfg.distributed.deepspeed
    elif cfg.distributed.zero_stage > 0:
        # Auto-generate a DeepSpeed config dict.
        ds_config = build_deepspeed_config(
            zero_stage=cfg.distributed.zero_stage,
            offload_optimizer=cfg.distributed.offload_optimizer,
            offload_param=cfg.distributed.offload_param,
            max_grad_norm=cfg.training.max_grad_norm,
            use_bf16=use_bf16,
        )

    grpo_kwargs: dict[str, Any] = {
        "output_dir": cfg.misc.output_dir,

        # --- GRPO algorithm parameters ---
        # For GSPO, we pass "grpo" as loss_type because the experimental
        # GSPO GRPOTrainer class still expects "grpo" as the base loss type.
        # The GSPO behavior is activated by importance_sampling_level.
        "loss_type": cfg.training.loss_type if cfg.training.loss_type != "gspo" else "grpo",
        "beta": cfg.training.beta,           # KL penalty coefficient
        "epsilon": cfg.training.epsilon,     # PPO clipping range
        # scale_rewards: "group" normalizes rewards within each prompt's
        # generation group.  "false" (the string) means no normalization.
        "scale_rewards": cfg.training.scale_rewards if cfg.training.scale_rewards != "false" else False,

        # --- Generation parameters (for rollouts) ---
        "num_generations": cfg.generation.num_generations,
        "max_completion_length": cfg.generation.max_new_tokens,
        "temperature": cfg.generation.temperature,
        "top_p": cfg.generation.top_p,
        "top_k": cfg.generation.top_k,
        "repetition_penalty": cfg.generation.repetition_penalty,

        # --- Standard training parameters ---
        "learning_rate": cfg.training.lr,
        "weight_decay": cfg.training.weight_decay,
        "per_device_train_batch_size": cfg.training.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
        "max_grad_norm": cfg.training.max_grad_norm,
        "num_train_epochs": cfg.training.num_train_epochs,
        "max_steps": cfg.training.max_steps,
        "warmup_ratio": cfg.training.warmup_ratio,
        "warmup_steps": cfg.training.warmup_steps,
        "save_steps": cfg.training.save_steps,
        "eval_steps": cfg.training.eval_steps,
        "logging_steps": cfg.training.logging_steps,
        "log_completions": cfg.training.log_completions,

        # --- Precision ---
        "bf16": use_bf16,
        "fp16": use_fp16,

        # --- DeepSpeed ---
        "deepspeed": ds_config,

        # --- Misc ---
        "seed": cfg.misc.seed,
        "report_to": cfg.misc.report_to or [],

        # --- Gradient checkpointing ---
        "gradient_checkpointing": cfg.model.gradient_checkpointing,
        "gradient_checkpointing_kwargs": cfg.model.gradient_checkpointing_kwargs,

        # CRITICAL: Do not let TRL remove "unused" columns from the dataset.
        # We need src_text and ref_text to be passed through to reward functions
        # as **kwargs.  By default, HuggingFace Trainer removes columns that
        # aren't model inputs — but TRL needs them for reward computation.
        "remove_unused_columns": False,
    }

    # GSPO: Generalized Sequence Policy Optimization.
    # Uses sequence-token level importance sampling: the probability ratio
    # r(θ) is computed at the sequence level (product of token probabilities),
    # but the loss is computed per-token with the sequence-level ratio.
    # Ref: https://huggingface.co/docs/trl/gspo_token
    if cfg.training.loss_type == "gspo":
        grpo_kwargs["importance_sampling_level"] = "sequence_token"

    # --- Asymmetric clipping (MiniRL, arxiv 2512.01374 Section 3.2) ---
    # Paper recommends ε_high=0.27, ε_low=0.2 instead of symmetric ε=0.2.
    # TRL >= 0.29.0 supports these as separate GRPOConfig parameters.
    if cfg.stability.epsilon_high > 0:
        grpo_kwargs["epsilon_high"] = cfg.stability.epsilon_high
    if cfg.stability.epsilon_low > 0:
        grpo_kwargs["epsilon_low"] = cfg.stability.epsilon_low

    return GRPOConfig(**grpo_kwargs)


# ---------------------------------------------------------------------------
# Reward function construction
# ---------------------------------------------------------------------------

def build_reward_funcs(cfg: DistributedRLConfig) -> list[Callable[..., list[float]]]:
    """Build TRL-compatible reward functions from config.

    === Rank-Aware Reward Architecture ===

    Only rank 0 creates actual reward model instances.  Why?

    1. Reward models (MetricX: ~5GB, XComet: ~4GB) need dedicated GPUs
    2. The training GPUs (0-5) are fully occupied by the policy model
    3. Reward models run on GPUs 6-7, which are only accessible from
       the master node via subprocess (CUDA_VISIBLE_DEVICES restriction)
    4. Running reward models on every rank would waste GPU memory

    So:
      - Rank 0: Creates MetricXReward/XCometReward → wraps with make_trl_reward_func()
      - Ranks 1-N: Use _dummy_reward_func that returns [0.0, 0.0, ...]

    TRL's GRPOTrainer internally broadcasts the real reward scores from rank 0
    to all other ranks, so they all end up with the same rewards for the
    advantage computation.

    Returns:
        A list of reward functions, one per enabled reward model.
        TRL calls each function and sums their outputs per completion.
    """
    reward_funcs: list[Callable[..., list[float]]] = []
    rank0 = is_main_process()

    # --- MetricX reward ---
    if cfg.reward.metricx.enabled:
        if rank0:
            # Create the actual MetricX scorer.
            # If python_executable is set in config, this spawns a subprocess
            # on the dedicated GPU.  Otherwise, loads the model in-process.
            metricx = MetricXReward(cfg.reward.metricx)
            # Wrap the RewardModel as a TRL-compatible function.
            # The wrapper extracts src_text/ref_text from **kwargs,
            # calls metricx.score(), and applies weight * (offset - raw_score).
            func = make_trl_reward_func(
                reward_model=metricx,
                src_column="src_text",
                ref_column="ref_text",
                weight=cfg.reward.w_metricx,
                offset=cfg.reward.metricx.offset,  # MetricX: lower=better → invert
                clip_value=cfg.stability.reward_clip_value,
            )
        else:
            # Non-rank-0 processes return zeros.  The real scores will be
            # broadcast from rank 0 by TRL's internal logic.
            func = _dummy_reward_func
        reward_funcs.append(func)

    # --- XComet reward ---
    if cfg.reward.xcomet.enabled:
        if rank0:
            xcomet = XCometReward(cfg.reward.xcomet)
            func = make_trl_reward_func(
                reward_model=xcomet,
                src_column="src_text",
                ref_column="ref_text",
                weight=cfg.reward.w_xcomet,
                offset=0.0,  # XComet: higher=better, no inversion needed
                clip_value=cfg.stability.reward_clip_value,
            )
        else:
            func = _dummy_reward_func
        reward_funcs.append(func)

    if not reward_funcs:
        raise ValueError("No reward functions configured. Enable at least one reward model.")

    return reward_funcs


def _dummy_reward_func(completions: list[str], **kwargs: Any) -> list[float]:
    """Placeholder reward function for non-rank-0 processes.

    Returns all zeros.  TRL will replace these with the real scores
    broadcast from rank 0.
    """
    return [0.0] * len(completions)


# ---------------------------------------------------------------------------
# GRPOTrainer class selection
# ---------------------------------------------------------------------------

def _get_grpo_trainer_class(loss_type: str) -> type:
    """Return the appropriate GRPOTrainer class based on loss type.

    For standard loss types (grpo, dapo, dr_grpo, sapo), TRL's built-in
    GRPOTrainer handles everything via the ``loss_type`` parameter.

    For GSPO, we need the experimental GRPOTrainer from
    ``trl.experimental.gspo_token`` which extends the standard one with
    sequence-token importance sampling.

    Ref: https://huggingface.co/docs/trl/gspo_token
    """
    if loss_type == "gspo":
        try:
            from trl.experimental.gspo_token import GRPOTrainer

            logger.info("Using GSPO-token GRPOTrainer from trl.experimental")
            return GRPOTrainer
        except ImportError:
            logger.warning(
                "trl.experimental.gspo_token not available. "
                "Falling back to standard GRPOTrainer. "
                "Upgrade TRL to >=0.29.0 for GSPO support."
            )

    from trl import GRPOTrainer

    return GRPOTrainer


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def run_training(cfg: DistributedRLConfig) -> None:
    """Load model, tokenizer, datasets, reward functions, and start GRPO training.

    This is the main function called by ``cli.py``.  It orchestrates everything:

      1. Build TRL GRPOConfig from our config
      2. Select the right GRPOTrainer class (standard or GSPO)
      3. Load the policy model from HuggingFace (or local path)
      4. Configure MoE-specific settings if applicable
      5. Load the tokenizer
      6. Prepare train/eval datasets in TRL format
      7. Build rank-aware reward functions
      8. Create and start the GRPOTrainer
      9. Save the final model (rank 0 only)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # --- TRL version check for stability features ---
    try:
        from trl import __version__ as _trl_ver
        _major, _minor = (int(x) for x in _trl_ver.split(".")[:2])
        if _major == 0 and _minor < 29:
            logger.warning(
                "TRL %s detected. Upgrade to >= 0.29.0 for: "
                "(1) no length normalization (MiniRL recommendation), "
                "(2) asymmetric epsilon_high/epsilon_low support.",
                _trl_ver,
            )
    except Exception:
        pass

    # --- Step 1: Build TRL GRPOConfig ---
    grpo_config = build_grpo_config(cfg)

    # --- Step 2: Select trainer class ---
    GRPOTrainer = _get_grpo_trainer_class(cfg.training.loss_type)

    # --- Step 3: Load the policy model ---
    # This is the LLM being fine-tuned.  It's loaded with the specified
    # precision (bf16/fp16) and attention implementation.
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": cfg.model.trust_remote_code,
        "torch_dtype": "auto",  # Auto-select dtype from model config
    }
    if cfg.model.attn_implementation and cfg.model.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = cfg.model.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.policy_name_or_path,
        **model_kwargs,
    )

    # --- Step 4: MoE-specific configuration ---
    # For Mixture-of-Experts models like Qwen3.5-35B-A3B:
    #
    # output_router_logits=True:
    #   Tells the model to output the router's logit distribution over experts.
    #   This is needed to compute the auxiliary load-balancing loss that
    #   prevents "expert collapse" (all tokens routed to the same few experts).
    #
    # router_aux_loss_coef:
    #   Weight for the auxiliary router loss in the total training loss:
    #   total_loss = RL_loss + router_aux_loss_coef * aux_loss
    #   Typical value: 0.001 (small enough not to dominate RL signal).
    if cfg.training.output_router_logits and hasattr(model.config, "output_router_logits"):
        model.config.output_router_logits = True
        logger.info("Enabled output_router_logits for MoE model")

    if cfg.training.router_aux_loss_coef > 0 and hasattr(model.config, "router_aux_loss_coef"):
        model.config.router_aux_loss_coef = cfg.training.router_aux_loss_coef
        logger.info("Set router_aux_loss_coef=%f", cfg.training.router_aux_loss_coef)

    # --- Step 5: Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.policy_name_or_path,
        use_fast=cfg.model.use_fast_tokenizer,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    # Ensure pad_token is set.  Many models (GPT-style) don't have one,
    # which causes errors during batch padding.  Using eos_token is the
    # standard workaround.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Step 6: Prepare datasets ---
    # Converts raw data to TRL format: {prompt: [...], src_text: ..., ref_text: ...}
    train_dataset = load_dataset_for_trl(cfg.data, cfg.prompt, split="train")
    eval_dataset = None
    has_eval = cfg.data.eval_file or cfg.data.hf_eval_split
    if has_eval:
        eval_dataset = load_dataset_for_trl(cfg.data, cfg.prompt, split="eval")

    # --- Step 7: Build reward functions ---
    # Rank 0 gets real reward models, other ranks get dummies.
    reward_funcs = build_reward_funcs(cfg)

    logger.info(
        "Starting GRPOTrainer: model=%s loss_type=%s num_generations=%d "
        "beta=%f epsilon=%f scale_rewards=%s deepspeed=%s",
        cfg.model.policy_name_or_path,
        cfg.training.loss_type,
        cfg.generation.num_generations,
        cfg.training.beta,
        cfg.training.epsilon,
        cfg.training.scale_rewards,
        "enabled" if grpo_config.deepspeed else "disabled",
    )

    # --- Step 8: Create and start GRPOTrainer ---
    # TRL's GRPOTrainer handles the entire training loop:
    #   - Generating rollouts (completions) from the policy model
    #   - Calling reward functions to score completions
    #   - Computing group-relative advantages
    #   - Computing the clipped surrogate loss
    #   - Adding KL penalty (if beta > 0)
    #   - Backpropagation and optimizer step via DeepSpeed
    #   - Checkpointing and logging
    #
    # ``reward_funcs`` is a list of callables.  TRL calls each one with:
    #   func(completions=["text1", ...], prompts=[...], src_text=[...], ...)
    # and sums their outputs.  Each function's weight is baked in.
    #
    # ``processing_class`` is the tokenizer — TRL uses it to tokenize prompts
    # and decode completions.

    # --- Step 7b: Build stability callbacks (MiniRL) ---
    callbacks = []
    if cfg.stability.enable_monitoring:
        from .stability import StabilityMonitorCallback

        callbacks.append(StabilityMonitorCallback(cfg.stability))
        logger.info(
            "StabilityMonitorCallback enabled: entropy_floor=%.2f kl_ceiling=%.2f "
            "halt_on_collapse_steps=%d",
            cfg.stability.entropy_floor,
            cfg.stability.kl_ceiling,
            cfg.stability.halt_on_collapse_steps,
        )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        callbacks=callbacks if callbacks else None,
    )

    # Start the training loop.
    trainer.train()

    # --- Step 9: Save final model ---
    # Only rank 0 saves to avoid multiple processes writing simultaneously.
    # DeepSpeed ZeRO-2: rank 0 has the full model.
    # DeepSpeed ZeRO-3: TRL/DeepSpeed internally gathers all parameter
    # shards to rank 0 before saving.
    if is_main_process():
        final_dir = os.path.join(cfg.misc.output_dir, "final")
        trainer.save_model(final_dir)
        logger.info("Final model saved to %s", final_dir)
