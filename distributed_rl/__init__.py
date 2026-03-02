"""distributed_rl — TRL-based multi-node GRPO training for translation models.

This package wraps HuggingFace TRL's GRPOTrainer with:
  - Pluggable reward models (MetricX, XComet) running on dedicated GPUs
  - DeepSpeed ZeRO-2/3 for multi-node model-parallel training
  - YAML-based config system mapping to TRL's GRPOConfig

Supported loss types (all implemented inside TRL, not custom code):
  - GRPO: Group Relative Policy Optimization
  - DAPO: Dynamic Advantage Policy Optimization
  - DR-GRPO: Doubly-Robust GRPO (variance-reduced)
  - SAPO: Self-Advantage Policy Optimization (no reference model needed)
  - GSPO: via ``trl.experimental.gspo_token`` (TRL >= 0.29.0)

Ref: TRL GRPOTrainer docs — https://huggingface.co/docs/trl/grpo_trainer
Ref: Original codebase — qwen3.5-35b-a3b/qwen35_moe_rl/
"""

__version__ = "0.1.0"
