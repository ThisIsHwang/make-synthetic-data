"""YAML config system → Python dataclass hierarchy → TRL GRPOConfig.

=== Overview ===

This module defines the complete configuration for GRPO training.  A single
YAML file controls everything: model choice, data source, reward models,
training hyperparameters, DeepSpeed settings, etc.

The config loading pipeline:
  1. ``load_config(path)`` reads a YAML file into a raw dict
  2. ``_coerce_dataclass()`` recursively converts the dict into a nested
     dataclass hierarchy (``DistributedRLConfig``)
  3. ``_resolve_*`` functions convert relative paths to absolute paths
     (relative to the YAML file's directory)
  4. ``_validate_config()`` checks for invalid combinations

Later, ``trainer.py:build_grpo_config()`` maps this into TRL's ``GRPOConfig``
which is what the actual GRPOTrainer accepts.

=== Dataclass Hierarchy ===

  DistributedRLConfig
  ├── ModelConfig        — which LLM to train (path, attention impl, grad ckpt)
  ├── DataConfig         — data source (JSONL file or HuggingFace dataset)
  ├── PromptConfig       — translation prompt template
  ├── GenerationConfig   — how the model generates rollouts (temp, top_p, etc.)
  ├── RewardConfig       — reward model weights and sub-configs
  │   ├── MetricXConfig  — MetricX scorer settings (GPU, model, subprocess)
  │   └── XCometConfig   — XComet scorer settings (GPU, model, subprocess)
  ├── TrainingConfig     — RL algorithm settings (loss type, beta, epsilon, LR)
  ├── DistributedConfig  — DeepSpeed / NCCL settings
  └── MiscConfig         — seed, dtype, output directory, HF cache

=== Key RL Hyperparameters (in TrainingConfig) ===

  loss_type:
    - "grpo": Group Relative Policy Optimization — the baseline algorithm.
      Computes advantage by comparing each completion against others in the
      same group (``num_generations`` completions per prompt).
    - "dapo": Dynamic Advantage — dynamically adjusts the clipping range.
    - "dr_grpo": Doubly-Robust — variance-reduced variant.
    - "sapo": Self-Advantage — uses model's own previous output as baseline,
      no reference model needed (beta is ignored).
    - "gspo": Generalized SPO — experimental in TRL, uses sequence-token
      level importance sampling.

  beta:
    KL divergence coefficient.  Penalizes the trained policy for diverging
    too far from the reference (initial) model.  In the GRPO loss:
      L = E[ min(r(θ) * A, clip(r(θ), 1±ε) * A) ] - β * KL(π_θ || π_ref)
    Setting beta=0 disables KL penalty (no reference model loaded → saves GPU).

  epsilon:
    PPO/GRPO clipping range.  The probability ratio r(θ) = π_θ/π_old is
    clipped to [1-ε, 1+ε] to prevent destructively large policy updates.

  scale_rewards:
    - "group": normalize rewards within each prompt's generation group
      (mean=0, std=1 per group).  This is the standard GRPO approach.
    - "batch": normalize across the entire batch.
    - "false": no normalization (raw reward values used as advantages).

  num_generations (in GenerationConfig):
    Number of completions sampled per prompt.  GRPO requires multiple
    completions to compute group-relative advantage.  Typical: 4-8.
    More generations → better advantage estimation but higher GPU cost.

Ref: Pattern adapted from qwen3.5-35b-a3b/qwen35_moe_rl/config.py
Ref: TRL GRPOConfig — https://huggingface.co/docs/trl/grpo_trainer#trl.GRPOConfig
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .prompting import DEFAULT_TRANSLATION_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclass hierarchy — each @dataclass maps to a top-level YAML key
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Which LLM to fine-tune and how to load it."""

    # HuggingFace model ID (e.g. "Qwen/Qwen3.5-35B-A3B") or local path.
    # This model is the "policy" — the model being trained via GRPO.
    policy_name_or_path: str = "Qwen/Qwen3.5-35B-A3B"

    # Whether to trust custom code in the model repo (required for Qwen).
    trust_remote_code: bool = True

    # Attention implementation:
    #   "auto" — let HuggingFace choose (usually FlashAttention2 if available)
    #   "flash_attention_2" — force FlashAttention2 (fastest, requires GPU)
    #   "sdpa" — PyTorch native scaled dot-product attention
    #   "eager" — standard attention (slowest, for debugging)
    attn_implementation: str = "auto"

    use_fast_tokenizer: bool = True

    # Gradient checkpointing trades compute for memory: instead of storing all
    # intermediate activations for backprop, recompute them on the fly.
    # Essential for large models (27B+) that wouldn't fit in GPU memory otherwise.
    # Ref: https://huggingface.co/docs/transformers/perf_train_gpu_one#gradient-checkpointing
    gradient_checkpointing: bool = True

    # use_reentrant=False is required for compatibility with FSDP and
    # DeepSpeed ZeRO-3.  The "reentrant" variant has subtle bugs with
    # unused parameters in MoE models.
    gradient_checkpointing_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"use_reentrant": False}
    )


@dataclass
class DataConfig:
    """Where to load training/eval data from and how to parse it.

    Supports two data sources (mutually exclusive):
      1. Local files: ``train_file`` (JSONL/JSON/Parquet)
      2. HuggingFace Hub: ``hf_dataset_name`` + ``hf_dataset_config_name``

    Field mappings (``*_field``) define which JSON keys contain the source text,
    reference translation, language codes, etc.  This makes the code agnostic
    to the specific dataset schema — just change the field names in YAML.
    """

    # Option 1: Local file paths (JSONL, JSON, or Parquet)
    train_file: str | None = None
    eval_file: str | None = None

    # Option 2: HuggingFace datasets (e.g. "google/wmt24pp")
    hf_dataset_name: str | None = None
    hf_dataset_config_name: str | None = None  # e.g. "en-ko_KR" for WMT
    hf_train_split: str = "train"
    hf_eval_split: str | None = None  # If None, reuses train split for eval
    hf_revision: str | None = None
    hf_streaming: bool = False  # Stream large datasets without downloading

    # Max number of examples to use (None = use all)
    limit: int | None = None
    eval_limit: int | None = None

    # --- Field mappings: which JSON/column names map to which roles ---
    # These tell data.py how to extract the right fields from any dataset.
    id_field: str = "id"
    src_text_field: str = "src_text"        # Source text to translate
    src_lang_field: str = "src_lang"        # Source language name
    tgt_lang_field: str = "tgt_lang"        # Target language name
    src_lang_code_field: str = "src_lang_code"  # ISO code (e.g. "en")
    tgt_lang_code_field: str = "tgt_lang_code"  # ISO code (e.g. "ko")
    ref_text_field: str = "ref_text"        # Reference translation (for ref-based reward)
    is_bad_source_field: str = "is_bad_source"  # Flag to skip bad source sentences
    skip_bad_source: bool = False

    # Default language values when the dataset doesn't have per-row language info
    default_src_lang: str = "English"
    default_tgt_lang: str = "Korean"
    default_src_lang_code: str = "en"
    default_tgt_lang_code: str = "ko"


@dataclass
class PromptConfig:
    """Translation prompt template with ``{text}``, ``{source_lang}`` etc. placeholders."""
    template: str = DEFAULT_TRANSLATION_PROMPT_TEMPLATE


@dataclass
class GenerationConfig:
    """Controls how the policy model generates rollout completions.

    In GRPO, each training step:
      1. Sample a batch of prompts
      2. Generate ``num_generations`` completions per prompt (the "rollouts")
      3. Score each completion with the reward model
      4. Compute group-relative advantage (compare within the group)
      5. Update the policy with the clipped surrogate objective

    These parameters control step 2 — how diverse and long the completions are.
    """

    # Maximum tokens to generate per completion.
    # For translation, 256 is typically enough for most sentence pairs.
    max_new_tokens: int = 256

    # Temperature controls randomness.  Higher = more diverse completions.
    # GRPO needs diversity to compute meaningful advantages:
    #   - Too low (< 0.3): all completions are nearly identical → zero advantage
    #   - Too high (> 1.0): completions are incoherent → noisy rewards
    #   - Sweet spot for translation: 0.4–0.8
    temperature: float = 0.8

    # Nucleus sampling: only consider tokens whose cumulative probability ≤ top_p.
    # 0.95 means ignore the bottom 5% probability mass (very unlikely tokens).
    top_p: float = 0.95

    # Top-k sampling: only consider the top-k most likely tokens.
    # 50 is a reasonable default that prevents sampling from the extreme tail.
    top_k: int = 50

    # Number of completions per prompt.  This is the "group size" in GRPO.
    # GRPO advantage = (reward_i - mean(rewards_in_group)) / std(rewards_in_group)
    # More generations → better advantage estimation but proportionally more
    # GPU time spent on generation and reward scoring.
    num_generations: int = 4

    do_sample: bool = True

    # Repetition penalty: values > 1.0 discourage the model from repeating tokens.
    # 1.0 means no penalty. Useful for preventing degenerate repetitive outputs.
    repetition_penalty: float = 1.0


@dataclass
class MetricXConfig:
    """Configuration for the MetricX translation quality reward model.

    MetricX is a regression model based on MT5 (encoder-decoder) that predicts
    a quality score for a translation.  The score is on a scale where
    **lower = better** (0 = perfect, 25 = worst).

    To use it as a reward signal (higher = better for RL), we apply:
      reward = offset - metricx_score
    So a perfect translation (score=0) gets reward=5.0 (with default offset=5).

    MetricX runs in a **subprocess** on a dedicated GPU (e.g. GPU 6) to avoid
    interfering with the training process's CUDA context and memory.
    The subprocess has its own Python environment to avoid dependency conflicts.
    """

    enabled: bool = True
    model_name: str = "google/metricx-24-hybrid-large-v2p6"
    tokenizer_name: str = "google/mt5-xl"

    # QE (Quality Estimation) mode: score without reference translation.
    # If True: uses "source: X candidate: Y reference: Z" input format.
    # If False: uses "source: X candidate: Y" input format (no ref needed).
    use_reference: bool = False

    batch_size: int = 4

    # Physical GPU index for this reward model.
    # The main training process uses GPUs 0-5 (via CUDA_VISIBLE_DEVICES).
    # MetricX subprocess sees ALL GPUs and places itself on gpu_id=6.
    gpu_id: int = 6

    dtype: str = "bfloat16"
    max_input_length: int = 2048

    # What to do when input exceeds max_input_length:
    #   "truncate" — silently truncate the input
    #   "skip" — skip and return a default score
    overflow_policy: str = "truncate"

    # The offset for converting MetricX score → reward.
    # reward = offset - metricx_score
    # With offset=5.0: perfect translation → reward=5, terrible → reward≈-20
    offset: float = 5.0

    # Path to Python executable for subprocess mode.
    # If set, MetricX runs in a separate process with its own venv.
    # If None, MetricX loads in the same process (for testing/debugging).
    python_executable: str | None = None

    subprocess_timeout_sec: float = 600.0


@dataclass
class XCometConfig:
    """Configuration for the XComet translation quality reward model.

    XComet (Unbabel/XCOMET-XL) is a learned metric that outputs a score
    in [0, 1] where **higher = better**.  Unlike MetricX, no offset inversion
    is needed — the raw score is directly used as reward.

    Like MetricX, it runs in a subprocess on a dedicated GPU (e.g. GPU 7).
    """

    enabled: bool = True
    model_name: str = "Unbabel/XCOMET-XL"
    batch_size: int = 2
    gpu_id: int = 7
    use_reference: bool = False
    python_executable: str | None = None
    subprocess_timeout_sec: float = 600.0


@dataclass
class RemoteLLMRewardConfig:
    """Configuration for a remote LLM-based reward model.

    Calls an OpenAI-compatible chat completion API to score translation
    quality.  The remote server can be vLLM, TGI, or any OpenAI-compatible
    endpoint.

    Two modes:
      - "judge": LLM-as-a-Judge.  Sends a prompt asking the LLM to rate
        the translation, parses a numeric score from the text response.
      - "scoring": Custom scoring endpoint.  The model returns a raw numeric
        score directly (the full response content is parsed as a float).
    """

    enabled: bool = False  # Disabled by default; opt-in

    # --- Connection ---
    # OpenAI SDK base_url pointing to the remote server.
    # Example: "http://10.0.0.5:8000/v1" for a vLLM server.
    base_url: str = "http://localhost:8000/v1"

    # API key: checked in this order:
    #   1. api_key field (direct value)
    #   2. Environment variable named by api_key_env
    # Follows the same pattern as misc.huggingface_token / huggingface_token_env.
    api_key: str | None = None
    api_key_env: str = "REMOTE_LLM_REWARD_API_KEY"

    # Model name passed to the OpenAI SDK.
    # For vLLM this is the HF model ID, e.g. "Qwen/Qwen2.5-72B-Instruct".
    model: str = "default"

    # --- Mode ---
    # "judge" = LLM-as-a-Judge with prompt template + score regex parsing.
    # "scoring" = endpoint returns numeric score directly (no prompt/parsing).
    mode: str = "judge"

    # --- Judge mode settings ---
    # Prompt template for judge mode.  Must contain {src} and {mt} placeholders.
    # May optionally contain {ref} (filled with reference or "N/A").
    # Empty string = use built-in default template.
    judge_prompt_template: str = ""
    judge_system_message: str = ""

    # Regex to extract numeric score from judge response.
    # Must contain one capture group matching a number.
    score_regex: str = r"(?i)(?:score\s*[:=]\s*)([\d]+(?:\.[\d]+)?)"

    # Expected score range for normalization to [0, 1]:
    #   normalized = (raw - score_min) / (score_max - score_min)
    score_min: float = 0.0
    score_max: float = 10.0

    # --- Generation parameters (judge mode) ---
    temperature: float = 0.0   # Deterministic for consistent scoring
    max_tokens: int = 256

    # --- Concurrency ---
    # Each sample = one API call.  Controls ThreadPoolExecutor parallelism.
    max_concurrent_requests: int = 16

    # Timeout per individual API request (seconds).
    request_timeout_sec: float = 60.0

    # --- Error handling ---
    # Score returned when API call or score parsing fails.
    fallback_score: float = 0.0

    # Max retries per request (handled by OpenAI SDK's built-in retry).
    max_retries: int = 2


@dataclass
class RewardConfig:
    """Reward model ensemble configuration.

    Multiple reward models can be combined via weighted sum:
      total_reward = w_metricx * metricx_reward + w_xcomet * xcomet_reward
                   + w_remote_llm * remote_llm_reward

    TRL's GRPOTrainer accepts a list of ``reward_funcs``.  It calls each one
    and sums their outputs per completion.  The weights here are baked into
    each function via ``make_trl_reward_func(weight=...)``.
    """

    # Weights for each reward model.  Set to 0.0 to disable without
    # removing the model config.
    w_metricx: float = 1.0
    w_xcomet: float = 0.0
    w_remote_llm: float = 0.0

    metricx: MetricXConfig = field(default_factory=MetricXConfig)
    xcomet: XCometConfig = field(default_factory=XCometConfig)
    remote_llm: RemoteLLMRewardConfig = field(default_factory=RemoteLLMRewardConfig)


@dataclass
class TrainingConfig:
    """RL algorithm and optimizer hyperparameters.

    These map to TRL's GRPOConfig fields.  See module docstring for
    detailed explanation of loss_type, beta, epsilon, scale_rewards.
    """

    loss_type: str = "grpo"  # grpo | dapo | dr_grpo | sapo | gspo
    beta: float = 0.01       # KL penalty coefficient (0 = no reference model)
    epsilon: float = 0.2     # PPO/GRPO clipping range for probability ratios

    # How to normalize rewards before computing advantages:
    #   "group" — within each prompt's generation group (standard GRPO)
    #   "batch" — across the full batch
    #   "false" — no normalization
    scale_rewards: str = "group"

    lr: float = 1e-6         # Learning rate (typically very small for RL fine-tuning)
    weight_decay: float = 0.0
    per_device_train_batch_size: int = 2  # Prompts per GPU per step
    gradient_accumulation_steps: int = 4  # Effective batch = num_gpus × batch × accum

    # Gradient clipping prevents explosion from high-variance RL gradients.
    max_grad_norm: float = 1.0

    num_train_epochs: float = 1.0
    max_steps: int = -1      # If > 0, overrides num_train_epochs
    warmup_ratio: float = 0.0
    warmup_steps: int = 0
    save_steps: int = 200
    eval_steps: int = 50
    logging_steps: int = 1
    log_completions: bool = False  # Log generated texts to W&B/TensorBoard

    # --- MoE (Mixture-of-Experts) specific settings ---
    # Qwen3.5-35B-A3B is a MoE model with 128 experts, 8 active per token.
    # Without output_router_logits=True, the router's load-balancing loss
    # cannot be computed and MoE training may collapse (all tokens routed
    # to the same few experts).
    output_router_logits: bool = False

    # Coefficient for the auxiliary router loss that encourages balanced
    # expert utilization.  Typical range: 0.001–0.01.
    # Total loss = RL_loss + router_aux_loss_coef * router_aux_loss
    router_aux_loss_coef: float = 0.0


@dataclass
class StabilityConfig:
    """MiniRL-inspired training stability controls.

    All features are opt-in: default values preserve existing behavior
    (zero overhead when disabled).

    Ref: arxiv 2512.01374 — "Stabilizing Reinforcement Learning with LLMs"
    Ref: TRL >= 0.29.0 already disables length normalization by default,
         which aligns with MiniRL's finding that it "invalidates the
         first-order approximation."
    """

    # --- Monitoring ---
    # Enable the StabilityMonitorCallback that tracks entropy, KL, rewards.
    enable_monitoring: bool = False

    # How often (in training steps) to compute stability metrics.
    monitoring_interval: int = 1

    # --- Collapse detection ---
    # If token-level entropy drops below this, log a warning.
    # Entropy drops precede training collapse (MiniRL Section 4).
    # Set to 0.0 to disable.
    entropy_floor: float = 0.0

    # If KL divergence exceeds this, log a warning.
    # Set to 0.0 to disable.
    kl_ceiling: float = 0.0

    # Halt training if entropy stays below floor for this many consecutive steps.
    # Set to 0 to disable (warning only).
    halt_on_collapse_steps: int = 0

    # --- Reward clipping ---
    # Clip transformed reward values to [-v, +v] before they enter advantage
    # computation.  Prevents extreme outliers from destabilizing group
    # normalization.  Set to 0.0 to disable.
    reward_clip_value: float = 0.0

    # --- Asymmetric clipping (MiniRL Section 3.2) ---
    # Paper recommends: epsilon_high=0.27, epsilon_low=0.2
    # TRL >= 0.29.0 supports these as separate GRPOConfig parameters.
    # When set to 0.0, falls back to the symmetric training.epsilon.
    epsilon_high: float = 0.0
    epsilon_low: float = 0.0

    # --- Routing Replay for MoE (Phase 2, not yet implemented) ---
    # "r2" = Vanilla Routing Replay (replay rollout expert assignments)
    # "r3" = Uniform Routing Replay
    # "none" = disabled (default)
    routing_replay: str = "none"

    # --- MoE router entropy monitoring ---
    # Requires training.output_router_logits=True.
    monitor_router_entropy: bool = False


@dataclass
class DistributedConfig:
    """DeepSpeed and NCCL distributed training settings.

    DeepSpeed ZeRO (Zero Redundancy Optimizer) shards model states across GPUs:
      - Stage 0: No sharding (standard DDP)
      - Stage 1: Shard optimizer states only (2× memory reduction)
      - Stage 2: Shard optimizer states + gradients (4× memory reduction)
      - Stage 3: Shard optimizer + gradients + parameters (8× memory reduction,
                  but adds communication overhead for parameter gathering)

    For 35B MoE models, ZeRO Stage 2 is typically sufficient because MoE
    models only activate ~3B parameters per token (8 of 128 experts).
    For dense 27B models, Stage 2 works with gradient checkpointing.
    Stage 3 is needed only if models don't fit with Stage 2.

    Ref: DeepSpeed ZeRO — https://www.deepspeed.ai/tutorials/zero/
    """

    # Path to DeepSpeed JSON config file.  If provided, this takes priority
    # over auto-generation.  If None, a config is auto-generated from the
    # settings below.
    deepspeed: str | None = None

    # Used only when deepspeed=None (auto-generate mode).
    zero_stage: int = 2
    offload_optimizer: bool = False   # Offload optimizer to CPU (saves GPU memory)
    offload_param: bool = False       # Offload parameters to CPU (ZeRO-3 only)

    # NCCL timeout in minutes for distributed collectives (all_reduce, broadcast).
    # Must be long enough to cover reward model scoring, which can take 10-30 min
    # for large batches.  During scoring, non-rank-0 processes are idle and waiting.
    nccl_timeout_minutes: int = 60


@dataclass
class MiscConfig:
    """Miscellaneous training settings."""

    seed: int = 42
    dtype: str = "bfloat16"  # bfloat16 | float16 | float32 (model loading precision)
    output_dir: str = "./outputs/distributed-rl"

    # HuggingFace Hub cache directory for model downloads.
    # Set this to a large disk (e.g. /media/sdd3) for multi-GB model files.
    huggingface_cache_dir: str | None = None

    # HuggingFace authentication for gated models.
    huggingface_token: str | None = None
    huggingface_token_env: str = "HF_TOKEN"  # Read token from this env var

    # Where to log training metrics: ["wandb"], ["tensorboard"], or [].
    report_to: list[str] = field(default_factory=list)


@dataclass
class DistributedRLConfig:
    """Top-level config that groups all sub-configs.

    Each field name corresponds to a top-level key in the YAML config file.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    misc: MiscConfig = field(default_factory=MiscConfig)


# ---------------------------------------------------------------------------
# Config loading helpers
# ---------------------------------------------------------------------------
# The _coerce_dataclass pattern is adapted from:
#   qwen3.5-35b-a3b/qwen35_moe_rl/config.py:204-217
# ---------------------------------------------------------------------------

def _coerce_dataclass(cls: type[Any], payload: dict[str, Any]) -> Any:
    """Recursively convert a nested dict into a nested dataclass instance.

    For each field in the dataclass:
      - If the field's default is itself a dataclass and the payload value is
        a dict, recurse into ``_coerce_dataclass`` for that sub-dataclass.
      - Otherwise, use the payload value as-is (Python duck-typing handles
        basic type coercion: int, float, str, bool, list).
      - If a field is not present in the payload, the dataclass default is used.

    Example:
      payload = {"model": {"policy_name_or_path": "my-model"}, "training": {"lr": 5e-7}}
      cfg = _coerce_dataclass(DistributedRLConfig, payload)
      # cfg.model.policy_name_or_path == "my-model"
      # cfg.training.lr == 5e-7
      # cfg.training.epsilon == 0.2  ← default preserved
    """
    defaults = cls()
    kwargs: dict[str, Any] = {}
    for field_info in cls.__dataclass_fields__.values():
        key = field_info.name
        if key not in payload:
            continue
        raw = payload[key]
        default_value = getattr(defaults, key)
        # If the default value is a dataclass and the payload is a dict,
        # recursively coerce the nested dict into the nested dataclass.
        if hasattr(default_value, "__dataclass_fields__") and isinstance(raw, dict):
            kwargs[key] = _coerce_dataclass(type(default_value), raw)
        else:
            kwargs[key] = raw
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------
# Paths in YAML are typically relative to the config file's directory.
# These functions resolve them to absolute paths so the rest of the code
# doesn't need to worry about working directory.
#
# Adapted from: qwen3.5-35b-a3b/qwen35_moe_rl/config.py:220-258
# ---------------------------------------------------------------------------

def _resolve_optional_path(value: str | None, base_dir: Path) -> str | None:
    """Resolve a relative file/directory path against the config file's directory."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return value
    path = Path(text).expanduser()
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _resolve_model_name_or_path(value: str | None, base_dir: Path) -> str | None:
    """Resolve a model name — could be a HuggingFace ID or a local path.

    If it looks like a HuggingFace model ID (e.g. "Qwen/Qwen3.5-35B-A3B"),
    return it as-is.  If it looks like a local path (starts with "./", "~",
    or the path exists on disk), resolve it relative to base_dir.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return value
    path = Path(text).expanduser()
    if path.is_absolute():
        return str(path)
    # Heuristic: if it starts with "." or "~" or the path exists, it's local.
    # Otherwise it's a HuggingFace model ID like "Qwen/Qwen3.5-35B-A3B".
    if text.startswith(".") or text.startswith("~") or path.exists():
        return str((base_dir / path).resolve())
    return text


def _resolve_command_or_path(value: str | None, base_dir: Path) -> str | None:
    """Resolve a command/executable path.

    Bare commands like "python" are returned as-is (found via PATH).
    Paths like "../../.venv-metrics/bin/python" are resolved relative to base_dir.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return value
    # If no path separators and doesn't start with "." or "~", it's a bare command.
    if "/" not in text and "\\" not in text and not text.startswith(".") and not text.startswith("~"):
        return text
    path = Path(text).expanduser()
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_config(cfg: DistributedRLConfig) -> None:
    """Check config for invalid or contradictory settings."""

    # --- Data source ---
    use_hf = bool(cfg.data.hf_dataset_name and str(cfg.data.hf_dataset_name).strip())
    if not use_hf:
        # Must have a valid local train file.
        if not cfg.data.train_file or not Path(cfg.data.train_file).exists():
            raise FileNotFoundError(f"data.train_file not found: {cfg.data.train_file}")
        if cfg.data.eval_file and not Path(cfg.data.eval_file).exists():
            raise FileNotFoundError(f"data.eval_file not found: {cfg.data.eval_file}")
    else:
        if not cfg.data.hf_train_split.strip():
            raise ValueError("data.hf_train_split must not be empty when hf_dataset_name is set.")

    # --- Training ---
    valid_loss_types = {"grpo", "dapo", "dr_grpo", "sapo", "gspo"}
    if cfg.training.loss_type not in valid_loss_types:
        raise ValueError(f"training.loss_type must be one of {valid_loss_types}")

    valid_scale = {"group", "batch", "false"}
    if cfg.training.scale_rewards not in valid_scale:
        raise ValueError(f"training.scale_rewards must be one of {valid_scale}")

    # --- Reward ---
    # At least one reward model must be enabled, otherwise there's no signal to train on.
    if (not cfg.reward.metricx.enabled
            and not cfg.reward.xcomet.enabled
            and not cfg.reward.remote_llm.enabled):
        raise ValueError("At least one reward model must be enabled.")

    # MetricX and XComet need different GPUs because they both occupy significant VRAM.
    if cfg.reward.metricx.enabled and cfg.reward.xcomet.enabled:
        if cfg.reward.metricx.gpu_id == cfg.reward.xcomet.gpu_id:
            raise ValueError("MetricX and XComet must use different GPUs.")

    # --- Distributed ---
    if cfg.distributed.zero_stage not in {0, 1, 2, 3}:
        raise ValueError("distributed.zero_stage must be 0, 1, 2, or 3.")

    # ZeRO param offloading only works at Stage 3 (where parameters are sharded).
    if cfg.distributed.offload_param and cfg.distributed.zero_stage < 3:
        logger.warning(
            "distributed.offload_param only effective with zero_stage=3. "
            "Current zero_stage=%s; will be ignored.",
            cfg.distributed.zero_stage,
        )

    # --- Remote LLM reward ---
    if cfg.reward.remote_llm.enabled:
        valid_remote_modes = {"judge", "scoring"}
        if cfg.reward.remote_llm.mode not in valid_remote_modes:
            raise ValueError(
                f"reward.remote_llm.mode must be one of {valid_remote_modes}, "
                f"got {cfg.reward.remote_llm.mode!r}"
            )
        if not cfg.reward.remote_llm.base_url.strip():
            raise ValueError("reward.remote_llm.base_url must not be empty when enabled.")
        if cfg.reward.remote_llm.score_max <= cfg.reward.remote_llm.score_min:
            raise ValueError(
                "reward.remote_llm.score_max must be greater than score_min."
            )
        if cfg.reward.remote_llm.max_concurrent_requests < 1:
            raise ValueError("reward.remote_llm.max_concurrent_requests must be >= 1.")

    # --- Stability (MiniRL) ---
    valid_routing = {"none", "r2", "r3"}
    if cfg.stability.routing_replay not in valid_routing:
        raise ValueError(f"stability.routing_replay must be one of {valid_routing}")

    if cfg.stability.routing_replay != "none" and not cfg.training.output_router_logits:
        raise ValueError(
            "stability.routing_replay requires training.output_router_logits=true "
            "(MoE model with router logits output)."
        )

    if cfg.stability.epsilon_high < 0 or cfg.stability.epsilon_low < 0:
        raise ValueError("stability.epsilon_high and epsilon_low must be >= 0.")

    if cfg.stability.reward_clip_value < 0:
        raise ValueError("stability.reward_clip_value must be >= 0.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> DistributedRLConfig:
    """Load a YAML config file and return a fully resolved, validated config.

    Pipeline: YAML → dict → _coerce_dataclass → path resolution → validation.
    """
    cfg_path = Path(path)
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    cfg = _coerce_dataclass(DistributedRLConfig, payload)

    # All relative paths in the config are resolved relative to the
    # directory containing the YAML file (not the current working directory).
    base_dir = cfg_path.parent.resolve()

    # Resolve paths
    cfg.data.train_file = _resolve_optional_path(cfg.data.train_file, base_dir)
    cfg.data.eval_file = _resolve_optional_path(cfg.data.eval_file, base_dir)
    cfg.model.policy_name_or_path = (
        _resolve_model_name_or_path(cfg.model.policy_name_or_path, base_dir)
        or cfg.model.policy_name_or_path
    )
    cfg.misc.output_dir = _resolve_optional_path(cfg.misc.output_dir, base_dir) or cfg.misc.output_dir
    cfg.misc.huggingface_cache_dir = _resolve_optional_path(cfg.misc.huggingface_cache_dir, base_dir)
    cfg.distributed.deepspeed = _resolve_optional_path(cfg.distributed.deepspeed, base_dir)
    cfg.reward.metricx.python_executable = _resolve_command_or_path(
        cfg.reward.metricx.python_executable, base_dir
    )
    cfg.reward.xcomet.python_executable = _resolve_command_or_path(
        cfg.reward.xcomet.python_executable, base_dir
    )

    _validate_config(cfg)
    return cfg


def dump_config(cfg: DistributedRLConfig, path: str | Path) -> None:
    """Serialize a config to YAML (for logging/reproducibility)."""
    Path(path).write_text(yaml.safe_dump(asdict(cfg), sort_keys=False), encoding="utf-8")
