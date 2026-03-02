"""MiniRL-inspired training stability monitoring callback.

Hooks into TRL's GRPOTrainer via the TrainerCallback protocol to track:
  - Token-level entropy trends (decline = potential collapse)
  - KL divergence trends (spike = potential divergence)
  - Reward statistics (mean, std, outliers)
  - Collapse detection with optional automatic training halt

The callback reads metrics that TRL already computes (entropy, kl, reward)
and adds trend analysis and collapse warnings on top.

All features are zero-cost when the callback is not injected (default).

Ref: arxiv 2512.01374 — "Stabilizing Reinforcement Learning with LLMs"
Ref: TRL TrainerCallback — https://huggingface.co/docs/transformers/main_classes/callback
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

try:
    from transformers import TrainerCallback, TrainerControl, TrainerState
    from transformers import TrainingArguments
except ImportError:  # pragma: no cover - lightweight test environments
    # Provide minimal stubs so the module can be imported without transformers.
    class TrainerCallback:  # type: ignore[no-redef]
        pass
    TrainerControl = Any  # type: ignore[assignment,misc]
    TrainerState = Any  # type: ignore[assignment,misc]
    TrainingArguments = Any  # type: ignore[assignment,misc]

from .config import StabilityConfig

logger = logging.getLogger(__name__)


@dataclass
class StabilityState:
    """Rolling window state for collapse detection.

    Uses fixed-size deques to track metric histories without
    unbounded memory growth.
    """

    entropy_history: deque = field(default_factory=lambda: deque(maxlen=100))
    kl_history: deque = field(default_factory=lambda: deque(maxlen=100))
    reward_history: deque = field(default_factory=lambda: deque(maxlen=100))
    consecutive_below_entropy_floor: int = 0
    consecutive_above_kl_ceiling: int = 0


class StabilityMonitorCallback(TrainerCallback):
    """TRL-compatible callback for training stability monitoring.

    Injected into GRPOTrainer via the ``callbacks=`` parameter in
    ``trainer.py:run_training()``.

    Hooks used:
      - on_log: Inspect TRL's logged metrics (entropy, kl, reward) and
        add trend analysis + collapse warnings.
      - on_step_end: Check collapse conditions and optionally halt training.

    This callback does NOT modify the training loss or gradients.
    It only reads metrics and logs warnings / additional diagnostics.

    Example config:
      stability:
        enable_monitoring: true
        entropy_floor: 0.5
        kl_ceiling: 10.0
        halt_on_collapse_steps: 20
    """

    def __init__(self, cfg: StabilityConfig) -> None:
        self.cfg = cfg
        self.state = StabilityState()

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called after TRL logs metrics.  Extract stability-relevant signals."""
        if logs is None:
            return

        # Skip if not on the monitoring interval.
        if self.cfg.monitoring_interval > 1:
            if state.global_step % self.cfg.monitoring_interval != 0:
                return

        # --- Extract TRL's built-in metrics ---
        entropy = logs.get("entropy")
        kl = logs.get("kl")
        reward_mean = logs.get("reward")

        # Track histories.
        if entropy is not None:
            self.state.entropy_history.append(float(entropy))
        if kl is not None:
            self.state.kl_history.append(float(kl))
        if reward_mean is not None:
            self.state.reward_history.append(float(reward_mean))

        # --- Entropy floor check ---
        if self.cfg.entropy_floor > 0 and entropy is not None:
            if float(entropy) < self.cfg.entropy_floor:
                self.state.consecutive_below_entropy_floor += 1
                logger.warning(
                    "[STABILITY] Entropy %.4f below floor %.4f "
                    "(consecutive: %d steps). Potential collapse.",
                    entropy,
                    self.cfg.entropy_floor,
                    self.state.consecutive_below_entropy_floor,
                )
            else:
                self.state.consecutive_below_entropy_floor = 0

        # --- KL ceiling check ---
        if self.cfg.kl_ceiling > 0 and kl is not None:
            if float(kl) > self.cfg.kl_ceiling:
                self.state.consecutive_above_kl_ceiling += 1
                logger.warning(
                    "[STABILITY] KL %.4f above ceiling %.4f "
                    "(consecutive: %d steps). Potential divergence.",
                    kl,
                    self.cfg.kl_ceiling,
                    self.state.consecutive_above_kl_ceiling,
                )
            else:
                self.state.consecutive_above_kl_ceiling = 0

        # --- Trend analysis ---
        # Compare recent 5-step window vs previous 5-step window.
        if len(self.state.entropy_history) >= 10:
            recent = list(self.state.entropy_history)[-5:]
            earlier = list(self.state.entropy_history)[-10:-5]
            logs["stability/entropy_trend"] = sum(recent) / 5 - sum(earlier) / 5

        if len(self.state.reward_history) >= 10:
            recent = list(self.state.reward_history)[-5:]
            earlier = list(self.state.reward_history)[-10:-5]
            logs["stability/reward_trend"] = sum(recent) / 5 - sum(earlier) / 5

        if len(self.state.kl_history) >= 10:
            recent = list(self.state.kl_history)[-5:]
            earlier = list(self.state.kl_history)[-10:-5]
            logs["stability/kl_trend"] = sum(recent) / 5 - sum(earlier) / 5

        # --- Inject comprehensive stability metrics for W&B / TensorBoard ---
        self._inject_metrics(logs, entropy, kl, reward_mean)

    def _inject_metrics(
        self,
        logs: dict[str, Any],
        entropy: float | None,
        kl: float | None,
        reward_mean: float | None,
    ) -> None:
        """Inject comprehensive stability metrics into the logs dict.

        These metrics are picked up by TRL's logging pipeline and forwarded
        to whatever ``report_to`` backends are configured (W&B, TensorBoard).

        All keys use the ``stability/`` prefix for clean dashboard grouping.
        """
        # --- Current values (re-emit under stability/ prefix) ---
        if entropy is not None:
            logs["stability/entropy"] = float(entropy)
        if kl is not None:
            logs["stability/kl"] = float(kl)
        if reward_mean is not None:
            logs["stability/reward_mean"] = float(reward_mean)

        # --- Rolling statistics from deque histories ---
        if self.state.entropy_history:
            vals = list(self.state.entropy_history)
            logs["stability/entropy_rolling_mean"] = sum(vals) / len(vals)

        if self.state.kl_history:
            vals = list(self.state.kl_history)
            logs["stability/kl_rolling_mean"] = sum(vals) / len(vals)

        if self.state.reward_history:
            vals = list(self.state.reward_history)
            n = len(vals)
            mean = sum(vals) / n
            logs["stability/reward_rolling_mean"] = mean
            if n >= 2:
                variance = sum((v - mean) ** 2 for v in vals) / (n - 1)
                logs["stability/reward_rolling_std"] = variance ** 0.5
            else:
                logs["stability/reward_rolling_std"] = 0.0

        # --- Collapse indicators (int -> float for W&B compatibility) ---
        logs["stability/consecutive_below_entropy_floor"] = float(
            self.state.consecutive_below_entropy_floor
        )
        logs["stability/consecutive_above_kl_ceiling"] = float(
            self.state.consecutive_above_kl_ceiling
        )

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Check collapse conditions and optionally halt training."""
        if self.cfg.halt_on_collapse_steps <= 0:
            return

        if (
            self.state.consecutive_below_entropy_floor
            >= self.cfg.halt_on_collapse_steps
        ):
            logger.error(
                "[STABILITY] HALTING TRAINING: Entropy below floor for %d "
                "consecutive steps. This indicates training collapse.",
                self.state.consecutive_below_entropy_floor,
            )
            control.should_training_stop = True
