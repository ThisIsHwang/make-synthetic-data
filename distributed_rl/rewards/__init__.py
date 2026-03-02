"""Pluggable reward model system for TRL GRPOTrainer.

This package provides:

  RewardModel (ABC):
    Abstract base class.  Any new reward model must subclass this and
    implement ``score(samples) -> RewardOutput``.

  make_trl_reward_func(reward_model, ...):
    Wraps a RewardModel into a callable that TRL's GRPOTrainer can use.
    TRL calls: ``func(completions=["text", ...], src_text=["s1", ...], ...)``
    The wrapper extracts the right columns, calls ``score()``, and applies
    weight/offset transformations.

  MetricXReward:
    MetricX translation quality scorer.  Lower raw score = better.
    Runs on a dedicated GPU via subprocess isolation.

  XCometReward:
    XComet translation quality scorer.  Higher raw score = better.
    Runs on a dedicated GPU via subprocess isolation.

Adding a new reward model:
  1. Create ``rewards/my_model.py`` with ``class MyReward(RewardModel)``
  2. Implement ``score(samples: list[SampleForScoring]) -> RewardOutput``
  3. Register it in ``trainer.py:build_reward_funcs()``
  4. Add config fields in ``config.py``
  5. Wrap with ``make_trl_reward_func(my_model, weight=..., offset=...)``
"""

from .base import RewardModel, SampleForScoring, RewardOutput, make_trl_reward_func
from .metricx import MetricXReward
from .remote_llm import RemoteLLMReward
from .xcomet import XCometReward

__all__ = [
    "RewardModel",
    "SampleForScoring",
    "RewardOutput",
    "make_trl_reward_func",
    "MetricXReward",
    "RemoteLLMReward",
    "XCometReward",
]
