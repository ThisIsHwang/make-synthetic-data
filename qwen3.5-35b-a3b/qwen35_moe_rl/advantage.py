from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def build_sequence_rewards(
    metricx_scores: list[float],
    xcomet_scores: list[float],
    metricx_offset: float,
    w_metricx: float,
    w_xcomet_seq: float,
    xcomet_seq_scale: float,
) -> list[float]:
    out: list[float] = []
    for mx, xc in zip(metricx_scores, xcomet_scores):
        metricx_reward = metricx_offset - mx
        seq_reward = (w_metricx * metricx_reward) + (w_xcomet_seq * (xc * xcomet_seq_scale))
        out.append(float(seq_reward))
    return out


def broadcast_sequence_reward(seq_reward: float, token_count: int) -> list[float]:
    return [float(seq_reward) for _ in range(token_count)]


def combine_advantages(sequence_rewards: list[float], token_rewards: list[float]) -> list[float]:
    if len(sequence_rewards) != len(token_rewards):
        raise ValueError(
            f"sequence/token reward length mismatch: {len(sequence_rewards)} != {len(token_rewards)}"
        )
    return [float(a + b) for a, b in zip(sequence_rewards, token_rewards)]


def apply_group_relative_advantage(
    raw_advantages: list[list[float]],
    group_ids: list[str],
    coef: float = 1.0,
    eps: float = 1e-8,
) -> tuple[list[list[float]], list[float]]:
    if len(raw_advantages) != len(group_ids):
        raise ValueError("raw_advantages and group_ids length mismatch")

    group_to_indices: dict[str, list[int]] = defaultdict(list)
    rollout_scalar: list[float] = []
    for idx, arr in enumerate(raw_advantages):
        scalar = sum(arr) / max(1, len(arr))
        rollout_scalar.append(float(scalar))
        group_to_indices[group_ids[idx]].append(idx)

    group_z = [0.0 for _ in raw_advantages]
    for indices in group_to_indices.values():
        vals = [rollout_scalar[i] for i in indices]
        mean = sum(vals) / max(1, len(vals))
        var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals))
        std = var**0.5
        denom = std + eps
        for i in indices:
            group_z[i] = (rollout_scalar[i] - mean) / denom

    adjusted: list[list[float]] = []
    for i, arr in enumerate(raw_advantages):
        shift = coef * group_z[i]
        adjusted.append([float(v + shift) for v in arr])

    return adjusted, group_z


def _flatten(values: Iterable[list[float]]) -> list[float]:
    flat: list[float] = []
    for row in values:
        flat.extend(row)
    return flat


def normalize_advantages(
    raw_advantages: list[list[float]],
    eps: float = 1e-8,
) -> tuple[list[list[float]], dict[str, float]]:
    flat = _flatten(raw_advantages)
    if not flat:
        return raw_advantages, {
            "raw_mean": 0.0,
            "raw_std": 0.0,
            "norm_mean": 0.0,
            "norm_std": 0.0,
        }

    raw_mean = sum(flat) / len(flat)
    raw_var = sum((v - raw_mean) ** 2 for v in flat) / len(flat)
    raw_std = raw_var**0.5

    normalized: list[list[float]] = []
    for row in raw_advantages:
        normalized.append([float((v - raw_mean) / (raw_std + eps)) for v in row])

    norm_flat = _flatten(normalized)
    norm_mean = sum(norm_flat) / len(norm_flat)
    norm_var = sum((v - norm_mean) ** 2 for v in norm_flat) / len(norm_flat)
    norm_std = norm_var**0.5

    return normalized, {
        "raw_mean": float(raw_mean),
        "raw_std": float(raw_std),
        "norm_mean": float(norm_mean),
        "norm_std": float(norm_std),
    }


# ---------------------------------------------------------------------------
# GSPO: Sequence-level advantage computation
# ---------------------------------------------------------------------------


def build_sequence_level_advantages(
    sequence_rewards: list[float],
    group_ids: list[str],
    normalize: bool = True,
    group_normalize: bool = True,
    group_coef: float = 1.0,
    eps: float = 1e-8,
) -> tuple[list[float], dict[str, float]]:
    """Compute sequence-level scalar advantages for GSPO.

    For GSPO the advantage is a single scalar per rollout/completion, derived
    from the sequence reward.  This function:
      1. Optionally applies group-relative normalization (within each prompt
         group, subtract mean and divide by std).
      2. Optionally normalizes globally across all rollouts.

    Returns:
        advantages: list of scalar advantages, one per rollout.
        stats: dict with raw_mean, raw_std, norm_mean, norm_std.
    """
    advantages = list(sequence_rewards)

    if group_normalize:
        group_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, gid in enumerate(group_ids):
            group_to_indices[gid].append(idx)

        for indices in group_to_indices.values():
            vals = [advantages[i] for i in indices]
            mean_val = sum(vals) / max(1, len(vals))
            var_val = sum((v - mean_val) ** 2 for v in vals) / max(1, len(vals))
            std_val = var_val ** 0.5
            denom = std_val + eps
            for i in indices:
                advantages[i] = (advantages[i] - mean_val) / denom * group_coef

    raw_mean = sum(advantages) / max(1, len(advantages))
    raw_var = sum((v - raw_mean) ** 2 for v in advantages) / max(1, len(advantages))
    raw_std = raw_var ** 0.5

    if normalize:
        norm_advantages = [(v - raw_mean) / (raw_std + eps) for v in advantages]
    else:
        norm_advantages = list(advantages)

    norm_mean = sum(norm_advantages) / max(1, len(norm_advantages))
    norm_var = sum((v - norm_mean) ** 2 for v in norm_advantages) / max(1, len(norm_advantages))
    norm_std = norm_var ** 0.5

    stats = {
        "raw_mean": float(raw_mean),
        "raw_std": float(raw_std),
        "norm_mean": float(norm_mean),
        "norm_std": float(norm_std),
    }
    return norm_advantages, stats
