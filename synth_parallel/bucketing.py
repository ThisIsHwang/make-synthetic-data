from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass
class BucketStats:
    seen: int
    kept: int


def assign_bucket(length: int, boundaries: list[int]) -> int:
    if len(boundaries) < 2:
        return 0
    for idx in range(len(boundaries) - 1):
        if boundaries[idx] <= length < boundaries[idx + 1]:
            return idx
    return len(boundaries) - 2


class BalancedBucketSampler:
    """Reservoir sampler that keeps balanced bucket pools and fills deficits from extras."""

    def __init__(
        self,
        boundaries: list[int],
        sample_size: int,
        oversample_factor: float,
        seed: int,
    ):
        if len(boundaries) < 2:
            raise ValueError("Need at least two bucket boundaries")
        self.boundaries = boundaries
        self.sample_size = sample_size
        self.rng = random.Random(seed)

        self.num_buckets = len(boundaries) - 1
        base = sample_size // self.num_buckets
        remainder = sample_size % self.num_buckets
        self.quotas = [base + (1 if i < remainder else 0) for i in range(self.num_buckets)]

        self.capacities = [
            max(quota, int(quota * max(1.0, oversample_factor))) for quota in self.quotas
        ]

        self.reservoirs: list[list[dict[str, Any]]] = [[] for _ in range(self.num_buckets)]
        self.seen_counts = [0 for _ in range(self.num_buckets)]

    def add(self, item: dict[str, Any], length_value: int) -> None:
        bucket = assign_bucket(length_value, self.boundaries)
        self.seen_counts[bucket] += 1

        reservoir = self.reservoirs[bucket]
        capacity = self.capacities[bucket]

        if len(reservoir) < capacity:
            reservoir.append(item)
            return

        j = self.rng.randint(0, self.seen_counts[bucket] - 1)
        if j < capacity:
            reservoir[j] = item

    def finalize(self) -> tuple[list[dict[str, Any]], dict[int, BucketStats]]:
        selected: list[dict[str, Any]] = []
        extras: list[dict[str, Any]] = []
        stats: dict[int, BucketStats] = {}

        for bucket_id in range(self.num_buckets):
            pool = self.reservoirs[bucket_id]
            quota = self.quotas[bucket_id]
            take = min(quota, len(pool))

            if take > 0:
                picked = self.rng.sample(pool, k=take)
                selected.extend(picked)
                picked_ids = {id(x) for x in picked}
                extras.extend([x for x in pool if id(x) not in picked_ids])
            else:
                extras.extend(pool)

            stats[bucket_id] = BucketStats(seen=self.seen_counts[bucket_id], kept=take)

        deficit = self.sample_size - len(selected)
        if deficit > 0 and extras:
            fill = extras if deficit >= len(extras) else self.rng.sample(extras, k=deficit)
            selected.extend(fill)

        if len(selected) > self.sample_size:
            selected = self.rng.sample(selected, k=self.sample_size)

        self.rng.shuffle(selected)
        return selected, stats
