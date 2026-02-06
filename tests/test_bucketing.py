from synth_parallel.bucketing import BalancedBucketSampler, assign_bucket


def test_assign_bucket():
    boundaries = [0, 10, 20, 40]
    assert assign_bucket(5, boundaries) == 0
    assert assign_bucket(10, boundaries) == 1
    assert assign_bucket(35, boundaries) == 2
    assert assign_bucket(999, boundaries) == 2


def test_balanced_sampler_quota_and_fill():
    boundaries = [0, 10, 20, 30]
    sampler = BalancedBucketSampler(boundaries, sample_size=6, oversample_factor=2.0, seed=7)

    # Bucket 0: many samples
    for i in range(20):
        sampler.add({"id": f"a{i}", "length_approx": 5}, 5)

    # Bucket 1: sparse
    for i in range(2):
        sampler.add({"id": f"b{i}", "length_approx": 15}, 15)

    # Bucket 2: many
    for i in range(20):
        sampler.add({"id": f"c{i}", "length_approx": 25}, 25)

    selected, stats = sampler.finalize()

    assert len(selected) == 6
    assert sum(s.kept for s in stats.values()) <= 6
