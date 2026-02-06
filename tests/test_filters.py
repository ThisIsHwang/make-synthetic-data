from synth_parallel.config import FiltersConfig
from synth_parallel.filters import apply_rule_based_filters


def test_rule_based_pass():
    cfg = FiltersConfig(min_chars=2, max_chars=200, max_copy_overlap=0.95)
    decision = apply_rule_based_filters("hello world", "안녕하세요 세계", cfg)
    assert decision.passed


def test_rule_based_reject_blocked_token():
    cfg = FiltersConfig(min_chars=2, max_chars=200)
    decision = apply_rule_based_filters("hello", "Here is the translation: 안녕하세요", cfg)
    assert not decision.passed
    assert decision.reason_code == "blocked_substring"


def test_rule_based_reject_length_ratio():
    cfg = FiltersConfig(min_chars=1, max_chars=200, length_ratio_min=0.5, length_ratio_max=1.5)
    decision = apply_rule_based_filters("hello world", "x", cfg)
    assert not decision.passed
    assert decision.reason_code == "ratio_too_small"
