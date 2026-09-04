"""
Phase 5 test suite: decision policy logic, evaluation, threshold
derivation, and simple-ranking comparison. Small, hand-verifiable
synthetic examples throughout.
"""

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.decision.policy import DecisionPolicy, InvalidThresholdError, DECISIONS
from src.decision.evaluation import evaluate_policy
from src.decision.threshold_selection import (
    score_at_top_fraction, derive_block_fraction_for_precision_target, build_policy_from_validation,
)
from src.decision.comparison import compare_policy_to_fixed_budget


# ---------------------------------------------------------------------------
# DecisionPolicy logic
# ---------------------------------------------------------------------------

def test_low_score_approves():
    policy = DecisionPolicy(approve_threshold=0.3, block_threshold=0.8)
    decisions = policy.decide([0.1])
    assert decisions[0] == "APPROVE"


def test_middle_score_reviews():
    policy = DecisionPolicy(approve_threshold=0.3, block_threshold=0.8)
    decisions = policy.decide([0.5])
    assert decisions[0] == "REVIEW"


def test_high_score_blocks():
    policy = DecisionPolicy(approve_threshold=0.3, block_threshold=0.8)
    decisions = policy.decide([0.9])
    assert decisions[0] == "BLOCK"


def test_exact_approve_threshold_boundary_is_review_not_approve():
    # Per the documented rule: score < approve_threshold -> APPROVE, so an
    # EXACT match falls through to REVIEW (strict inequality).
    policy = DecisionPolicy(approve_threshold=0.3, block_threshold=0.8)
    decisions = policy.decide([0.3])
    assert decisions[0] == "REVIEW"


def test_exact_block_threshold_boundary_is_block():
    policy = DecisionPolicy(approve_threshold=0.3, block_threshold=0.8)
    decisions = policy.decide([0.8])
    assert decisions[0] == "BLOCK"


def test_invalid_threshold_ordering_raises():
    with pytest.raises(InvalidThresholdError):
        DecisionPolicy(approve_threshold=0.8, block_threshold=0.3)


def test_threshold_out_of_range_raises():
    with pytest.raises(InvalidThresholdError):
        DecisionPolicy(approve_threshold=-0.1, block_threshold=0.5)
    with pytest.raises(InvalidThresholdError):
        DecisionPolicy(approve_threshold=0.1, block_threshold=1.5)


def test_equal_thresholds_allowed_empty_review_bucket():
    # A degenerate but valid two-bucket policy (no REVIEW bucket).
    policy = DecisionPolicy(approve_threshold=0.5, block_threshold=0.5)
    decisions = policy.decide([0.3, 0.5, 0.7])
    assert list(decisions) == ["APPROVE", "BLOCK", "BLOCK"]


def test_every_transaction_receives_exactly_one_decision():
    rng = np.random.default_rng(0)
    scores = rng.random(1000)
    policy = DecisionPolicy(approve_threshold=0.3, block_threshold=0.8)
    decisions = policy.decide(scores)
    assert len(decisions) == 1000
    assert set(decisions) <= set(DECISIONS)
    assert all(d in DECISIONS for d in decisions)


def test_nan_score_raises():
    policy = DecisionPolicy(approve_threshold=0.3, block_threshold=0.8)
    with pytest.raises(ValueError):
        policy.decide([0.1, float("nan"), 0.9])


def test_policy_to_dict_from_dict_roundtrip():
    policy = DecisionPolicy(approve_threshold=0.2, block_threshold=0.9, name="test_policy")
    d = policy.to_dict()
    policy2 = DecisionPolicy.from_dict(d)
    assert policy2.approve_threshold == policy.approve_threshold
    assert policy2.block_threshold == policy.block_threshold
    assert policy2.name == policy.name


# ---------------------------------------------------------------------------
# evaluate_policy
# ---------------------------------------------------------------------------

def test_evaluate_policy_hand_verified():
    # 10 transactions, fraud at indices 8, 9.
    y_true = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    scores = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.9, 0.95]
    # approve_threshold=0.5 -> everything <0.5 APPROVE (idx 0-7)
    # block_threshold=0.85 -> everything >=0.85 BLOCK (idx 8,9)
    # nothing in between -> REVIEW empty
    policy = DecisionPolicy(approve_threshold=0.5, block_threshold=0.85, name="test")
    result = evaluate_policy(y_true, scores, policy)

    assert result["buckets"]["APPROVE"]["count"] == 8
    assert result["buckets"]["APPROVE"]["fraud_count"] == 0
    assert result["buckets"]["REVIEW"]["count"] == 0
    assert result["buckets"]["BLOCK"]["count"] == 2
    assert result["buckets"]["BLOCK"]["fraud_count"] == 2
    assert result["fraud_captured_block_alone"] == 2
    assert result["fraud_captured_review_plus_block"] == 2
    assert result["fraud_missed_in_approve"] == 0
    assert result["precision_among_blocked"] == pytest.approx(1.0)
    assert result["fraud_recall_review_plus_block"] == pytest.approx(1.0)


def test_evaluate_policy_with_review_bucket_and_missed_fraud():
    # 10 transactions, fraud at idx 2 (low score, will be missed/approved),
    # idx 5 (mid score, reviewed), idx 9 (high score, blocked).
    y_true = [0, 0, 1, 0, 0, 1, 0, 0, 0, 1]
    scores = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.35, 0.4, 0.45, 0.95]
    policy = DecisionPolicy(approve_threshold=0.3, block_threshold=0.9, name="test")
    result = evaluate_policy(y_true, scores, policy)

    # APPROVE: scores < 0.3 -> idx 0,1,2,3 (values .05,.1,.15,.2)
    assert result["buckets"]["APPROVE"]["count"] == 4
    assert result["buckets"]["APPROVE"]["fraud_count"] == 1  # idx2 fraud, missed
    assert result["fraud_missed_in_approve"] == 1

    # BLOCK: scores >= 0.9 -> idx 9 only
    assert result["buckets"]["BLOCK"]["count"] == 1
    assert result["buckets"]["BLOCK"]["fraud_count"] == 1

    # REVIEW: the rest -> idx 4,5,6,7,8 (5 txns), fraud at idx5 only
    assert result["buckets"]["REVIEW"]["count"] == 5
    assert result["buckets"]["REVIEW"]["fraud_count"] == 1

    assert result["fraud_captured_review_plus_block"] == 2
    assert result["total_fraud_cases"] == 3
    assert result["fraud_recall_review_plus_block"] == pytest.approx(2 / 3)
    assert result["legitimate_transactions_reviewed"] == 4
    assert result["legitimate_transactions_blocked"] == 0
    assert result["pct_legitimate_blocked"] == pytest.approx(0.0)


def test_evaluate_policy_decision_percentages_sum_to_one():
    rng = np.random.default_rng(1)
    n = 500
    y_true = (rng.random(n) < 0.05).astype(int)
    scores = rng.random(n)
    policy = DecisionPolicy(approve_threshold=0.3, block_threshold=0.8, name="test")
    result = evaluate_policy(y_true, scores, policy)
    total_pct = sum(result["buckets"][d]["pct_of_total"] for d in DECISIONS)
    assert total_pct == pytest.approx(1.0)
    total_count = sum(result["buckets"][d]["count"] for d in DECISIONS)
    assert total_count == n


# ---------------------------------------------------------------------------
# threshold_selection
# ---------------------------------------------------------------------------

def test_score_at_top_fraction_hand_verified():
    scores = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    # top 20% of 10 = top 2 -> {9,10}; boundary (min of selected) = 9
    threshold = score_at_top_fraction(scores, 0.2)
    assert threshold == pytest.approx(9.0)


def test_derive_block_fraction_meets_target():
    # 100 transactions, first 5 (highest scores, idx 95-99) all fraud.
    n = 100
    y_true = np.zeros(n, dtype=int)
    y_true[-5:] = 1
    scores = np.arange(n, dtype=float)
    result = derive_block_fraction_for_precision_target(
        y_true, scores, max_fraction=0.10, min_precision=1.0, step=0.01,
    )
    # top 5% (5 txns) are exactly the 5 fraud cases -> precision 1.0 achievable
    assert result["target_achieved"] is True
    assert result["achieved_precision"] == pytest.approx(1.0)
    assert result["block_fraction"] <= 0.05 + 1e-9


def test_derive_block_fraction_target_not_achievable_reports_honestly():
    n = 100
    rng = np.random.default_rng(2)
    y_true = (rng.random(n) < 0.05).astype(int)
    scores = rng.random(n)
    result = derive_block_fraction_for_precision_target(
        y_true, scores, max_fraction=0.10, min_precision=0.999, step=0.01,  # near-impossible target
    )
    assert result["target_achieved"] is False
    assert result["achieved_precision"] < 0.999


def test_build_policy_from_validation_produces_valid_policy():
    n = 2000
    rng = np.random.default_rng(3)
    y_val = (rng.random(n) < 0.05).astype(int)
    val_scores = rng.random(n)
    # Bias: give fraud cases higher scores on average so precision targets are achievable.
    val_scores = np.where(y_val == 1, val_scores + 0.5, val_scores)
    val_scores = np.clip(val_scores, 0, 1)

    policy, diagnostics = build_policy_from_validation(
        y_val, val_scores, review_capacity=0.05, block_precision_target=0.5, name="test_policy",
    )
    assert isinstance(policy, DecisionPolicy)
    assert policy.approve_threshold <= policy.block_threshold
    assert diagnostics["name"] == "test_policy"
    assert diagnostics["review_capacity_target"] == 0.05


def test_build_policy_from_validation_signature_has_no_test_data_parameter():
    """
    Structural leakage guard: the threshold-derivation function must not
    even be ABLE to accept test data — checked via its signature, not just
    by convention.
    """
    sig = inspect.signature(build_policy_from_validation)
    param_names = set(sig.parameters.keys())
    assert not any("test" in p.lower() for p in param_names)


def test_frozen_policy_decide_requires_no_labels():
    """The frozen policy's decide() must work from scores alone — applying
    it to test data must not require test labels."""
    policy = DecisionPolicy(approve_threshold=0.3, block_threshold=0.8, name="frozen")
    sig = inspect.signature(policy.decide)
    assert list(sig.parameters.keys()) == ["scores"]  # no y/labels parameter at all
    decisions = policy.decide([0.1, 0.5, 0.9])  # works with scores only
    assert len(decisions) == 3


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------

def test_compare_policy_to_fixed_budget_matched_intervention():
    n = 200
    rng = np.random.default_rng(4)
    y_true = (rng.random(n) < 0.05).astype(int)
    scores = rng.random(n)
    policy, _ = build_policy_from_validation(y_true, scores, review_capacity=0.10, block_precision_target=0.3, name="test")
    policy_eval = evaluate_policy(y_true, scores, policy)

    comparison = compare_policy_to_fixed_budget(y_true, scores, policy_eval)
    expected_fraction = (
        policy_eval["buckets"]["REVIEW"]["pct_of_total"] + policy_eval["buckets"]["BLOCK"]["pct_of_total"]
    )
    assert comparison["matched_intervention_fraction"] == pytest.approx(expected_fraction)
    # Matched budget -> both approaches flag (approximately) the same count.
    n_three_way = comparison["three_way_policy"]["n_intervened"]
    n_simple = comparison["simple_fixed_budget_ranking"]["n_intervened"]
    assert abs(n_three_way - n_simple) <= 1  # rounding at most 1 transaction


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
