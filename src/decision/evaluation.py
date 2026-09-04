"""
Phase 5: Decision-policy evaluation — turns (y_true, decisions) into the
full operational report the Phase 5 brief asks for: decision distribution,
per-bucket fraud capture, and customer-friction metrics.

Deliberately separate from `src/evaluation/operational_eval.py` (which
evaluates a continuous ranking under a fixed top-K budget) because a
three-way policy's unit of analysis is a DECISION BUCKET, not a budget
fraction — reused where it overlaps (see `compare_policy_to_fixed_budget`
in `src/decision/comparison.py`), not duplicated.
"""

from __future__ import annotations

import numpy as np

from src.decision.policy import DecisionPolicy, DECISIONS


def evaluate_policy(y_true, scores, policy: DecisionPolicy) -> dict:
    """
    Full operational evaluation of a frozen `DecisionPolicy` against
    (y_true, scores). Does not select or adjust anything — purely reports.
    """
    y_true = np.asarray(y_true).astype(int)
    decisions = policy.decide(scores)
    n = len(y_true)
    total_fraud = int(y_true.sum())
    total_legit = n - total_fraud

    buckets = {}
    for d in DECISIONS:
        mask = decisions == d
        n_in_bucket = int(mask.sum())
        fraud_in_bucket = int(y_true[mask].sum())
        legit_in_bucket = n_in_bucket - fraud_in_bucket
        buckets[d] = {
            "count": n_in_bucket,
            "pct_of_total": (n_in_bucket / n) if n > 0 else float("nan"),
            "fraud_count": fraud_in_bucket,
            "legitimate_count": legit_in_bucket,
            "fraud_rate": (fraud_in_bucket / n_in_bucket) if n_in_bucket > 0 else float("nan"),
        }

    fraud_in_review = buckets["REVIEW"]["fraud_count"]
    fraud_in_block = buckets["BLOCK"]["fraud_count"]
    fraud_in_approve = buckets["APPROVE"]["fraud_count"]
    legit_in_review = buckets["REVIEW"]["legitimate_count"]
    legit_in_block = buckets["BLOCK"]["legitimate_count"]

    precision_review = (fraud_in_review / buckets["REVIEW"]["count"]) if buckets["REVIEW"]["count"] > 0 else float("nan")
    precision_block = (fraud_in_block / buckets["BLOCK"]["count"]) if buckets["BLOCK"]["count"] > 0 else float("nan")

    return {
        "policy_name": policy.name,
        "approve_threshold": policy.approve_threshold,
        "block_threshold": policy.block_threshold,
        "n_transactions_total": n,
        "total_fraud_cases": total_fraud,
        "total_legitimate_cases": total_legit,
        "buckets": buckets,
        "fraud_captured_review_plus_block": fraud_in_review + fraud_in_block,
        "fraud_captured_block_alone": fraud_in_block,
        "fraud_missed_in_approve": fraud_in_approve,
        "fraud_recall_review_plus_block": (
            (fraud_in_review + fraud_in_block) / total_fraud if total_fraud > 0 else float("nan")
        ),
        "precision_among_reviewed": precision_review,
        "precision_among_blocked": precision_block,
        "pct_legitimate_sent_to_review": (legit_in_review / total_legit) if total_legit > 0 else float("nan"),
        "pct_legitimate_blocked": (legit_in_block / total_legit) if total_legit > 0 else float("nan"),
        "pct_legitimate_intervened": (
            (legit_in_review + legit_in_block) / total_legit if total_legit > 0 else float("nan")
        ),
        "legitimate_transactions_reviewed": legit_in_review,
        "legitimate_transactions_blocked": legit_in_block,
    }
