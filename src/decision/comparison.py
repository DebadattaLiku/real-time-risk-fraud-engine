"""
Phase 5: Compare the three-way APPROVE/REVIEW/BLOCK policy against the
simpler strategy: "top K% highest-risk transactions -> REVIEW (manual
queue), everything else -> APPROVE" — i.e. exactly what Phases 2A-4 already
evaluated via `fixed_budget_report`.

Fairness rule: the simple ranking's budget K is set equal to the policy's
TOTAL intervention rate (REVIEW + BLOCK combined) — the same number of
transactions get flagged for some form of attention either way. This is
the only apples-to-apples comparison; comparing against a different budget
would not answer "does the three-way structure add anything beyond
ranking," it would just compare two different budgets.
"""

from __future__ import annotations

from src.evaluation.operational_eval import fixed_budget_report


def compare_policy_to_fixed_budget(y_true, scores, policy_eval: dict) -> dict:
    """
    `policy_eval` is the output of `evaluate_policy(...)` for the SAME
    (y_true, scores). Builds the matched-budget simple-ranking comparison
    and returns both results side by side plus the delta.
    """
    total_intervention_fraction = (
        policy_eval["buckets"]["REVIEW"]["pct_of_total"] + policy_eval["buckets"]["BLOCK"]["pct_of_total"]
    )
    simple_ranking = fixed_budget_report(y_true, scores, total_intervention_fraction)

    three_way_recall = policy_eval["fraud_recall_review_plus_block"]
    simple_recall = simple_ranking["fraud_recall"]

    return {
        "matched_intervention_fraction": total_intervention_fraction,
        "three_way_policy": {
            "policy_name": policy_eval["policy_name"],
            "n_intervened": policy_eval["buckets"]["REVIEW"]["count"] + policy_eval["buckets"]["BLOCK"]["count"],
            "fraud_captured": policy_eval["fraud_captured_review_plus_block"],
            "fraud_recall": three_way_recall,
            "precision_among_blocked": policy_eval["precision_among_blocked"],
            "precision_among_reviewed": policy_eval["precision_among_reviewed"],
        },
        "simple_fixed_budget_ranking": {
            "n_intervened": simple_ranking["n_transactions_reviewed"],
            "fraud_captured": simple_ranking["fraud_cases_captured"],
            "fraud_recall": simple_recall,
            "precision_among_flagged": simple_ranking["precision_among_reviewed"],
        },
        "fraud_recall_delta_three_way_minus_simple": (
            three_way_recall - simple_recall if not (three_way_recall != three_way_recall or simple_recall != simple_recall) else float("nan")
        ),
        "note": (
            "The three-way policy's key structural difference from simple "
            "ranking is not raw recall at the matched budget (both flag the "
            "same NUMBER of transactions, so recall is necessarily very "
            "similar or identical) — it is that the same intervention "
            "budget is split into a high-confidence automated BLOCK tier "
            "and a lower-confidence manual REVIEW tier, which a flat ranked "
            "list does not distinguish. See the Phase 5 report for the "
            "operational interpretation."
        ),
    }
