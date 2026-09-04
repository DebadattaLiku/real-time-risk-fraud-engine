"""
Phase 11: Champion vs. candidate evaluation.

Reuses the project's already-approved evaluation functions directly —
`compute_ranking_metrics` (Phase 2A), `compute_threshold_metrics` (Phase
2A), and `evaluate_policy` (Phase 5) — rather than reimplementing metric
computation. This module's only job is to run those same functions once
for the champion's scores and once for the candidate's scores (on the
SAME labeled dataset) and structure the results side by side.

## Dataset discipline

`compare_champion_vs_candidate` takes an explicit `dataset_label`
parameter and requires the caller to say which partition the scores came
from. This project's established rule (Phase 0 onward) is that the final
TEST partition is evaluated once, not used as a repeated tuning signal —
this module does not enforce that by code (it cannot know how many times
a given score array has been computed), but it DOES make the dataset
explicit in every report so the discipline is auditable rather than
implicit, and `src/run_phase11_register_champion.py` /
`scripts/demo_model_governance.py` both use VALIDATION for gate decisions,
reporting test-set numbers only informationally.
"""

from __future__ import annotations

from src.evaluation.metrics import compute_ranking_metrics, compute_threshold_metrics
from src.decision.policy import DecisionPolicy
from src.decision.evaluation import evaluate_policy


def evaluate_model_scores(y_true, scores, policy: DecisionPolicy | None = None, threshold: float | None = None) -> dict:
    """One model's full evaluation bundle on one labeled dataset."""
    result = {"ranking": compute_ranking_metrics(y_true, scores)}
    if threshold is not None:
        result["threshold_metrics"] = compute_threshold_metrics(y_true, scores, threshold)
    if policy is not None:
        result["policy_evaluation"] = evaluate_policy(y_true, scores, policy)
    return result


def compare_champion_vs_candidate(
    champion_eval: dict,
    candidate_eval: dict,
    dataset_label: str,
) -> dict:
    """
    `champion_eval`/`candidate_eval` are outputs of `evaluate_model_scores`
    on the SAME labeled dataset (say so explicitly via `dataset_label`,
    e.g. "validation" or "test — final evaluation only").
    """
    champ_ranking = champion_eval["ranking"]
    cand_ranking = candidate_eval["ranking"]

    comparison = {
        "dataset_label": dataset_label,
        "ranking_metrics": {
            "champion": champ_ranking,
            "candidate": cand_ranking,
            "delta_pr_auc": cand_ranking["pr_auc"] - champ_ranking["pr_auc"],
            "delta_roc_auc": cand_ranking["roc_auc"] - champ_ranking["roc_auc"],
        },
    }

    if "policy_evaluation" in champion_eval and "policy_evaluation" in candidate_eval:
        champ_pol = champion_eval["policy_evaluation"]
        cand_pol = candidate_eval["policy_evaluation"]
        comparison["policy_metrics"] = {
            "champion": {
                "decision_distribution": {k: v["pct_of_total"] for k, v in champ_pol["buckets"].items()},
                "fraud_recall_review_plus_block": champ_pol["fraud_recall_review_plus_block"],
                "precision_among_blocked": champ_pol["precision_among_blocked"],
                "pct_legitimate_blocked": champ_pol["pct_legitimate_blocked"],
            },
            "candidate": {
                "decision_distribution": {k: v["pct_of_total"] for k, v in cand_pol["buckets"].items()},
                "fraud_recall_review_plus_block": cand_pol["fraud_recall_review_plus_block"],
                "precision_among_blocked": cand_pol["precision_among_blocked"],
                "pct_legitimate_blocked": cand_pol["pct_legitimate_blocked"],
            },
            "delta_fraud_recall": cand_pol["fraud_recall_review_plus_block"] - champ_pol["fraud_recall_review_plus_block"],
            "delta_pct_legitimate_blocked": cand_pol["pct_legitimate_blocked"] - champ_pol["pct_legitimate_blocked"],
        }

    return comparison
