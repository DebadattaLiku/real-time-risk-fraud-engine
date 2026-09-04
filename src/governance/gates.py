"""
Phase 11: Promotion gates.

Six explicit, individually-explainable gates. Each returns exactly one of
`PASS` / `FAIL` / `REQUIRES_REVIEW` with a human-readable explanation —
never a hidden weighted score. The final recommendation
(`aggregate_gate_results`) is a simple, stated rule:

    any gate FAILS               -> REJECT
    no FAIL, any REQUIRES_REVIEW -> REQUIRES_REVIEW
    all gates PASS                -> PROMOTE

All numeric thresholds below (`MIN_CANDIDATE_PR_AUC`,
`MAX_ACCEPTABLE_PR_AUC_DEGRADATION`, `MAX_ACCEPTABLE_LEGIT_BLOCKED_INCREASE`)
are explicitly documented as PROJECT GOVERNANCE THRESHOLDS chosen for this
repository, not claimed as universal fraud-industry standards.
"""

from __future__ import annotations

from src.governance.compatibility import run_compatibility_checks

GATE_RESULTS = ("PASS", "FAIL", "REQUIRES_REVIEW")
RECOMMENDATIONS = ("PROMOTE", "REJECT", "REQUIRES_REVIEW")

# --- Project governance thresholds (documented, configurable) --------------
MIN_CANDIDATE_PR_AUC = 0.30
MAX_ACCEPTABLE_PR_AUC_DEGRADATION = 0.02   # candidate may be up to this much WORSE than champion on PR-AUC and still pass Gate 4
REVIEW_ZONE_PR_AUC_DEGRADATION = 0.005     # within this much worse -> REQUIRES_REVIEW rather than an outright PASS
MAX_ACCEPTABLE_LEGIT_BLOCKED_INCREASE = 0.005  # 0.5 percentage points of additional legitimate-customer blocking
# -----------------------------------------------------------------------


def gate_metric_availability(champion_eval: dict, candidate_eval: dict) -> dict:
    def _valid(ranking: dict) -> bool:
        pr = ranking.get("pr_auc")
        roc = ranking.get("roc_auc")
        return pr is not None and roc is not None and pr == pr and roc == roc  # not NaN

    ok = _valid(champion_eval["ranking"]) and _valid(candidate_eval["ranking"])
    return {
        "gate": "required_metric_availability",
        "result": "PASS" if ok else "FAIL",
        "explanation": (
            "Both champion and candidate have valid PR-AUC/ROC-AUC on the evaluation dataset."
            if ok else
            "One or both models are missing valid PR-AUC/ROC-AUC — cannot evaluate promotion without them."
        ),
    }


def gate_schema_compatibility(champion_metadata: dict, candidate_metadata: dict) -> dict:
    check = run_compatibility_checks(champion_metadata, candidate_metadata)[0]
    return {
        "gate": "feature_schema_compatibility",
        "result": "PASS" if check["compatible"] else "FAIL",
        "explanation": check["explanation"],
    }


def gate_candidate_quality(candidate_eval: dict) -> dict:
    pr_auc = candidate_eval["ranking"]["pr_auc"]
    ok = pr_auc >= MIN_CANDIDATE_PR_AUC
    return {
        "gate": "candidate_quality_threshold",
        "result": "PASS" if ok else "FAIL",
        "explanation": (
            f"Candidate PR-AUC ({pr_auc:.4f}) meets the project minimum ({MIN_CANDIDATE_PR_AUC})."
            if ok else
            f"Candidate PR-AUC ({pr_auc:.4f}) is below the project minimum ({MIN_CANDIDATE_PR_AUC}) — "
            f"too weak to be considered regardless of how it compares to the champion."
        ),
    }


def gate_no_unacceptable_degradation(champion_eval: dict, candidate_eval: dict) -> dict:
    champ_pr = champion_eval["ranking"]["pr_auc"]
    cand_pr = candidate_eval["ranking"]["pr_auc"]
    delta = cand_pr - champ_pr  # negative = candidate is worse

    # A small epsilon absorbs ordinary floating-point representation noise
    # (e.g. `0.50 - 0.02` does not evaluate to exactly `-0.02` in binary
    # floating point) so a candidate landing exactly at a configured
    # boundary is judged by the INTENDED threshold, not by which way a
    # sub-1e-9 rounding error happened to fall — caught by a real test
    # failure at the exact boundary before this fix.
    _EPS = 1e-9

    if delta >= -REVIEW_ZONE_PR_AUC_DEGRADATION - _EPS:
        result = "PASS"
        explanation = (
            f"Candidate PR-AUC ({cand_pr:.4f}) is at or above the champion's ({champ_pr:.4f}), "
            f"or within the negligible {REVIEW_ZONE_PR_AUC_DEGRADATION} review-free tolerance."
        )
    elif delta >= -MAX_ACCEPTABLE_PR_AUC_DEGRADATION - _EPS:
        result = "REQUIRES_REVIEW"
        explanation = (
            f"Candidate PR-AUC ({cand_pr:.4f}) is {abs(delta):.4f} below the champion's ({champ_pr:.4f}) — "
            f"within the {MAX_ACCEPTABLE_PR_AUC_DEGRADATION} maximum tolerated degradation, but not "
            f"negligible; a human should confirm this tradeoff is acceptable (e.g. for a gain elsewhere)."
        )
    else:
        result = "FAIL"
        explanation = (
            f"Candidate PR-AUC ({cand_pr:.4f}) is {abs(delta):.4f} below the champion's ({champ_pr:.4f}), "
            f"exceeding the {MAX_ACCEPTABLE_PR_AUC_DEGRADATION} maximum tolerated degradation."
        )
    return {"gate": "no_unacceptable_degradation", "result": result, "explanation": explanation,
            "delta_pr_auc": delta}


def gate_operational_compatibility(champion_eval: dict, candidate_eval: dict) -> dict:
    if "policy_evaluation" not in champion_eval or "policy_evaluation" not in candidate_eval:
        return {
            "gate": "operational_compatibility",
            "result": "REQUIRES_REVIEW",
            "explanation": "No decision-policy evaluation was supplied for one or both models — "
                           "cannot assess operational/customer-friction impact automatically.",
        }
    champ_blocked = champion_eval["policy_evaluation"]["pct_legitimate_blocked"]
    cand_blocked = candidate_eval["policy_evaluation"]["pct_legitimate_blocked"]
    delta = cand_blocked - champ_blocked

    ok = delta <= MAX_ACCEPTABLE_LEGIT_BLOCKED_INCREASE
    return {
        "gate": "operational_compatibility",
        "result": "PASS" if ok else "REQUIRES_REVIEW",
        "explanation": (
            f"Legitimate-customer blocked rate changes by {delta:+.4%} under the candidate — "
            f"within the {MAX_ACCEPTABLE_LEGIT_BLOCKED_INCREASE:.2%} tolerance."
            if ok else
            f"Legitimate-customer blocked rate increases by {delta:+.4%} under the candidate, "
            f"exceeding the {MAX_ACCEPTABLE_LEGIT_BLOCKED_INCREASE:.2%} tolerance — a human should "
            f"confirm the added friction is acceptable before promotion."
        ),
        "delta_pct_legitimate_blocked": delta,
    }


def gate_decision_policy_compatibility(champion_metadata: dict, candidate_metadata: dict) -> dict:
    check = run_compatibility_checks(champion_metadata, candidate_metadata)[1]
    return {
        "gate": "decision_policy_compatibility",
        "result": "PASS" if check["compatible"] else "REQUIRES_REVIEW",
        "explanation": check["explanation"],
    }


def run_all_gates(
    champion_metadata: dict, candidate_metadata: dict,
    champion_eval: dict, candidate_eval: dict,
) -> list:
    return [
        gate_metric_availability(champion_eval, candidate_eval),
        gate_schema_compatibility(champion_metadata, candidate_metadata),
        gate_candidate_quality(candidate_eval),
        gate_no_unacceptable_degradation(champion_eval, candidate_eval),
        gate_operational_compatibility(champion_eval, candidate_eval),
        gate_decision_policy_compatibility(champion_metadata, candidate_metadata),
    ]


def aggregate_gate_results(gate_results: list) -> dict:
    results = [g["result"] for g in gate_results]
    failed = [g["gate"] for g in gate_results if g["result"] == "FAIL"]
    review = [g["gate"] for g in gate_results if g["result"] == "REQUIRES_REVIEW"]

    if failed:
        recommendation = "REJECT"
        explanation = f"Rejected: failed gate(s): {', '.join(failed)}."
    elif review:
        recommendation = "REQUIRES_REVIEW"
        explanation = f"Requires human review: gate(s) needing attention: {', '.join(review)}."
    else:
        recommendation = "PROMOTE"
        explanation = "All promotion gates passed."

    return {
        "recommendation": recommendation,
        "explanation": explanation,
        "failed_gates": failed,
        "review_gates": review,
        "passed_gates": [g["gate"] for g in gate_results if g["result"] == "PASS"],
    }
