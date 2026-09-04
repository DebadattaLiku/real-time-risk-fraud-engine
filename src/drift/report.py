"""
Phase 10: Drift report aggregation.

The overall batch severity is the MAXIMUM severity across every monitored
feature and output ("worst-feature-wins") — a simple, fully explainable
rule, not a black-box weighted score. The explanation names exactly which
features/outputs are at that maximum severity level, so a person reading
the report can immediately see what's driving it.
"""

from __future__ import annotations

from src.drift.detectors import SEVERITY_ORDER


def aggregate_overall_severity(named_results: list) -> dict:
    """
    `named_results`: list of dicts, each with at least `name` and
    `severity` keys (severity one of `SEVERITY_ORDER`).
    """
    if not named_results:
        return {
            "severity": "NO_SIGNIFICANT_DRIFT",
            "explanation": "No monitored features or outputs were present in this batch.",
            "contributing": [],
        }

    max_idx = max(SEVERITY_ORDER.index(r["severity"]) for r in named_results)
    overall_severity = SEVERITY_ORDER[max_idx]
    contributing = [r["name"] for r in named_results if r["severity"] == overall_severity]

    if overall_severity == "NO_SIGNIFICANT_DRIFT":
        explanation = "No monitored feature or output showed significant drift from the reference distribution."
    else:
        label = overall_severity.replace("_", " ").title()
        explanation = f"{label} driven primarily by: " + ", ".join(contributing)

    return {"severity": overall_severity, "explanation": explanation, "contributing": contributing}


def build_drift_report(
    batch_size: int,
    reference_metadata: dict,
    feature_results: list,
    risk_score_result: dict | None,
    decision_result: dict | None,
    low_confidence: bool,
    warnings: list,
    timestamp: str,
) -> dict:
    named_results = [
        {"name": r["feature"], "severity": r["severity"]} for r in feature_results
    ]
    if risk_score_result is not None:
        named_results.append({"name": "risk_score", "severity": risk_score_result["severity"]})
    if decision_result is not None:
        named_results.append({"name": "decision_distribution", "severity": decision_result["severity"]})

    overall = aggregate_overall_severity(named_results)

    return {
        "timestamp": timestamp,
        "batch_size": batch_size,
        "reference_version": reference_metadata.get("reference_version"),
        "reference_partition": reference_metadata.get("reference_partition"),
        "low_confidence": low_confidence,
        "warnings": list(warnings),
        "feature_results": feature_results,
        "risk_score_result": risk_score_result,
        "decision_result": decision_result,
        "overall_severity": overall["severity"],
        "overall_explanation": overall["explanation"],
        "contributing_signals": overall["contributing"],
    }
