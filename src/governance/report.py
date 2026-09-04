"""
Phase 11: Champion/challenger governance report generation.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.governance.evaluation import compare_champion_vs_candidate
from src.governance.compatibility import run_compatibility_checks
from src.governance.gates import run_all_gates, aggregate_gate_results


def build_governance_report(
    champion_metadata: dict,
    candidate_metadata: dict,
    champion_eval: dict,
    candidate_eval: dict,
    dataset_label: str,
) -> dict:
    comparison = compare_champion_vs_candidate(champion_eval, candidate_eval, dataset_label)
    compatibility_checks = run_compatibility_checks(champion_metadata, candidate_metadata)
    gate_results = run_all_gates(champion_metadata, candidate_metadata, champion_eval, candidate_eval)
    aggregate = aggregate_gate_results(gate_results)

    return {
        "report_generated_at": datetime.now(timezone.utc).isoformat(),
        "champion": {
            "model_id": champion_metadata["model_id"],
            "model_version": champion_metadata["model_version"],
            "status": champion_metadata["status"],
        },
        "candidate": {
            "model_id": candidate_metadata["model_id"],
            "model_version": candidate_metadata["model_version"],
            "status": candidate_metadata["status"],
        },
        "evaluation_context": {
            "dataset_label": dataset_label,
            "feature_schema_version_champion": champion_metadata["feature_schema_version"],
            "feature_schema_version_candidate": candidate_metadata["feature_schema_version"],
        },
        "metrics_comparison": comparison,
        "compatibility_checks": compatibility_checks,
        "promotion_gates": gate_results,
        "recommendation": aggregate["recommendation"],
        "recommendation_explanation": aggregate["explanation"],
        "failed_gates": aggregate["failed_gates"],
        "review_gates": aggregate["review_gates"],
        "passed_gates": aggregate["passed_gates"],
    }


def save_governance_report(report: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)


def render_markdown_summary(report: dict) -> str:
    lines = [
        f"# Champion vs. Candidate Governance Report",
        "",
        f"Generated: {report['report_generated_at']}",
        "",
        f"- **Champion**: `{report['champion']['model_id']}` ({report['champion']['status']})",
        f"- **Candidate**: `{report['candidate']['model_id']}` ({report['candidate']['status']})",
        f"- **Evaluation dataset**: {report['evaluation_context']['dataset_label']}",
        "",
        f"## Recommendation: **{report['recommendation']}**",
        "",
        report["recommendation_explanation"],
        "",
        "## Promotion Gates",
        "",
        "| Gate | Result | Explanation |",
        "|---|---|---|",
    ]
    for g in report["promotion_gates"]:
        lines.append(f"| {g['gate']} | {g['result']} | {g['explanation']} |")

    rm = report["metrics_comparison"]["ranking_metrics"]
    lines += [
        "",
        "## Ranking Metrics",
        "",
        "| Metric | Champion | Candidate | Delta |",
        "|---|---|---|---|",
        f"| PR-AUC | {rm['champion']['pr_auc']:.4f} | {rm['candidate']['pr_auc']:.4f} | {rm['delta_pr_auc']:+.4f} |",
        f"| ROC-AUC | {rm['champion']['roc_auc']:.4f} | {rm['candidate']['roc_auc']:.4f} | {rm['delta_roc_auc']:+.4f} |",
    ]
    return "\n".join(lines)
