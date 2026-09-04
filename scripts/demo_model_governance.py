#!/usr/bin/env python3
"""
Phase 11 demonstration script — model lifecycle & governance walkthrough.

    Current Champion (real, Phase 4/5 model)
        -> Register Candidate (MOCK — see below)
        -> Evaluate Candidate (real validation labels, mock candidate scores)
        -> Apply Promotion Gates
        -> Generate Recommendation
        -> Explicit Promotion Example
        -> Rollback Example

**Important**: the "candidate" in this demonstration is a MOCK artifact —
synthetic scores derived from the real champion's real scores, NOT a
second trained fraud model. This exercises the governance MECHANICS
(registration, comparison, gates, promotion, rollback) honestly, without
claiming any synthetic score array is a better real fraud detector. Every
print statement below labels this explicitly.

Uses its own throwaway registry file (NOT the real
`artifacts/models/registry.json` used by the API) so this demo's
promote/rollback actions never affect the actual governed state.

Does not retrain the real model. Does not modify the frozen decision
policy.

Usage (from the repository root):

    python scripts/demo_model_governance.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import yaml

from src.data.load import load_train_transaction, load_config
from src.data.split import compute_temporal_split
from src.features.behavioral import compute_behavioral_features, get_behavioral_feature_names
from src.run_phase6_simulation import build_or_load_bundle
from src.decision.policy import DecisionPolicy
from src.governance.registry import ModelRegistry
from src.governance.metadata import build_model_metadata, compute_artifact_sha256
from src.governance.evaluation import evaluate_model_scores
from src.governance.report import build_governance_report, render_markdown_summary, save_governance_report

DEMO_REGISTRY_PATH = REPO_ROOT / "artifacts" / "models" / "demo_registry.json"
CHAMPION_MODEL_ID = "fraud-risk-lightgbm-v1"


def main() -> int:
    print("=" * 70)
    print("Phase 11 model governance demonstration")
    print("(uses a THROWAWAY demo registry — does not touch the real one)")
    print("=" * 70)

    if DEMO_REGISTRY_PATH.is_file():
        DEMO_REGISTRY_PATH.unlink()  # start clean each run
    registry = ModelRegistry(DEMO_REGISTRY_PATH)

    print("\nStep 1: Register the current (real) champion...")
    config = load_config()
    bundle = build_or_load_bundle(config)
    fp, lgbm_pre, model = bundle["feature_pipeline"], bundle["lgbm_preprocessor"], bundle["model"]
    bhv_names = bundle["bhv_names"]

    with open(REPO_ROOT / "config" / "decision_policy.yaml") as f:
        pc = yaml.safe_load(f)
    policy = DecisionPolicy(approve_threshold=pc["approve_threshold"], block_threshold=pc["block_threshold"], name=pc["policy_name"])

    id_col, time_col, entity_col, target_col = "TransactionID", "TransactionDT", "card1", "isFraud"
    df, _ = load_train_transaction(config=config)
    train_df, val_df, test_df, _ = compute_temporal_split(
        df, time_col, id_col, config["split"]["train_ratio"], config["split"]["val_ratio"], config["split"]["test_ratio"],
    )
    del df

    combined_raw = pd.concat([
        train_df[[id_col, time_col, entity_col, "TransactionAmt"]],
        val_df[[id_col, time_col, entity_col, "TransactionAmt"]],
        test_df[[id_col, time_col, entity_col, "TransactionAmt"]],
    ], ignore_index=True)
    bhv_features = compute_behavioral_features(combined_raw, entity_col=entity_col, time_col=time_col, id_col=id_col, amount_col="TransactionAmt")
    bhv_indexed = bhv_features.set_index(id_col)

    X_val = fp.transform(val_df)
    Z_val = lgbm_pre.transform(X_val)
    val_bhv = bhv_indexed.loc[val_df[id_col].values, bhv_names].reset_index(drop=True)
    Z_val_full = pd.concat([Z_val.reset_index(drop=True), val_bhv], axis=1)
    y_val = val_df[target_col].reset_index(drop=True)
    champion_val_scores = model.predict_proba(Z_val_full)[:, 1]

    champion_metadata = build_model_metadata(
        model_id=CHAMPION_MODEL_ID, model_version="v1", model_type="LightGBM (real, Phase 4/5)",
        status="CHAMPION", feature_schema_version="phase1-v1",
        training_data_reference="TRAIN partition (real)", validation_data_reference="VALIDATION partition (real)",
        evaluation_metrics={"val_pr_auc": float(pd.Series(champion_val_scores).mean())},
        artifact_reference="data/interim/phase6_model_bundle.pkl",
        artifact_sha256=compute_artifact_sha256(REPO_ROOT / "data" / "interim" / "phase6_model_bundle.pkl"),
        decision_policy_version=pc["policy_name"],
        notes="REAL champion model — the actual approved Phase 4/5 LightGBM model.",
    )
    registry.register_champion(champion_metadata)
    print(f"  Registered '{CHAMPION_MODEL_ID}' as CHAMPION in the demo registry.")

    print("\nStep 2: Register a MOCK candidate (SYNTHETIC scores, NOT a real trained model)...")
    print("  >>> The 'candidate' below is champion scores + synthetic noise/blend.")
    print("  >>> This demonstrates governance MECHANICS only — it is NOT a claim")
    print("  >>> that any real competing fraud model was trained or is better.")
    rng = np.random.default_rng(42)

    # MOCK candidate A: champion scores + noise -> expected to be slightly WORSE.
    mock_worse_scores = np.clip(champion_val_scores + rng.normal(0, 0.05, size=len(champion_val_scores)), 0, 1)
    # MOCK candidate B: a synthetic "improved" blend (partially informed by
    # true labels — an explicit, labeled SYNTHETIC construction, not a
    # real modeling technique) -> expected to be slightly BETTER, purely
    # to demonstrate the PROMOTE path mechanically.
    mock_better_scores = np.clip(0.85 * champion_val_scores + 0.15 * y_val.to_numpy(), 0, 1)

    candidate_worse_metadata = build_model_metadata(
        model_id="mock-candidate-worse-v1", model_version="v1", model_type="MOCK (synthetic, demonstration only)",
        status="CANDIDATE", feature_schema_version="phase1-v1",
        training_data_reference="N/A — synthetic demo artifact, not trained",
        validation_data_reference="VALIDATION partition (real labels, synthetic scores)",
        evaluation_metrics={},
        artifact_reference="N/A (no real artifact — synthetic scores only)",
        decision_policy_version=pc["policy_name"],
        notes="MOCK candidate for demonstration only: champion scores + Gaussian noise. NOT a real fraud model.",
    )
    candidate_better_metadata = build_model_metadata(
        model_id="mock-candidate-better-v1", model_version="v1", model_type="MOCK (synthetic, demonstration only)",
        status="CANDIDATE", feature_schema_version="phase1-v1",
        training_data_reference="N/A — synthetic demo artifact, not trained",
        validation_data_reference="VALIDATION partition (real labels, synthetic scores)",
        evaluation_metrics={},
        artifact_reference="N/A (no real artifact — synthetic scores only)",
        decision_policy_version=pc["policy_name"],
        notes="MOCK candidate for demonstration only: a synthetic label-informed blend. NOT a real fraud model.",
    )
    registry.register_candidate(candidate_worse_metadata)
    registry.register_candidate(candidate_better_metadata)
    print(f"  Registered 2 MOCK candidates as CANDIDATE.")

    print("\nStep 3: Evaluate champion vs. each candidate on REAL validation labels...")
    champion_eval = evaluate_model_scores(y_val, champion_val_scores, policy=policy)
    worse_eval = evaluate_model_scores(y_val, mock_worse_scores, policy=policy)
    better_eval = evaluate_model_scores(y_val, mock_better_scores, policy=policy)
    print(f"  Champion val PR-AUC:        {champion_eval['ranking']['pr_auc']:.4f}")
    print(f"  Mock 'worse' candidate PR-AUC: {worse_eval['ranking']['pr_auc']:.4f}")
    print(f"  Mock 'better' candidate PR-AUC: {better_eval['ranking']['pr_auc']:.4f}")

    print("\nStep 4: Apply promotion gates + generate governance reports...")
    report_worse = build_governance_report(
        champion_metadata, candidate_worse_metadata, champion_eval, worse_eval, dataset_label="validation",
    )
    report_better = build_governance_report(
        champion_metadata, candidate_better_metadata, champion_eval, better_eval, dataset_label="validation",
    )

    print(f"\n  --- Mock 'worse' candidate: {report_worse['recommendation']} ---")
    print(f"  {report_worse['recommendation_explanation']}")
    for g in report_worse["promotion_gates"]:
        print(f"    [{g['result']:16s}] {g['gate']}: {g['explanation']}")

    print(f"\n  --- Mock 'better' candidate: {report_better['recommendation']} ---")
    print(f"  {report_better['recommendation_explanation']}")
    for g in report_better["promotion_gates"]:
        print(f"    [{g['result']:16s}] {g['gate']}: {g['explanation']}")

    reports_dir = REPO_ROOT / "reports" / "model_governance"
    save_governance_report(report_worse, reports_dir / "champion_vs_mock_worse_candidate_report.json")
    save_governance_report(report_better, reports_dir / "champion_vs_mock_better_candidate_report.json")
    (reports_dir / "champion_vs_mock_better_candidate_report.md").write_text(render_markdown_summary(report_better))
    print(f"\n  Saved reports under: {reports_dir}")

    print("\n  Confirming evaluation/reporting alone did NOT change the champion:")
    assert registry.get_champion()["model_id"] == CHAMPION_MODEL_ID
    print(f"  Champion is still: {registry.get_champion()['model_id']} (correct — reports never auto-promote)")

    print("\nStep 5: Explicit promotion (only if the recommendation supports it)...")
    if report_better["recommendation"] == "PROMOTE":
        registry.promote(
            "mock-candidate-better-v1", approved_by="demo-script (simulated human reviewer)",
            reason="Demonstration: mock candidate passed all promotion gates.",
        )
        print(f"  PROMOTED 'mock-candidate-better-v1' to CHAMPION (demo registry only).")
        print(f"  New champion: {registry.get_champion()['model_id']}")
        print(f"  Previous champion status: {registry.get_model(CHAMPION_MODEL_ID)['status']}")
    else:
        print(f"  Skipping promotion — recommendation was '{report_better['recommendation']}', not PROMOTE.")

    print("\nStep 6: Rollback demonstration...")
    if registry.get_champion()["model_id"] != CHAMPION_MODEL_ID:
        registry.rollback(
            CHAMPION_MODEL_ID, approved_by="demo-script (simulated human reviewer)",
            reason="Demonstration: rolling back to the real champion after the mock promotion.",
        )
        print(f"  ROLLED BACK to '{CHAMPION_MODEL_ID}'.")
        print(f"  Current champion: {registry.get_champion()['model_id']}")
    else:
        print("  (No promotion occurred above, so no rollback is needed for this run.)")

    registry.save()
    print(f"\nFinal demo registry summary:")
    print(json.dumps(registry.summary(), indent=2))

    print("\n" + "=" * 70)
    print("Demonstration complete. The REAL registry (artifacts/models/registry.json) "
          "was never touched by this script. Model was NOT retrained; decision "
          "policy was NOT modified.")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
