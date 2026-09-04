#!/usr/bin/env python3
"""
Phase 10 demonstration script — drift monitoring walkthrough.

    Load the approved reference profile (built from real VALIDATION data)
        -> Scenario A: stable batch (sampled from real TEST data)          -> expect NO_SIGNIFICANT_DRIFT / LOW_DRIFT
        -> Scenario B: transaction-amount shift (synthetic)                -> expect amount drift detected
        -> Scenario C: categorical (ProductCD) shift (synthetic)           -> expect categorical drift detected
        -> Scenario D: risk-score shift (synthetic)                       -> expect output drift detected

Scenarios B/C/D are EXPLICITLY CONTROLLED/SYNTHETIC — real transaction data
in this project does not naturally exhibit dramatic drift within a single
182-day window, so these scenarios construct deliberate distributional
shifts to demonstrate the detector actually works, clearly labeled as such
throughout (never presented as naturally observed behavior).

Does not retrain the model. Does not modify the decision policy.

Usage (from the repository root):

    python scripts/demo_drift_monitoring.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from src.data.load import load_train_transaction, load_config
from src.data.split import compute_temporal_split
from src.features.behavioral import compute_behavioral_features, get_behavioral_feature_names
from src.run_phase6_simulation import build_or_load_bundle
from src.run_phase10_build_reference import REFERENCE_PROFILE_PATH
from src.drift.reference import load_reference_profile, MONITORED_NUMERIC_FEATURES, MONITORED_CATEGORICAL_FEATURES
from src.drift.monitor import DriftMonitor


def _print_report(title: str, report: dict) -> None:
    print(f"\n--- {title} ---")
    print(f"  batch_size={report['batch_size']}  low_confidence={report['low_confidence']}")
    if report["warnings"]:
        for w in report["warnings"]:
            print(f"  WARNING: {w}")
    print(f"  OVERALL: {report['overall_severity']} — {report['overall_explanation']}")
    print("  feature-level results:")
    for r in report["feature_results"]:
        print(f"    {r['feature']:24s} [{r['type']:11s}] psi={r['psi']:.4f}  severity={r['severity']}")
    if report["risk_score_result"]:
        rr = report["risk_score_result"]
        print(f"    {'risk_score':24s} [{'output':11s}] psi={rr['psi']:.4f}  severity={rr['severity']}  "
              f"(ref_mean={rr['reference_mean']:.4f}, cur_mean={rr['current_mean']:.4f})")
    if report["decision_result"]:
        dr = report["decision_result"]
        print(f"    {'decision_distribution':24s} [{'output':11s}] psi={dr['psi']:.4f}  severity={dr['severity']}")


def main() -> int:
    print("=" * 70)
    print("Phase 10 drift monitoring demonstration")
    print("=" * 70)

    if not REFERENCE_PROFILE_PATH.is_file():
        print(f"\nReference profile not found at {REFERENCE_PROFILE_PATH}.")
        print("Run: python -m src.run_phase10_build_reference")
        return 1

    print(f"\nLoading approved reference profile from {REFERENCE_PROFILE_PATH} "
          f"(built from real VALIDATION data)...")
    profile = load_reference_profile(REFERENCE_PROFILE_PATH)
    print(f"  reference_partition={profile['metadata']['reference_partition']}, "
          f"n_reference_transactions={profile['metadata']['n_reference_transactions']}")
    monitor = DriftMonitor(profile)

    print("\nLoading real model bundle + a slice of real TEST-partition data "
          "(for Scenario A only — NOT used anywhere else in this project as a reference)...")
    config = load_config()
    bundle = build_or_load_bundle(config)
    fp, lgbm_pre, model = bundle["feature_pipeline"], bundle["lgbm_preprocessor"], bundle["model"]
    bhv_names = bundle["bhv_names"]

    id_col, time_col, entity_col = "TransactionID", "TransactionDT", "card1"
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

    sample = test_df.iloc[:500].copy()
    X_sample = fp.transform(sample)
    Z_sample = lgbm_pre.transform(X_sample)
    sample_bhv = bhv_indexed.loc[sample[id_col].values, bhv_names].reset_index(drop=True)
    Z_sample_full = pd.concat([Z_sample.reset_index(drop=True), sample_bhv], axis=1)
    sample_scores = model.predict_proba(Z_sample_full)[:, 1]
    from src.decision.policy import DecisionPolicy
    import yaml
    with open(REPO_ROOT / "config" / "decision_policy.yaml") as f:
        pc = yaml.safe_load(f)
    policy = DecisionPolicy(approve_threshold=pc["approve_threshold"], block_threshold=pc["block_threshold"], name=pc["policy_name"])
    sample_decisions = policy.decide(sample_scores)

    monitored_raw_cols = [c for c in (MONITORED_NUMERIC_FEATURES + MONITORED_CATEGORICAL_FEATURES) if c in sample.columns]
    sample_batch = sample[monitored_raw_cols].reset_index(drop=True).copy()
    for f in bhv_names:
        if f in sample_bhv.columns:
            sample_batch[f] = sample_bhv[f].values

    # --- Scenario A: stable — sampled from the SAME partition (validation)
    # the reference profile itself was built from, per the brief's own
    # definition ("current batch is sampled similarly to reference"), NOT
    # from the test partition. This distinction matters concretely here:
    # Phase 4's behavioral features (bhv_prev_txn_count, bhv_hist_mean_amt,
    # etc.) are CUMULATIVE counters that grow over the dataset's timeline
    # by construction — test-partition entities have structurally seen
    # MORE prior history than validation-partition entities did at their
    # point in time, simply because more of the timeline precedes them.
    # Comparing reference (validation) against test would therefore show
    # real, measurable behavioral-feature "drift" that reflects the
    # chronological split design, not a genuine change in transaction
    # behavior — an important, honestly-reported finding in its own right
    # (see the Phase 10 report), but not what "Scenario A: stable" is
    # supposed to demonstrate.
    rng_a = np.random.default_rng(2026)
    stable_idx = rng_a.choice(len(val_df), size=500, replace=False)
    stable_sample = val_df.iloc[stable_idx].copy()
    X_stable = fp.transform(stable_sample)
    Z_stable = lgbm_pre.transform(X_stable)
    stable_bhv = bhv_indexed.loc[stable_sample[id_col].values, bhv_names].reset_index(drop=True)
    Z_stable_full = pd.concat([Z_stable.reset_index(drop=True), stable_bhv], axis=1)
    stable_scores = model.predict_proba(Z_stable_full)[:, 1]
    from src.decision.policy import DecisionPolicy
    import yaml
    with open(REPO_ROOT / "config" / "decision_policy.yaml") as f:
        pc = yaml.safe_load(f)
    policy = DecisionPolicy(approve_threshold=pc["approve_threshold"], block_threshold=pc["block_threshold"], name=pc["policy_name"])
    stable_decisions = policy.decide(stable_scores)

    stable_batch = stable_sample[monitored_raw_cols].reset_index(drop=True).copy()
    for f in bhv_names:
        if f in stable_bhv.columns:
            stable_batch[f] = stable_bhv[f].values

    report_a = monitor.analyze_batch(stable_batch, risk_scores=stable_scores, decisions=stable_decisions)
    _print_report("Scenario A — Stable (random sample FROM the reference/validation partition itself)", report_a)

    print("\n(For comparison/honest reporting only — NOT part of the four "
          "required scenarios — a real TEST-partition sample shows "
          "measurable behavioral-feature 'drift' against the validation "
          "reference, for the structural reason explained above, not a "
          "bug in the detector:)")
    reference_vs_test_report = monitor.analyze_batch(sample_batch, risk_scores=sample_scores, decisions=sample_decisions)
    _print_report("(Informational) Reference (validation) vs. real TEST-partition sample", reference_vs_test_report)

    # --- Scenario B: transaction-amount shift (CONTROLLED/SYNTHETIC) ---
    batch_b = stable_batch.copy()
    batch_b["TransactionAmt"] = batch_b["TransactionAmt"] * 15.0 + 500.0  # deliberate, large synthetic shift
    report_b = monitor.analyze_batch(batch_b, risk_scores=stable_scores, decisions=stable_decisions)
    _print_report("Scenario B — CONTROLLED/SYNTHETIC transaction-amount shift (15x + 500)", report_b)

    # --- Scenario C: categorical (ProductCD) shift (CONTROLLED/SYNTHETIC) ---
    batch_c = stable_batch.copy()
    batch_c["ProductCD"] = "C"  # force every transaction to a single category
    report_c = monitor.analyze_batch(batch_c, risk_scores=stable_scores, decisions=stable_decisions)
    _print_report("Scenario C — CONTROLLED/SYNTHETIC categorical shift (ProductCD forced to 'C')", report_c)

    # --- Scenario D: risk-score shift (CONTROLLED/SYNTHETIC) ---
    rng = np.random.default_rng(0)
    synthetic_high_scores = rng.uniform(0.7, 1.0, size=len(stable_batch))
    synthetic_decisions = policy.decide(synthetic_high_scores)
    report_d = monitor.analyze_batch(stable_batch, risk_scores=synthetic_high_scores, decisions=synthetic_decisions)
    _print_report("Scenario D — CONTROLLED/SYNTHETIC risk-score shift (uniform[0.7, 1.0])", report_d)

    print("\n" + "=" * 70)
    print("Drift monitor summary (GET /drift/summary equivalent):")
    summary = monitor.summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 70)
    assert report_a["overall_severity"] in ("NO_SIGNIFICANT_DRIFT", "LOW_DRIFT"), "Scenario A should be stable"
    assert report_b["overall_severity"] in ("MODERATE_DRIFT", "HIGH_DRIFT"), "Scenario B should detect drift"
    assert report_c["overall_severity"] in ("MODERATE_DRIFT", "HIGH_DRIFT"), "Scenario C should detect drift"
    assert report_d["overall_severity"] in ("MODERATE_DRIFT", "HIGH_DRIFT"), "Scenario D should detect drift"
    print("All scenario expectations confirmed. Model was NOT retrained; "
          "decision policy was NOT modified.")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
