"""
Phase 3: Isolation Forest anomaly detection + complementarity analysis
against the Phase 2B LightGBM model.

Because Phase 2B did not persist the trained LightGBM model object to disk
(only its metrics), this script retrains the exact SAME selected Phase 2B
variant (standard/unweighted, identical hyperparameters, identical fixed
random_state=42, identical deterministic single-threaded training, identical
data/split) purely to obtain its prediction scores for comparison. This is
NOT a new experiment and does not change the Phase 2B benchmark — the
retrained model's validation/test PR-AUC is checked against the original
Phase 2B numbers before anything else proceeds, to confirm bit-for-bit
reproducibility rather than assuming it.

Protocol enforced:
    Train      -> fit AnomalyPreprocessor AND Isolation Forest (unsupervised,
                  no y passed)
    Validation -> select Isolation Forest configuration, derive thresholds
                  for the conditional analysis
    Test       -> final evaluation only

Usage:
    python -m src.run_phase3_anomaly
(run from the repository root so the `src` package resolves)
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.data.load import load_train_transaction, load_config, DataValidationError
from src.data.split import compute_temporal_split, verify_split
from src.features.schema import build_feature_schema, schema_summary
from src.features.pipeline import FeaturePipeline
from src.models.lightgbm_preprocessing import LightGBMPreprocessor
from src.models.lightgbm_model import train_lightgbm
from src.models.anomaly_preprocessing import AnomalyPreprocessor
from src.models.anomaly import AnomalyDetector
from src.evaluation.metrics import compute_ranking_metrics
from src.evaluation.operational_eval import multi_budget_table, fixed_budget_report
from src.evaluation.complementarity import (
    score_correlation, overlap_analysis, union_diagnostic,
    conditional_low_lgbm_high_anomaly,
)
from src.evaluation.plots import plot_pr_curves

BUDGETS = [0.01, 0.02, 0.05]

# A modest number of Isolation Forest configurations, per the brief.
ISO_CONFIGS = {
    "default": dict(n_estimators=100, max_samples="auto", max_features=1.0),
    "more_trees_larger_subsample": dict(n_estimators=200, max_samples=1024, max_features=1.0),
    "more_trees_half_features": dict(n_estimators=200, max_samples="auto", max_features=0.5),
}


def evaluate_scores(y_val, val_scores, y_test, test_scores):
    return {
        "val_ranking": compute_ranking_metrics(y_val, val_scores),
        "test_ranking": compute_ranking_metrics(y_test, test_scores),
        "val_budget_table": multi_budget_table(y_val, val_scores, BUDGETS),
        "test_budget_table": multi_budget_table(y_test, test_scores, BUDGETS),
        "val_fixed_budgets": {f"{int(b*100)}pct": fixed_budget_report(y_val, val_scores, b) for b in BUDGETS},
        "test_fixed_budgets": {f"{int(b*100)}pct": fixed_budget_report(y_test, test_scores, b) for b in BUDGETS},
    }


def main() -> int:
    config = load_config()
    id_col = config["split"]["id_column"]
    time_col = config["split"]["time_column"]
    target_col = config["split"]["target_column"]
    seed = config.get("random_seed", 42)

    print("=" * 70)
    print("Step 1/8: Load + temporal split (identical to Phase 1/2A/2B)")
    print("=" * 70)
    df, load_report = load_train_transaction(config=config)
    train_df, val_df, test_df, split_meta = compute_temporal_split(
        df, time_col, id_col,
        config["split"]["train_ratio"], config["split"]["val_ratio"], config["split"]["test_ratio"],
    )
    del df
    gc.collect()
    checks = verify_split(train_df, val_df, test_df, time_col, id_col)
    assert all(checks[k] for k in (
        "no_row_overlap_train_val", "no_row_overlap_val_test", "no_row_overlap_train_test",
        "max_train_dt_lte_min_val_dt", "max_val_dt_lte_min_test_dt", "all_ids_unique_across_splits",
    )), f"Split integrity check failed: {checks}"
    print(f"  train: {train_df.shape}, val: {val_df.shape}, test: {test_df.shape}")

    print("\n" + "=" * 70)
    print("Step 2/8: Phase 1 feature schema + FeaturePipeline")
    print("=" * 70)
    schema = build_feature_schema(train_df)
    fp = FeaturePipeline(schema)
    fp.fit(train_df)
    X_train = fp.transform(train_df)
    y_train = fp.get_target(train_df, target_col)
    X_val = fp.transform(val_df)
    y_val = fp.get_target(val_df, target_col)
    X_test = fp.transform(test_df)
    y_test = fp.get_target(test_df, target_col)
    del train_df, val_df, test_df
    gc.collect()

    print("\n" + "=" * 70)
    print("Step 3/8: Reproduce Phase 2B LightGBM (standard variant) for comparison scores")
    print("=" * 70)
    lgbm_pre = LightGBMPreprocessor(schema)
    Z_train_lgbm = lgbm_pre.fit_transform(X_train)
    Z_val_lgbm = lgbm_pre.transform(X_val)
    Z_test_lgbm = lgbm_pre.transform(X_test)
    lgbm_model, lgbm_info = train_lightgbm(
        Z_train_lgbm, y_train, Z_val_lgbm, y_val,
        categorical_features=lgbm_pre.get_categorical_feature_names(),
        scale_pos_weight=None,
    )
    lgbm_val_scores = lgbm_model.predict_proba(Z_val_lgbm)[:, 1]
    lgbm_test_scores = lgbm_model.predict_proba(Z_test_lgbm)[:, 1]
    lgbm_val_pr_auc = compute_ranking_metrics(y_val, lgbm_val_scores)["pr_auc"]
    lgbm_test_pr_auc = compute_ranking_metrics(y_test, lgbm_test_scores)["pr_auc"]
    print(f"  Reproduced LightGBM val PR-AUC={lgbm_val_pr_auc:.4f}, test PR-AUC={lgbm_test_pr_auc:.4f}")

    phase2b_path = REPO_ROOT / config["paths"]["interim_dir"] / "phase2b_metrics.json"
    reproducibility_check = {"reproduced": True}
    if phase2b_path.is_file():
        with open(phase2b_path) as f:
            phase2b = json.load(f)
        orig = phase2b["lightgbm_standard"]
        orig_val = orig["val_ranking"]["pr_auc"]
        orig_test = orig["test_ranking"]["pr_auc"]
        val_match = abs(orig_val - lgbm_val_pr_auc) < 1e-9
        test_match = abs(orig_test - lgbm_test_pr_auc) < 1e-9
        reproducibility_check = {
            "original_val_pr_auc": orig_val, "reproduced_val_pr_auc": lgbm_val_pr_auc, "val_match": val_match,
            "original_test_pr_auc": orig_test, "reproduced_test_pr_auc": lgbm_test_pr_auc, "test_match": test_match,
        }
        print(f"  Reproducibility check vs saved Phase 2B metrics: val_match={val_match}, test_match={test_match}")
        if not (val_match and test_match):
            print("  WARNING: reproduced LightGBM does not exactly match saved Phase 2B metrics.")
    del Z_train_lgbm
    gc.collect()

    print("\n" + "=" * 70)
    print("Step 4/8: Anomaly-specific preprocessing (fit on TRAIN ONLY, V columns excluded)")
    print("=" * 70)
    anomaly_pre = AnomalyPreprocessor(schema)
    Z_train_anom = anomaly_pre.fit_transform(X_train)
    Z_val_anom = anomaly_pre.transform(X_val)
    Z_test_anom = anomaly_pre.transform(X_test)
    del X_train, X_val, X_test
    gc.collect()
    print(f"  Z_train_anom: {Z_train_anom.shape} ({len(anomaly_pre.get_feature_names())} features, V columns excluded)")

    print("\n" + "=" * 70)
    print("Step 5/8: Test Isolation Forest configurations, select by VALIDATION only")
    print("=" * 70)
    config_results = {}
    for name, params in ISO_CONFIGS.items():
        detector = AnomalyDetector(random_state=seed, n_jobs=1, **params)
        detector.fit(Z_train_anom)  # unsupervised — no y passed
        val_scores = detector.anomaly_score(Z_val_anom)
        val_pr_auc = compute_ranking_metrics(y_val, val_scores)["pr_auc"]
        val_roc_auc = compute_ranking_metrics(y_val, val_scores)["roc_auc"]
        val_recall_2pct = fixed_budget_report(y_val, val_scores, 0.02)["fraud_recall"]
        config_results[name] = {
            "params": params, "val_pr_auc": val_pr_auc, "val_roc_auc": val_roc_auc,
            "val_recall_at_2pct": val_recall_2pct, "detector": detector, "val_scores": val_scores,
        }
        print(f"  Config '{name}' {params}: val PR-AUC={val_pr_auc:.4f}, "
              f"val ROC-AUC={val_roc_auc:.4f}, val Recall@2%={val_recall_2pct:.4f}")

    selected_name = max(config_results, key=lambda k: config_results[k]["val_pr_auc"])
    selected = config_results[selected_name]
    print(f"\n  Selected configuration (by validation PR-AUC): '{selected_name}'")

    print("\n" + "=" * 70)
    print("Step 6/8: Final test evaluation of selected Isolation Forest configuration")
    print("=" * 70)
    anomaly_test_scores = selected["detector"].anomaly_score(Z_test_anom)
    anomaly_val_scores = selected["val_scores"]
    anomaly_eval = evaluate_scores(y_val, anomaly_val_scores, y_test, anomaly_test_scores)
    print(f"  Isolation Forest val PR-AUC={anomaly_eval['val_ranking']['pr_auc']:.4f}, "
          f"test PR-AUC={anomaly_eval['test_ranking']['pr_auc']:.4f}")
    print(f"  LightGBM (for reference) val PR-AUC={lgbm_val_pr_auc:.4f}, test PR-AUC={lgbm_test_pr_auc:.4f}")

    print("\n" + "=" * 70)
    print("Step 7/8: Complementarity analysis (test set, primary; validation shown too)")
    print("=" * 70)
    corr_val = score_correlation(lgbm_val_scores, anomaly_val_scores)
    corr_test = score_correlation(lgbm_test_scores, anomaly_test_scores)
    print(f"  Score correlation (test): Pearson r={corr_test['pearson_r']:.4f}, "
          f"Spearman r={corr_test['spearman_r']:.4f}")

    overlap_results = {}
    union_results = {}
    for b in BUDGETS:
        overlap_results[f"{int(b*100)}pct"] = overlap_analysis(
            y_test, lgbm_test_scores, anomaly_test_scores, b, name_a="lightgbm", name_b="isolation_forest",
        )
        union_results[f"{int(b*100)}pct"] = union_diagnostic(
            y_test, lgbm_test_scores, anomaly_test_scores, b,
        )
    for b_key, r in overlap_results.items():
        print(f"  [{b_key}] both={r['fraud_captured_by_both']}, "
              f"only_lgbm={r['fraud_captured_only_by_lightgbm']}, "
              f"only_anomaly={r['fraud_captured_only_by_isolation_forest']}, "
              f"total_fraud={r['total_fraud_cases']}")

    # Conditional analysis: thresholds derived from VALIDATION ONLY.
    lgbm_low_threshold = float(np.percentile(lgbm_val_scores, 50))   # bottom half by LightGBM
    anomaly_high_threshold = float(np.percentile(anomaly_val_scores, 90))  # top 10% by anomaly score
    print(f"\n  Conditional analysis thresholds (from validation): "
          f"lgbm_low<{lgbm_low_threshold:.4f} (P50), anomaly_high>{anomaly_high_threshold:.4f} (P90)")
    conditional_val = conditional_low_lgbm_high_anomaly(
        y_val, lgbm_val_scores, anomaly_val_scores, lgbm_low_threshold, anomaly_high_threshold,
    )
    conditional_test = conditional_low_lgbm_high_anomaly(
        y_test, lgbm_test_scores, anomaly_test_scores, lgbm_low_threshold, anomaly_high_threshold,
    )
    print(f"  Conditional (val): n={conditional_val['n_transactions_matching_condition']}, "
          f"fraud={conditional_val['fraud_cases_in_condition']}, "
          f"rate={conditional_val['fraud_rate_in_condition']:.4f}, lift={conditional_val['lift_over_overall_rate']:.2f}x")
    print(f"  Conditional (test): n={conditional_test['n_transactions_matching_condition']}, "
          f"fraud={conditional_test['fraud_cases_in_condition']}, "
          f"rate={conditional_test['fraud_rate_in_condition']:.4f}, lift={conditional_test['lift_over_overall_rate']:.2f}x")

    print("\n" + "=" * 70)
    print("Step 8/8: Figures and metrics artifact")
    print("=" * 70)
    figures_dir = REPO_ROOT / config["paths"]["figures_dir"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    y_test_arr = np.asarray(y_test)
    ax.hist(anomaly_test_scores[y_test_arr == 0], bins=60, alpha=0.6, density=True, label="Non-fraud")
    ax.hist(anomaly_test_scores[y_test_arr == 1], bins=60, alpha=0.6, density=True, label="Fraud")
    ax.set_xlabel("Isolation Forest anomaly score (higher = more anomalous)")
    ax.set_ylabel("Density")
    ax.set_title("Anomaly Score Distribution by True Label (Test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "phase3_anomaly_score_distribution.png", dpi=120)
    plt.close(fig)

    plot_pr_curves(
        {"Isolation Forest": (y_test, anomaly_test_scores), "LightGBM (reference)": (y_test, lgbm_test_scores)},
        figures_dir / "phase3_pr_curve_test.png",
        title="Precision-Recall Curve: Isolation Forest vs LightGBM (Test)",
    )

    rng = np.random.default_rng(seed)
    n_sample = min(5000, len(lgbm_test_scores))
    sample_idx = rng.choice(len(lgbm_test_scores), size=n_sample, replace=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    is_fraud_sample = y_test_arr[sample_idx] == 1
    ax.scatter(lgbm_test_scores[sample_idx][~is_fraud_sample], anomaly_test_scores[sample_idx][~is_fraud_sample],
               s=4, alpha=0.3, label="Non-fraud", color="#4C72B0")
    ax.scatter(lgbm_test_scores[sample_idx][is_fraud_sample], anomaly_test_scores[sample_idx][is_fraud_sample],
               s=10, alpha=0.8, label="Fraud", color="#C44E52")
    ax.set_xlabel("LightGBM fraud probability")
    ax.set_ylabel("Isolation Forest anomaly score")
    ax.set_title(f"LightGBM vs Anomaly Score (Test, n={n_sample} sampled)\nSpearman r={corr_test['spearman_r']:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "phase3_lgbm_vs_anomaly_scatter.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    budget_keys = list(overlap_results.keys())
    both_vals = [overlap_results[k]["fraud_captured_by_both"] for k in budget_keys]
    only_lgbm_vals = [overlap_results[k]["fraud_captured_only_by_lightgbm"] for k in budget_keys]
    only_anom_vals = [overlap_results[k]["fraud_captured_only_by_isolation_forest"] for k in budget_keys]
    x = np.arange(len(budget_keys))
    width = 0.25
    ax.bar(x - width, both_vals, width, label="Captured by both")
    ax.bar(x, only_lgbm_vals, width, label="Only LightGBM")
    ax.bar(x + width, only_anom_vals, width, label="Only Isolation Forest")
    ax.set_xticks(x)
    ax.set_xticklabels(budget_keys)
    ax.set_ylabel("Fraud cases captured")
    ax.set_title("Fraud Capture Overlap at Matched Review Budgets (Test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "phase3_fraud_overlap.png", dpi=120)
    plt.close(fig)

    print(f"  Saved figures under: {figures_dir}")

    all_config_summaries = {
        name: {k: v for k, v in r.items() if k not in ("detector", "val_scores")}
        for name, r in config_results.items()
    }
    interim_dir = REPO_ROOT / config["paths"]["interim_dir"]
    metrics_path = interim_dir / "phase3_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "lightgbm_reproducibility_check": reproducibility_check,
            "isolation_forest_configs_tested": all_config_summaries,
            "selected_config": selected_name,
            "anomaly_evaluation": anomaly_eval,
            "lightgbm_reference": {
                "val_pr_auc": lgbm_val_pr_auc, "test_pr_auc": lgbm_test_pr_auc,
            },
            "score_correlation_val": corr_val,
            "score_correlation_test": corr_test,
            "overlap_analysis_test": overlap_results,
            "union_diagnostic_test": union_results,
            "conditional_analysis": {
                "thresholds": {"lgbm_low_threshold": lgbm_low_threshold, "anomaly_high_threshold": anomaly_high_threshold},
                "validation": conditional_val,
                "test": conditional_test,
            },
        }, f, indent=2)
    print(f"  Saved metrics: {metrics_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
