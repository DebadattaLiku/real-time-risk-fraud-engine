"""
Phase 2B: Train and evaluate two LightGBM variants (standard,
scale_pos_weight-adjusted) on the real IEEE-CIS training data, using the
EXACT SAME Phase 1 temporal split as Phase 2A, and compare against the
Phase 2A Logistic Regression baseline (loaded from its saved metrics —
Phase 2A is NOT rerun).

Protocol enforced:
    Train      -> fit category vocabulary AND both LightGBM models
    Validation -> early stopping, variant selection, threshold selection
    Test       -> final evaluation only, reported once

Usage:
    python -m src.run_phase2b_lightgbm
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
from src.models.lightgbm_model import train_lightgbm, get_feature_importance
from src.evaluation.metrics import (
    compute_ranking_metrics, compute_threshold_metrics, select_threshold_by_f1,
)
from src.evaluation.operational_eval import multi_budget_table, fixed_budget_report
from src.evaluation.plots import plot_pr_curves, plot_roc_curves, plot_recall_at_k_curve


BUDGETS = [0.01, 0.02, 0.05]
HEADLINE_BUDGET = 0.02


def evaluate_model(name, y_val, val_scores, y_test, test_scores, val_threshold):
    """Identical evaluation block to Phase 2A's — same functions, same protocol."""
    result = {"model": name}
    result["val_ranking"] = compute_ranking_metrics(y_val, val_scores)
    result["test_ranking"] = compute_ranking_metrics(y_test, test_scores)
    result["val_threshold_0.5"] = compute_threshold_metrics(y_val, val_scores, 0.5)
    result["test_threshold_0.5"] = compute_threshold_metrics(y_test, test_scores, 0.5)
    result["selected_threshold"] = val_threshold["threshold"]
    result["val_at_selected_threshold"] = compute_threshold_metrics(y_val, val_scores, val_threshold["threshold"])
    result["test_at_selected_threshold"] = compute_threshold_metrics(y_test, test_scores, val_threshold["threshold"])
    result["val_budget_table"] = multi_budget_table(y_val, val_scores, BUDGETS)
    result["test_budget_table"] = multi_budget_table(y_test, test_scores, BUDGETS)
    result["val_fixed_budget"] = fixed_budget_report(y_val, val_scores, HEADLINE_BUDGET)
    result["test_fixed_budget"] = fixed_budget_report(y_test, test_scores, HEADLINE_BUDGET)
    return result


def main() -> int:
    config = load_config()
    id_col = config["split"]["id_column"]
    time_col = config["split"]["time_column"]
    target_col = config["split"]["target_column"]
    seed = config.get("random_seed", 42)

    print("=" * 70)
    print("Step 1/7: Load + validate (reusing Phase 1 loader)")
    print("=" * 70)
    try:
        df, load_report = load_train_transaction(config=config)
    except (FileNotFoundError, DataValidationError) as e:
        print(f"ERROR: {e}")
        return 1
    print(f"  Loaded shape: {df.shape}")

    print("\n" + "=" * 70)
    print("Step 2/7: Temporal split (identical boundaries to Phase 1 and Phase 2A)")
    print("=" * 70)
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
    print(f"  Split integrity verified — identical to Phase 1/2A boundaries "
          f"(train_dt_max={train_df[time_col].max()}, val_dt_min={val_df[time_col].min()}, "
          f"val_dt_max={val_df[time_col].max()}, test_dt_min={test_df[time_col].min()})")

    print("\n" + "=" * 70)
    print("Step 3/7: Phase 1 feature schema + FeaturePipeline (train-only schema)")
    print("=" * 70)
    schema = build_feature_schema(train_df)
    summary = schema_summary(schema)
    print(f"  Used features: {summary['n_used']} ({summary['n_numeric_used']} numeric, "
          f"{summary['n_categorical_used']} categorical)")

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
    print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

    print("\n" + "=" * 70)
    print("Step 4/7: LightGBM-specific preprocessing (category vocab fit on TRAIN ONLY)")
    print("=" * 70)
    pre = LightGBMPreprocessor(schema)
    Z_train = pre.fit_transform(X_train)
    del X_train
    gc.collect()
    Z_val = pre.transform(X_val)
    del X_val
    gc.collect()
    Z_test = pre.transform(X_test)
    del X_test
    gc.collect()
    print(f"  Z_train: {Z_train.shape}, Z_val: {Z_val.shape}, Z_test: {Z_test.shape}")
    print(f"  Categorical features (native, not one-hot): {pre.get_categorical_feature_names()}")
    print("  Numeric NaNs preserved (no imputation) — LightGBM native missing-value handling.")
    print("  Categorical vocab: 'missing' (Phase 1 placeholder) and 'unseen' (val/test-only "
          "category values) are kept as two distinct, explicit buckets.")

    print("\n" + "=" * 70)
    print("Step 5/7: Train LightGBM A (standard) and B (scale_pos_weight)")
    print("=" * 70)
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = n_neg / n_pos
    print(f"  Train class counts: neg={n_neg}, pos={n_pos}, "
          f"scale_pos_weight (n_neg/n_pos) = {scale_pos_weight:.4f}")
    print("  Note: scale_pos_weight is NOT assumed to help — both variants are trained and "
          "compared on validation PR-AUC; the better one is selected empirically (see report).")

    model_a, info_a = train_lightgbm(
        Z_train, y_train, Z_val, y_val,
        categorical_features=pre.get_categorical_feature_names(),
        scale_pos_weight=None,
    )
    print(f"  LightGBM A (standard) — best_iteration={info_a['best_iteration']}, "
          f"best val PR-AUC during training={info_a['best_val_pr_auc_during_training']:.4f}")

    model_b, info_b = train_lightgbm(
        Z_train, y_train, Z_val, y_val,
        categorical_features=pre.get_categorical_feature_names(),
        scale_pos_weight=scale_pos_weight,
    )
    print(f"  LightGBM B (scale_pos_weight={scale_pos_weight:.2f}) — best_iteration={info_b['best_iteration']}, "
          f"best val PR-AUC during training={info_b['best_val_pr_auc_during_training']:.4f}")

    val_scores_a = model_a.predict_proba(Z_val)[:, 1]
    test_scores_a = model_a.predict_proba(Z_test)[:, 1]
    val_scores_b = model_b.predict_proba(Z_val)[:, 1]
    test_scores_b = model_b.predict_proba(Z_test)[:, 1]

    threshold_a = select_threshold_by_f1(y_val, val_scores_a)
    threshold_b = select_threshold_by_f1(y_val, val_scores_b)
    print(f"  LightGBM A selected threshold (val F1-optimal): {threshold_a['threshold']:.4f}")
    print(f"  LightGBM B selected threshold (val F1-optimal): {threshold_b['threshold']:.4f}")

    print("\n" + "=" * 70)
    print("Step 6/7: Evaluate both variants; select by VALIDATION PR-AUC only")
    print("=" * 70)
    result_a = evaluate_model("lightgbm_standard", y_val, val_scores_a, y_test, test_scores_a, threshold_a)
    result_b = evaluate_model("lightgbm_scale_pos_weight", y_val, val_scores_b, y_test, test_scores_b, threshold_b)

    print(f"  A val PR-AUC={result_a['val_ranking']['pr_auc']:.4f}  "
          f"B val PR-AUC={result_b['val_ranking']['pr_auc']:.4f}")
    print(f"  A test PR-AUC={result_a['test_ranking']['pr_auc']:.4f}  "
          f"B test PR-AUC={result_b['test_ranking']['pr_auc']:.4f}")

    selected = "lightgbm_scale_pos_weight" if result_b["val_ranking"]["pr_auc"] > result_a["val_ranking"]["pr_auc"] else "lightgbm_standard"
    selected_model = model_b if selected == "lightgbm_scale_pos_weight" else model_a
    print(f"\n  Selected variant (by validation PR-AUC): {selected}")

    print("\n" + "=" * 70)
    print("Step 7/7: Feature importance, comparison with Phase 2A, figures")
    print("=" * 70)
    importances = get_feature_importance(selected_model, list(Z_train.columns), importance_type="gain")
    top_20 = importances[:20]
    print("  Top 10 features by gain:")
    for name, val in top_20[:10]:
        print(f"    {name}: {val:.2f}")

    # --- Load Phase 2A results for comparison (NOT rerun) ---
    interim_dir = REPO_ROOT / config["paths"]["interim_dir"]
    phase2a_path = interim_dir / "phase2a_metrics.json"
    comparison = None
    if phase2a_path.is_file():
        with open(phase2a_path) as f:
            phase2a = json.load(f)
        logreg_selected = phase2a["selected"]
        logreg_result = phase2a[logreg_selected]
        comparison = {
            "logreg_selected_variant": logreg_selected,
            "logreg_val_pr_auc": logreg_result["val_ranking"]["pr_auc"],
            "logreg_test_pr_auc": logreg_result["test_ranking"]["pr_auc"],
            "logreg_test_recall_at_2pct": logreg_result["test_budget_table"][1]["recall_at_k"],
            "lightgbm_selected_variant": selected,
            "lightgbm_val_pr_auc": (result_b if selected == "lightgbm_scale_pos_weight" else result_a)["val_ranking"]["pr_auc"],
            "lightgbm_test_pr_auc": (result_b if selected == "lightgbm_scale_pos_weight" else result_a)["test_ranking"]["pr_auc"],
            "lightgbm_test_recall_at_2pct": (result_b if selected == "lightgbm_scale_pos_weight" else result_a)["test_budget_table"][1]["recall_at_k"],
        }
        print(f"\n  Comparison vs Phase 2A ({logreg_selected}):")
        print(f"    LogReg  test PR-AUC={comparison['logreg_test_pr_auc']:.4f}  "
              f"Recall@2%={comparison['logreg_test_recall_at_2pct']:.4f}")
        print(f"    LightGBM test PR-AUC={comparison['lightgbm_test_pr_auc']:.4f}  "
              f"Recall@2%={comparison['lightgbm_test_recall_at_2pct']:.4f}")
    else:
        print("  WARNING: Phase 2A metrics file not found — skipping comparison block.")

    figures_dir = REPO_ROOT / config["paths"]["figures_dir"]
    models_val = {"LightGBM Standard": (y_val, val_scores_a), "LightGBM scale_pos_weight": (y_val, val_scores_b)}
    models_test = {"LightGBM Standard": (y_test, test_scores_a), "LightGBM scale_pos_weight": (y_test, test_scores_b)}

    plot_pr_curves(models_val, figures_dir / "phase2b_pr_curve_validation.png",
                    title="LightGBM Precision-Recall Curve (Validation)")
    plot_pr_curves(models_test, figures_dir / "phase2b_pr_curve_test.png",
                    title="LightGBM Precision-Recall Curve (Test)")
    plot_roc_curves(models_test, figures_dir / "phase2b_roc_curve_test.png",
                     title="LightGBM ROC Curve (Test)")
    plot_recall_at_k_curve(models_test, figures_dir / "phase2b_recall_at_k_test.png",
                            title="LightGBM Recall@K Review-Budget Curve (Test)")

    # Feature importance figure (simple, single horizontal bar chart)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [n for n, _ in top_20][::-1]
    vals = [v for _, v in top_20][::-1]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(names, vals, color="#4C72B0")
    ax.set_xlabel("Total split gain")
    ax.set_title(f"Top 20 LightGBM Feature Importances ({selected}, by gain)")
    fig.tight_layout()
    fig.savefig(figures_dir / "phase2b_feature_importance.png", dpi=120)
    plt.close(fig)
    print(f"\n  Saved figures under: {figures_dir}")

    # --- Lightweight metrics artifact ---
    metrics_path = interim_dir / "phase2b_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "lightgbm_standard": result_a,
            "lightgbm_scale_pos_weight": result_b,
            "selected": selected,
            "training_info": {"standard": info_a, "scale_pos_weight": info_b},
            "feature_importance_top20": top_20,
            "comparison_vs_phase2a": comparison,
        }, f, indent=2)
    print(f"  Saved metrics: {metrics_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
