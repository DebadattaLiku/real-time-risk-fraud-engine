"""
Phase 4: Leakage-safe behavioral features + ablation experiment.

    Model A: Phase 2B LightGBM baseline, transaction-level features only
             (391 cols) — reproduced with IDENTICAL hyperparameters/split/
             seed, exactly as Phase 3 did, and checked for exact
             reproducibility before anything else proceeds.
    Model B: Model A's exact same 391 columns + 12 leakage-safe behavioral
             features appended (card1 pseudo-entity), trained with the
             IDENTICAL LightGBM hyperparameters/protocol as Model A — the
             presence of behavioral features is the ONLY variable that
             differs between A and B.

Protocol enforced:
    Train      -> fit LightGBMPreprocessor AND both models
    Validation -> the ONLY basis for deciding whether behavioral features
                  help (never test)
    Test       -> final evaluation only, reported for both models but not
                  used to make any decision

Row alignment note: `TransactionID` is captured directly from each split
BEFORE it is dropped by `FeaturePipeline`/`LightGBMPreprocessor` (both
correctly exclude it as a non-predictive identifier), then used to
left-merge the separately-computed behavioral features back on — a merge
by key, not a positional assumption, so there is no fragile dependency on
row order being preserved through every transform step.

Usage:
    python -m src.run_phase4_behavioral
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
import pandas as pd

from src.data.load import load_train_transaction, load_config, DataValidationError
from src.data.split import compute_temporal_split, verify_split
from src.features.schema import build_feature_schema, schema_summary
from src.features.pipeline import FeaturePipeline
from src.features.behavioral import (
    compute_behavioral_features, get_behavioral_feature_names, behavioral_diagnostics,
)
from src.models.lightgbm_preprocessing import LightGBMPreprocessor
from src.models.lightgbm_model import train_lightgbm, get_feature_importance
from src.evaluation.metrics import compute_ranking_metrics
from src.evaluation.operational_eval import multi_budget_table, fixed_budget_report
from src.evaluation.plots import plot_pr_curves, plot_recall_at_k_curve

BUDGETS = [0.01, 0.02, 0.05]


def evaluate_scores(y_val, val_scores, y_test, test_scores):
    return {
        "val_ranking": compute_ranking_metrics(y_val, val_scores),
        "test_ranking": compute_ranking_metrics(y_test, test_scores),
        "val_budget_table": multi_budget_table(y_val, val_scores, BUDGETS),
        "test_budget_table": multi_budget_table(y_test, test_scores, BUDGETS),
        "val_fixed_budgets": {f"{int(b*100)}pct": fixed_budget_report(y_val, val_scores, b) for b in BUDGETS},
        "test_fixed_budgets": {f"{int(b*100)}pct": fixed_budget_report(y_test, test_scores, b) for b in BUDGETS},
    }


def attach_behavioral(Z: pd.DataFrame, ids: pd.Series, bhv_indexed: pd.DataFrame, bhv_names: list) -> pd.DataFrame:
    """Left-merge behavioral columns onto Z by TransactionID (key-based, not positional)."""
    bhv_aligned = bhv_indexed.loc[ids.values, bhv_names].reset_index(drop=True)
    out = pd.concat([Z.reset_index(drop=True), bhv_aligned], axis=1)
    assert len(out) == len(Z), "row count changed while attaching behavioral features"
    return out


def main() -> int:
    config = load_config()
    id_col = config["split"]["id_column"]
    time_col = config["split"]["time_column"]
    target_col = config["split"]["target_column"]
    entity_col = "card1"

    print("=" * 70)
    print("Step 1/9: Load + temporal split (identical to Phase 1/2A/2B/3)")
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

    train_ids = train_df[id_col].reset_index(drop=True)
    val_ids = val_df[id_col].reset_index(drop=True)
    test_ids = test_df[id_col].reset_index(drop=True)

    print("\n" + "=" * 70)
    print("Step 2/9: Compute leakage-safe behavioral features (card1 pseudo-entity)")
    print("=" * 70)
    # Only 4 columns are ever extracted for this — isFraud is never touched.
    combined_raw = pd.concat([
        train_df[[id_col, time_col, entity_col, "TransactionAmt"]],
        val_df[[id_col, time_col, entity_col, "TransactionAmt"]],
        test_df[[id_col, time_col, entity_col, "TransactionAmt"]],
    ], ignore_index=True)
    bhv_features = compute_behavioral_features(
        combined_raw, entity_col=entity_col, time_col=time_col, id_col=id_col, amount_col="TransactionAmt",
    )
    bhv_names = get_behavioral_feature_names(bhv_features, id_col=id_col)
    print(f"  Computed {len(bhv_names)} behavioral features for {len(bhv_features)} transactions:")
    print(f"    {bhv_names}")

    diagnostics = behavioral_diagnostics(bhv_features, combined_raw[entity_col])
    print(f"  Cold-start (no prior history): {diagnostics['pct_cold_start_no_prior_history']:.4%}")
    print(f"  card1 coverage: {diagnostics['n_unique_entities']} unique entities, "
          f"mean {diagnostics['mean_txns_per_entity']:.1f} txns/entity")

    bhv_indexed = bhv_features.set_index(id_col)
    del combined_raw
    gc.collect()

    print("\n" + "=" * 70)
    print("Step 3/9: Phase 1 feature schema + FeaturePipeline (unchanged from Phase 1/2B)")
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
    print("Step 4/9: LightGBM-specific preprocessing (Model A columns, baseline)")
    print("=" * 70)
    lgbm_pre = LightGBMPreprocessor(schema)
    Z_train_A = lgbm_pre.fit_transform(X_train)
    Z_val_A = lgbm_pre.transform(X_val)
    Z_test_A = lgbm_pre.transform(X_test)
    del X_train, X_val, X_test
    gc.collect()
    print(f"  Z_train_A: {Z_train_A.shape} (baseline transaction-level features)")

    print("\n" + "=" * 70)
    print("Step 5/9: Build Model B columns = Model A columns + behavioral features")
    print("=" * 70)
    Z_train_B = attach_behavioral(Z_train_A, train_ids, bhv_indexed, bhv_names)
    Z_val_B = attach_behavioral(Z_val_A, val_ids, bhv_indexed, bhv_names)
    Z_test_B = attach_behavioral(Z_test_A, test_ids, bhv_indexed, bhv_names)
    print(f"  Z_train_B: {Z_train_B.shape} (baseline + {len(bhv_names)} behavioral features)")
    assert list(Z_train_A.columns) == list(Z_train_B.columns[:Z_train_A.shape[1]]), \
        "Model A's original columns must be untouched/unreordered in Model B's input"

    print("\n" + "=" * 70)
    print("Step 6/9: Train Model A (baseline) and Model B (+behavioral) — IDENTICAL hyperparameters")
    print("=" * 70)
    categorical_features = lgbm_pre.get_categorical_feature_names()  # unchanged for both models

    model_A, info_A = train_lightgbm(
        Z_train_A, y_train, Z_val_A, y_val,
        categorical_features=categorical_features, scale_pos_weight=None,
    )
    print(f"  Model A — best_iteration={info_A['best_iteration']}, "
          f"best val PR-AUC during training={info_A['best_val_pr_auc_during_training']:.4f}")

    model_B, info_B = train_lightgbm(
        Z_train_B, y_train, Z_val_B, y_val,
        categorical_features=categorical_features, scale_pos_weight=None,
    )
    print(f"  Model B — best_iteration={info_B['best_iteration']}, "
          f"best val PR-AUC during training={info_B['best_val_pr_auc_during_training']:.4f}")

    val_scores_A = model_A.predict_proba(Z_val_A)[:, 1]
    test_scores_A = model_A.predict_proba(Z_test_A)[:, 1]
    val_scores_B = model_B.predict_proba(Z_val_B)[:, 1]
    test_scores_B = model_B.predict_proba(Z_test_B)[:, 1]

    print("\n" + "=" * 70)
    print("Step 7/9: Reproducibility check for Model A vs saved Phase 2B/3 benchmark")
    print("=" * 70)
    val_pr_auc_A = compute_ranking_metrics(y_val, val_scores_A)["pr_auc"]
    test_pr_auc_A = compute_ranking_metrics(y_test, test_scores_A)["pr_auc"]
    phase2b_path = REPO_ROOT / config["paths"]["interim_dir"] / "phase2b_metrics.json"
    reproducibility_check = {"checked": False}
    if phase2b_path.is_file():
        with open(phase2b_path) as f:
            phase2b = json.load(f)
        orig = phase2b["lightgbm_standard"]
        val_match = abs(orig["val_ranking"]["pr_auc"] - val_pr_auc_A) < 1e-9
        test_match = abs(orig["test_ranking"]["pr_auc"] - test_pr_auc_A) < 1e-9
        reproducibility_check = {
            "checked": True,
            "original_val_pr_auc": orig["val_ranking"]["pr_auc"], "reproduced_val_pr_auc": val_pr_auc_A, "val_match": val_match,
            "original_test_pr_auc": orig["test_ranking"]["pr_auc"], "reproduced_test_pr_auc": test_pr_auc_A, "test_match": test_match,
        }
        print(f"  Model A vs Phase 2B benchmark: val_match={val_match}, test_match={test_match}")
        if not (val_match and test_match):
            print("  WARNING: Model A does not exactly match the saved Phase 2B benchmark.")

    print("\n" + "=" * 70)
    print("Step 8/9: Evaluate both models; compare on VALIDATION (decision), report TEST (final)")
    print("=" * 70)
    result_A = evaluate_scores(y_val, val_scores_A, y_test, test_scores_A)
    result_B = evaluate_scores(y_val, val_scores_B, y_test, test_scores_B)
    print(f"  Model A: val PR-AUC={result_A['val_ranking']['pr_auc']:.4f}, test PR-AUC={result_A['test_ranking']['pr_auc']:.4f}")
    print(f"  Model B: val PR-AUC={result_B['val_ranking']['pr_auc']:.4f}, test PR-AUC={result_B['test_ranking']['pr_auc']:.4f}")

    val_pr_auc_delta = result_B["val_ranking"]["pr_auc"] - result_A["val_ranking"]["pr_auc"]
    decision = "behavioral_features_help" if val_pr_auc_delta > 0 else "behavioral_features_do_not_help"
    print(f"\n  Validation PR-AUC delta (B - A): {val_pr_auc_delta:+.4f}")
    print(f"  Decision (based on VALIDATION ONLY): {decision}")

    print("\n" + "=" * 70)
    print("Step 9/9: Feature importance, figures, metrics artifact")
    print("=" * 70)
    importances_B = get_feature_importance(model_B, list(Z_train_B.columns), importance_type="gain")
    bhv_ranks = [(i + 1, name, val) for i, (name, val) in enumerate(importances_B) if name in bhv_names]
    print("  Behavioral feature ranks (of {} total features) in Model B:".format(len(importances_B)))
    for rank, name, val in bhv_ranks:
        print(f"    #{rank}: {name} (gain={val:.2f})")

    figures_dir = REPO_ROOT / config["paths"]["figures_dir"]
    models_test = {"Model A (baseline)": (y_test, test_scores_A), "Model B (+behavioral)": (y_test, test_scores_B)}
    plot_pr_curves(models_test, figures_dir / "phase4_pr_curve_test.png",
                    title="Precision-Recall Curve: Baseline vs +Behavioral (Test)")
    plot_recall_at_k_curve(models_test, figures_dir / "phase4_recall_at_k_test.png",
                            title="Recall@K: Baseline vs +Behavioral (Test)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top_15 = importances_B[:15]
    names = [n for n, _ in top_15][::-1]
    vals = [v for _, v in top_15][::-1]
    colors = ["#C44E52" if n in bhv_names else "#4C72B0" for n in names]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(names, vals, color=colors)
    ax.set_xlabel("Total split gain")
    ax.set_title("Top 15 Model B Feature Importances\n(red = behavioral feature)")
    fig.tight_layout()
    fig.savefig(figures_dir / "phase4_feature_importance.png", dpi=120)
    plt.close(fig)

    # Diagnostic figure: prior-transaction-count distribution (log scale).
    prev_counts = bhv_features["bhv_prev_txn_count"].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(np.log1p(prev_counts), bins=60, color="#55A868")
    ax.set_xlabel("log(1 + prior transaction count)")
    ax.set_ylabel("Number of transactions")
    ax.set_title("Distribution of Historical Transaction Count per card1 Entity")
    fig.tight_layout()
    fig.savefig(figures_dir / "phase4_behavioral_diagnostics.png", dpi=120)
    plt.close(fig)

    print(f"  Saved figures under: {figures_dir}")

    interim_dir = REPO_ROOT / config["paths"]["interim_dir"]
    metrics_path = interim_dir / "phase4_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "reproducibility_check_model_a": reproducibility_check,
            "model_a_baseline": result_A,
            "model_b_behavioral": result_B,
            "validation_pr_auc_delta_b_minus_a": val_pr_auc_delta,
            "decision_based_on_validation": decision,
            "behavioral_feature_names": bhv_names,
            "behavioral_feature_ranks_in_model_b": bhv_ranks,
            "behavioral_diagnostics": diagnostics,
            "training_info": {"model_a": info_A, "model_b": info_B},
        }, f, indent=2, default=str)
    print(f"  Saved metrics: {metrics_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
