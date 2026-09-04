"""
Phase 5: Validation-calibrated APPROVE/REVIEW/BLOCK decision policy, built
on top of the Phase 4 selected model (transaction-level + behavioral
features, LightGBM, standard/unweighted).

Because Phase 4 did not persist the trained model object, this script
retrains Model B with IDENTICAL hyperparameters/features/split/seed and
checks it against the saved Phase 4 metrics for exact reproducibility
before trusting anything downstream — the same discipline used in Phase 3
and Phase 4 to compare against unsaved prior models.

Protocol enforced:
    Train      -> fit LightGBMPreprocessor AND the model
    Validation -> ALL policy design (candidate construction, selection)
    Test       -> the frozen, selected policy applied ONCE, unchanged

Usage:
    python -m src.run_phase5_decision_policy
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
import yaml

from src.data.load import load_train_transaction, load_config, DataValidationError
from src.data.split import compute_temporal_split, verify_split
from src.features.schema import build_feature_schema
from src.features.pipeline import FeaturePipeline
from src.features.behavioral import compute_behavioral_features, get_behavioral_feature_names
from src.models.lightgbm_preprocessing import LightGBMPreprocessor
from src.models.lightgbm_model import train_lightgbm
from src.evaluation.metrics import compute_ranking_metrics
from src.decision.policy import DecisionPolicy
from src.decision.evaluation import evaluate_policy
from src.decision.threshold_selection import build_policy_from_validation
from src.decision.comparison import compare_policy_to_fixed_budget

# Three candidate operating policies, per the brief. Each pair
# (review_capacity, block_precision_target) is a documented, explicit
# experimental scenario — not a claimed universal recommendation.
CANDIDATE_POLICIES = {
    "conservative": dict(review_capacity=0.01, block_precision_target=0.98),
    "balanced": dict(review_capacity=0.02, block_precision_target=0.90),
    "aggressive": dict(review_capacity=0.05, block_precision_target=0.70),
}


def attach_behavioral(Z: pd.DataFrame, ids: pd.Series, bhv_indexed: pd.DataFrame, bhv_names: list) -> pd.DataFrame:
    bhv_aligned = bhv_indexed.loc[ids.values, bhv_names].reset_index(drop=True)
    out = pd.concat([Z.reset_index(drop=True), bhv_aligned], axis=1)
    assert len(out) == len(Z)
    return out


def main() -> int:
    config = load_config()
    id_col = config["split"]["id_column"]
    time_col = config["split"]["time_column"]
    target_col = config["split"]["target_column"]
    entity_col = "card1"

    print("=" * 70)
    print("Step 1/7: Load + split + Phase 4 model reproduction")
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

    combined_raw = pd.concat([
        train_df[[id_col, time_col, entity_col, "TransactionAmt"]],
        val_df[[id_col, time_col, entity_col, "TransactionAmt"]],
        test_df[[id_col, time_col, entity_col, "TransactionAmt"]],
    ], ignore_index=True)
    bhv_features = compute_behavioral_features(
        combined_raw, entity_col=entity_col, time_col=time_col, id_col=id_col, amount_col="TransactionAmt",
    )
    bhv_names = get_behavioral_feature_names(bhv_features, id_col=id_col)
    bhv_indexed = bhv_features.set_index(id_col)
    del combined_raw
    gc.collect()

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

    lgbm_pre = LightGBMPreprocessor(schema)
    Z_train = lgbm_pre.fit_transform(X_train)
    Z_val = lgbm_pre.transform(X_val)
    Z_test = lgbm_pre.transform(X_test)
    del X_train, X_val, X_test
    gc.collect()

    Z_train_B = attach_behavioral(Z_train, train_ids, bhv_indexed, bhv_names)
    Z_val_B = attach_behavioral(Z_val, val_ids, bhv_indexed, bhv_names)
    Z_test_B = attach_behavioral(Z_test, test_ids, bhv_indexed, bhv_names)
    del Z_train, Z_val, Z_test
    gc.collect()
    print(f"  Z_train_B: {Z_train_B.shape} (391 transaction + {len(bhv_names)} behavioral)")

    model, info = train_lightgbm(
        Z_train_B, y_train, Z_val_B, y_val,
        categorical_features=lgbm_pre.get_categorical_feature_names(), scale_pos_weight=None,
    )
    val_scores = model.predict_proba(Z_val_B)[:, 1]
    test_scores = model.predict_proba(Z_test_B)[:, 1]
    val_pr_auc = compute_ranking_metrics(y_val, val_scores)["pr_auc"]
    test_pr_auc = compute_ranking_metrics(y_test, test_scores)["pr_auc"]
    print(f"  Reproduced Model B: val PR-AUC={val_pr_auc:.4f}, test PR-AUC={test_pr_auc:.4f}")

    phase4_path = REPO_ROOT / config["paths"]["interim_dir"] / "phase4_metrics.json"
    reproducibility_check = {"checked": False}
    if phase4_path.is_file():
        with open(phase4_path) as f:
            phase4 = json.load(f)
        orig = phase4["model_b_behavioral"]
        val_match = abs(orig["val_ranking"]["pr_auc"] - val_pr_auc) < 1e-9
        test_match = abs(orig["test_ranking"]["pr_auc"] - test_pr_auc) < 1e-9
        reproducibility_check = {
            "checked": True, "val_match": val_match, "test_match": test_match,
            "original_val_pr_auc": orig["val_ranking"]["pr_auc"], "reproduced_val_pr_auc": val_pr_auc,
            "original_test_pr_auc": orig["test_ranking"]["pr_auc"], "reproduced_test_pr_auc": test_pr_auc,
        }
        print(f"  Reproducibility vs Phase 4: val_match={val_match}, test_match={test_match}")
        if not (val_match and test_match):
            print("  WARNING: Model B does not exactly match the saved Phase 4 benchmark.")

    print("\n" + "=" * 70)
    print("Step 2/7: Build candidate policies from VALIDATION data only")
    print("=" * 70)
    candidates = {}
    for name, params in CANDIDATE_POLICIES.items():
        policy, diag = build_policy_from_validation(
            y_val, val_scores, review_capacity=params["review_capacity"],
            block_precision_target=params["block_precision_target"], name=name,
        )
        val_eval = evaluate_policy(y_val, val_scores, policy)
        candidates[name] = {"policy": policy, "diagnostics": diag, "val_eval": val_eval}
        print(f"  [{name}] approve_thr={policy.approve_threshold:.4f}, block_thr={policy.block_threshold:.4f}")
        print(f"    val: APPROVE={val_eval['buckets']['APPROVE']['pct_of_total']:.4%}, "
              f"REVIEW={val_eval['buckets']['REVIEW']['pct_of_total']:.4%}, "
              f"BLOCK={val_eval['buckets']['BLOCK']['pct_of_total']:.4%}")
        print(f"    val: fraud recall(REVIEW+BLOCK)={val_eval['fraud_recall_review_plus_block']:.4f}, "
              f"precision(BLOCK)={val_eval['precision_among_blocked']:.4f}, "
              f"legit blocked%={val_eval['pct_legitimate_blocked']:.4%}, "
              f"legit reviewed%={val_eval['pct_legitimate_sent_to_review']:.4%}")

    print("\n" + "=" * 70)
    print("Step 3/7: Select preferred policy (VALIDATION ONLY)")
    print("=" * 70)
    for name, c in candidates.items():
        ve = c["val_eval"]
        print(f"  {name}: recall={ve['fraud_recall_review_plus_block']:.4f}, "
              f"block_precision={ve['precision_among_blocked']:.4f}, "
              f"legit_blocked={ve['pct_legitimate_blocked']:.4%}")

    selected_name = "balanced"
    selected_policy = candidates[selected_name]["policy"]
    print(f"\n  Selected policy: '{selected_name}' "
          f"(approve_thr={selected_policy.approve_threshold:.4f}, block_thr={selected_policy.block_threshold:.4f})")
    print("  Rationale: balances fraud capture against legitimate-customer "
          "friction better than 'conservative' (which captures less fraud) "
          "and imposes less friction than 'aggressive' (which blocks/reviews "
          "far more legitimate transactions for a comparatively small "
          "additional recall gain) — see full validation comparison in the report.")

    print("\n" + "=" * 70)
    print("Step 4/7: Freeze and save policy configuration")
    print("=" * 70)
    policy_config = {
        "policy_name": selected_policy.name,
        "approve_threshold": selected_policy.approve_threshold,
        "block_threshold": selected_policy.block_threshold,
        "selection_criteria": (
            "Selected on VALIDATION data only: balances fraud recall, "
            "BLOCK-bucket precision, and legitimate-customer friction "
            "(pct legitimate blocked/reviewed) across three candidate "
            "operating policies (conservative/balanced/aggressive)."
        ),
        "review_capacity_assumption": CANDIDATE_POLICIES[selected_name]["review_capacity"],
        "block_precision_target": CANDIDATE_POLICIES[selected_name]["block_precision_target"],
        "candidate_policies_considered": {
            name: {
                "approve_threshold": c["policy"].approve_threshold,
                "block_threshold": c["policy"].block_threshold,
                "review_capacity_target": CANDIDATE_POLICIES[name]["review_capacity"],
                "block_precision_target": CANDIDATE_POLICIES[name]["block_precision_target"],
            }
            for name, c in candidates.items()
        },
        "note": (
            "These thresholds are modeling-policy decisions calibrated on "
            "historical validation data, not guaranteed real-world "
            "financial or legal rules. See reports/phase5_decision_policy_summary.md."
        ),
    }
    policy_config_path = REPO_ROOT / "config" / "decision_policy.yaml"
    with open(policy_config_path, "w") as f:
        yaml.safe_dump(policy_config, f, default_flow_style=False, sort_keys=False)
    print(f"  Saved frozen policy to: {policy_config_path}")

    print("\n" + "=" * 70)
    print("Step 5/7: Apply frozen policy to UNTOUCHED test data")
    print("=" * 70)
    test_eval = evaluate_policy(y_test, test_scores, selected_policy)
    print(f"  test: APPROVE={test_eval['buckets']['APPROVE']['pct_of_total']:.4%}, "
          f"REVIEW={test_eval['buckets']['REVIEW']['pct_of_total']:.4%}, "
          f"BLOCK={test_eval['buckets']['BLOCK']['pct_of_total']:.4%}")
    print(f"  test: fraud recall(REVIEW+BLOCK)={test_eval['fraud_recall_review_plus_block']:.4f}, "
          f"precision(BLOCK)={test_eval['precision_among_blocked']:.4f}, "
          f"precision(REVIEW)={test_eval['precision_among_reviewed']:.4f}")
    print(f"  test: legit blocked={test_eval['legitimate_transactions_blocked']} "
          f"({test_eval['pct_legitimate_blocked']:.4%}), "
          f"legit reviewed={test_eval['legitimate_transactions_reviewed']} "
          f"({test_eval['pct_legitimate_sent_to_review']:.4%})")

    print("\n" + "=" * 70)
    print("Step 6/7: Compare vs simple fixed-budget ranking (matched intervention)")
    print("=" * 70)
    comparison_val = compare_policy_to_fixed_budget(y_val, val_scores, candidates[selected_name]["val_eval"])
    comparison_test = compare_policy_to_fixed_budget(y_test, test_scores, test_eval)
    print(f"  val: three-way recall={comparison_val['three_way_policy']['fraud_recall']:.4f} vs "
          f"simple recall={comparison_val['simple_fixed_budget_ranking']['fraud_recall']:.4f}")
    print(f"  test: three-way recall={comparison_test['three_way_policy']['fraud_recall']:.4f} vs "
          f"simple recall={comparison_test['simple_fixed_budget_ranking']['fraud_recall']:.4f}")

    print("\n" + "=" * 70)
    print("Step 7/7: Diagnostics, figures, metrics artifact")
    print("=" * 70)
    figures_dir = REPO_ROOT / config["paths"]["figures_dir"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_test_arr = np.asarray(y_test)
    test_decisions = selected_policy.decide(test_scores)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"APPROVE": "#55A868", "REVIEW": "#DD8452", "BLOCK": "#C44E52"}
    for d in ("APPROVE", "REVIEW", "BLOCK"):
        mask = test_decisions == d
        ax.hist(test_scores[mask], bins=60, alpha=0.6, label=f"{d} (n={mask.sum()})", color=colors[d])
    ax.axvline(selected_policy.approve_threshold, color="gray", linestyle="--", linewidth=1)
    ax.axvline(selected_policy.block_threshold, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("LightGBM fraud probability")
    ax.set_ylabel("Count")
    ax.set_title(f"Risk-Score Distribution by Decision Bucket (Test, policy='{selected_name}')")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "phase5_score_distribution_by_decision.png", dpi=120)
    plt.close(fig)

    n_bins = 20
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(test_scores, bin_edges) - 1, 0, n_bins - 1)
    fraud_rate_by_bin = np.array([
        y_test_arr[bin_idx == i].mean() if (bin_idx == i).sum() > 0 else np.nan
        for i in range(n_bins)
    ])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(bin_edges[:-1] + 0.025, fraud_rate_by_bin, marker="o")
    ax.axvline(selected_policy.approve_threshold, color="gray", linestyle="--", linewidth=1, label="approve_threshold")
    ax.axvline(selected_policy.block_threshold, color="black", linestyle="--", linewidth=1, label="block_threshold")
    ax.set_xlabel("LightGBM fraud probability (bucketed)")
    ax.set_ylabel("Actual fraud rate")
    ax.set_title("Fraud Rate by Risk-Score Region (Test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "phase5_fraud_rate_by_score_region.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    names = ["APPROVE", "REVIEW", "BLOCK"]
    counts = [test_eval["buckets"][d]["count"] for d in names]
    ax.bar(names, counts, color=[colors[d] for d in names])
    for i, c in enumerate(counts):
        ax.text(i, c, f"{c:,}\n({c/len(y_test_arr):.2%})", ha="center", va="bottom")
    ax.set_ylabel("Transaction count")
    ax.set_title(f"Decision Distribution (Test, policy='{selected_name}')")
    fig.tight_layout()
    fig.savefig(figures_dir / "phase5_decision_distribution.png", dpi=120)
    plt.close(fig)

    print(f"  Saved figures under: {figures_dir}")

    interim_dir = REPO_ROOT / config["paths"]["interim_dir"]
    metrics_path = interim_dir / "phase5_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "reproducibility_check": reproducibility_check,
            "candidate_policies": {
                name: {
                    "diagnostics": c["diagnostics"],
                    "val_eval": {k: v for k, v in c["val_eval"].items()},
                }
                for name, c in candidates.items()
            },
            "selected_policy_name": selected_name,
            "policy_config": policy_config,
            "test_eval": test_eval,
            "comparison_val": comparison_val,
            "comparison_test": comparison_test,
        }, f, indent=2, default=str)
    print(f"  Saved metrics: {metrics_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
