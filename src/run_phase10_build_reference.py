"""
Phase 10: Build and persist the real drift-reference profile from the
approved VALIDATION partition, reusing every existing pipeline piece
unchanged (Phase 1 FeaturePipeline, Phase 4 behavioral features, Phase 4/5
LightGBMPreprocessor + trained model, Phase 5 frozen decision policy).

Run once (or whenever the approved model/policy legitimately changes);
downstream drift analysis only ever reads the saved JSON artifact, never
the raw dataset.

Usage:
    python -m src.run_phase10_build_reference
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import yaml

from src.data.load import load_train_transaction, load_config
from src.data.split import compute_temporal_split
from src.features.behavioral import compute_behavioral_features, get_behavioral_feature_names
from src.decision.policy import DecisionPolicy
from src.run_phase6_simulation import build_or_load_bundle
from src.drift.reference import (
    build_reference_profile, save_reference_profile,
    MONITORED_NUMERIC_FEATURES, MONITORED_CATEGORICAL_FEATURES,
)

REFERENCE_PROFILE_PATH = REPO_ROOT / "artifacts" / "drift" / "reference_profile.json"


def main() -> int:
    config = load_config()
    id_col, time_col, target_col, entity_col = "TransactionID", "TransactionDT", "isFraud", "card1"

    print("Step 1/5: Load/build model bundle (reused unchanged from Phase 6)")
    bundle = build_or_load_bundle(config)
    schema, fp, lgbm_pre, model = bundle["schema"], bundle["feature_pipeline"], bundle["lgbm_preprocessor"], bundle["model"]
    bhv_names = bundle["bhv_names"]

    print("Step 2/5: Load + temporal split (identical boundaries to all prior phases)")
    df, _ = load_train_transaction(config=config)
    train_df, val_df, test_df, _ = compute_temporal_split(
        df, time_col, id_col, config["split"]["train_ratio"], config["split"]["val_ratio"], config["split"]["test_ratio"],
    )
    del df
    gc.collect()
    print(f"  train: {train_df.shape}, val: {val_df.shape} (reference partition), test: {test_df.shape}")

    print("Step 3/5: Compute behavioral features (Phase 4, unchanged) for the validation partition")
    combined_raw = pd.concat([
        train_df[[id_col, time_col, entity_col, "TransactionAmt"]],
        val_df[[id_col, time_col, entity_col, "TransactionAmt"]],
        test_df[[id_col, time_col, entity_col, "TransactionAmt"]],
    ], ignore_index=True)
    bhv_features = compute_behavioral_features(combined_raw, entity_col=entity_col, time_col=time_col, id_col=id_col, amount_col="TransactionAmt")
    bhv_indexed = bhv_features.set_index(id_col)
    del combined_raw
    gc.collect()

    print("Step 4/5: Compute model risk scores + frozen-policy decisions on validation (Phase 4/5, unchanged)")
    X_val = fp.transform(val_df)
    Z_val = lgbm_pre.transform(X_val)
    val_bhv = bhv_indexed.loc[val_df[id_col].values, bhv_names].reset_index(drop=True)
    Z_val_full = pd.concat([Z_val.reset_index(drop=True), val_bhv], axis=1)
    val_scores = model.predict_proba(Z_val_full)[:, 1]

    with open(REPO_ROOT / "config" / "decision_policy.yaml") as f:
        policy_config = yaml.safe_load(f)
    policy = DecisionPolicy(
        approve_threshold=policy_config["approve_threshold"],
        block_threshold=policy_config["block_threshold"],
        name=policy_config["policy_name"],
    )
    val_decisions = policy.decide(val_scores)
    print(f"  val risk score mean={val_scores.mean():.4f}, decisions={pd.Series(val_decisions).value_counts().to_dict()}")

    print("Step 5/5: Assemble reference dataframe (monitored columns only) and build + save the profile")
    monitored_raw_cols = [c for c in (MONITORED_NUMERIC_FEATURES + MONITORED_CATEGORICAL_FEATURES) if c in val_df.columns]
    reference_df = val_df[monitored_raw_cols].reset_index(drop=True).copy()
    for f in bhv_names:
        if f in val_bhv.columns:
            reference_df[f] = val_bhv[f].values

    profile = build_reference_profile(reference_df, val_scores, val_decisions, reference_partition_name="validation")
    save_reference_profile(profile, REFERENCE_PROFILE_PATH)
    print(f"  Saved reference profile to: {REFERENCE_PROFILE_PATH}")
    print(f"  n_reference_transactions={profile['metadata']['n_reference_transactions']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
