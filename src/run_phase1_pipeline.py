"""
Phase 1: End-to-end run of the leakage-safe data pipeline.

    load -> temporal split -> feature schema (train-only) -> feature pipeline
        (fit on train, transform train/val/test)

Produces X_train/y_train, X_validation/y_validation, X_test/y_test as
in-memory pandas objects (no huge duplicate CSVs written), plus two small
artifacts:
    data/interim/phase1_split_metadata.json
    reports/phase1_feature_schema.csv

Usage:
    python -m src.run_phase1_pipeline
(run from the repository root so the `src` package resolves)
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.load import load_train_transaction, load_config, DataValidationError
from src.data.split import compute_temporal_split, verify_split
from src.features.schema import build_feature_schema, schema_to_dataframe, schema_summary
from src.features.pipeline import FeaturePipeline


def main() -> int:
    config = load_config()
    id_col = config["split"]["id_column"]
    time_col = config["split"]["time_column"]
    target_col = config["split"]["target_column"]

    print("Step 1/5: Loading and validating train_transaction.csv ...")
    try:
        df, load_report = load_train_transaction(config=config)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1
    except DataValidationError as e:
        print(f"ERROR: {e}")
        return 1
    print(f"  Loaded shape: {df.shape}")
    print(f"  TransactionID unique: {load_report.transaction_id_unique}")
    print(f"  Target binary: {load_report.target_is_binary}, values={load_report.target_unique_values}")
    print(f"  TransactionDT valid: {load_report.transaction_dt_valid}")

    print("\nStep 2/5: Computing strict chronological train/val/test split ...")
    train_df, val_df, test_df, split_meta = compute_temporal_split(
        df, time_col, id_col,
        config["split"]["train_ratio"],
        config["split"]["val_ratio"],
        config["split"]["test_ratio"],
    )
    del df
    gc.collect()
    print(f"  train: {train_df.shape}, val: {val_df.shape}, test: {test_df.shape}")

    print("\nStep 3/5: Verifying split integrity ...")
    split_checks = verify_split(train_df, val_df, test_df, time_col, id_col)
    for k, v in split_checks.items():
        print(f"  {k}: {v}")
    critical_checks = [
        "no_row_overlap_train_val", "no_row_overlap_val_test",
        "no_row_overlap_train_test", "max_train_dt_lte_min_val_dt",
        "max_val_dt_lte_min_test_dt", "all_ids_unique_across_splits",
    ]
    if not all(split_checks[c] for c in critical_checks):
        print("ERROR: split integrity check failed — aborting before feature pipeline.")
        return 1

    print("\nStep 4/5: Building feature schema (train-only statistics) ...")
    schema = build_feature_schema(train_df)
    summary = schema_summary(schema)
    print(f"  Total columns: {summary['n_total_columns']}")
    print(f"  Used as features: {summary['n_used']}")
    print(f"  Excluded: {summary['n_excluded']} -> {summary['excluded_columns']}")
    print(f"  Numeric used: {summary['n_numeric_used']}, Categorical used: {summary['n_categorical_used']}")
    print(f"  Used by group: {summary['used_by_group']}")

    print("\nStep 5/5: Fitting pipeline on TRAIN ONLY, transforming train/val/test ...")
    pipeline = FeaturePipeline(schema)
    pipeline.fit(train_df)  # train only

    X_train = pipeline.transform(train_df)
    y_train = pipeline.get_target(train_df, target_col)
    X_val = pipeline.transform(val_df)
    y_val = pipeline.get_target(val_df, target_col)
    X_test = pipeline.transform(test_df)
    y_test = pipeline.get_target(test_df, target_col)

    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    same_cols = (
        list(X_train.columns) == list(X_val.columns) == list(X_test.columns)
    )
    print(f"  Feature columns identical across train/val/test: {same_cols}")
    assert target_col not in X_train.columns
    assert id_col not in X_train.columns
    assert time_col not in X_train.columns
    print(f"  Confirmed excluded from X: {target_col}, {id_col}, {time_col}")

    # --- lightweight artifacts only, no huge duplicate CSVs ---
    interim_dir = REPO_ROOT / config["paths"]["interim_dir"]
    split_meta.save_json(interim_dir / "phase1_split_metadata.json")
    print(f"\nSaved split metadata: {interim_dir / 'phase1_split_metadata.json'}")

    reports_dir = REPO_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    schema_df = schema_to_dataframe(schema)
    schema_csv_path = reports_dir / "phase1_feature_schema.csv"
    schema_df.to_csv(schema_csv_path, index=False)
    print(f"Saved feature schema: {schema_csv_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
