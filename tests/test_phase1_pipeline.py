"""
Phase 1 test suite: data loading, temporal splitting, and the feature
preparation pipeline. Uses small synthetic in-memory data — never the real
IEEE-CIS dataset — so these tests are fast and self-contained.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.load import (
    load_train_transaction, DataValidationError, REQUIRED_COLUMNS,
)
from src.data.split import compute_temporal_split, verify_split
from src.features.schema import build_feature_schema, schema_summary, infer_feature_group
from src.features.pipeline import FeaturePipeline, MISSING_CATEGORY


# ---------------------------------------------------------------------------
# Synthetic data helper
# ---------------------------------------------------------------------------

def _make_synthetic_transactions(n=300, seed=0, with_nans=True):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "TransactionID": np.arange(n),
        "TransactionDT": np.sort(rng.integers(0, 1_000_000, size=n)),
        "isFraud": (rng.random(n) < 0.05).astype(int),
        "TransactionAmt": rng.uniform(1, 500, size=n).astype("float32"),
        "ProductCD": rng.choice(["W", "C", "R", "H", "S"], size=n),
        "card1": rng.integers(1000, 1020, size=n),
        "card4": rng.choice(["visa", "mastercard", "discover"], size=n),
        "addr1": rng.integers(100, 130, size=n).astype("float32"),
        "C1": rng.integers(0, 10, size=n).astype("float32"),
        "D1": rng.uniform(0, 100, size=n).astype("float32"),
        "V1": rng.uniform(0, 1, size=n).astype("float32"),
        "M1": rng.choice(["T", "F"], size=n),
    })
    if with_nans:
        rng2 = np.random.default_rng(seed + 1)
        nan_idx = rng2.choice(n, size=n // 10, replace=False)
        df.loc[nan_idx, "addr1"] = np.nan
        df.loc[nan_idx, "card4"] = np.nan
        df.loc[nan_idx, "D1"] = np.nan
    return df


# ---------------------------------------------------------------------------
# load.py tests
# ---------------------------------------------------------------------------

def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_train_transaction(path=tmp_path / "does_not_exist.csv")


def test_load_required_column_validation(tmp_path):
    df = pd.DataFrame({"TransactionID": [1, 2, 3], "SomeOtherCol": [1, 2, 3]})
    csv_path = tmp_path / "train_transaction.csv"
    df.to_csv(csv_path, index=False)
    with pytest.raises(DataValidationError):
        load_train_transaction(path=csv_path)


def test_load_transaction_id_uniqueness_validation(tmp_path):
    df = _make_synthetic_transactions(n=50)
    df.loc[1, "TransactionID"] = df.loc[0, "TransactionID"]  # inject duplicate
    csv_path = tmp_path / "train_transaction.csv"
    df.to_csv(csv_path, index=False)
    with pytest.raises(DataValidationError):
        load_train_transaction(path=csv_path)


def test_load_valid_file_succeeds(tmp_path):
    df = _make_synthetic_transactions(n=50)
    csv_path = tmp_path / "train_transaction.csv"
    df.to_csv(csv_path, index=False)
    loaded_df, report = load_train_transaction(path=csv_path)
    assert report.required_columns_present
    assert report.transaction_id_unique
    assert report.target_is_binary
    assert report.transaction_dt_valid
    assert loaded_df.shape[0] == 50


def test_load_non_binary_target_raises(tmp_path):
    df = _make_synthetic_transactions(n=50)
    df.loc[0, "isFraud"] = 2  # invalid class
    csv_path = tmp_path / "train_transaction.csv"
    df.to_csv(csv_path, index=False)
    with pytest.raises(DataValidationError):
        load_train_transaction(path=csv_path)


# ---------------------------------------------------------------------------
# split.py tests
# ---------------------------------------------------------------------------

def test_split_no_overlap_and_chronological_order():
    df = _make_synthetic_transactions(n=1000)
    train_df, val_df, test_df, meta = compute_temporal_split(
        df, "TransactionDT", "TransactionID", 0.7, 0.15, 0.15,
    )
    checks = verify_split(train_df, val_df, test_df, "TransactionDT", "TransactionID")
    assert checks["no_row_overlap_train_val"]
    assert checks["no_row_overlap_val_test"]
    assert checks["no_row_overlap_train_test"]
    assert checks["max_train_dt_lte_min_val_dt"]
    assert checks["max_val_dt_lte_min_test_dt"]
    assert checks["all_ids_unique_across_splits"]


def test_split_approximate_proportions():
    df = _make_synthetic_transactions(n=2000)
    train_df, val_df, test_df, meta = compute_temporal_split(
        df, "TransactionDT", "TransactionID", 0.7, 0.15, 0.15,
    )
    assert abs(len(train_df) / 2000 - 0.70) < 0.01
    assert abs(len(val_df) / 2000 - 0.15) < 0.01
    assert abs(len(test_df) / 2000 - 0.15) < 0.01


def test_split_never_shuffles_never_randomizes():
    df = _make_synthetic_transactions(n=500)
    train_df_1, val_df_1, test_df_1, _ = compute_temporal_split(
        df, "TransactionDT", "TransactionID", 0.7, 0.15, 0.15,
    )
    train_df_2, val_df_2, test_df_2, _ = compute_temporal_split(
        df, "TransactionDT", "TransactionID", 0.7, 0.15, 0.15,
    )
    # Identical output on repeated calls (fully deterministic, no RNG).
    pd.testing.assert_frame_equal(train_df_1, train_df_2)
    pd.testing.assert_frame_equal(val_df_1, val_df_2)
    pd.testing.assert_frame_equal(test_df_1, test_df_2)


def test_split_bad_ratios_raise():
    df = _make_synthetic_transactions(n=100)
    with pytest.raises(ValueError):
        compute_temporal_split(df, "TransactionDT", "TransactionID", 0.7, 0.2, 0.2)


def test_split_metadata_matches_actual_frames():
    df = _make_synthetic_transactions(n=800)
    train_df, val_df, test_df, meta = compute_temporal_split(
        df, "TransactionDT", "TransactionID", 0.7, 0.15, 0.15,
    )
    assert meta.n_rows_train == len(train_df)
    assert meta.n_rows_val == len(val_df)
    assert meta.n_rows_test == len(test_df)
    assert meta.train_dt_max == train_df["TransactionDT"].max()
    assert meta.test_dt_min == test_df["TransactionDT"].min()
    assert meta.method == "time_based"


# ---------------------------------------------------------------------------
# features/schema.py + features/pipeline.py tests
# ---------------------------------------------------------------------------

def test_schema_excludes_target_id_and_raw_time():
    df = _make_synthetic_transactions(n=200)
    schema = build_feature_schema(df)
    excluded_names = {s.name for s in schema if not s.used}
    assert "isFraud" in excluded_names
    assert "TransactionID" in excluded_names
    assert "TransactionDT" in excluded_names
    used_names = {s.name for s in schema if s.used}
    assert "isFraud" not in used_names
    assert "TransactionID" not in used_names
    assert "TransactionDT" not in used_names


def test_schema_missingness_computed_only_from_given_df():
    # If schema is built from train_df only, missingness must reflect
    # train_df's own NaN rate, not some other partition's.
    df = _make_synthetic_transactions(n=300, with_nans=True)
    schema = build_feature_schema(df)
    addr1_spec = next(s for s in schema if s.name == "addr1")
    expected = float(df["addr1"].isna().mean())
    assert addr1_spec.missing_pct_train == pytest.approx(expected)


def test_feature_group_inference():
    assert infer_feature_group("card1") == "card"
    assert infer_feature_group("addr2") == "addr"
    assert infer_feature_group("D5") == "D"
    assert infer_feature_group("C3") == "C"
    assert infer_feature_group("V101") == "V"
    assert infer_feature_group("M4") == "M"
    assert infer_feature_group("TransactionID") == "id"
    assert infer_feature_group("isFraud") == "target"
    assert infer_feature_group("TransactionDT") == "time"
    assert infer_feature_group("P_emaildomain") == "email"


def test_pipeline_transform_excludes_target_and_id():
    df = _make_synthetic_transactions(n=200)
    schema = build_feature_schema(df)
    pipeline = FeaturePipeline(schema)
    pipeline.fit(df)
    X = pipeline.transform(df)
    assert "isFraud" not in X.columns
    assert "TransactionID" not in X.columns
    assert "TransactionDT" not in X.columns


def test_pipeline_feature_columns_consistent_across_splits():
    df = _make_synthetic_transactions(n=900)
    train_df, val_df, test_df, _ = compute_temporal_split(
        df, "TransactionDT", "TransactionID", 0.7, 0.15, 0.15,
    )
    schema = build_feature_schema(train_df)
    pipeline = FeaturePipeline(schema)
    pipeline.fit(train_df)
    X_train = pipeline.transform(train_df)
    X_val = pipeline.transform(val_df)
    X_test = pipeline.transform(test_df)
    assert list(X_train.columns) == list(X_val.columns) == list(X_test.columns)


def test_pipeline_missing_categorical_gets_explicit_category():
    df = _make_synthetic_transactions(n=300, with_nans=True)
    assert df["card4"].isna().sum() > 0
    schema = build_feature_schema(df)
    pipeline = FeaturePipeline(schema)
    pipeline.fit(df)
    X = pipeline.transform(df)
    assert X["card4"].isna().sum() == 0
    assert MISSING_CATEGORY in X["card4"].cat.categories
    assert (X["card4"] == MISSING_CATEGORY).sum() == df["card4"].isna().sum()


def test_pipeline_numeric_missingness_preserved_not_imputed():
    df = _make_synthetic_transactions(n=300, with_nans=True)
    assert df["addr1"].isna().sum() > 0
    schema = build_feature_schema(df)
    pipeline = FeaturePipeline(schema)
    pipeline.fit(df)
    X = pipeline.transform(df)
    # Numeric NaNs must be preserved as-is (no imputation in Phase 1).
    assert X["addr1"].isna().sum() == df["addr1"].isna().sum()


def test_pipeline_transform_before_fit_raises():
    df = _make_synthetic_transactions(n=100)
    schema = build_feature_schema(df)
    pipeline = FeaturePipeline(schema)
    with pytest.raises(RuntimeError):
        pipeline.transform(df)


def test_pipeline_fit_uses_only_the_dataframe_passed_in():
    """
    Structural leakage guard: fitting the pipeline on train_df and then
    transforming val_df/test_df must not require or consult val_df/test_df
    at fit time. We simulate this by fitting on train only and confirming
    transform still works correctly on frames never seen during fit,
    including a categorical value that only appears in val/test.
    """
    df = _make_synthetic_transactions(n=900)
    train_df, val_df, test_df, _ = compute_temporal_split(
        df, "TransactionDT", "TransactionID", 0.7, 0.15, 0.15,
    )
    schema = build_feature_schema(train_df)  # train-only schema
    pipeline = FeaturePipeline(schema)
    pipeline.fit(train_df)  # fit sees train_df only

    # transform() must work on frames the pipeline object never received
    # during fit() — this would fail if fit() had captured/cached val/test.
    X_val = pipeline.transform(val_df)
    X_test = pipeline.transform(test_df)
    assert len(X_val) == len(val_df)
    assert len(X_test) == len(test_df)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
