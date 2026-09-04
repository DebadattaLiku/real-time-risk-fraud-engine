"""
Phase 3 test suite: anomaly-detection-specific preprocessing
(`src/models/anomaly_preprocessing.py`). Synthetic data through the real
Phase 1 pipeline.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.split import compute_temporal_split
from src.features.schema import build_feature_schema
from src.features.pipeline import FeaturePipeline
from src.models.anomaly_preprocessing import AnomalyPreprocessor, get_anomaly_feature_columns


def _make_synthetic_transactions(n=600, seed=0, with_nans=True, with_rare_category=False):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "TransactionID": np.arange(n),
        "TransactionDT": np.sort(rng.integers(0, 1_000_000, size=n)),
        "isFraud": (rng.random(n) < 0.05).astype(int),
        "TransactionAmt": rng.uniform(1, 500, size=n).astype("float32"),
        "ProductCD": rng.choice(["W", "C", "R"], size=n),
        "card1": rng.integers(1000, 1020, size=n),
        "card4": rng.choice(["visa", "mastercard"], size=n),
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
    if with_rare_category:
        df.loc[df.index[-3:], "ProductCD"] = "NEVER_SEEN_IN_TRAIN"
    return df


def _build_phase1_splits(df):
    train_df, val_df, test_df, _ = compute_temporal_split(
        df, "TransactionDT", "TransactionID", 0.7, 0.15, 0.15,
    )
    schema = build_feature_schema(train_df)
    fp = FeaturePipeline(schema)
    fp.fit(train_df)
    X_train = fp.transform(train_df)
    X_val = fp.transform(val_df)
    X_test = fp.transform(test_df)
    return schema, X_train, X_val, X_test


def test_v_columns_excluded_from_anomaly_feature_set():
    df = _make_synthetic_transactions(n=300)
    schema, X_train, _, _ = _build_phase1_splits(df)
    numeric, categorical = get_anomaly_feature_columns(schema)
    assert "V1" not in numeric
    assert "V1" not in categorical


def test_fit_only_on_train_then_transform_others():
    df = _make_synthetic_transactions(n=600)
    schema, X_train, X_val, X_test = _build_phase1_splits(df)
    pre = AnomalyPreprocessor(schema)
    pre.fit(X_train)  # fit sees train only
    Z_train = pre.transform(X_train)
    Z_val = pre.transform(X_val)
    Z_test = pre.transform(X_test)
    assert Z_train.shape[0] == len(X_train)
    assert Z_val.shape[0] == len(X_val)
    assert Z_test.shape[0] == len(X_test)
    assert Z_train.shape[1] == Z_val.shape[1] == Z_test.shape[1]


def test_transform_before_fit_raises():
    df = _make_synthetic_transactions(n=200)
    schema, X_train, _, _ = _build_phase1_splits(df)
    pre = AnomalyPreprocessor(schema)
    with pytest.raises(RuntimeError):
        pre.transform(X_train)


def test_no_nans_in_output():
    df = _make_synthetic_transactions(n=600, with_nans=True)
    schema, X_train, X_val, X_test = _build_phase1_splits(df)
    pre = AnomalyPreprocessor(schema)
    pre.fit(X_train)
    for X in (X_train, X_val, X_test):
        Z = pre.transform(X)
        assert not np.isnan(Z).any()


def test_output_is_fully_numeric_array():
    df = _make_synthetic_transactions(n=300)
    schema, X_train, _, _ = _build_phase1_splits(df)
    pre = AnomalyPreprocessor(schema)
    Z = pre.fit_transform(X_train)
    assert isinstance(Z, np.ndarray)
    assert np.issubdtype(Z.dtype, np.floating)


def test_unseen_category_gets_zero_frequency():
    df = _make_synthetic_transactions(n=600, with_rare_category=True)
    schema, X_train, X_val, X_test = _build_phase1_splits(df)
    assert "NEVER_SEEN_IN_TRAIN" not in X_train["ProductCD"].astype(str).unique()

    pre = AnomalyPreprocessor(schema)
    pre.fit(X_train)
    # Locate ProductCD's column index in the output.
    col_idx = pre.get_feature_names().index("ProductCD")

    combined = pd.concat([X_val, X_test])
    rare_rows = combined[combined["ProductCD"].astype(str) == "NEVER_SEEN_IN_TRAIN"]
    assert len(rare_rows) >= 1
    Z_rare = pre.transform(rare_rows)
    assert np.all(Z_rare[:, col_idx] == 0.0)


def test_median_imputation_uses_train_statistics_only():
    df = _make_synthetic_transactions(n=600, with_nans=True)
    schema, X_train, X_val, X_test = _build_phase1_splits(df)
    pre = AnomalyPreprocessor(schema)
    pre.fit(X_train)
    expected_median = float(X_train["addr1"].median())
    assert pre._medians["addr1"] == pytest.approx(expected_median)


def test_no_target_id_time_in_feature_columns():
    df = _make_synthetic_transactions(n=300)
    schema, X_train, _, _ = _build_phase1_splits(df)
    pre = AnomalyPreprocessor(schema)
    names = pre.get_feature_names()
    assert "isFraud" not in names
    assert "TransactionID" not in names
    assert "TransactionDT" not in names


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
