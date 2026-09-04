"""
Phase 2A test suite: Logistic-Regression-specific preprocessing
(`src/models/preprocessing.py`). Uses small synthetic data, built through
the real Phase 1 pipeline (schema + FeaturePipeline) so these tests exercise
the actual integration path, not a mocked one.
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
from src.models.preprocessing import LogRegPreprocessor, get_numeric_and_categorical_columns


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
        # Inject a category value that ONLY appears in the last few rows,
        # so a chronological split puts it exclusively in val/test.
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


def test_preprocessor_fit_only_on_train_then_transform_others():
    df = _make_synthetic_transactions(n=600)
    schema, X_train, X_val, X_test = _build_phase1_splits(df)

    pre = LogRegPreprocessor(schema)
    pre.fit(X_train)  # fit sees train only
    Z_train = pre.transform(X_train)
    Z_val = pre.transform(X_val)
    Z_test = pre.transform(X_test)

    assert Z_train.shape[0] == len(X_train)
    assert Z_val.shape[0] == len(X_val)
    assert Z_test.shape[0] == len(X_test)


def test_preprocessor_output_shapes_compatible_across_splits():
    df = _make_synthetic_transactions(n=600)
    schema, X_train, X_val, X_test = _build_phase1_splits(df)

    pre = LogRegPreprocessor(schema)
    pre.fit(X_train)
    Z_train = pre.transform(X_train)
    Z_val = pre.transform(X_val)
    Z_test = pre.transform(X_test)

    # Same number of columns across all three transformed matrices.
    assert Z_train.shape[1] == Z_val.shape[1] == Z_test.shape[1]


def test_preprocessor_transform_before_fit_raises():
    df = _make_synthetic_transactions(n=200)
    schema, X_train, _, _ = _build_phase1_splits(df)
    pre = LogRegPreprocessor(schema)
    with pytest.raises(RuntimeError):
        pre.transform(X_train)


def test_preprocessor_no_nans_in_output():
    df = _make_synthetic_transactions(n=600, with_nans=True)
    schema, X_train, X_val, X_test = _build_phase1_splits(df)
    pre = LogRegPreprocessor(schema)
    pre.fit(X_train)
    for X in (X_train, X_val, X_test):
        Z = pre.transform(X)
        assert not np.isnan(Z).any(), "LogReg preprocessing output must never contain NaN"


def test_preprocessor_handles_unseen_category_without_crashing():
    df = _make_synthetic_transactions(n=600, with_rare_category=True)
    schema, X_train, X_val, X_test = _build_phase1_splits(df)
    # The rare category only appears in the last 3 rows -> with a 70/15/15
    # chronological split those land in val/test, not train.
    assert "NEVER_SEEN_IN_TRAIN" not in X_train["ProductCD"].astype(str).unique()

    pre = LogRegPreprocessor(schema)
    pre.fit(X_train)
    # Must not raise, even though val/test contain a category never seen
    # during fit.
    Z_val = pre.transform(X_val)
    Z_test = pre.transform(X_test)
    assert Z_val.shape[0] == len(X_val)
    assert Z_test.shape[0] == len(X_test)


def test_preprocessor_numeric_scaled_using_train_statistics_only():
    df = _make_synthetic_transactions(n=600)
    schema, X_train, X_val, X_test = _build_phase1_splits(df)
    pre = LogRegPreprocessor(schema)
    pre.fit(X_train)
    Z_train = pre.transform(X_train)

    numeric_cols, _ = get_numeric_and_categorical_columns(schema)
    n_numeric = len(numeric_cols)
    # Scaled training numeric block should have ~zero mean (StandardScaler
    # fit on this exact data), confirming scaling parameters came from train.
    train_numeric_block = Z_train[:, :n_numeric]
    assert np.allclose(train_numeric_block.mean(axis=0), 0.0, atol=1e-3)


def test_preprocessor_excludes_target_id_time_via_schema():
    df = _make_synthetic_transactions(n=300)
    schema, X_train, _, _ = _build_phase1_splits(df)
    # X_train already comes from Phase 1's FeaturePipeline, which excludes
    # these — confirm the LogReg preprocessor's column list agrees.
    numeric_cols, categorical_cols = get_numeric_and_categorical_columns(schema)
    all_cols = set(numeric_cols) | set(categorical_cols)
    assert "isFraud" not in all_cols
    assert "TransactionID" not in all_cols
    assert "TransactionDT" not in all_cols


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
