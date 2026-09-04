"""
Phase 2B test suite: LightGBM-specific preprocessing
(`src/models/lightgbm_preprocessing.py`). Synthetic data, built through the
real Phase 1 pipeline (schema + FeaturePipeline) so these tests exercise the
actual integration path.
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
from src.models.lightgbm_preprocessing import LightGBMPreprocessor, UNSEEN_CATEGORY


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


def test_category_vocabulary_built_from_train_only():
    df = _make_synthetic_transactions(n=600, with_rare_category=True)
    schema, X_train, X_val, X_test = _build_phase1_splits(df)
    assert "NEVER_SEEN_IN_TRAIN" not in X_train["ProductCD"].astype(str).unique()

    pre = LightGBMPreprocessor(schema)
    pre.fit(X_train)
    # The vocabulary for ProductCD must only contain values seen in train,
    # plus the reserved UNSEEN_CATEGORY bucket.
    train_values = set(X_train["ProductCD"].astype(str).unique())
    assert set(pre._categories["ProductCD"]) == train_values | {UNSEEN_CATEGORY}


def test_unseen_category_mapped_to_explicit_bucket_not_crash():
    df = _make_synthetic_transactions(n=600, with_rare_category=True)
    schema, X_train, X_val, X_test = _build_phase1_splits(df)

    pre = LightGBMPreprocessor(schema)
    pre.fit(X_train)
    out_val = pre.transform(X_val)
    out_test = pre.transform(X_test)

    # Rows with "NEVER_SEEN_IN_TRAIN" should land in val or test (rare
    # category injected at the end of the chronological dataset) and be
    # mapped to UNSEEN_CATEGORY, not raise and not silently become NaN.
    all_mapped = pd.concat([out_val["ProductCD"], out_test["ProductCD"]])
    assert (all_mapped.astype(str) == UNSEEN_CATEGORY).sum() >= 1


def test_missing_and_unseen_are_distinct_buckets():
    df = _make_synthetic_transactions(n=600, with_nans=True, with_rare_category=True)
    schema, X_train, X_val, X_test = _build_phase1_splits(df)
    pre = LightGBMPreprocessor(schema)
    pre.fit(X_train)
    cats = pre._categories["ProductCD"]
    assert "__missing__" in cats or True  # ProductCD has no injected NaNs in this fixture; check card4 instead
    cats_card4 = pre._categories["card4"]
    assert "__missing__" in cats_card4
    assert UNSEEN_CATEGORY in cats_card4
    assert "__missing__" != UNSEEN_CATEGORY


def test_transform_before_fit_raises():
    df = _make_synthetic_transactions(n=200)
    schema, X_train, _, _ = _build_phase1_splits(df)
    pre = LightGBMPreprocessor(schema)
    with pytest.raises(RuntimeError):
        pre.transform(X_train)


def test_numeric_missing_values_preserved_not_imputed():
    df = _make_synthetic_transactions(n=600, with_nans=True)
    schema, X_train, X_val, X_test = _build_phase1_splits(df)
    assert X_train["addr1"].isna().sum() > 0 or X_val["addr1"].isna().sum() > 0 or X_test["addr1"].isna().sum() > 0

    pre = LightGBMPreprocessor(schema)
    pre.fit(X_train)
    out_train = pre.transform(X_train)
    # LightGBM path must NOT impute numeric NaNs (unlike LogReg path).
    assert out_train["addr1"].isna().sum() == X_train["addr1"].isna().sum()


def test_feature_columns_consistent_across_splits():
    df = _make_synthetic_transactions(n=900)
    schema, X_train, X_val, X_test = _build_phase1_splits(df)
    pre = LightGBMPreprocessor(schema)
    pre.fit(X_train)
    out_train = pre.transform(X_train)
    out_val = pre.transform(X_val)
    out_test = pre.transform(X_test)
    assert list(out_train.columns) == list(out_val.columns) == list(out_test.columns)


def test_categorical_columns_use_pandas_category_dtype():
    df = _make_synthetic_transactions(n=300)
    schema, X_train, _, _ = _build_phase1_splits(df)
    pre = LightGBMPreprocessor(schema)
    pre.fit(X_train)
    out = pre.transform(X_train)
    for col in pre.categorical_cols:
        assert isinstance(out[col].dtype, pd.CategoricalDtype)


def test_no_target_id_time_in_output_columns():
    df = _make_synthetic_transactions(n=300)
    schema, X_train, _, _ = _build_phase1_splits(df)
    pre = LightGBMPreprocessor(schema)
    pre.fit(X_train)
    out = pre.transform(X_train)
    assert "isFraud" not in out.columns
    assert "TransactionID" not in out.columns
    assert "TransactionDT" not in out.columns


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
