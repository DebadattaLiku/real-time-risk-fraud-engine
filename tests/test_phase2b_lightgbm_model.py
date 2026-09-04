"""
Phase 2B test suite: LightGBM training module
(`src/models/lightgbm_model.py`). Uses a small synthetic dataset so this
runs fast — real-data results are validated separately by the Phase 2B
runner script against the actual IEEE-CIS data.
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
from src.models.lightgbm_preprocessing import LightGBMPreprocessor
from src.models.lightgbm_model import train_lightgbm, get_feature_importance


def _make_synthetic_transactions(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    # Make fraud weakly but genuinely predictable from a couple of features
    # so the model has something real to learn (not pure noise), keeping
    # the smoke test meaningful without needing real data.
    amt = rng.uniform(1, 500, size=n).astype("float32")
    c1 = rng.integers(0, 10, size=n).astype("float32")
    fraud_logit = -3.5 + 0.01 * amt + 0.3 * c1
    fraud_prob = 1 / (1 + np.exp(-fraud_logit))
    is_fraud = (rng.random(n) < fraud_prob).astype(int)

    df = pd.DataFrame({
        "TransactionID": np.arange(n),
        "TransactionDT": np.sort(rng.integers(0, 1_000_000, size=n)),
        "isFraud": is_fraud,
        "TransactionAmt": amt,
        "ProductCD": rng.choice(["W", "C", "R"], size=n),
        "card1": rng.integers(1000, 1020, size=n),
        "card4": rng.choice(["visa", "mastercard"], size=n),
        "addr1": rng.integers(100, 130, size=n).astype("float32"),
        "C1": c1,
        "D1": rng.uniform(0, 100, size=n).astype("float32"),
        "V1": rng.uniform(0, 1, size=n).astype("float32"),
        "M1": rng.choice(["T", "F"], size=n),
    })
    return df


def _build_splits_and_preprocess(df):
    train_df, val_df, test_df, _ = compute_temporal_split(
        df, "TransactionDT", "TransactionID", 0.7, 0.15, 0.15,
    )
    schema = build_feature_schema(train_df)
    fp = FeaturePipeline(schema)
    fp.fit(train_df)
    X_train = fp.transform(train_df)
    y_train = fp.get_target(train_df, "isFraud")
    X_val = fp.transform(val_df)
    y_val = fp.get_target(val_df, "isFraud")
    X_test = fp.transform(test_df)
    y_test = fp.get_target(test_df, "isFraud")

    pre = LightGBMPreprocessor(schema)
    Z_train = pre.fit_transform(X_train)
    Z_val = pre.transform(X_val)
    Z_test = pre.transform(X_test)
    return pre, Z_train, y_train, Z_val, y_val, Z_test, y_test


def test_model_trains_on_synthetic_data():
    df = _make_synthetic_transactions(n=2000)
    pre, Z_train, y_train, Z_val, y_val, Z_test, y_test = _build_splits_and_preprocess(df)
    model, info = train_lightgbm(
        Z_train, y_train, Z_val, y_val,
        categorical_features=pre.get_categorical_feature_names(),
        params={"n_estimators": 50},
        early_stopping_rounds=10,
    )
    assert model is not None
    assert info["best_iteration"] >= 1


def test_prediction_probabilities_valid_and_correct_length():
    df = _make_synthetic_transactions(n=2000)
    pre, Z_train, y_train, Z_val, y_val, Z_test, y_test = _build_splits_and_preprocess(df)
    model, info = train_lightgbm(
        Z_train, y_train, Z_val, y_val,
        categorical_features=pre.get_categorical_feature_names(),
        params={"n_estimators": 50},
        early_stopping_rounds=10,
    )
    test_scores = model.predict_proba(Z_test)[:, 1]
    assert len(test_scores) == len(Z_test)
    assert np.all(test_scores >= 0.0) and np.all(test_scores <= 1.0)
    assert not np.isnan(test_scores).any()


def test_model_never_receives_target_id_or_raw_time():
    df = _make_synthetic_transactions(n=1000)
    pre, Z_train, y_train, Z_val, y_val, Z_test, y_test = _build_splits_and_preprocess(df)
    for Z in (Z_train, Z_val, Z_test):
        assert "isFraud" not in Z.columns
        assert "TransactionID" not in Z.columns
        assert "TransactionDT" not in Z.columns


def test_scale_pos_weight_changes_training_without_erroring():
    df = _make_synthetic_transactions(n=2000)
    pre, Z_train, y_train, Z_val, y_val, Z_test, y_test = _build_splits_and_preprocess(df)
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    spw = n_neg / max(n_pos, 1)
    model, info = train_lightgbm(
        Z_train, y_train, Z_val, y_val,
        categorical_features=pre.get_categorical_feature_names(),
        scale_pos_weight=spw,
        params={"n_estimators": 50},
        early_stopping_rounds=10,
    )
    assert info["params"]["scale_pos_weight"] == pytest.approx(spw)


def test_feature_importance_returns_sorted_pairs_covering_all_features():
    df = _make_synthetic_transactions(n=2000)
    pre, Z_train, y_train, Z_val, y_val, Z_test, y_test = _build_splits_and_preprocess(df)
    model, info = train_lightgbm(
        Z_train, y_train, Z_val, y_val,
        categorical_features=pre.get_categorical_feature_names(),
        params={"n_estimators": 50},
        early_stopping_rounds=10,
    )
    importances = get_feature_importance(model, list(Z_train.columns))
    assert len(importances) == Z_train.shape[1]
    values = [v for _, v in importances]
    assert values == sorted(values, reverse=True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
