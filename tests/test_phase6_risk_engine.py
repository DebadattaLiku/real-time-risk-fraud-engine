"""
Phase 6 test suite: RiskDecisionEngine (`src/engine/risk_engine.py`), built
on a small synthetic model/pipeline fixture (same style as Phase 2B/4's
synthetic fixtures) so this runs fast and independent of real data.
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
from src.features.behavioral import compute_behavioral_features, get_behavioral_feature_names
from src.models.lightgbm_preprocessing import LightGBMPreprocessor
from src.models.lightgbm_model import train_lightgbm
from src.decision.policy import DecisionPolicy
from src.engine.state import BehavioralStateManager
from src.engine.risk_engine import RiskDecisionEngine, TransactionValidationError


def _make_synthetic_transactions(n=800, seed=0):
    rng = np.random.default_rng(seed)
    amt = rng.uniform(1, 500, size=n).astype("float32")
    c1 = rng.integers(0, 10, size=n).astype("float32")
    fraud_logit = -3.0 + 0.01 * amt + 0.3 * c1
    fraud_prob = 1 / (1 + np.exp(-fraud_logit))
    is_fraud = (rng.random(n) < fraud_prob).astype(int)
    df = pd.DataFrame({
        "TransactionID": np.arange(n),
        "TransactionDT": np.sort(rng.integers(0, 500000, size=n)),
        "isFraud": is_fraud,
        "TransactionAmt": amt,
        "ProductCD": rng.choice(["W", "C", "R"], size=n),
        "card1": rng.integers(1000, 1015, size=n),
        "card4": rng.choice(["visa", "mastercard"], size=n),
        "addr1": rng.integers(100, 130, size=n).astype("float32"),
        "C1": c1,
        "D1": rng.uniform(0, 100, size=n).astype("float32"),
        "V1": rng.uniform(0, 1, size=n).astype("float32"),
        "M1": rng.choice(["T", "F"], size=n),
    })
    return df


def _build_engine_fixture(n=800, seed=0):
    """Builds a small, real (not mocked) engine from synthetic data."""
    df = _make_synthetic_transactions(n=n, seed=seed)
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

    lgbm_pre = LightGBMPreprocessor(schema)
    Z_train = lgbm_pre.fit_transform(X_train)
    Z_val = lgbm_pre.transform(X_val)

    # Attach behavioral features (small offline computation for fixture setup).
    combined_raw = pd.concat([
        train_df[["TransactionID", "TransactionDT", "card1", "TransactionAmt"]],
        val_df[["TransactionID", "TransactionDT", "card1", "TransactionAmt"]],
        test_df[["TransactionID", "TransactionDT", "card1", "TransactionAmt"]],
    ], ignore_index=True)
    bhv = compute_behavioral_features(combined_raw)
    bhv_names = get_behavioral_feature_names(bhv)
    bhv_indexed = bhv.set_index("TransactionID")

    def attach(Z, ids):
        b = bhv_indexed.loc[ids.values, bhv_names].reset_index(drop=True)
        return pd.concat([Z.reset_index(drop=True), b], axis=1)

    train_ids = train_df["TransactionID"].reset_index(drop=True)
    val_ids = val_df["TransactionID"].reset_index(drop=True)
    Z_train_full = attach(Z_train, train_ids)
    Z_val_full = attach(Z_val, val_ids)

    model, info = train_lightgbm(
        Z_train_full, y_train, Z_val_full, y_val,
        categorical_features=lgbm_pre.get_categorical_feature_names(),
        params={"n_estimators": 50}, early_stopping_rounds=10,
    )

    policy = DecisionPolicy(approve_threshold=0.3, block_threshold=0.7, name="test_policy")
    state_manager = BehavioralStateManager()
    engine = RiskDecisionEngine(
        schema=schema, feature_pipeline=fp, lgbm_preprocessor=lgbm_pre,
        model=model, policy=policy, state_manager=state_manager,
    )
    return engine, train_df, val_df, test_df, schema


def _row_to_transaction_dict(row: pd.Series, exclude=("isFraud",)) -> dict:
    d = row.to_dict()
    for k in exclude:
        d.pop(k, None)
    return d


# ---------------------------------------------------------------------------
# Processing order
# ---------------------------------------------------------------------------

def test_state_not_updated_before_prediction():
    engine, train_df, val_df, test_df, schema = _build_engine_fixture()
    row = test_df.iloc[0]
    txn = _row_to_transaction_dict(row)
    entity = txn["card1"]

    count_before = engine.state_manager.get_state_snapshot(entity)
    count_before = count_before["count"] if count_before else 0

    result = engine.process_transaction(txn)

    # After process_transaction returns, state MUST be updated (count += 1).
    snap_after = engine.state_manager.get_state_snapshot(entity)
    assert snap_after["count"] == count_before + 1
    assert result["state_updated"] is True


def test_behavioral_features_in_result_reflect_pre_update_state():
    engine, train_df, val_df, test_df, schema = _build_engine_fixture()
    row1 = test_df.iloc[0]
    txn1 = _row_to_transaction_dict(row1)
    result1 = engine.process_transaction(txn1)
    assert result1["behavioral_features"]["bhv_prev_txn_count"] == 0  # first ever for this run


# ---------------------------------------------------------------------------
# Self-history / future leakage through the engine
# ---------------------------------------------------------------------------

def test_self_history_leakage_first_second_third():
    engine, train_df, val_df, test_df, schema = _build_engine_fixture()
    # Build 3 synthetic transactions for the SAME entity, distinct amounts.
    base = _row_to_transaction_dict(test_df.iloc[0])
    entity = 9999999  # a fresh entity id not used elsewhere
    txn1 = dict(base, TransactionID=900001, TransactionDT=1000, card1=entity, TransactionAmt=10.0)
    txn2 = dict(base, TransactionID=900002, TransactionDT=2000, card1=entity, TransactionAmt=20.0)
    txn3 = dict(base, TransactionID=900003, TransactionDT=3000, card1=entity, TransactionAmt=999.0)

    r1 = engine.process_transaction(txn1)
    assert r1["behavioral_features"]["bhv_prev_txn_count"] == 0

    r2 = engine.process_transaction(txn2)
    assert r2["behavioral_features"]["bhv_prev_txn_count"] == 1
    assert r2["behavioral_features"]["bhv_hist_mean_amt"] == pytest.approx(10.0)

    r3 = engine.process_transaction(txn3)
    assert r3["behavioral_features"]["bhv_prev_txn_count"] == 2
    assert r3["behavioral_features"]["bhv_hist_mean_amt"] == pytest.approx(15.0)
    assert r3["behavioral_features"]["bhv_hist_max_amt"] == pytest.approx(20.0)  # NOT 999


def test_future_transaction_does_not_change_earlier_results():
    entity = 8888888
    base_engine, train_df, val_df, test_df, schema = _build_engine_fixture()
    base_txn = _row_to_transaction_dict(test_df.iloc[0])
    txn1 = dict(base_txn, TransactionID=800001, TransactionDT=1000, card1=entity, TransactionAmt=10.0)
    txn2 = dict(base_txn, TransactionID=800002, TransactionDT=2000, card1=entity, TransactionAmt=20.0)
    txn3 = dict(base_txn, TransactionID=800003, TransactionDT=3000, card1=entity, TransactionAmt=99999.0)

    # Scenario A: process only txn1, txn2.
    engine_a, *_ = _build_engine_fixture()
    ra1 = engine_a.process_transaction(txn1)
    ra2 = engine_a.process_transaction(txn2)

    # Scenario B: process txn1, txn2, txn3.
    engine_b, *_ = _build_engine_fixture()
    rb1 = engine_b.process_transaction(txn1)
    rb2 = engine_b.process_transaction(txn2)
    rb3 = engine_b.process_transaction(txn3)

    assert ra1["risk_score"] == pytest.approx(rb1["risk_score"])
    assert ra2["risk_score"] == pytest.approx(rb2["risk_score"])
    assert ra1["decision"] == rb1["decision"]
    assert ra2["decision"] == rb2["decision"]
    for k in ra2["behavioral_features"]:
        v1, v2 = ra2["behavioral_features"][k], rb2["behavioral_features"][k]
        if v1 != v1 and v2 != v2:  # both NaN
            continue
        assert v1 == pytest.approx(v2)


# ---------------------------------------------------------------------------
# State reset
# ---------------------------------------------------------------------------

def test_reset_state_produces_clean_behavior():
    engine, train_df, val_df, test_df, schema = _build_engine_fixture()
    txn = _row_to_transaction_dict(test_df.iloc[0])
    engine.process_transaction(txn)
    engine.reset_state()
    txn2 = dict(txn, TransactionID=int(txn["TransactionID"]) + 1)
    result = engine.process_transaction(txn2)
    if result["behavioral_features"]["bhv_prev_txn_count"] != 0:
        # Only meaningful if txn2 shares the same entity as txn.
        assert txn2["card1"] != txn["card1"]
    else:
        assert result["behavioral_features"]["bhv_prev_txn_count"] == 0


# ---------------------------------------------------------------------------
# Invalid input
# ---------------------------------------------------------------------------

def test_missing_required_field_raises_and_does_not_update_state():
    engine, train_df, val_df, test_df, schema = _build_engine_fixture()
    txn = _row_to_transaction_dict(test_df.iloc[0])
    entity = txn["card1"]
    del txn["TransactionAmt"]

    snap_before = engine.state_manager.get_state_snapshot(entity)
    with pytest.raises(TransactionValidationError):
        engine.process_transaction(txn)
    snap_after = engine.state_manager.get_state_snapshot(entity)
    assert snap_before == snap_after  # no partial update


def test_isfraud_in_input_raises():
    engine, train_df, val_df, test_df, schema = _build_engine_fixture()
    txn = _row_to_transaction_dict(test_df.iloc[0], exclude=())  # keep isFraud
    with pytest.raises(TransactionValidationError):
        engine.process_transaction(txn)


def test_negative_amount_raises():
    engine, train_df, val_df, test_df, schema = _build_engine_fixture()
    txn = _row_to_transaction_dict(test_df.iloc[0])
    txn["TransactionAmt"] = -5.0
    with pytest.raises(TransactionValidationError):
        engine.process_transaction(txn)


def test_non_numeric_amount_raises():
    engine, train_df, val_df, test_df, schema = _build_engine_fixture()
    txn = _row_to_transaction_dict(test_df.iloc[0])
    txn["TransactionAmt"] = "not_a_number"
    with pytest.raises(TransactionValidationError):
        engine.process_transaction(txn)


def test_invalid_transaction_does_not_partially_update_state():
    engine, train_df, val_df, test_df, schema = _build_engine_fixture()
    txn = _row_to_transaction_dict(test_df.iloc[0])
    entity = txn["card1"]
    bad_txn = dict(txn, TransactionAmt=float("nan"))
    snap_before = engine.state_manager.get_state_snapshot(entity)
    with pytest.raises(TransactionValidationError):
        engine.process_transaction(bad_txn)
    snap_after = engine.state_manager.get_state_snapshot(entity)
    assert snap_before == snap_after


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

def test_result_contains_required_fields():
    engine, train_df, val_df, test_df, schema = _build_engine_fixture()
    txn = _row_to_transaction_dict(test_df.iloc[0])
    result = engine.process_transaction(txn)
    assert "transaction_id" in result
    assert "risk_score" in result
    assert "decision" in result
    assert "behavioral_features" in result
    assert 0.0 <= result["risk_score"] <= 1.0
    assert result["decision"] in ("APPROVE", "REVIEW", "BLOCK")


def test_none_valued_numeric_field_does_not_break_prediction():
    """
    Regression test: a JSON-null -> Python-`None` value for a numeric
    field (as opposed to `float('nan')`) must not leave that column as
    pandas dtype `object` in the single-row transaction DataFrame, which
    LightGBM's native predict() rejects outright. Caught during real
    end-to-end API testing with genuine high-missingness transactions.
    """
    engine, train_df, val_df, test_df, schema = _build_engine_fixture()
    txn = _row_to_transaction_dict(test_df.iloc[0])
    txn["D1"] = None  # simulate a JSON null arriving for a numeric field
    txn["addr1"] = None
    result = engine.process_transaction(txn)  # must not raise
    assert 0.0 <= result["risk_score"] <= 1.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
