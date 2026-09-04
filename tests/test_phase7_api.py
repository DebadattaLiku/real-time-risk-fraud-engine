"""
Phase 7 test suite: FastAPI risk scoring service.

Uses a small synthetic engine (same pattern as Phase 6's synthetic
fixture) injected directly via `src.api.dependencies.set_engine`, and
`TestClient(app)` WITHOUT the `with` context manager — this deliberately
skips the app's `lifespan` startup (which would otherwise load the real
~600MB dataset), confirmed empirically to work with FastAPI/Starlette's
TestClient. This keeps the automated suite fast and independent of real
data, while manual verification against the real bundle is documented
separately in the Phase 7 report.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

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
from src.engine.risk_engine import RiskDecisionEngine

from src.api import dependencies as deps
from src.api.main import app


# ---------------------------------------------------------------------------
# Synthetic engine fixture (mirrors Phase 6's pattern)
# ---------------------------------------------------------------------------

def _make_synthetic_transactions(n=800, seed=0):
    rng = np.random.default_rng(seed)
    amt = rng.uniform(1, 500, size=n).astype("float32")
    c1 = rng.integers(0, 10, size=n).astype("float32")
    fraud_logit = -3.0 + 0.01 * amt + 0.3 * c1
    fraud_prob = 1 / (1 + np.exp(-fraud_logit))
    is_fraud = (rng.random(n) < fraud_prob).astype(int)
    return pd.DataFrame({
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


def _build_synthetic_engine(n=800, seed=0):
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

    Z_train_full = attach(Z_train, train_df["TransactionID"].reset_index(drop=True))
    Z_val_full = attach(Z_val, val_df["TransactionID"].reset_index(drop=True))

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
    return engine, {"policy_name": "test_policy", "approve_threshold": 0.3, "block_threshold": 0.7}, test_df


@pytest.fixture()
def client():
    engine, policy_config, test_df = _build_synthetic_engine()
    deps.set_engine(engine, policy_config)
    yield TestClient(app), engine, test_df
    deps.set_engine(None, None)  # cleanup so tests don't leak state into each other


def _txn_payload(row: pd.Series) -> dict:
    d = row.to_dict()
    d.pop("isFraud", None)
    return {k: (None if (isinstance(v, float) and v != v) else v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health_ok_when_engine_initialized(client):
    c, engine, test_df = client
    resp = c.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["policy_loaded"] is True


def test_health_degraded_when_engine_not_initialized():
    deps.set_engine(None, None)
    c = TestClient(app)
    resp = c.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def test_metadata_returns_model_and_policy_info(client):
    c, engine, test_df = client
    resp = c.get("/metadata")
    assert resp.status_code == 200
    body = resp.json()
    assert body["policy_name"] == "test_policy"
    assert body["approve_threshold"] == pytest.approx(0.3)
    assert body["block_threshold"] == pytest.approx(0.7)
    assert set(body["supported_decisions"]) == {"APPROVE", "REVIEW", "BLOCK"}
    assert "model_type" in body and "api_version" in body


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def test_predict_valid_request_returns_expected_fields(client):
    c, engine, test_df = client
    payload = _txn_payload(test_df.iloc[0])
    resp = c.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["transaction_id"] == payload["TransactionID"]
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["decision"] in ("APPROVE", "REVIEW", "BLOCK")
    assert body["processing_status"] == "success"
    assert "model_version" in body and "policy_version" in body


def test_predict_missing_required_field_returns_422(client):
    c, engine, test_df = client
    payload = _txn_payload(test_df.iloc[0])
    del payload["TransactionAmt"]
    resp = c.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_negative_amount_returns_422(client):
    c, engine, test_df = client
    payload = _txn_payload(test_df.iloc[0])
    payload["TransactionAmt"] = -10.0
    resp = c.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_with_isfraud_field_rejected(client):
    c, engine, test_df = client
    payload = _txn_payload(test_df.iloc[0])
    payload["isFraud"] = 0
    resp = c.post("/predict", json=payload)
    assert resp.status_code == 422
    assert "isFraud" in resp.text  # pydantic's RequestValidationError body, not our custom handler


def test_predict_response_never_leaks_traceback(client):
    c, engine, test_df = client
    payload = _txn_payload(test_df.iloc[0])
    payload["TransactionAmt"] = "not_a_number"
    resp = c.post("/predict", json=payload)
    assert resp.status_code == 422
    assert "Traceback" not in resp.text
    assert ".py" not in resp.text  # no file path / stack frame leakage


# ---------------------------------------------------------------------------
# Stateful behavior
# ---------------------------------------------------------------------------

def test_sequential_requests_preserve_behavioral_state(client):
    c, engine, test_df = client
    entity = 777777
    base = _txn_payload(test_df.iloc[0])

    txn1 = dict(base, TransactionID=700001, TransactionDT=1000, card1=entity, TransactionAmt=10.0)
    resp1 = c.post("/predict", json=txn1)
    assert resp1.status_code == 200

    # Directly verify state advanced (the response itself deliberately
    # does not expose behavioral features — see schemas.py).
    snap = engine.state_manager.get_state_snapshot(entity)
    assert snap["count"] == 1

    txn2 = dict(base, TransactionID=700002, TransactionDT=2000, card1=entity, TransactionAmt=20.0)
    resp2 = c.post("/predict", json=txn2)
    assert resp2.status_code == 200
    snap2 = engine.state_manager.get_state_snapshot(entity)
    assert snap2["count"] == 2
    assert snap2["sum_amt"] == pytest.approx(30.0)


def test_dev_reset_state_clears_history(client):
    c, engine, test_df = client
    entity = 666666
    base = _txn_payload(test_df.iloc[0])
    txn = dict(base, TransactionID=600001, TransactionDT=1000, card1=entity, TransactionAmt=10.0)
    c.post("/predict", json=txn)
    assert engine.state_manager.get_state_snapshot(entity) is not None

    resp = c.post("/dev/reset-state")
    assert resp.status_code == 200
    assert engine.state_manager.get_state_snapshot(entity) is None


# ---------------------------------------------------------------------------
# Invalid input must not update state
# ---------------------------------------------------------------------------

def test_invalid_request_does_not_update_state(client):
    c, engine, test_df = client
    entity = 555555
    base = _txn_payload(test_df.iloc[0])
    bad_txn = dict(base, TransactionID=500001, TransactionDT=1000, card1=entity, TransactionAmt=-5.0)
    resp = c.post("/predict", json=bad_txn)
    assert resp.status_code == 422
    assert engine.state_manager.get_state_snapshot(entity) is None  # never touched


# ---------------------------------------------------------------------------
# Direct engine vs API parity
# ---------------------------------------------------------------------------

def test_direct_engine_vs_api_parity():
    """
    Build TWO independent engines from the SAME synthetic data/seed (so
    they start with identical state), process the SAME transaction
    directly through one and via the API through the other, and confirm
    scores/decisions match within tolerance.
    """
    engine_direct, policy_config, test_df = _build_synthetic_engine(seed=42)
    engine_api, _, _ = _build_synthetic_engine(seed=42)
    deps.set_engine(engine_api, policy_config)
    c = TestClient(app)

    payload = _txn_payload(test_df.iloc[5])

    direct_result = engine_direct.process_transaction(dict(payload))
    api_resp = c.post("/predict", json=payload)
    assert api_resp.status_code == 200
    api_body = api_resp.json()

    assert abs(direct_result["risk_score"] - api_body["risk_score"]) < 1e-6
    assert direct_result["decision"] == api_body["decision"]
    assert direct_result["transaction_id"] == api_body["transaction_id"]

    deps.set_engine(None, None)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
