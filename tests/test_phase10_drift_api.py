"""
Phase 10 test suite: `/drift/summary` and `/drift/analyze` API integration,
plus the critical non-interference validation (drift monitoring must not
change any inference behavior).
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
from src.monitoring.metrics import MetricsRegistry
from src.drift.monitor import DriftMonitor
from src.drift.reference import build_reference_profile


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


def _txn_payload(row: pd.Series) -> dict:
    d = row.to_dict()
    d.pop("isFraud", None)
    return {k: (None if (isinstance(v, float) and v != v) else v) for k, v in d.items()}


def _make_synthetic_reference_profile(seed=0):
    rng = np.random.default_rng(seed)
    n = 500
    ref_df = pd.DataFrame({
        "TransactionAmt": rng.gamma(2.0, 50.0, size=n),
        "C1": rng.poisson(3, size=n).astype(float),
        "C13": rng.poisson(5, size=n).astype(float),
        "C14": rng.poisson(2, size=n).astype(float),
        "D2": rng.uniform(0, 100, size=n),
        "V258": rng.normal(0, 1, size=n),
        "ProductCD": rng.choice(["W", "C", "R"], size=n),
        "R_emaildomain": rng.choice(["gmail.com", "yahoo.com"], size=n),
        "bhv_prev_txn_count": rng.poisson(20, size=n).astype(float),
        "bhv_hist_mean_amt": rng.gamma(2.0, 50.0, size=n),
        "bhv_time_since_prev_txn": rng.exponential(3600, size=n),
        "bhv_prior_count_24h": rng.poisson(2, size=n).astype(float),
    })
    scores = rng.beta(1, 20, size=n)
    decisions = rng.choice(["APPROVE", "REVIEW", "BLOCK"], size=n, p=[0.97, 0.02, 0.01])
    return build_reference_profile(ref_df, scores, decisions)


@pytest.fixture()
def client():
    engine, policy_config, test_df = _build_synthetic_engine()
    deps.set_engine(engine, policy_config)
    deps.set_metrics_registry(MetricsRegistry())
    deps.set_drift_monitor(DriftMonitor(_make_synthetic_reference_profile()))
    yield TestClient(app), engine, test_df
    deps.set_engine(None, None)
    deps.set_metrics_registry(None)
    deps.set_drift_monitor(None)


# ---------------------------------------------------------------------------
# /drift/summary
# ---------------------------------------------------------------------------

def test_drift_summary_available_when_monitor_loaded(client):
    c, engine, test_df = client
    resp = c.get("/drift/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True
    assert body["batches_analyzed"] == 0  # nothing analyzed yet


def test_drift_summary_unavailable_when_no_reference_profile():
    deps.set_drift_monitor(None)
    c = TestClient(app)
    resp = c.get("/drift/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is False


def test_drift_summary_does_not_trigger_new_analysis(client):
    """Calling /drift/summary repeatedly must not itself count as analyzing
    a batch — it only reports the LATEST state."""
    c, engine, test_df = client
    c.get("/drift/summary")
    c.get("/drift/summary")
    c.get("/drift/summary")
    body = c.get("/drift/summary").json()
    assert body["batches_analyzed"] == 0


# ---------------------------------------------------------------------------
# /drift/analyze
# ---------------------------------------------------------------------------

def test_drift_analyze_valid_batch(client):
    c, engine, test_df = client
    records = [
        {"TransactionAmt": 50.0 + i, "ProductCD": "W", "bhv_prev_txn_count": 10.0, "risk_score": 0.05, "decision": "APPROVE"}
        for i in range(40)
    ]
    resp = c.post("/drift/analyze", json={"records": records})
    assert resp.status_code == 200
    body = resp.json()
    assert body["batch_size"] == 40
    assert body["overall_severity"] in ("NO_SIGNIFICANT_DRIFT", "LOW_DRIFT", "MODERATE_DRIFT", "HIGH_DRIFT")
    assert "feature_results" in body


def test_drift_analyze_empty_records_rejected(client):
    c, engine, test_df = client
    resp = c.post("/drift/analyze", json={"records": []})
    assert resp.status_code in (422,)


def test_drift_analyze_updates_summary(client):
    c, engine, test_df = client
    records = [{"TransactionAmt": 60.0, "ProductCD": "C", "risk_score": 0.1, "decision": "REVIEW"} for _ in range(35)]
    c.post("/drift/analyze", json={"records": records})
    summary = c.get("/drift/summary").json()
    assert summary["batches_analyzed"] == 1


def test_drift_analyze_unavailable_returns_503():
    deps.set_drift_monitor(None)
    c = TestClient(app)
    resp = c.post("/drift/analyze", json={"records": [{"TransactionAmt": 1.0}]})
    assert resp.status_code == 503


def test_drift_analyze_shifted_batch_detects_drift(client):
    c, engine, test_df = client
    # Reference TransactionAmt ~ gamma(2, 50) (mean ~100). Send a batch
    # with amounts far outside that range.
    records = [{"TransactionAmt": 5000.0 + i, "ProductCD": "W", "risk_score": 0.5, "decision": "REVIEW"} for i in range(40)]
    resp = c.post("/drift/analyze", json={"records": records})
    body = resp.json()
    amt_result = next(r for r in body["feature_results"] if r["feature"] == "TransactionAmt")
    assert amt_result["severity"] in ("MODERATE_DRIFT", "HIGH_DRIFT")


# ---------------------------------------------------------------------------
# CRITICAL: non-interference — drift monitoring must not change inference
# ---------------------------------------------------------------------------

def test_drift_monitoring_does_not_change_predict_behavior():
    """Two identical engines, one queried through an API WITHOUT a drift
    monitor at all, one WITH a drift monitor loaded and actively used
    between predict calls — /predict results must be identical either way."""
    engine_a, policy_config, test_df = _build_synthetic_engine(seed=55)
    engine_b, _, _ = _build_synthetic_engine(seed=55)

    deps.set_engine(engine_a, policy_config)
    deps.set_metrics_registry(MetricsRegistry())
    deps.set_drift_monitor(None)
    c_a = TestClient(app)

    payload = _txn_payload(test_df.iloc[3])
    resp_a = c_a.post("/predict", json=payload)
    body_a = resp_a.json()

    deps.set_engine(engine_b, policy_config)
    deps.set_metrics_registry(MetricsRegistry())
    deps.set_drift_monitor(DriftMonitor(_make_synthetic_reference_profile()))
    c_b = TestClient(app)
    # Exercise drift analysis in between, to prove it has no side effect
    # on the engine or subsequent predictions.
    c_b.post("/drift/analyze", json={"records": [
        {"TransactionAmt": 999.0, "ProductCD": "W", "risk_score": 0.9, "decision": "BLOCK"} for _ in range(40)
    ]})
    resp_b = c_b.post("/predict", json=payload)
    body_b = resp_b.json()

    assert abs(body_a["risk_score"] - body_b["risk_score"]) < 1e-9
    assert body_a["decision"] == body_b["decision"]

    deps.set_engine(None, None)
    deps.set_metrics_registry(None)
    deps.set_drift_monitor(None)


def test_drift_analyze_does_not_touch_behavioral_state(client):
    """POST /drift/analyze must never call the engine or mutate
    behavioral state for any entity — it only reads a batch DataFrame."""
    c, engine, test_df = client
    entity = 313131
    snap_before = engine.state_manager.get_state_snapshot(entity)  # None, never touched
    records = [{"TransactionAmt": 50.0, "ProductCD": "W", "card1": entity, "risk_score": 0.1, "decision": "APPROVE"} for _ in range(35)]
    c.post("/drift/analyze", json={"records": records})
    snap_after = engine.state_manager.get_state_snapshot(entity)
    assert snap_before == snap_after == None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
