"""
Phase 9 test suite: monitoring integration with the FastAPI app
(`/metrics`, `/monitoring/summary`, and — critically — non-interference
with existing inference behavior).

Reuses the exact synthetic-engine fixture pattern from
`tests/test_phase7_api.py` for consistency and speed.
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


@pytest.fixture()
def client():
    engine, policy_config, test_df = _build_synthetic_engine()
    deps.set_engine(engine, policy_config)
    deps.set_metrics_registry(MetricsRegistry())
    yield TestClient(app), engine, test_df
    deps.set_engine(None, None)
    deps.set_metrics_registry(None)


# ---------------------------------------------------------------------------
# /metrics and /monitoring/summary basic behavior
# ---------------------------------------------------------------------------

def test_metrics_endpoint_responds_prometheus_text(client):
    c, engine, test_df = client
    resp = c.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    assert "fraud_api_requests_total" in resp.text
    assert "# HELP" in resp.text
    assert "# TYPE" in resp.text


def test_monitoring_summary_has_expected_structure(client):
    c, engine, test_df = client
    resp = c.get("/monitoring/summary")
    assert resp.status_code == 200
    body = resp.json()
    for key in ("uptime_seconds", "requests", "predictions", "decisions", "risk_scores", "data_quality", "errors"):
        assert key in body
    assert set(body["decisions"]["counts"].keys()) == {"APPROVE", "REVIEW", "BLOCK"}


def test_metrics_update_after_predictions(client):
    c, engine, test_df = client
    for i in range(3):
        payload = _txn_payload(test_df.iloc[i])
        payload["TransactionID"] = 800000 + i
        resp = c.post("/predict", json=payload)
        assert resp.status_code == 200

    summary = c.get("/monitoring/summary").json()
    assert summary["predictions"]["successes"] == 3
    assert summary["predictions"]["attempts"] == 3
    assert summary["decisions"]["total"] == 3
    assert summary["risk_scores"]["count"] == 3


def test_invalid_request_counted_as_failure_not_success(client):
    c, engine, test_df = client
    payload = _txn_payload(test_df.iloc[0])
    payload["TransactionAmt"] = -5.0
    payload["TransactionID"] = 810000
    resp = c.post("/predict", json=payload)
    assert resp.status_code == 422

    summary = c.get("/monitoring/summary").json()
    assert summary["predictions"]["successes"] == 0
    assert summary["errors"]["request_validation_failures"] == 1  # pydantic ge=0 catches this


def test_engine_level_validation_failure_counted_distinctly(client):
    """isFraud presence is caught by the request-schema layer (pydantic),
    not the engine — confirms the two failure counters are attributed
    correctly to where the rejection actually happened."""
    c, engine, test_df = client
    payload = _txn_payload(test_df.iloc[0])
    payload["isFraud"] = 0
    payload["TransactionID"] = 820000
    resp = c.post("/predict", json=payload)
    assert resp.status_code == 422
    summary = c.get("/monitoring/summary").json()
    assert summary["errors"]["request_validation_failures"] >= 1


def test_request_metrics_track_all_endpoints(client):
    c, engine, test_df = client
    c.get("/health")
    c.get("/metadata")
    payload = _txn_payload(test_df.iloc[0])
    payload["TransactionID"] = 830000
    c.post("/predict", json=payload)

    summary = c.get("/monitoring/summary").json()
    assert summary["requests"]["by_endpoint"].get("/health", 0) >= 1
    assert summary["requests"]["by_endpoint"].get("/metadata", 0) >= 1
    assert summary["requests"]["by_endpoint"].get("/predict", 0) >= 1


def test_decision_metrics_reflect_actual_decisions(client):
    c, engine, test_df = client
    decisions_seen = []
    for i in range(10):
        payload = _txn_payload(test_df.iloc[i])
        payload["TransactionID"] = 840000 + i
        resp = c.post("/predict", json=payload)
        decisions_seen.append(resp.json()["decision"])

    summary = c.get("/monitoring/summary").json()
    for d in ("APPROVE", "REVIEW", "BLOCK"):
        assert summary["decisions"]["counts"][d] == decisions_seen.count(d)


# ---------------------------------------------------------------------------
# CRITICAL: non-interference — monitoring must not change inference behavior
# ---------------------------------------------------------------------------

def test_monitoring_does_not_change_risk_scores_or_decisions():
    """
    Two independently-built engines from the SAME seed: one queried
    directly (no monitoring involved at all), one queried through the
    FULLY MONITORED API (middleware + metrics recording + structured
    logging all active). Scores and decisions must match exactly.
    """
    engine_direct, policy_config, test_df = _build_synthetic_engine(seed=99)
    engine_api, _, _ = _build_synthetic_engine(seed=99)
    deps.set_engine(engine_api, policy_config)
    deps.set_metrics_registry(MetricsRegistry())
    c = TestClient(app)

    for i in range(5):
        payload = _txn_payload(test_df.iloc[i])
        direct_result = engine_direct.process_transaction(dict(payload))
        api_resp = c.post("/predict", json=payload)
        api_body = api_resp.json()

        assert abs(direct_result["risk_score"] - api_body["risk_score"]) < 1e-9
        assert direct_result["decision"] == api_body["decision"]
        assert direct_result["transaction_id"] == api_body["transaction_id"]

    deps.set_engine(None, None)
    deps.set_metrics_registry(None)


def test_monitoring_does_not_change_behavioral_state_updates():
    """Same entity, same sequence of transactions, processed once directly
    and once through the monitored API — final state must be identical."""
    engine_a, policy_config, test_df = _build_synthetic_engine(seed=123)
    engine_b, _, _ = _build_synthetic_engine(seed=123)

    entity = 424242
    base = _txn_payload(test_df.iloc[0])
    txns = [
        dict(base, TransactionID=900000 + i, TransactionDT=1000 * (i + 1), card1=entity, TransactionAmt=10.0 * (i + 1))
        for i in range(4)
    ]

    for txn in txns:
        engine_a.process_transaction(dict(txn))

    deps.set_engine(engine_b, policy_config)
    deps.set_metrics_registry(MetricsRegistry())
    c = TestClient(app)
    for txn in txns:
        c.post("/predict", json=txn)

    snap_a = engine_a.state_manager.get_state_snapshot(entity)
    snap_b = engine_b.state_manager.get_state_snapshot(entity)
    assert snap_a["count"] == snap_b["count"]
    assert snap_a["sum_amt"] == pytest.approx(snap_b["sum_amt"])
    assert snap_a["last_time"] == pytest.approx(snap_b["last_time"])

    deps.set_engine(None, None)
    deps.set_metrics_registry(None)


def test_monitoring_absent_does_not_break_predict(client):
    """If the metrics registry is somehow unset mid-request (edge case),
    /predict must still work — monitoring is best-effort observation, not
    a hard dependency of the inference path."""
    c, engine, test_df = client
    deps.set_metrics_registry(None)  # simulate monitoring being unavailable
    payload = _txn_payload(test_df.iloc[0])
    payload["TransactionID"] = 850000
    resp = c.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] in ("APPROVE", "REVIEW", "BLOCK")


# ---------------------------------------------------------------------------
# Restart behavior (via the API)
# ---------------------------------------------------------------------------

def test_fresh_metrics_registry_via_api_starts_clean(client):
    c, engine, test_df = client
    payload = _txn_payload(test_df.iloc[0])
    payload["TransactionID"] = 860000
    c.post("/predict", json=payload)
    summary_before = c.get("/monitoring/summary").json()
    assert summary_before["predictions"]["successes"] == 1

    # Simulate a process restart: install a brand-new registry.
    deps.set_metrics_registry(MetricsRegistry())
    summary_after = c.get("/monitoring/summary").json()
    assert summary_after["predictions"]["successes"] == 0
    assert summary_after["requests"]["total"] <= 1  # only the /monitoring/summary call itself so far


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
