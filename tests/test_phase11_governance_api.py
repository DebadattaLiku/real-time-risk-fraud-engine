"""
Phase 11 test suite: `/model-governance/summary` API integration, plus the
critical non-interference validation (model governance must not change
any inference behavior).
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
from src.governance.registry import ModelRegistry
from src.governance.metadata import build_model_metadata


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


def _make_test_registry(tmp_path) -> ModelRegistry:
    reg = ModelRegistry(tmp_path / "registry.json")
    meta = build_model_metadata(
        model_id="test-champ-v1", model_version="v1", model_type="LightGBM", status="CHAMPION",
        feature_schema_version="phase1-v1", training_data_reference="train", validation_data_reference="val",
        evaluation_metrics={"val_ranking": {"pr_auc": 0.5, "roc_auc": 0.9}}, artifact_reference="fake.pkl",
        decision_policy_version="balanced",
    )
    reg.register_champion(meta)
    return reg


@pytest.fixture()
def client(tmp_path):
    engine, policy_config, test_df = _build_synthetic_engine()
    deps.set_engine(engine, policy_config)
    deps.set_metrics_registry(MetricsRegistry())
    deps.set_model_registry(_make_test_registry(tmp_path))
    yield TestClient(app), engine, test_df
    deps.set_engine(None, None)
    deps.set_metrics_registry(None)
    deps.set_model_registry(None)


# ---------------------------------------------------------------------------
# /model-governance/summary
# ---------------------------------------------------------------------------

def test_governance_summary_available_when_registry_loaded(client):
    c, engine, test_df = client
    resp = c.get("/model-governance/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True
    assert body["champion_model_id"] == "test-champ-v1"
    assert body["champion_status"] == "CHAMPION"


def test_governance_summary_unavailable_when_no_registry():
    deps.set_model_registry(None)
    c = TestClient(app)
    resp = c.get("/model-governance/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is False


def test_governance_summary_reflects_candidate_registration(client, tmp_path):
    c, engine, test_df = client
    registry = deps.get_model_registry_or_none()
    cand_meta = build_model_metadata(
        model_id="test-cand-v1", model_version="v1", model_type="LightGBM", status="CANDIDATE",
        feature_schema_version="phase1-v1", training_data_reference="train", validation_data_reference="val",
        evaluation_metrics={"val_ranking": {"pr_auc": 0.55, "roc_auc": 0.91}}, artifact_reference="fake2.pkl",
        decision_policy_version="balanced",
    )
    registry.register_candidate(cand_meta)

    body = c.get("/model-governance/summary").json()
    assert body["n_candidates"] == 1


# ---------------------------------------------------------------------------
# CRITICAL: non-interference — governance must not change inference
# ---------------------------------------------------------------------------

def test_governance_registry_does_not_change_predict_behavior(tmp_path):
    """Two identical engines: one queried through an API with NO governance
    registry loaded, one with a registry loaded (and a candidate
    registered in between predict calls) — /predict results identical."""
    engine_a, policy_config, test_df = _build_synthetic_engine(seed=77)
    engine_b, _, _ = _build_synthetic_engine(seed=77)

    deps.set_engine(engine_a, policy_config)
    deps.set_metrics_registry(MetricsRegistry())
    deps.set_model_registry(None)
    c_a = TestClient(app)

    payload = _txn_payload(test_df.iloc[2])
    resp_a = c_a.post("/predict", json=payload)
    body_a = resp_a.json()

    deps.set_engine(engine_b, policy_config)
    deps.set_metrics_registry(MetricsRegistry())
    registry = _make_test_registry(tmp_path)
    deps.set_model_registry(registry)
    c_b = TestClient(app)
    # Exercise governance in between — register a candidate, check summary.
    cand_meta = build_model_metadata(
        model_id="test-cand-v1", model_version="v1", model_type="LightGBM", status="CANDIDATE",
        feature_schema_version="phase1-v1", training_data_reference="train", validation_data_reference="val",
        evaluation_metrics={"val_ranking": {"pr_auc": 0.55, "roc_auc": 0.91}}, artifact_reference="fake2.pkl",
        decision_policy_version="balanced",
    )
    registry.register_candidate(cand_meta)
    c_b.get("/model-governance/summary")

    resp_b = c_b.post("/predict", json=payload)
    body_b = resp_b.json()

    assert abs(body_a["risk_score"] - body_b["risk_score"]) < 1e-9
    assert body_a["decision"] == body_b["decision"]

    deps.set_engine(None, None)
    deps.set_metrics_registry(None)
    deps.set_model_registry(None)


def test_governance_summary_never_promotes_or_mutates_champion(client):
    """Calling /model-governance/summary repeatedly must never change
    which model is champion — it is strictly read-only."""
    c, engine, test_df = client
    registry = deps.get_model_registry_or_none()
    champ_before = registry.get_champion()["model_id"]
    for _ in range(5):
        c.get("/model-governance/summary")
    champ_after = registry.get_champion()["model_id"]
    assert champ_before == champ_after


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
