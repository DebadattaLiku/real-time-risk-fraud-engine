"""
Production upgrade — SHAP explainability tests, against the REAL champion
model bundle (not a synthetic model) — explanation quality/plausibility
isn't asserted (that's inherently subjective), but the critical safety
property IS: explanations must never change risk_score or decision.
"""

import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.api.dependencies import build_engine
from src.explainability.shap_explainer import FraudExplainer


@pytest.fixture(scope="module")
def real_engine_with_explainer():
    engine, _ = build_engine(warm_start=False)
    engine.explainer = FraudExplainer(engine.model, top_k=5)
    return engine


def _real_transaction(engine, **overrides) -> dict:
    base = {c: None for c in engine.get_expected_raw_columns()}
    base.update({"TransactionID": 1, "TransactionDT": 100000, "TransactionAmt": 100.0, "card1": 55555, "ProductCD": "W"})
    base.update(overrides)
    return base


def test_explain_false_by_default_no_explanation_key_populated(real_engine_with_explainer):
    engine = real_engine_with_explainer
    result = engine.process_transaction(_real_transaction(engine, TransactionID=101))
    assert result["explanation"] is None


def test_explain_true_returns_top_features(real_engine_with_explainer):
    engine = real_engine_with_explainer
    result = engine.process_transaction(_real_transaction(engine, TransactionID=102), explain=True)
    assert result["explanation"] is not None
    assert "top_features" in result["explanation"]
    assert len(result["explanation"]["top_features"]) == 5
    for f in result["explanation"]["top_features"]:
        assert "feature" in f and "impact" in f
        assert isinstance(f["impact"], float)


def test_explanation_does_not_change_risk_score_or_decision(real_engine_with_explainer):
    """The critical safety property: explain=True vs explain=False must
    produce IDENTICAL risk_score and decision for the same transaction.
    Uses two DIFFERENT entities so neither call's state update (which
    happens for every process_transaction call, by design — see
    src/engine/risk_engine.py) can influence the other's behavioral
    features; otherwise this would be comparing two genuinely different
    inputs, not testing the explanation's non-interference property."""
    engine = real_engine_with_explainer
    txn_a = _real_transaction(engine, TransactionID=201, TransactionAmt=2500.0, card1=910001)
    txn_b = _real_transaction(engine, TransactionID=202, TransactionAmt=2500.0, card1=910002)

    result_no_explain = engine.process_transaction(txn_a, explain=False)
    result_with_explain = engine.process_transaction(txn_b, explain=True)

    assert result_no_explain["risk_score"] == pytest.approx(result_with_explain["risk_score"])
    assert result_no_explain["decision"] == result_with_explain["decision"]


def test_build_engine_configures_shap_explainer_by_default():
    """New default behavior: ENABLE_SHAP_EXPLAINABILITY defaults to true,
    so a freshly built engine should have a real, working explainer
    without any extra configuration."""
    engine, _ = build_engine(warm_start=False)
    assert engine.explainer is not None
    result = engine.process_transaction(_real_transaction(engine, TransactionID=999), explain=True)
    assert "top_features" in result["explanation"]


def test_explanation_without_configured_explainer_does_not_crash():
    """If explain=True is requested but no explainer was configured on
    this engine instance, the engine must degrade gracefully, never raise.
    build_engine() now configures a real explainer by default
    (ENABLE_SHAP_EXPLAINABILITY=true) — explicitly clear it here to
    simulate the "not configured" case this test targets."""
    engine, _ = build_engine(warm_start=False)
    engine.explainer = None
    result = engine.process_transaction(_real_transaction(engine, TransactionID=301), explain=True)
    assert result["risk_score"] is not None  # prediction still succeeded
    assert result["explanation"] is not None
    assert "error" in result["explanation"]


def test_explanation_latency_is_measured_and_reported(real_engine_with_explainer, capsys):
    """Real measurement — not a correctness assertion beyond sanity."""
    engine = real_engine_with_explainer
    latencies = []
    for i in range(20):
        result = engine.process_transaction(_real_transaction(engine, TransactionID=400 + i), explain=True)
        latencies.append(result["explanation"]["explanation_time_ms"])
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    print(f"\n[SHAP explanation latency, n=20] p50={p50:.3f}ms max={latencies[-1]:.3f}ms")
    assert p50 > 0  # real measurement, not asserting a specific threshold


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
