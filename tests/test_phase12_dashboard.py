"""
Phase 12 test suite: dashboard imports, API client helper functions,
artifact loaders, and engine fallback — all against synthetic/real data
where practical, without launching an actual Streamlit server (Streamlit
apps are not designed to be imported/run as plain functions the way
FastAPI routes are; page-level correctness is instead validated by the
real `streamlit run` smoke test documented in the Phase 12 report).
"""

import sys
from pathlib import Path

import pytest
import responses

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.utils import api_client, artifacts, ui


# ---------------------------------------------------------------------------
# Dashboard imports successfully
# ---------------------------------------------------------------------------

def test_dashboard_utils_import_cleanly():
    import dashboard.utils.api_client  # noqa: F401
    import dashboard.utils.artifacts  # noqa: F401
    import dashboard.utils.engine_fallback  # noqa: F401
    import dashboard.utils.session  # noqa: F401
    import dashboard.utils.ui  # noqa: F401


def test_dashboard_page_files_are_syntactically_valid():
    import ast
    pages_dir = REPO_ROOT / "dashboard" / "pages"
    page_files = [f for f in pages_dir.glob("*.py") if f.name != "__init__.py"]
    assert len(page_files) >= 6
    for f in page_files:
        ast.parse(f.read_text(encoding="utf-8"))  # raises SyntaxError if invalid
    ast.parse((REPO_ROOT / "dashboard" / "app.py").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# API client — helper functions (mocked HTTP, no real server needed)
# ---------------------------------------------------------------------------

@responses.activate
def test_is_api_available_true_on_200():
    responses.add(responses.GET, "http://fake-host:8000/health", json={"status": "ok"}, status=200)
    assert api_client.is_api_available("http://fake-host:8000") is True


def test_is_api_available_false_on_connection_error():
    # No responses mocked/active -> the real request fails to connect.
    assert api_client.is_api_available("http://this-host-does-not-exist.invalid:9999", timeout=1.0) is False


@responses.activate
def test_predict_success():
    responses.add(
        responses.POST, "http://fake-host:8000/predict",
        json={"transaction_id": 1, "risk_score": 0.1, "decision": "APPROVE"}, status=200,
    )
    ok, body = api_client.predict({"TransactionID": 1}, base_url="http://fake-host:8000")
    assert ok is True
    assert body["decision"] == "APPROVE"


@responses.activate
def test_predict_validation_error_returns_false_not_exception():
    responses.add(
        responses.POST, "http://fake-host:8000/predict",
        json={"detail": "invalid"}, status=422,
    )
    ok, body = api_client.predict({"TransactionID": 1}, base_url="http://fake-host:8000")
    assert ok is False
    assert body["status_code"] == 422


@responses.activate
def test_get_monitoring_summary():
    responses.add(
        responses.GET, "http://fake-host:8000/monitoring/summary",
        json={"requests": {"total": 5}}, status=200,
    )
    ok, body = api_client.get_monitoring_summary("http://fake-host:8000")
    assert ok is True
    assert body["requests"]["total"] == 5


@responses.activate
def test_get_drift_summary_unavailable():
    responses.add(
        responses.GET, "http://fake-host:8000/drift/summary",
        json={"available": False, "detail": "no reference profile"}, status=200,
    )
    ok, body = api_client.get_drift_summary("http://fake-host:8000")
    assert ok is True
    assert body["available"] is False


@responses.activate
def test_get_governance_summary():
    responses.add(
        responses.GET, "http://fake-host:8000/model-governance/summary",
        json={"available": True, "champion_model_id": "fraud-risk-lightgbm-v1"}, status=200,
    )
    ok, body = api_client.get_governance_summary("http://fake-host:8000")
    assert ok is True
    assert body["champion_model_id"] == "fraud-risk-lightgbm-v1"


@responses.activate
def test_analyze_drift_batch():
    responses.add(
        responses.POST, "http://fake-host:8000/drift/analyze",
        json={"overall_severity": "NO_SIGNIFICANT_DRIFT"}, status=200,
    )
    ok, body = api_client.analyze_drift_batch([{"TransactionAmt": 10.0}], base_url="http://fake-host:8000")
    assert ok is True
    assert body["overall_severity"] == "NO_SIGNIFICANT_DRIFT"


# ---------------------------------------------------------------------------
# Artifact loaders — real repository artifacts
# ---------------------------------------------------------------------------

def test_load_phase4_metrics_if_present():
    result = artifacts.load_phase4_metrics()
    if result is not None:
        assert "model_b_behavioral" in result


def test_get_champion_test_metrics_structure():
    result = artifacts.get_champion_test_metrics()
    if result is not None:
        assert "test_ranking" in result
        assert "pr_auc" in result["test_ranking"]
        assert "test_budget_table" in result


def test_get_champion_policy_test_eval_structure():
    result = artifacts.get_champion_policy_test_eval()
    if result is not None:
        assert "buckets" in result
        assert set(result["buckets"].keys()) == {"APPROVE", "REVIEW", "BLOCK"}


def test_load_reference_profile_structure():
    result = artifacts.load_reference_profile()
    if result is not None:
        assert "metadata" in result
        assert "numeric" in result and "categorical" in result and "behavioral" in result


def test_load_model_registry_structure():
    result = artifacts.load_model_registry()
    if result is not None:
        assert "champion_model_id" in result
        assert "models" in result


def test_load_decision_policy_structure():
    result = artifacts.load_decision_policy()
    if result is not None:
        assert "approve_threshold" in result
        assert "block_threshold" in result


def test_missing_artifact_returns_none_not_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "PHASE4_METRICS_PATH", tmp_path / "does_not_exist.json")
    assert artifacts.load_phase4_metrics() is None


def test_read_cached_test_summary_returns_none_or_dict():
    result = artifacts.read_cached_test_summary()
    assert result is None or isinstance(result, dict)


# ---------------------------------------------------------------------------
# UI helper functions
# ---------------------------------------------------------------------------

def test_severity_badge_covers_all_known_severities():
    for sev in ("NO_SIGNIFICANT_DRIFT", "LOW_DRIFT", "MODERATE_DRIFT", "HIGH_DRIFT"):
        badge = ui.severity_badge(sev)
        assert sev.split("_")[0].title() in badge or "No" in badge


def test_decision_badge_covers_all_decisions():
    for d in ("APPROVE", "REVIEW", "BLOCK"):
        assert d in ui.decision_badge(d)


def test_recommendation_badge_covers_all_recommendations():
    for r in ("PROMOTE", "REJECT", "REQUIRES_REVIEW"):
        assert r in ui.recommendation_badge(r)


def test_api_status_badge_reflects_availability():
    assert "Live" in ui.api_status_badge(True)
    assert "fallback" in ui.api_status_badge(False)


# ---------------------------------------------------------------------------
# Engine fallback — real local RiskDecisionEngine (no live API needed)
# ---------------------------------------------------------------------------

def test_build_engine_for_dashboard_and_score_valid_transaction():
    from dashboard.utils.engine_fallback import build_engine_for_dashboard, score_transaction_locally

    engine = build_engine_for_dashboard(warm_start=False)
    transaction = {
        "TransactionID": 9999901, "TransactionDT": 100000, "TransactionAmt": 55.0,
        "ProductCD": "W", "card1": 12345, "card4": "visa", "card6": "debit",
    }
    ok, result = score_transaction_locally(engine, transaction)
    assert ok is True
    assert 0.0 <= result["risk_score"] <= 1.0
    assert result["decision"] in ("APPROVE", "REVIEW", "BLOCK")
    assert "behavioral_features" in result  # available via the direct engine, unlike the API response


def test_score_transaction_locally_invalid_input_handled_cleanly():
    from dashboard.utils.engine_fallback import build_engine_for_dashboard, score_transaction_locally

    engine = build_engine_for_dashboard(warm_start=False)
    bad_transaction = {
        "TransactionID": 9999902, "TransactionDT": 100000, "TransactionAmt": -5.0,  # invalid: negative
        "ProductCD": "W", "card1": 12345,
    }
    ok, result = score_transaction_locally(engine, bad_transaction)
    assert ok is False
    assert "error" in result


def test_score_transaction_locally_isfraud_rejected():
    from dashboard.utils.engine_fallback import build_engine_for_dashboard, score_transaction_locally

    engine = build_engine_for_dashboard(warm_start=False)
    bad_transaction = {
        "TransactionID": 9999903, "TransactionDT": 100000, "TransactionAmt": 20.0,
        "ProductCD": "W", "card1": 12345, "isFraud": 0,
    }
    ok, result = score_transaction_locally(engine, bad_transaction)
    assert ok is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
