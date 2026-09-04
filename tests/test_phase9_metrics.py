"""
Phase 9 test suite: MetricsRegistry (`src/monitoring/metrics.py`) — unit
tests using hand-verified small sequences, independent of FastAPI.
"""

import sys
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.monitoring.metrics import MetricsRegistry, _bucket_label


# ---------------------------------------------------------------------------
# Request metrics
# ---------------------------------------------------------------------------

def test_request_count_increases():
    reg = MetricsRegistry()
    reg.record_request("/predict", 200, 10.0)
    reg.record_request("/predict", 200, 20.0)
    reg.record_request("/health", 200, 1.0)
    snap = reg.snapshot()
    assert snap["requests"]["total"] == 3
    assert snap["requests"]["by_endpoint"]["/predict"] == 2
    assert snap["requests"]["by_endpoint"]["/health"] == 1


def test_request_status_class_buckets():
    reg = MetricsRegistry()
    reg.record_request("/predict", 200, 5.0)
    reg.record_request("/predict", 422, 5.0)
    reg.record_request("/predict", 503, 5.0)
    snap = reg.snapshot()
    assert snap["requests"]["by_status_class"]["2xx"] == 1
    assert snap["requests"]["by_status_class"]["4xx"] == 1
    assert snap["requests"]["by_status_class"]["5xx"] == 1


def test_request_avg_latency():
    reg = MetricsRegistry()
    reg.record_request("/predict", 200, 10.0)
    reg.record_request("/predict", 200, 30.0)
    snap = reg.snapshot()
    assert snap["requests"]["avg_latency_ms"] == pytest.approx(20.0)
    assert snap["requests"]["avg_latency_ms_by_endpoint"]["/predict"] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Prediction metrics
# ---------------------------------------------------------------------------

def test_prediction_attempts_successes_failures():
    reg = MetricsRegistry()
    reg.record_prediction_attempt()
    reg.record_prediction_success(decision="APPROVE", risk_score=0.1, duration_ms=5.0)
    reg.record_prediction_attempt()
    reg.record_engine_validation_failure()
    snap = reg.snapshot()
    assert snap["predictions"]["attempts"] == 2
    assert snap["predictions"]["successes"] == 1
    assert snap["predictions"]["failures"] == 1


def test_prediction_avg_engine_latency():
    reg = MetricsRegistry()
    reg.record_prediction_success(decision="APPROVE", risk_score=0.1, duration_ms=10.0)
    reg.record_prediction_success(decision="APPROVE", risk_score=0.1, duration_ms=20.0)
    snap = reg.snapshot()
    assert snap["predictions"]["avg_engine_latency_ms"] == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# Decision metrics
# ---------------------------------------------------------------------------

def test_decision_counters_hand_verified():
    reg = MetricsRegistry()
    reg.record_prediction_success(decision="APPROVE", risk_score=0.1, duration_ms=1.0)
    reg.record_prediction_success(decision="APPROVE", risk_score=0.2, duration_ms=1.0)
    reg.record_prediction_success(decision="REVIEW", risk_score=0.5, duration_ms=1.0)
    reg.record_prediction_success(decision="BLOCK", risk_score=0.9, duration_ms=1.0)
    snap = reg.snapshot()
    assert snap["decisions"]["counts"] == {"APPROVE": 2, "REVIEW": 1, "BLOCK": 1}
    assert snap["decisions"]["total"] == 4
    assert snap["decisions"]["distribution"]["APPROVE"] == pytest.approx(0.5)
    assert snap["decisions"]["distribution"]["REVIEW"] == pytest.approx(0.25)
    assert snap["decisions"]["distribution"]["BLOCK"] == pytest.approx(0.25)


def test_decision_totals_consistent_with_successes():
    reg = MetricsRegistry()
    for d in ("APPROVE", "APPROVE", "REVIEW", "BLOCK", "APPROVE"):
        reg.record_prediction_success(decision=d, risk_score=0.3, duration_ms=1.0)
    snap = reg.snapshot()
    assert snap["decisions"]["total"] == snap["predictions"]["successes"]


# ---------------------------------------------------------------------------
# Risk-score metrics
# ---------------------------------------------------------------------------

def test_risk_score_count_mean_min_max():
    reg = MetricsRegistry()
    scores = [0.1, 0.5, 0.9, 0.3]
    for s in scores:
        reg.record_prediction_success(decision="APPROVE", risk_score=s, duration_ms=1.0)
    snap = reg.snapshot()
    rs = snap["risk_scores"]
    assert rs["count"] == 4
    assert rs["mean"] == pytest.approx(sum(scores) / 4)
    assert rs["min"] == pytest.approx(0.1)
    assert rs["max"] == pytest.approx(0.9)


def test_risk_score_bucket_label_hand_verified():
    assert _bucket_label(0.0) == "0.0-0.1"
    assert _bucket_label(0.05) == "0.0-0.1"
    assert _bucket_label(0.15) == "0.1-0.2"
    assert _bucket_label(0.99) == "0.9-1.0"
    assert _bucket_label(1.0) == "0.9-1.0"  # exact 1.0 falls in the last bucket, not a new one


def test_risk_score_bucket_counts():
    reg = MetricsRegistry()
    for s in (0.05, 0.05, 0.15, 0.95):
        reg.record_prediction_success(decision="APPROVE", risk_score=s, duration_ms=1.0)
    snap = reg.snapshot()
    buckets = snap["risk_scores"]["bucket_counts"]
    assert buckets["0.0-0.1"] == 2
    assert buckets["0.1-0.2"] == 1
    assert buckets["0.9-1.0"] == 1


def test_risk_score_metrics_empty_state_returns_none_not_error():
    reg = MetricsRegistry()
    snap = reg.snapshot()
    assert snap["risk_scores"]["count"] == 0
    assert snap["risk_scores"]["mean"] is None
    assert snap["risk_scores"]["min"] is None
    assert snap["risk_scores"]["max"] is None


# ---------------------------------------------------------------------------
# Data-quality / error metrics
# ---------------------------------------------------------------------------

def test_transaction_amount_summary():
    reg = MetricsRegistry()
    reg.record_prediction_success(decision="APPROVE", risk_score=0.1, duration_ms=1.0, transaction_amount=10.0)
    reg.record_prediction_success(decision="APPROVE", risk_score=0.1, duration_ms=1.0, transaction_amount=50.0)
    snap = reg.snapshot()
    amt = snap["data_quality"]["transaction_amount"]
    assert amt["count"] == 2
    assert amt["mean"] == pytest.approx(30.0)
    assert amt["min"] == pytest.approx(10.0)
    assert amt["max"] == pytest.approx(50.0)


def test_validation_failure_counters_distinct():
    reg = MetricsRegistry()
    reg.record_request_validation_failure()
    reg.record_request_validation_failure()
    reg.record_engine_validation_failure()
    snap = reg.snapshot()
    assert snap["errors"]["request_validation_failures"] == 2
    assert snap["errors"]["engine_validation_failures"] == 1


def test_error_counters():
    reg = MetricsRegistry()
    reg.record_engine_not_initialized()
    reg.record_internal_error()
    reg.record_internal_error()
    snap = reg.snapshot()
    assert snap["errors"]["engine_not_initialized_errors"] == 1
    assert snap["errors"]["internal_errors"] == 2


# ---------------------------------------------------------------------------
# Thread safety (basic)
# ---------------------------------------------------------------------------

def test_concurrent_recording_no_lost_updates():
    reg = MetricsRegistry()
    n_threads = 8
    n_per_thread = 200

    def worker():
        for _ in range(n_per_thread):
            reg.record_prediction_success(decision="APPROVE", risk_score=0.5, duration_ms=1.0)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snap = reg.snapshot()
    assert snap["predictions"]["successes"] == n_threads * n_per_thread
    assert snap["risk_scores"]["count"] == n_threads * n_per_thread


# ---------------------------------------------------------------------------
# Restart behavior
# ---------------------------------------------------------------------------

def test_new_registry_starts_clean():
    reg1 = MetricsRegistry()
    reg1.record_request("/predict", 200, 1.0)
    reg1.record_prediction_success(decision="APPROVE", risk_score=0.5, duration_ms=1.0)

    reg2 = MetricsRegistry()  # simulates a process restart
    snap = reg2.snapshot()
    assert snap["requests"]["total"] == 0
    assert snap["predictions"]["successes"] == 0
    assert snap["decisions"]["total"] == 0
    assert snap["risk_scores"]["count"] == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
