"""
Phase 10 test suite: drift reference profile, statistical detectors, and
batch monitor — synthetic data throughout, independent of the real
dataset (fast, deterministic).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.drift.reference import (
    build_reference_profile, save_reference_profile, load_reference_profile,
    numeric_feature_profile, categorical_feature_profile,
    MONITORED_NUMERIC_FEATURES, MONITORED_CATEGORICAL_FEATURES, MONITORED_BEHAVIORAL_FEATURES,
)
from src.drift.detectors import (
    compute_numeric_drift, compute_categorical_drift, compute_decision_drift,
    severity_from_psi, _psi_from_percentages, SEVERITY_ORDER,
)
from src.drift.report import aggregate_overall_severity
from src.drift.monitor import DriftMonitor, MIN_BATCH_SIZE


def _make_reference_df(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "TransactionAmt": rng.gamma(shape=2.0, scale=50.0, size=n),
        "C1": rng.poisson(3, size=n).astype(float),
        "C13": rng.poisson(5, size=n).astype(float),
        "C14": rng.poisson(2, size=n).astype(float),
        "D2": rng.uniform(0, 100, size=n),
        "V258": rng.normal(0, 1, size=n),
        "ProductCD": rng.choice(["W", "C", "R", "H", "S"], size=n, p=[0.6, 0.15, 0.1, 0.1, 0.05]),
        "R_emaildomain": rng.choice(["gmail.com", "yahoo.com", "hotmail.com"], size=n, p=[0.5, 0.3, 0.2]),
        "bhv_prev_txn_count": rng.poisson(20, size=n).astype(float),
        "bhv_hist_mean_amt": rng.gamma(shape=2.0, scale=50.0, size=n),
        "bhv_time_since_prev_txn": rng.exponential(3600, size=n),
        "bhv_prior_count_24h": rng.poisson(2, size=n).astype(float),
    })


def _make_reference_profile(n=2000, seed=0):
    ref_df = _make_reference_df(n=n, seed=seed)
    rng = np.random.default_rng(seed + 1)
    risk_scores = rng.beta(1, 20, size=n)  # skewed toward low risk, like the real model
    decisions = rng.choice(["APPROVE", "REVIEW", "BLOCK"], size=n, p=[0.97, 0.02, 0.01])
    return build_reference_profile(ref_df, risk_scores, decisions), ref_df


# ---------------------------------------------------------------------------
# Reference profile
# ---------------------------------------------------------------------------

def test_reference_profile_can_be_created():
    profile, _ = _make_reference_profile()
    assert "metadata" in profile
    assert "numeric" in profile and "categorical" in profile and "behavioral" in profile
    assert "risk_score" in profile and "decision" in profile


def test_reference_profile_contains_all_monitored_features():
    profile, _ = _make_reference_profile()
    for f in MONITORED_NUMERIC_FEATURES:
        assert f in profile["numeric"]
    for f in MONITORED_CATEGORICAL_FEATURES:
        assert f in profile["categorical"]
    for f in MONITORED_BEHAVIORAL_FEATURES:
        assert f in profile["behavioral"]


def test_reference_profile_save_and_load_roundtrip(tmp_path):
    profile, _ = _make_reference_profile()
    path = tmp_path / "reference_profile.json"
    save_reference_profile(profile, path)
    assert path.is_file()
    loaded = load_reference_profile(path)
    assert loaded["metadata"]["n_reference_transactions"] == profile["metadata"]["n_reference_transactions"]
    assert loaded["numeric"]["TransactionAmt"]["mean"] == pytest.approx(profile["numeric"]["TransactionAmt"]["mean"])


def test_reference_profile_reproducible_given_same_data():
    ref_df = _make_reference_df(n=500, seed=7)
    scores = np.full(500, 0.1)
    decisions = ["APPROVE"] * 500
    p1 = build_reference_profile(ref_df, scores, decisions)
    p2 = build_reference_profile(ref_df, scores, decisions)
    assert p1["numeric"]["TransactionAmt"]["mean"] == p2["numeric"]["TransactionAmt"]["mean"]
    assert p1["numeric"]["TransactionAmt"]["bin_edges"] == p2["numeric"]["TransactionAmt"]["bin_edges"]


def test_numeric_feature_profile_handles_missing_values():
    values = pd.Series([1.0, 2.0, np.nan, 3.0, np.nan])
    profile = numeric_feature_profile(values)
    assert profile["n"] == 3
    assert profile["missing_pct"] == pytest.approx(2 / 5)


def test_categorical_feature_profile_frequencies_sum_to_one():
    values = ["A", "A", "B", "C", "A"]
    profile = categorical_feature_profile(values)
    assert sum(profile["category_frequencies"].values()) == pytest.approx(1.0)
    assert profile["category_frequencies"]["A"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Numerical drift
# ---------------------------------------------------------------------------

def test_psi_zero_for_identical_distributions():
    ref_pct = np.array([0.1, 0.2, 0.3, 0.4])
    cur_pct = np.array([0.1, 0.2, 0.3, 0.4])
    assert _psi_from_percentages(ref_pct, cur_pct) == pytest.approx(0.0, abs=1e-9)


def test_psi_positive_for_shifted_distribution():
    ref_pct = np.array([0.25, 0.25, 0.25, 0.25])
    cur_pct = np.array([0.7, 0.1, 0.1, 0.1])
    psi = _psi_from_percentages(ref_pct, cur_pct)
    assert psi > 0


def test_severity_from_psi_thresholds():
    assert severity_from_psi(0.01) == "NO_SIGNIFICANT_DRIFT"
    assert severity_from_psi(0.15) == "LOW_DRIFT"
    assert severity_from_psi(0.25) == "MODERATE_DRIFT"
    assert severity_from_psi(0.5) == "HIGH_DRIFT"


def test_stable_numeric_distribution_produces_low_drift():
    profile, ref_df = _make_reference_profile(n=3000, seed=1)
    rng = np.random.default_rng(999)
    # Resample from the SAME generating distribution -> should be stable.
    current = rng.gamma(shape=2.0, scale=50.0, size=500)
    result = compute_numeric_drift(profile["numeric"]["TransactionAmt"], current)
    assert result["severity"] in ("NO_SIGNIFICANT_DRIFT", "LOW_DRIFT")
    assert result["psi"] < 0.2


def test_shifted_numeric_distribution_produces_measurable_drift():
    profile, ref_df = _make_reference_profile(n=3000, seed=2)
    rng = np.random.default_rng(888)
    # A large, deliberate shift: 10x the scale.
    current = rng.gamma(shape=2.0, scale=500.0, size=500)
    result = compute_numeric_drift(profile["numeric"]["TransactionAmt"], current)
    assert result["psi"] > 0.2
    assert result["severity"] in ("MODERATE_DRIFT", "HIGH_DRIFT")
    assert result["current_mean"] > result["reference_mean"]


def test_numeric_drift_handles_zero_frequency_bins_safely():
    profile, ref_df = _make_reference_profile(n=1000, seed=3)
    # A degenerate current batch: all identical values, far outside the
    # reference range entirely (falls in one extreme bin only).
    current = np.full(50, 999999.0)
    result = compute_numeric_drift(profile["numeric"]["TransactionAmt"], current)
    assert np.isfinite(result["psi"])
    assert result["severity"] in SEVERITY_ORDER


def test_numeric_drift_handles_empty_current_batch():
    profile, ref_df = _make_reference_profile(n=500, seed=4)
    result = compute_numeric_drift(profile["numeric"]["TransactionAmt"], [])
    assert np.isfinite(result["psi"])  # must not be NaN/inf
    assert result["current_mean"] is None


def test_numeric_drift_ks_statistic_present_for_adequate_samples():
    profile, ref_df = _make_reference_profile(n=2000, seed=5)
    current = np.random.default_rng(1).gamma(2.0, 50.0, size=200)
    result = compute_numeric_drift(profile["numeric"]["TransactionAmt"], current)
    assert result["ks_statistic"] is not None
    assert 0.0 <= result["ks_statistic"] <= 1.0
    assert result["ks_p_value"] is not None


# ---------------------------------------------------------------------------
# Categorical drift
# ---------------------------------------------------------------------------

def test_stable_categorical_distribution_produces_low_drift():
    profile, ref_df = _make_reference_profile(n=3000, seed=6)
    rng = np.random.default_rng(777)
    current = rng.choice(["W", "C", "R", "H", "S"], size=500, p=[0.6, 0.15, 0.1, 0.1, 0.05])
    result = compute_categorical_drift(profile["categorical"]["ProductCD"], current)
    assert result["severity"] in ("NO_SIGNIFICANT_DRIFT", "LOW_DRIFT")
    assert result["new_categories"] == []


def test_changed_categorical_distribution_is_detected():
    profile, ref_df = _make_reference_profile(n=3000, seed=8)
    # Deliberately flip the distribution: mostly "S" instead of mostly "W".
    current = ["S"] * 450 + ["W"] * 50
    result = compute_categorical_drift(profile["categorical"]["ProductCD"], current)
    assert result["psi"] > 0.2
    assert result["severity"] in ("MODERATE_DRIFT", "HIGH_DRIFT")


def test_new_category_surfaced_explicitly():
    profile, ref_df = _make_reference_profile(n=1000, seed=9)
    current = ["W"] * 90 + ["ZZZ_NEVER_SEEN"] * 10
    result = compute_categorical_drift(profile["categorical"]["ProductCD"], current)
    assert "ZZZ_NEVER_SEEN" in result["new_categories"]


def test_missing_category_handled_safely_not_as_error():
    profile, ref_df = _make_reference_profile(n=1000, seed=10)
    # Only "W" present in the current batch -> every other reference
    # category is "missing" from this batch.
    current = ["W"] * 100
    result = compute_categorical_drift(profile["categorical"]["ProductCD"], current)
    assert np.isfinite(result["psi"])
    assert set(result["missing_categories"]) == {"C", "H", "R", "S"}


# ---------------------------------------------------------------------------
# Behavioral-feature drift
# ---------------------------------------------------------------------------

def test_behavioral_feature_drift_uses_exact_existing_names():
    profile, ref_df = _make_reference_profile(n=1000, seed=11)
    for f in MONITORED_BEHAVIORAL_FEATURES:
        assert f.startswith("bhv_")  # exact Phase 4 naming convention, not invented


def test_behavioral_feature_stable_vs_shifted():
    profile, ref_df = _make_reference_profile(n=2000, seed=12)
    rng = np.random.default_rng(222)
    stable_current = rng.poisson(20, size=300).astype(float)
    stable_result = compute_numeric_drift(profile["behavioral"]["bhv_prev_txn_count"], stable_current)
    assert stable_result["severity"] in ("NO_SIGNIFICANT_DRIFT", "LOW_DRIFT")

    shifted_current = rng.poisson(200, size=300).astype(float)  # 10x higher velocity
    shifted_result = compute_numeric_drift(profile["behavioral"]["bhv_prev_txn_count"], shifted_current)
    assert shifted_result["psi"] > stable_result["psi"]


# ---------------------------------------------------------------------------
# Risk-score drift
# ---------------------------------------------------------------------------

def test_similar_risk_scores_produce_low_drift():
    profile, ref_df = _make_reference_profile(n=3000, seed=13)
    rng = np.random.default_rng(333)
    current_scores = rng.beta(1, 20, size=500)
    result = compute_numeric_drift(profile["risk_score"], current_scores)
    assert result["severity"] in ("NO_SIGNIFICANT_DRIFT", "LOW_DRIFT")


def test_shifted_risk_scores_produce_detected_drift():
    profile, ref_df = _make_reference_profile(n=3000, seed=14)
    rng = np.random.default_rng(444)
    # Reference is beta(1,20) (skewed low); shift heavily toward high risk.
    current_scores = rng.beta(20, 1, size=500)
    result = compute_numeric_drift(profile["risk_score"], current_scores)
    assert result["psi"] > 0.2
    assert result["severity"] in ("MODERATE_DRIFT", "HIGH_DRIFT")
    assert result["current_mean"] > result["reference_mean"]


# ---------------------------------------------------------------------------
# Decision-distribution drift
# ---------------------------------------------------------------------------

def test_decision_drift_stable():
    profile, ref_df = _make_reference_profile(n=3000, seed=15)
    rng = np.random.default_rng(555)
    current_decisions = rng.choice(["APPROVE", "REVIEW", "BLOCK"], size=500, p=[0.97, 0.02, 0.01])
    result = compute_decision_drift(profile["decision"], current_decisions)
    assert result["severity"] in ("NO_SIGNIFICANT_DRIFT", "LOW_DRIFT")


def test_decision_drift_detects_shift():
    profile, ref_df = _make_reference_profile(n=3000, seed=16)
    # Reference: ~97% APPROVE. Current: much more intervention.
    current_decisions = (["APPROVE"] * 60 + ["REVIEW"] * 25 + ["BLOCK"] * 15)
    result = compute_decision_drift(profile["decision"], current_decisions)
    assert result["psi"] > 0.1
    assert result["severity"] != "NO_SIGNIFICANT_DRIFT"


# ---------------------------------------------------------------------------
# Small-batch behavior
# ---------------------------------------------------------------------------

def test_small_batch_flagged_low_confidence():
    profile, ref_df = _make_reference_profile(n=2000, seed=17)
    monitor = DriftMonitor(profile)
    small_batch = ref_df.iloc[:5].copy()
    report = monitor.analyze_batch(small_batch)
    assert report["low_confidence"] is True
    assert any("minimum recommended size" in w for w in report["warnings"])


def test_adequate_batch_not_flagged_low_confidence():
    profile, ref_df = _make_reference_profile(n=2000, seed=18)
    monitor = DriftMonitor(profile)
    batch = ref_df.iloc[:MIN_BATCH_SIZE + 10].copy()
    report = monitor.analyze_batch(batch)
    assert report["low_confidence"] is False


def test_small_batch_still_produces_valid_severity_not_pretending_certainty():
    profile, ref_df = _make_reference_profile(n=2000, seed=19)
    monitor = DriftMonitor(profile)
    tiny_batch = ref_df.iloc[:3].copy()
    report = monitor.analyze_batch(tiny_batch)
    assert report["overall_severity"] in SEVERITY_ORDER  # still a real, valid result
    assert report["low_confidence"] is True  # but honestly caveated


# ---------------------------------------------------------------------------
# Severity aggregation (explainability)
# ---------------------------------------------------------------------------

def test_aggregate_overall_severity_worst_wins():
    results = [
        {"name": "TransactionAmt", "severity": "NO_SIGNIFICANT_DRIFT"},
        {"name": "ProductCD", "severity": "HIGH_DRIFT"},
        {"name": "D2", "severity": "LOW_DRIFT"},
    ]
    agg = aggregate_overall_severity(results)
    assert agg["severity"] == "HIGH_DRIFT"
    assert agg["contributing"] == ["ProductCD"]
    assert "ProductCD" in agg["explanation"]


def test_aggregate_overall_severity_multiple_contributors():
    results = [
        {"name": "A", "severity": "MODERATE_DRIFT"},
        {"name": "B", "severity": "MODERATE_DRIFT"},
        {"name": "C", "severity": "LOW_DRIFT"},
    ]
    agg = aggregate_overall_severity(results)
    assert agg["severity"] == "MODERATE_DRIFT"
    assert set(agg["contributing"]) == {"A", "B"}


def test_aggregate_overall_severity_empty_input():
    agg = aggregate_overall_severity([])
    assert agg["severity"] == "NO_SIGNIFICANT_DRIFT"


# ---------------------------------------------------------------------------
# DriftMonitor.analyze_batch end-to-end + summary tracking
# ---------------------------------------------------------------------------

def test_analyze_batch_full_report_structure():
    profile, ref_df = _make_reference_profile(n=2000, seed=20)
    monitor = DriftMonitor(profile)
    batch = ref_df.iloc[:200].copy()
    scores = np.random.default_rng(0).beta(1, 20, size=200)
    decisions = np.random.default_rng(1).choice(["APPROVE", "REVIEW", "BLOCK"], size=200, p=[0.97, 0.02, 0.01])
    report = monitor.analyze_batch(batch, risk_scores=scores, decisions=decisions)

    for key in ("timestamp", "batch_size", "reference_version", "feature_results",
                "risk_score_result", "decision_result", "overall_severity", "overall_explanation"):
        assert key in report
    assert report["batch_size"] == 200
    assert len(report["feature_results"]) == len(MONITORED_NUMERIC_FEATURES) + len(MONITORED_CATEGORICAL_FEATURES) + len(MONITORED_BEHAVIORAL_FEATURES)


def test_monitor_summary_tracks_batches_and_severity():
    profile, ref_df = _make_reference_profile(n=2000, seed=21)
    monitor = DriftMonitor(profile)
    for _ in range(3):
        monitor.analyze_batch(ref_df.iloc[:100].copy())
    summary = monitor.summary()
    assert summary["batches_analyzed"] == 3
    assert summary["latest_status"] in SEVERITY_ORDER
    assert sum(summary["severity_counts"].values()) == 3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
