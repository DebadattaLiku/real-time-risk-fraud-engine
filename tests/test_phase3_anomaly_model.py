"""
Phase 3 test suite: Isolation Forest wrapper (`src/models/anomaly.py`) and
complementarity analysis (`src/evaluation/complementarity.py`), using small
synthetic examples with hand-verifiable answers.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.anomaly import AnomalyDetector
from src.evaluation.complementarity import (
    score_correlation, overlap_analysis, union_diagnostic,
    conditional_low_lgbm_high_anomaly,
)


# ---------------------------------------------------------------------------
# AnomalyDetector
# ---------------------------------------------------------------------------

def _make_synthetic_numeric(n=500, seed=0, n_outliers=20):
    rng = np.random.default_rng(seed)
    normal = rng.normal(0, 1, size=(n - n_outliers, 5))
    outliers = rng.normal(8, 1, size=(n_outliers, 5))  # clearly separated cluster
    X = np.vstack([normal, outliers]).astype(np.float32)
    y = np.array([0] * (n - n_outliers) + [1] * n_outliers)
    # shuffle deterministically
    idx = rng.permutation(n)
    return X[idx], y[idx]


def test_model_trains_on_synthetic_data():
    X, y = _make_synthetic_numeric()
    detector = AnomalyDetector(n_estimators=50)
    detector.fit(X)
    scores = detector.anomaly_score(X)
    assert len(scores) == len(X)


def test_score_output_length_matches_input_length():
    X, y = _make_synthetic_numeric(n=300)
    detector = AnomalyDetector(n_estimators=50)
    detector.fit(X)
    X_test = X[:123]
    scores = detector.anomaly_score(X_test)
    assert len(scores) == 123


def test_anomaly_score_direction_higher_is_more_anomalous():
    X, y = _make_synthetic_numeric(n=1000, n_outliers=50)
    detector = AnomalyDetector(n_estimators=100)
    detector.fit(X)
    scores = detector.anomaly_score(X)
    # The synthetic outlier cluster (y==1) should have systematically
    # higher anomaly scores than the normal cluster (y==0).
    mean_outlier_score = scores[y == 1].mean()
    mean_normal_score = scores[y == 0].mean()
    assert mean_outlier_score > mean_normal_score


def test_score_before_fit_raises():
    X, y = _make_synthetic_numeric(n=100)
    detector = AnomalyDetector()
    with pytest.raises(RuntimeError):
        detector.anomaly_score(X)


def test_fit_accepts_no_label_argument():
    """
    Isolation Forest is unsupervised — `fit` must not accept or require a
    label array. This is a structural check (fit's signature has no `y`
    parameter at all), not just a documentation claim.
    """
    import inspect
    sig = inspect.signature(AnomalyDetector.fit)
    assert "y" not in sig.parameters


# ---------------------------------------------------------------------------
# complementarity.py
# ---------------------------------------------------------------------------

def test_score_correlation_perfectly_correlated():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([2.0, 4.0, 6.0, 8.0, 10.0])  # perfectly linearly correlated
    result = score_correlation(a, b)
    assert result["pearson_r"] == pytest.approx(1.0)
    assert result["spearman_r"] == pytest.approx(1.0)


def test_score_correlation_uncorrelated_ranks():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([3.0, 1.0, 4.0, 1.0, 5.0])  # not monotonic w.r.t. a
    result = score_correlation(a, b)
    assert -1.0 <= result["spearman_r"] <= 1.0  # sane range, not asserting a specific value


def test_overlap_analysis_hand_verified():
    # 10 transactions, fraud at idx 0 and 5.
    y_true = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    # model_a top-20% (2 txns) picks idx 0,1 (highest scores)
    score_a = [10, 9, 1, 2, 3, 4, 5, 6, 7, 8]
    # model_b top-20% (2 txns) picks idx 5,6 (highest scores)
    score_b = [1, 2, 3, 4, 5, 20, 19, 6, 7, 8]

    result = overlap_analysis(y_true, score_a, score_b, k_fraction=0.2, name_a="lgbm", name_b="anomaly")
    assert result["n_reviewed_each"] == 2
    assert result["row_overlap_count"] == 0  # {0,1} vs {5,6} -> no overlap
    assert result["fraud_captured_by_lgbm"] == 1     # idx 0
    assert result["fraud_captured_by_anomaly"] == 1  # idx 5
    assert result["fraud_captured_by_both"] == 0
    assert result["fraud_captured_only_by_lgbm"] == 1
    assert result["fraud_captured_only_by_anomaly"] == 1
    assert result["total_fraud_cases"] == 2


def test_union_diagnostic_hand_verified():
    y_true = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    score_a = [10, 9, 1, 2, 3, 4, 5, 6, 7, 8]   # top-2 -> idx 0,1
    score_b = [1, 2, 3, 4, 5, 20, 19, 6, 7, 8]  # top-2 -> idx 5,6

    result = union_diagnostic(y_true, score_a, score_b, k_fraction=0.2)
    assert result["n_reviewed_union"] == 4  # {0,1,5,6}, no overlap -> union of 4
    assert result["fraud_captured_by_union"] == 2  # both fraud cases (0 and 5) captured
    assert result["fraud_recall_union"] == pytest.approx(1.0)
    assert result["fraud_captured_by_model_a_alone_at_same_k"] == 1
    assert result["incremental_fraud_from_union_vs_model_a_alone"] == 1
    assert "DIAGNOSTIC ONLY" in result["note"]


def test_conditional_low_lgbm_high_anomaly_hand_verified():
    y_true = np.array([1, 0, 1, 0, 0])
    lgbm_score = np.array([0.1, 0.2, 0.05, 0.9, 0.8])   # low risk: idx 0,1,2
    anomaly_score = np.array([0.9, 0.1, 0.95, 0.2, 0.3])  # high anomaly: idx 0,2

    # condition: lgbm < 0.5 AND anomaly > 0.5 -> idx 0 and 2
    result = conditional_low_lgbm_high_anomaly(
        y_true, lgbm_score, anomaly_score,
        lgbm_low_threshold=0.5, anomaly_high_threshold=0.5,
    )
    assert result["n_transactions_matching_condition"] == 2
    assert result["fraud_cases_in_condition"] == 2  # both idx 0 and 2 are fraud
    assert result["fraud_rate_in_condition"] == pytest.approx(1.0)
    assert result["overall_fraud_rate"] == pytest.approx(2 / 5)
    assert result["lift_over_overall_rate"] == pytest.approx(1.0 / (2 / 5))


def test_conditional_analysis_no_matches_handled_safely():
    y_true = np.array([0, 0, 0])
    lgbm_score = np.array([0.9, 0.9, 0.9])  # nothing below threshold
    anomaly_score = np.array([0.1, 0.1, 0.1])
    result = conditional_low_lgbm_high_anomaly(
        y_true, lgbm_score, anomaly_score,
        lgbm_low_threshold=0.5, anomaly_high_threshold=0.5,
    )
    assert result["n_transactions_matching_condition"] == 0
    assert np.isnan(result["fraud_rate_in_condition"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
