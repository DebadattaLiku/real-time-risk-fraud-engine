"""
Phase 6 test suite: offline/online parity validation utilities
(`src/engine/parity.py`), and an end-to-end parity check using a small
synthetic engine + offline pipeline pair.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.engine.parity import (
    validate_behavioral_feature_parity, validate_score_parity, validate_decision_parity,
)
from src.features.behavioral import compute_behavioral_features
from src.engine.state import BehavioralStateManager


# ---------------------------------------------------------------------------
# validate_score_parity / validate_decision_parity — unit-level
# ---------------------------------------------------------------------------

def test_validate_score_parity_hand_verified():
    offline_scores = {1: 0.5, 2: 0.8, 3: 0.1}
    online_results = [
        {"transaction_id": 1, "risk_score": 0.5000001},
        {"transaction_id": 2, "risk_score": 0.8},
        {"transaction_id": 3, "risk_score": 0.15},  # deliberate mismatch beyond tolerance
    ]
    result = validate_score_parity(offline_scores, online_results, tolerance=1e-6)
    assert result["n_compared"] == 3
    assert result["n_mismatches"] == 1
    assert result["max_abs_diff"] == pytest.approx(0.05, abs=1e-9)


def test_validate_decision_parity_hand_verified():
    offline_decisions = {1: "APPROVE", 2: "BLOCK", 3: "REVIEW"}
    online_results = [
        {"transaction_id": 1, "decision": "APPROVE"},
        {"transaction_id": 2, "decision": "BLOCK"},
        {"transaction_id": 3, "decision": "BLOCK"},  # mismatch
    ]
    result = validate_decision_parity(offline_decisions, online_results)
    assert result["n_compared"] == 3
    assert result["n_matching"] == 2
    assert result["pct_matching"] == pytest.approx(2 / 3)
    assert result["n_mismatches"] == 1


# ---------------------------------------------------------------------------
# End-to-end behavioral feature parity: offline batch vs online sequential
# ---------------------------------------------------------------------------

def test_behavioral_feature_parity_offline_vs_online_sequential():
    n = 300
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "TransactionID": np.arange(n),
        "TransactionDT": np.sort(rng.integers(0, 300000, size=n)),
        "card1": rng.integers(1, 8, size=n),
        "TransactionAmt": rng.uniform(1, 200, size=n),
    })
    df_sorted = df.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)

    # Offline: batch vectorized computation.
    offline = compute_behavioral_features(df_sorted)

    # Online: sequential state manager, processed in the SAME deterministic order.
    mgr = BehavioralStateManager()
    online_results = []
    for _, row in df_sorted.iterrows():
        features = mgr.compute_features(row["card1"], row["TransactionDT"], row["TransactionAmt"])
        online_results.append({"transaction_id": row["TransactionID"], "behavioral_features": features})
        mgr.update(row["card1"], row["TransactionDT"], row["TransactionAmt"])

    result = validate_behavioral_feature_parity(offline, online_results, tolerance=1e-6)
    assert result["n_mismatches"] == 0
    assert result["max_abs_diff"] < 1e-6
    assert result["n_transactions_fully_matching"] == n


def test_behavioral_feature_parity_detects_a_real_mismatch():
    """Sanity check that the parity utility actually catches a real
    discrepancy, rather than trivially reporting success."""
    offline = pd.DataFrame({
        "TransactionID": [1, 2],
        "bhv_prev_txn_count": [0.0, 1.0],
        "bhv_prev_txn_count_log1p": [0.0, 0.693],
        "bhv_hist_mean_amt": [np.nan, 10.0],
        "bhv_hist_std_amt": [np.nan, np.nan],
        "bhv_hist_min_amt": [np.nan, 10.0],
        "bhv_hist_max_amt": [np.nan, 10.0],
        "bhv_time_since_prev_txn": [np.nan, 100.0],
        "bhv_amt_to_hist_mean_ratio": [np.nan, 2.0],
        "bhv_amt_diff_from_hist_mean": [np.nan, 10.0],
        "bhv_amt_zscore": [np.nan, np.nan],
        "bhv_prior_count_1h": [0.0, 1.0],
        "bhv_prior_count_24h": [0.0, 1.0],
    })
    online_results = [
        {"transaction_id": 1, "behavioral_features": {
            "bhv_prev_txn_count": 0.0, "bhv_prev_txn_count_log1p": 0.0,
            "bhv_hist_mean_amt": float("nan"), "bhv_hist_std_amt": float("nan"),
            "bhv_hist_min_amt": float("nan"), "bhv_hist_max_amt": float("nan"),
            "bhv_time_since_prev_txn": float("nan"), "bhv_amt_to_hist_mean_ratio": float("nan"),
            "bhv_amt_diff_from_hist_mean": float("nan"), "bhv_amt_zscore": float("nan"),
            "bhv_prior_count_1h": 0.0, "bhv_prior_count_24h": 0.0,
        }},
        {"transaction_id": 2, "behavioral_features": {
            "bhv_prev_txn_count": 1.0, "bhv_prev_txn_count_log1p": 0.693,
            "bhv_hist_mean_amt": 999.0,  # deliberately wrong
            "bhv_hist_std_amt": float("nan"), "bhv_hist_min_amt": 10.0, "bhv_hist_max_amt": 10.0,
            "bhv_time_since_prev_txn": 100.0, "bhv_amt_to_hist_mean_ratio": 2.0,
            "bhv_amt_diff_from_hist_mean": 10.0, "bhv_amt_zscore": float("nan"),
            "bhv_prior_count_1h": 1.0, "bhv_prior_count_24h": 1.0,
        }},
    ]
    result = validate_behavioral_feature_parity(offline, online_results, tolerance=1e-6)
    assert result["n_mismatches"] == 1
    assert result["n_transactions_fully_matching"] == 1
    assert result["max_abs_diff"] == pytest.approx(989.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
