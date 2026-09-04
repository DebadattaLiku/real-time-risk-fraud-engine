"""
Phase 6 test suite: BehavioralStateManager (`src/engine/state.py`).
Hand-verified small sequences, mirroring the Phase 4 offline test style.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.engine.state import BehavioralStateManager


def _isnan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


# ---------------------------------------------------------------------------
# Basic sequential behavior (mirrors Phase 4 offline hand-verified tests)
# ---------------------------------------------------------------------------

def test_first_transaction_no_history():
    mgr = BehavioralStateManager()
    features = mgr.compute_features("A", current_time=100, current_amount=50.0)
    assert features["bhv_prev_txn_count"] == 0
    assert _isnan(features["bhv_hist_mean_amt"])
    assert _isnan(features["bhv_hist_std_amt"])
    assert _isnan(features["bhv_hist_min_amt"])
    assert _isnan(features["bhv_hist_max_amt"])
    assert _isnan(features["bhv_time_since_prev_txn"])


def test_compute_features_is_read_only():
    mgr = BehavioralStateManager()
    mgr.compute_features("A", 100, 50.0)
    mgr.compute_features("A", 100, 50.0)
    mgr.compute_features("A", 100, 50.0)
    # Reading three times must not have created any state.
    features = mgr.compute_features("A", 200, 10.0)
    assert features["bhv_prev_txn_count"] == 0


def test_second_transaction_sees_only_first():
    mgr = BehavioralStateManager()
    f1 = mgr.compute_features("A", 100, 50.0)
    mgr.update("A", 100, 50.0)
    f2 = mgr.compute_features("A", 200, 70.0)
    assert f2["bhv_prev_txn_count"] == 1
    assert f2["bhv_hist_mean_amt"] == pytest.approx(50.0)
    assert f2["bhv_hist_min_amt"] == pytest.approx(50.0)
    assert f2["bhv_hist_max_amt"] == pytest.approx(50.0)
    assert _isnan(f2["bhv_hist_std_amt"])  # single prior point
    assert f2["bhv_time_since_prev_txn"] == pytest.approx(100)


def test_third_transaction_sees_only_first_and_second_not_itself():
    mgr = BehavioralStateManager()
    mgr.compute_features("A", 100, 50.0)
    mgr.update("A", 100, 50.0)
    mgr.compute_features("A", 200, 70.0)
    mgr.update("A", 200, 70.0)
    f3 = mgr.compute_features("A", 300, 999.0)  # large amount, must not affect itself
    assert f3["bhv_prev_txn_count"] == 2
    assert f3["bhv_hist_mean_amt"] == pytest.approx(60.0)
    assert f3["bhv_hist_min_amt"] == pytest.approx(50.0)
    assert f3["bhv_hist_max_amt"] == pytest.approx(70.0)  # NOT 999
    assert f3["bhv_hist_std_amt"] == pytest.approx(np.std([50.0, 70.0], ddof=1))


def test_update_only_after_prediction_convention_state_unaffected_by_repeated_reads():
    """State must only change via update(), never via compute_features()."""
    mgr = BehavioralStateManager()
    mgr.update("A", 100, 10.0)
    before = mgr.get_state_snapshot("A")
    mgr.compute_features("A", 200, 999999.0)
    mgr.compute_features("A", 300, -1.0)  # even a nonsensical read shouldn't mutate
    after = mgr.get_state_snapshot("A")
    assert before == after


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_zero_historical_std_when_prior_amounts_identical():
    mgr = BehavioralStateManager()
    mgr.update("A", 100, 10.0)
    mgr.update("A", 200, 10.0)
    f3 = mgr.compute_features("A", 300, 10.0)
    assert f3["bhv_hist_std_amt"] == pytest.approx(0.0)
    assert _isnan(f3["bhv_amt_zscore"])  # zero std -> guarded NaN, not div-by-zero


def test_zero_historical_mean_ratio_is_nan_not_inf():
    mgr = BehavioralStateManager()
    mgr.update("A", 100, 0.0)
    f2 = mgr.compute_features("A", 200, 50.0)
    assert f2["bhv_hist_mean_amt"] == pytest.approx(0.0)
    assert _isnan(f2["bhv_amt_to_hist_mean_ratio"])


def test_sparse_entities_independent():
    mgr = BehavioralStateManager()
    mgr.update("A", 100, 10.0)
    fB = mgr.compute_features("B", 100, 999.0)
    assert fB["bhv_prev_txn_count"] == 0  # entity B unaffected by entity A


# ---------------------------------------------------------------------------
# Velocity windows
# ---------------------------------------------------------------------------

def test_velocity_windows_hand_verified():
    mgr = BehavioralStateManager()
    mgr.update("A", 0, 10.0)
    mgr.update("A", 1800, 10.0)     # 30 min later
    mgr.update("A", 5000, 10.0)     # ~1h23min after first
    f = mgr.compute_features("A", 5100, 10.0)  # 100s after the third update
    # at time 5100: entries at t=0,1800,5000. window 1h = [1500,5100) -> t=1800 and t=5000 qualify (2 count)
    # window 24h = [5100-86400, 5100) -> all three qualify (3 counts)
    assert f["bhv_prior_count_1h"] == 2
    assert f["bhv_prior_count_24h"] == 3


def test_velocity_prunes_entries_older_than_24h():
    mgr = BehavioralStateManager()
    mgr.update("A", 0, 10.0)
    f = mgr.compute_features("A", 100000, 10.0)  # 100,000s later, > 24h (86400s)
    assert f["bhv_prior_count_24h"] == 0
    assert f["bhv_prior_count_1h"] == 0
    # but prev_txn_count (lifetime count) must still reflect it
    assert f["bhv_prev_txn_count"] == 1


# ---------------------------------------------------------------------------
# State lifecycle
# ---------------------------------------------------------------------------

def test_reset_produces_clean_state():
    mgr = BehavioralStateManager()
    mgr.update("A", 100, 10.0)
    mgr.update("A", 200, 20.0)
    mgr.reset()
    f = mgr.compute_features("A", 300, 30.0)
    assert f["bhv_prev_txn_count"] == 0


def test_serialize_deserialize_roundtrip():
    mgr = BehavioralStateManager()
    mgr.update("A", 100, 10.0)
    mgr.update("A", 200, 20.0)
    mgr.update("B", 150, 5.0)

    serialized = mgr.serialize()
    mgr2 = BehavioralStateManager()
    mgr2.load_serialized(serialized, key_type=str)

    f_orig = mgr.compute_features("A", 300, 30.0)
    f_restored = mgr2.compute_features("A", 300, 30.0)
    assert f_orig == f_restored


# ---------------------------------------------------------------------------
# bulk_initialize equivalence
# ---------------------------------------------------------------------------

def test_bulk_initialize_equivalent_to_sequential_updates():
    n = 60
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "TransactionID": np.arange(n),
        "TransactionDT": np.sort(rng.integers(0, 200000, size=n)),
        "card1": rng.integers(1, 4, size=n),
        "TransactionAmt": rng.uniform(1, 100, size=n),
    })
    df = df.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)

    split_at = 40
    historical = df.iloc[:split_at]
    remainder = df.iloc[split_at:]

    # Manager A: bulk_initialize on historical, then sequential update on remainder.
    mgr_a = BehavioralStateManager()
    mgr_a.bulk_initialize(historical, as_of_time=float(historical["TransactionDT"].max()))
    for _, row in remainder.iterrows():
        mgr_a.compute_features(row["card1"], row["TransactionDT"], row["TransactionAmt"])
        mgr_a.update(row["card1"], row["TransactionDT"], row["TransactionAmt"])

    # Manager B: pure sequential update over the ENTIRE sequence.
    mgr_b = BehavioralStateManager()
    for _, row in df.iterrows():
        mgr_b.compute_features(row["card1"], row["TransactionDT"], row["TransactionAmt"])
        mgr_b.update(row["card1"], row["TransactionDT"], row["TransactionAmt"])

    # Compare final state for every entity.
    for entity in df["card1"].unique():
        snap_a = mgr_a.get_state_snapshot(entity)
        snap_b = mgr_b.get_state_snapshot(entity)
        assert snap_a["count"] == snap_b["count"]
        assert snap_a["sum_amt"] == pytest.approx(snap_b["sum_amt"])
        assert snap_a["m2"] == pytest.approx(snap_b["m2"])
        assert snap_a["min_amt"] == pytest.approx(snap_b["min_amt"])
        assert snap_a["max_amt"] == pytest.approx(snap_b["max_amt"])
        assert snap_a["last_time"] == pytest.approx(snap_b["last_time"])


def test_numeric_stability_welford_avoids_catastrophic_cancellation():
    """
    Phase 13 audit: a direct, quantitative demonstration that the current
    Welford-based implementation avoids the catastrophic-cancellation
    failure mode of the earlier `sum_sq_amt - count*mean**2` formula
    (documented in reports/phase6_realtime_engine_summary.md).

    Construct a scenario known to break the naive formula: many
    transactions clustered very tightly around a LARGE value (so
    `sum_sq_amt` and `count * mean**2` are both huge, nearly-equal
    numbers whose small true difference — the real variance — is lost to
    floating-point rounding, occasionally even going negative). Compare
    against numpy's own (independently, differently implemented) variance
    calculation as ground truth.
    """
    rng = np.random.default_rng(7)
    n = 20000
    true_std = 1e-3
    base = 1_000_000.0
    amounts = base + rng.normal(0, true_std, size=n)

    # Ground truth via an independent, well-tested implementation.
    expected_std = float(np.std(amounts, ddof=1))

    # The OLD, catastrophically-unstable formula, reproduced inline here
    # ONLY for this comparison (not present anywhere in src/ anymore).
    naive_sum = float(np.sum(amounts))
    naive_sumsq = float(np.sum(amounts ** 2))
    naive_mean = naive_sum / n
    naive_var = (naive_sumsq - n * naive_mean ** 2) / (n - 1)
    naive_std = math.sqrt(naive_var) if naive_var > 0 else float("nan")

    # The CURRENT implementation (Welford, via the real state manager).
    mgr = BehavioralStateManager()
    for i, amt in enumerate(amounts):
        mgr.update("A", float(i), float(amt))
    # Read back the accumulated std via a feature computation "as of" one
    # more transaction (compute_features never mutates state).
    features = mgr.compute_features("A", float(n), base)
    welford_std = features["bhv_hist_std_amt"]

    welford_error = abs(welford_std - expected_std)
    naive_error = abs(naive_std - expected_std) if naive_std == naive_std else float("inf")

    # The Welford-based result must be close to the true standard
    # deviation ...
    assert welford_error < 1e-6, f"Welford std ({welford_std}) should closely match numpy's ({expected_std})"
    # ... and must be at least as accurate as the naive formula would have
    # been for this exact adversarial input (usually dramatically more
    # accurate; the naive formula can even go negative/NaN here).
    assert welford_error <= naive_error


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
