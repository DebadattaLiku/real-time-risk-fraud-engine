"""
Phase 4 test suite: leakage-safe behavioral feature engineering
(`src/features/behavioral.py`). These tests are the most important safety
net in this phase — behavioral/historical features are exactly the kind of
feature most prone to accidental future-information leakage, so they get
hand-verified, small, fully-traceable synthetic examples rather than only
statistical/shape checks.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.behavioral import (
    compute_behavioral_features, get_behavioral_feature_names, behavioral_diagnostics,
)


def _mk(id_, dt, card1, amt):
    return {"TransactionID": id_, "TransactionDT": dt, "card1": card1, "TransactionAmt": amt}


# ---------------------------------------------------------------------------
# Strictly historical behavior
# ---------------------------------------------------------------------------

def test_first_transaction_has_no_history():
    df = pd.DataFrame([_mk(1, 100, "A", 50.0)])
    out = compute_behavioral_features(df)
    row = out.iloc[0]
    assert row["bhv_prev_txn_count"] == 0
    assert pd.isna(row["bhv_hist_mean_amt"])
    assert pd.isna(row["bhv_hist_std_amt"])
    assert pd.isna(row["bhv_hist_min_amt"])
    assert pd.isna(row["bhv_hist_max_amt"])
    assert pd.isna(row["bhv_time_since_prev_txn"])


def test_second_transaction_sees_only_first():
    df = pd.DataFrame([
        _mk(1, 100, "A", 50.0),
        _mk(2, 200, "A", 70.0),
    ])
    out = compute_behavioral_features(df).set_index("TransactionID")
    row2 = out.loc[2]
    assert row2["bhv_prev_txn_count"] == 1
    assert row2["bhv_hist_mean_amt"] == pytest.approx(50.0)
    assert row2["bhv_hist_min_amt"] == pytest.approx(50.0)
    assert row2["bhv_hist_max_amt"] == pytest.approx(50.0)
    assert pd.isna(row2["bhv_hist_std_amt"])  # single point -> undefined variance
    assert row2["bhv_time_since_prev_txn"] == pytest.approx(100)


def test_third_transaction_sees_only_first_and_second_not_itself():
    df = pd.DataFrame([
        _mk(1, 100, "A", 50.0),
        _mk(2, 200, "A", 70.0),
        _mk(3, 300, "A", 999.0),  # large amount — must NOT affect its own history
    ])
    out = compute_behavioral_features(df).set_index("TransactionID")
    row3 = out.loc[3]
    assert row3["bhv_prev_txn_count"] == 2
    assert row3["bhv_hist_mean_amt"] == pytest.approx((50.0 + 70.0) / 2)
    assert row3["bhv_hist_min_amt"] == pytest.approx(50.0)
    assert row3["bhv_hist_max_amt"] == pytest.approx(70.0)  # NOT 999 — self-exclusion holds
    assert row3["bhv_hist_std_amt"] == pytest.approx(np.std([50.0, 70.0], ddof=1))


def test_current_transaction_excluded_from_its_own_history_large_outlier():
    """A transaction with an extreme amount must not inflate its own
    historical mean/std/min/max — this is the single most important
    leakage check in this suite."""
    df = pd.DataFrame([
        _mk(1, 100, "A", 10.0),
        _mk(2, 200, "A", 1_000_000.0),  # extreme value at the row being tested
    ])
    out = compute_behavioral_features(df).set_index("TransactionID")
    row2 = out.loc[2]
    assert row2["bhv_hist_mean_amt"] == pytest.approx(10.0)  # not influenced by 1,000,000
    assert row2["bhv_hist_max_amt"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Future leakage
# ---------------------------------------------------------------------------

def test_adding_future_transaction_does_not_change_earlier_features():
    df_without_future = pd.DataFrame([
        _mk(1, 100, "A", 50.0),
        _mk(2, 200, "A", 70.0),
    ])
    df_with_future = pd.DataFrame([
        _mk(1, 100, "A", 50.0),
        _mk(2, 200, "A", 70.0),
        _mk(3, 300, "A", 99999.0),  # future transaction, should not affect rows 1/2
    ])

    out_without = compute_behavioral_features(df_without_future).set_index("TransactionID")
    out_with = compute_behavioral_features(df_with_future).set_index("TransactionID")

    for col in get_behavioral_feature_names(out_without):
        for txn_id in (1, 2):
            v1 = out_without.loc[txn_id, col]
            v2 = out_with.loc[txn_id, col]
            if pd.isna(v1) and pd.isna(v2):
                continue
            assert v1 == pytest.approx(v2), f"{col} for txn {txn_id} changed when a future row was added"


def test_future_transaction_far_in_time_does_not_leak_into_velocity_windows():
    df = pd.DataFrame([
        _mk(1, 0, "A", 10.0),
        _mk(2, 5000, "A", 20.0),          # within 1h (3600s) of neither prior nor future
        _mk(3, 10_000_000, "A", 30.0),    # far future — must not affect earlier velocity counts
    ])
    out_full = compute_behavioral_features(df).set_index("TransactionID")
    df_partial = df.iloc[:2]
    out_partial = compute_behavioral_features(df_partial).set_index("TransactionID")
    assert out_full.loc[2, "bhv_prior_count_1h"] == out_partial.loc[2, "bhv_prior_count_1h"]
    assert out_full.loc[2, "bhv_prior_count_24h"] == out_partial.loc[2, "bhv_prior_count_24h"]


# ---------------------------------------------------------------------------
# Same-timestamp deterministic ordering
# ---------------------------------------------------------------------------

def test_same_timestamp_ordered_by_transaction_id():
    # Two transactions share the exact same TransactionDT; TransactionID
    # must be the deterministic tiebreaker (lower ID = earlier).
    df = pd.DataFrame([
        _mk(20, 100, "A", 50.0),   # higher ID, should sort AFTER id=10 despite same DT
        _mk(10, 100, "A", 30.0),
    ])
    out = compute_behavioral_features(df).set_index("TransactionID")
    # id=10 comes first in (DT, ID) order -> no history.
    assert out.loc[10, "bhv_prev_txn_count"] == 0
    # id=20 comes second -> sees id=10's amount as its only history.
    assert out.loc[20, "bhv_prev_txn_count"] == 1
    assert out.loc[20, "bhv_hist_mean_amt"] == pytest.approx(30.0)


def test_ordering_is_independent_of_input_row_order():
    df_order1 = pd.DataFrame([_mk(10, 100, "A", 30.0), _mk(20, 100, "A", 50.0)])
    df_order2 = pd.DataFrame([_mk(20, 100, "A", 50.0), _mk(10, 100, "A", 30.0)])
    out1 = compute_behavioral_features(df_order1).set_index("TransactionID")
    out2 = compute_behavioral_features(df_order2).set_index("TransactionID")
    pd.testing.assert_frame_equal(out1.sort_index(), out2.sort_index())


# ---------------------------------------------------------------------------
# Split continuity (train -> validation -> test)
# ---------------------------------------------------------------------------

def test_validation_row_sees_prior_train_history():
    # Simulate: train has entity A's txns 1,2; "validation" adds txn 3.
    train_like = pd.DataFrame([_mk(1, 100, "A", 10.0), _mk(2, 200, "A", 20.0)])
    combined = pd.concat([train_like, pd.DataFrame([_mk(3, 300, "A", 30.0)])], ignore_index=True)
    out = compute_behavioral_features(combined).set_index("TransactionID")
    row3 = out.loc[3]
    assert row3["bhv_prev_txn_count"] == 2  # sees both train rows
    assert row3["bhv_hist_mean_amt"] == pytest.approx(15.0)


def test_test_row_sees_prior_train_and_validation_history():
    train_like = pd.DataFrame([_mk(1, 100, "A", 10.0)])
    val_like = pd.DataFrame([_mk(2, 200, "A", 20.0)])
    test_like = pd.DataFrame([_mk(3, 300, "A", 30.0)])
    combined = pd.concat([train_like, val_like, test_like], ignore_index=True)
    out = compute_behavioral_features(combined).set_index("TransactionID")
    row3 = out.loc[3]
    assert row3["bhv_prev_txn_count"] == 2  # sees train row + validation row
    assert row3["bhv_hist_mean_amt"] == pytest.approx(15.0)


def test_no_future_split_information_used():
    # test-partition transaction must not affect a validation-partition row.
    train_like = pd.DataFrame([_mk(1, 100, "A", 10.0)])
    val_like = pd.DataFrame([_mk(2, 200, "A", 20.0)])
    test_like = pd.DataFrame([_mk(3, 300, "A", 9999.0)])  # extreme, must not leak backward
    combined = pd.concat([train_like, val_like, test_like], ignore_index=True)
    out = compute_behavioral_features(combined).set_index("TransactionID")
    row2 = out.loc[2]
    assert row2["bhv_prev_txn_count"] == 1
    assert row2["bhv_hist_mean_amt"] == pytest.approx(10.0)  # unaffected by future test row


# ---------------------------------------------------------------------------
# Target exclusion
# ---------------------------------------------------------------------------

def test_isfraud_column_raises_assertion():
    df = pd.DataFrame([_mk(1, 100, "A", 50.0)])
    df["isFraud"] = 0
    with pytest.raises(AssertionError):
        compute_behavioral_features(df)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_single_historical_transaction_std_is_nan_not_zero():
    df = pd.DataFrame([_mk(1, 100, "A", 10.0), _mk(2, 200, "A", 20.0)])
    out = compute_behavioral_features(df).set_index("TransactionID")
    assert pd.isna(out.loc[2, "bhv_hist_std_amt"])  # exactly one prior point -> undefined, not 0


def test_zero_historical_std_when_prior_amounts_identical():
    df = pd.DataFrame([
        _mk(1, 100, "A", 10.0),
        _mk(2, 200, "A", 10.0),
        _mk(3, 300, "A", 10.0),  # third row: two identical priors -> std should be exactly 0
    ])
    out = compute_behavioral_features(df).set_index("TransactionID")
    row3 = out.loc[3]
    assert row3["bhv_hist_std_amt"] == pytest.approx(0.0)
    # z-score with a zero historical std must be NaN (guarded), not inf/div-by-zero.
    assert pd.isna(row3["bhv_amt_zscore"])


def test_zero_historical_mean_ratio_is_nan_not_inf():
    df = pd.DataFrame([
        _mk(1, 100, "A", 0.0),
        _mk(2, 200, "A", 50.0),
    ])
    out = compute_behavioral_features(df).set_index("TransactionID")
    row2 = out.loc[2]
    assert row2["bhv_hist_mean_amt"] == pytest.approx(0.0)
    assert pd.isna(row2["bhv_amt_to_hist_mean_ratio"])  # 50/0 guarded to NaN, not inf


def test_missing_amount_does_not_crash_and_propagates_as_nan():
    df = pd.DataFrame([
        _mk(1, 100, "A", np.nan),
        _mk(2, 200, "A", 20.0),
    ])
    out = compute_behavioral_features(df).set_index("TransactionID")
    # Should not raise; a NaN historical input propagates to NaN downstream
    # rather than being silently treated as 0.
    assert pd.isna(out.loc[2, "bhv_hist_mean_amt"])


def test_sparse_pseudo_entity_each_appears_once():
    df = pd.DataFrame([
        _mk(1, 100, "A", 10.0),
        _mk(2, 200, "B", 20.0),
        _mk(3, 300, "C", 30.0),
    ])
    out = compute_behavioral_features(df).set_index("TransactionID")
    for txn_id in (1, 2, 3):
        assert out.loc[txn_id, "bhv_prev_txn_count"] == 0  # each entity's only transaction


def test_output_has_one_row_per_input_row_and_covers_all_ids():
    n = 50
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "TransactionID": np.arange(n),
        "TransactionDT": np.sort(rng.integers(0, 100000, size=n)),
        "card1": rng.integers(1, 5, size=n),
        "TransactionAmt": rng.uniform(1, 100, size=n),
    })
    out = compute_behavioral_features(df)
    assert len(out) == n
    assert set(out["TransactionID"]) == set(df["TransactionID"])


def test_velocity_counts_non_negative_and_le_prev_count():
    n = 200
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "TransactionID": np.arange(n),
        "TransactionDT": np.sort(rng.integers(0, 50000, size=n)),
        "card1": rng.integers(1, 6, size=n),
        "TransactionAmt": rng.uniform(1, 100, size=n),
    })
    out = compute_behavioral_features(df)
    assert (out["bhv_prior_count_1h"] >= 0).all()
    assert (out["bhv_prior_count_24h"] >= 0).all()
    assert (out["bhv_prior_count_1h"] <= out["bhv_prev_txn_count"]).all()
    assert (out["bhv_prior_count_24h"] <= out["bhv_prev_txn_count"]).all()
    assert (out["bhv_prior_count_1h"] <= out["bhv_prior_count_24h"]).all()


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def test_behavioral_diagnostics_runs_and_reports_cold_start_pct():
    df = pd.DataFrame([
        _mk(1, 100, "A", 10.0),  # cold start
        _mk(2, 200, "A", 20.0),  # has history
        _mk(3, 100, "B", 15.0),  # cold start
    ])
    out = compute_behavioral_features(df)
    diag = behavioral_diagnostics(out)
    assert diag["n_transactions"] == 3
    assert diag["pct_cold_start_no_prior_history"] == pytest.approx(2 / 3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
