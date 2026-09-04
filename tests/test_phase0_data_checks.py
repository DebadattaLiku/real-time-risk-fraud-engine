"""
Phase 0 test suite.

IMPORTANT: These tests validate the *logic* of the availability checks and
leakage-guard functions in src/data/inspect_data.py. They do this using small
synthetic in-memory DataFrames constructed purely to exercise the code paths
(uniqueness checks, split-boundary checks, missingness aggregation, etc.).

This is NOT a substitute for the real IEEE-CIS dataset and none of this
synthetic data is used anywhere outside this test file. No claims about the
real dataset are derived from these tests.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "data"))

import inspect_data as idata  # noqa: E402


def test_check_dataset_availability_reports_missing(tmp_path):
    report = idata.check_dataset_availability(tmp_path)
    assert report.all_present is False
    assert report.any_present is False
    for fname in idata.REQUIRED_FILES:
        assert report.found[fname] is False


def test_check_dataset_availability_reports_present(tmp_path):
    for fname in idata.REQUIRED_FILES:
        (tmp_path / fname).write_text("id,val\n1,2\n")
    report = idata.check_dataset_availability(tmp_path)
    assert report.all_present is True
    assert all(report.found.values())


def test_check_dataset_availability_partial(tmp_path):
    (tmp_path / "train_transaction.csv").write_text("id,val\n1,2\n")
    report = idata.check_dataset_availability(tmp_path)
    assert report.found["train_transaction.csv"] is True
    assert report.found["train_identity.csv"] is False
    assert report.all_present is False
    assert report.any_present is True


def _make_synthetic_df(n=200, seed=42):
    rng = np.random.default_rng(seed)
    # Approximately 3% fraud rate, matching the real-world imbalance this
    # project is designed around. Deterministic given `seed`.
    approx_fraud_rate = 0.03
    return pd.DataFrame({
        "TransactionID": np.arange(n),
        "TransactionDT": np.sort(rng.integers(0, 1_000_000, size=n)),
        "isFraud": (rng.random(n) < approx_fraud_rate).astype(int),
        "card1": rng.integers(1000, 1010, size=n),
        "card2": rng.integers(100, 105, size=n),
        "addr1": rng.integers(200, 210, size=n),
        "addr2": rng.integers(10, 12, size=n),
        "D1": rng.random(n),
        "C1": rng.integers(0, 5, size=n),
        "V1": rng.random(n),
    })


def test_structure_report_detects_duplicates():
    df = _make_synthetic_df(n=50)
    df_with_dupe = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    result = idata.structure_report(df_with_dupe, "TransactionID", "isFraud")
    assert result["duplicate_transaction_ids"] == 1
    assert result["id_is_unique"] is False


def test_structure_report_unique_ids():
    df = _make_synthetic_df(n=50)
    result = idata.structure_report(df, "TransactionID", "isFraud")
    assert result["duplicate_transaction_ids"] == 0
    assert result["id_is_unique"] is True


def test_temporal_analysis_boundaries_are_within_range():
    df = _make_synthetic_df(n=500)
    result = idata.temporal_analysis(df, "TransactionDT", "isFraud", 0.7, 0.15, 0.15)
    assert result["dt_min"] <= result["proposed_train_end_dt"] <= result["dt_max"]
    assert result["proposed_train_end_dt"] <= result["proposed_val_end_dt"] <= result["dt_max"]


def test_pseudo_entity_analysis_does_not_use_isfraud():
    df = _make_synthetic_df(n=200)
    candidates = [["card1"], ["card1", "card2", "addr1"]]
    results = idata.pseudo_entity_analysis(df, candidates)
    assert len(results) == 2
    for r in results:
        assert "isFraud" not in r["key_columns"]
        assert r["cardinality"] > 0


def test_pseudo_entity_analysis_excludes_rows_with_missing_key_components():
    df = _make_synthetic_df(n=100)
    # Introduce missing values in one of the key columns for a known subset.
    df = df.copy()
    df.loc[:9, "addr2"] = np.nan  # 10 rows now have a missing key component

    results = idata.pseudo_entity_analysis(df, [["card1", "addr2"]])
    r = results[0]

    assert r["rows_with_missing_key_component"] == 10
    assert r["rows_with_complete_key"] == 90
    assert r["pct_rows_with_missing_key_component"] == pytest.approx(0.10)

    # Rows dropped for missing components must not have been merged into a
    # bogus "nan"-matched group: cardinality should be computed only over
    # the 90 complete-key rows, never referencing the string "nan".
    valid_df = df.loc[~df[["card1", "addr2"]].isna().any(axis=1), ["card1", "addr2"]]
    expected_cardinality = valid_df.astype(str).agg("_".join, axis=1).nunique()
    assert r["cardinality"] == expected_cardinality


def test_pseudo_entity_analysis_all_missing_key_returns_no_cardinality():
    df = _make_synthetic_df(n=20)
    df = df.copy()
    df["addr2"] = np.nan  # every row missing this key component
    results = idata.pseudo_entity_analysis(df, [["addr2"]])
    r = results[0]
    assert r["rows_with_complete_key"] == 0
    assert r["cardinality"] is None


def test_run_integrity_tests_no_overlap_and_correct_ordering():
    df = _make_synthetic_df(n=1000)
    tests = idata.run_integrity_tests(df, "TransactionID", "TransactionDT", 0.7, 0.15)
    assert tests["transaction_id_unique"] is True
    assert tests["no_row_overlap_train_val"] is True
    assert tests["no_row_overlap_val_test"] is True
    assert tests["no_row_overlap_train_test"] is True
    assert tests["max_train_dt_lte_min_val_dt"] is True
    assert tests["max_val_dt_lte_min_test_dt"] is True


def test_missingness_analysis_handles_missing_values():
    df = _make_synthetic_df(n=100)
    df.loc[:19, "D1"] = np.nan  # 20% missing in D1
    result = idata.missingness_analysis(df)
    assert result["D"]["n_columns"] >= 1
    assert result["D"]["max_missing_pct"] >= 0.19


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
