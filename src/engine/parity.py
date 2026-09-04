"""
Phase 6: Offline vs. online parity validation.

Compares the online `RiskDecisionEngine` simulation's outputs against the
offline batch pipeline's outputs for the SAME transactions, in three
respects: behavioral features, risk scores, and final decisions. Reports
exact counts and maximum differences rather than a pass/fail boolean alone
— mismatches must be visible, not silently summarized away.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

BEHAVIORAL_FEATURE_NAMES = [
    "bhv_prev_txn_count", "bhv_prev_txn_count_log1p", "bhv_hist_mean_amt",
    "bhv_hist_std_amt", "bhv_hist_min_amt", "bhv_hist_max_amt",
    "bhv_time_since_prev_txn", "bhv_amt_to_hist_mean_ratio",
    "bhv_amt_diff_from_hist_mean", "bhv_amt_zscore",
    "bhv_prior_count_1h", "bhv_prior_count_24h",
]


def _values_match(a, b, tolerance: float) -> bool:
    a_nan = a is None or (isinstance(a, float) and math.isnan(a))
    b_nan = b is None or (isinstance(b, float) and math.isnan(b))
    if a_nan and b_nan:
        return True
    if a_nan != b_nan:
        return False
    return abs(float(a) - float(b)) <= tolerance


def validate_behavioral_feature_parity(
    offline_features_df: pd.DataFrame, online_results: list, id_col: str = "TransactionID",
    tolerance: float = 1e-6,
) -> dict:
    """
    `offline_features_df`: the direct output of
    `compute_behavioral_features` (one row per transaction, `id_col` +
    `bhv_*` columns).
    `online_results`: list of `RiskDecisionEngine.process_transaction()`
    result dicts, each with `transaction_id` and `behavioral_features`.
    """
    offline_indexed = offline_features_df.set_index(id_col)
    mismatches = []
    max_abs_diff = 0.0
    n_compared = 0
    n_matching_transactions = 0

    for r in online_results:
        txn_id = r["transaction_id"]
        if txn_id not in offline_indexed.index:
            mismatches.append({"transaction_id": txn_id, "reason": "not found in offline output"})
            continue
        offline_row = offline_indexed.loc[txn_id]
        online_features = r["behavioral_features"]
        txn_ok = True
        for name in BEHAVIORAL_FEATURE_NAMES:
            n_compared += 1
            offline_val = offline_row[name]
            online_val = online_features.get(name)
            if not _values_match(offline_val, online_val, tolerance):
                txn_ok = False
                diff = (
                    abs(float(offline_val) - float(online_val))
                    if not (pd.isna(offline_val) or online_val is None or (isinstance(online_val, float) and math.isnan(online_val)))
                    else float("inf")
                )
                max_abs_diff = max(max_abs_diff, diff if diff != float("inf") else max_abs_diff)
                mismatches.append({
                    "transaction_id": txn_id, "feature": name,
                    "offline_value": None if pd.isna(offline_val) else float(offline_val),
                    "online_value": online_val,
                })
        if txn_ok:
            n_matching_transactions += 1

    return {
        "n_transactions_compared": len(online_results),
        "n_features_compared": n_compared,
        "n_transactions_fully_matching": n_matching_transactions,
        "n_mismatches": len(mismatches),
        "max_abs_diff": max_abs_diff,
        "tolerance": tolerance,
        "mismatches": mismatches[:50],  # cap for report readability; count above is exact
    }


def validate_score_parity(offline_scores: dict, online_results: list, tolerance: float = 1e-6) -> dict:
    """
    `offline_scores`: {transaction_id: risk_score} from the offline batch
    pipeline (same model, same preprocessing, applied in one batch call).
    `online_results`: list of engine result dicts.
    """
    diffs = []
    mismatches = []
    for r in online_results:
        txn_id = r["transaction_id"]
        if txn_id not in offline_scores:
            mismatches.append({"transaction_id": txn_id, "reason": "not found in offline scores"})
            continue
        offline_score = offline_scores[txn_id]
        online_score = r["risk_score"]
        diff = abs(offline_score - online_score)
        diffs.append(diff)
        if diff > tolerance:
            mismatches.append({
                "transaction_id": txn_id, "offline_score": offline_score,
                "online_score": online_score, "abs_diff": diff,
            })

    diffs_arr = np.array(diffs) if diffs else np.array([0.0])
    return {
        "n_compared": len(online_results),
        "n_mismatches": len(mismatches),
        "max_abs_diff": float(diffs_arr.max()),
        "mean_abs_diff": float(diffs_arr.mean()),
        "correlation": (
            float(np.corrcoef(
                [offline_scores[r["transaction_id"]] for r in online_results if r["transaction_id"] in offline_scores],
                [r["risk_score"] for r in online_results if r["transaction_id"] in offline_scores],
            )[0, 1]) if len(diffs) > 1 else float("nan")
        ),
        "tolerance": tolerance,
        "mismatches": mismatches[:50],
    }


def validate_decision_parity(offline_decisions: dict, online_results: list) -> dict:
    """
    `offline_decisions`: {transaction_id: decision_str} from applying the
    frozen Phase 5 policy to the offline batch scores.
    """
    n_compared = 0
    n_matching = 0
    mismatches = []
    for r in online_results:
        txn_id = r["transaction_id"]
        if txn_id not in offline_decisions:
            continue
        n_compared += 1
        offline_decision = offline_decisions[txn_id]
        online_decision = r["decision"]
        if offline_decision == online_decision:
            n_matching += 1
        else:
            mismatches.append({
                "transaction_id": txn_id, "offline_decision": offline_decision, "online_decision": online_decision,
            })

    return {
        "n_compared": n_compared,
        "n_matching": n_matching,
        "pct_matching": (n_matching / n_compared) if n_compared > 0 else float("nan"),
        "n_mismatches": len(mismatches),
        "mismatches": mismatches[:50],
    }
