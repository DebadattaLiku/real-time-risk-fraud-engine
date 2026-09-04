"""
Phase 3: Complementarity analysis between a supervised score (LightGBM
fraud probability) and an unsupervised score (Isolation Forest anomaly
score).

Reuses `src/evaluation/operational_eval.top_k_mask` for all top-K selection
so tie-breaking is identical to every other top-K calculation in this
project — no separate implementation of "what counts as the top K%" exists
anywhere in the codebase.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr

from src.evaluation.operational_eval import top_k_mask


def score_correlation(score_a: np.ndarray, score_b: np.ndarray) -> dict:
    """
    Pearson (linear) and Spearman (rank/monotonic) correlation between two
    scores. Spearman is arguably the more relevant one here since both
    scores are ultimately used for ranking/top-K selection, not as
    calibrated probabilities — but both are reported since they answer
    slightly different questions (linear relationship vs. rank agreement).
    """
    score_a = np.asarray(score_a)
    score_b = np.asarray(score_b)
    pearson_r, pearson_p = pearsonr(score_a, score_b)
    spearman_r, spearman_p = spearmanr(score_a, score_b)
    return {
        "pearson_r": float(pearson_r),
        "pearson_p_value": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p_value": float(spearman_p),
    }


def overlap_analysis(
    y_true, score_a: np.ndarray, score_b: np.ndarray, k_fraction: float,
    name_a: str = "model_a", name_b: str = "model_b",
) -> dict:
    """
    At a given review budget, compares which transactions each model's
    top-K selects and which fraud cases each one captures.

    Directly answers: does model_b (typically the anomaly detector) catch
    fraud that model_a (typically LightGBM) misses at the SAME budget —
    not a larger, budget-exceeding pool.
    """
    y_true = np.asarray(y_true)
    mask_a, n_a = top_k_mask(score_a, k_fraction)
    mask_b, n_b = top_k_mask(score_b, k_fraction)

    both_mask = mask_a & mask_b
    only_a_mask = mask_a & ~mask_b
    only_b_mask = mask_b & ~mask_a

    total_fraud = int(y_true.sum())
    fraud_a = int(y_true[mask_a].sum())
    fraud_b = int(y_true[mask_b].sum())
    fraud_both = int(y_true[both_mask].sum())
    fraud_only_a = int(y_true[only_a_mask].sum())
    fraud_only_b = int(y_true[only_b_mask].sum())

    return {
        "k_fraction": k_fraction,
        "n_reviewed_each": n_a,  # n_a == n_b by construction (same k_fraction, same n)
        "row_overlap_count": int(both_mask.sum()),
        "row_overlap_fraction_of_topk": float(both_mask.sum() / n_a) if n_a > 0 else float("nan"),
        f"fraud_captured_by_{name_a}": fraud_a,
        f"fraud_captured_by_{name_b}": fraud_b,
        "fraud_captured_by_both": fraud_both,
        f"fraud_captured_only_by_{name_a}": fraud_only_a,
        f"fraud_captured_only_by_{name_b}": fraud_only_b,
        "total_fraud_cases": total_fraud,
    }


def union_diagnostic(
    y_true, score_a: np.ndarray, score_b: np.ndarray, k_fraction: float,
) -> dict:
    """
    DIAGNOSTIC ONLY — the union of model_a's top-K and model_b's top-K can
    contain up to 2*K% of transactions (more than the fixed review budget
    the rest of this project evaluates under). This function exists only
    to measure the theoretical upper bound of combined coverage, NOT to
    propose it as an operational policy. Callers must not present this as
    a deployable review budget.
    """
    y_true = np.asarray(y_true)
    mask_a, n_a = top_k_mask(score_a, k_fraction)
    mask_b, n_b = top_k_mask(score_b, k_fraction)
    union_mask = mask_a | mask_b

    total_fraud = int(y_true.sum())
    fraud_union = int(y_true[union_mask].sum())
    fraud_a_only_topk = int(y_true[mask_a].sum())

    return {
        "k_fraction_each_model": k_fraction,
        "n_reviewed_union": int(union_mask.sum()),
        "effective_combined_budget_fraction": float(union_mask.sum() / len(y_true)),
        "fraud_captured_by_union": fraud_union,
        "fraud_recall_union": (fraud_union / total_fraud) if total_fraud > 0 else float("nan"),
        "fraud_captured_by_model_a_alone_at_same_k": fraud_a_only_topk,
        "incremental_fraud_from_union_vs_model_a_alone": fraud_union - fraud_a_only_topk,
        "note": (
            "DIAGNOSTIC ONLY. The union exceeds the fixed review budget "
            "(up to 2x the per-model k_fraction of transactions) and is "
            "NOT a valid operational policy on its own."
        ),
    }


def conditional_low_lgbm_high_anomaly(
    y_true, lgbm_score, anomaly_score,
    lgbm_low_threshold: float, anomaly_high_threshold: float,
) -> dict:
    """
    Transactions where LightGBM assigns LOW fraud risk (score below
    `lgbm_low_threshold`) but Isolation Forest assigns HIGH anomaly risk
    (score above `anomaly_high_threshold`). Both thresholds must be
    determined from VALIDATION data by the caller — this function only
    applies whatever numeric threshold values it's given; it does not
    derive them, so it cannot itself introduce test-set leakage as long as
    the caller passes validation-derived numbers.
    """
    lgbm_score = np.asarray(lgbm_score)
    anomaly_score = np.asarray(anomaly_score)
    y_true = np.asarray(y_true)

    condition_mask = (lgbm_score < lgbm_low_threshold) & (anomaly_score > anomaly_high_threshold)
    n_flagged = int(condition_mask.sum())
    fraud_in_flagged = int(y_true[condition_mask].sum())
    fraud_rate_in_flagged = (fraud_in_flagged / n_flagged) if n_flagged > 0 else float("nan")

    overall_fraud_rate = float(y_true.mean())

    return {
        "lgbm_low_threshold": lgbm_low_threshold,
        "anomaly_high_threshold": anomaly_high_threshold,
        "n_transactions_matching_condition": n_flagged,
        "fraud_cases_in_condition": fraud_in_flagged,
        "fraud_rate_in_condition": fraud_rate_in_flagged,
        "overall_fraud_rate": overall_fraud_rate,
        "lift_over_overall_rate": (
            fraud_rate_in_flagged / overall_fraud_rate
            if n_flagged > 0 and overall_fraud_rate > 0 else float("nan")
        ),
    }
