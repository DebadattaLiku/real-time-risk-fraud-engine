"""
Phase 2A: Operational fraud-review metrics.

These answer the question the project's evaluation goals are actually built
around: "if operations can only manually review a fixed slice of
transactions, how much fraud do we catch, and how much of what we flag is
real?" This is deliberately kept separate from `metrics.py` (threshold-
independent ranking / single-threshold classification metrics) because it
has a different unit of analysis: a REVIEW BUDGET (a fraction or count of
transactions), not a probability threshold.
"""

from __future__ import annotations

import math

import numpy as np


def top_k_mask(y_score, k_fraction: float) -> tuple[np.ndarray, int]:
    """
    Rank all transactions by score descending and select the top
    `k_fraction` of them. Ties at the cutoff score are broken by stable
    sort order (original row order), documented rather than hidden —
    operationally, some tie-breaking rule is unavoidable when a review
    queue has a hard capacity, and this makes the same tie-breaking rule
    apply consistently to every call.

    Public (Phase 3 onward): exposed so other modules — e.g.
    `src/evaluation/complementarity.py`'s top-K overlap analysis — reuse
    the exact same row-selection/tie-breaking logic instead of
    reimplementing it, per the project's "don't duplicate evaluation
    logic" principle. Behavior is unchanged from the original
    Phase-2A-private `_top_k_mask`; this is a visibility change only.

    Returns (boolean mask of selected rows, n_selected).
    """
    y_score = np.asarray(y_score)
    n = len(y_score)
    if not (0 < k_fraction <= 1):
        raise ValueError(f"k_fraction must be in (0, 1], got {k_fraction}")
    n_selected = max(1, math.ceil(n * k_fraction))
    # argsort ascending, take the last n_selected (highest scores); stable
    # sort so ties resolve by original row order, not arbitrarily.
    order = np.argsort(y_score, kind="mergesort")
    top_idx = order[-n_selected:]
    mask = np.zeros(n, dtype=bool)
    mask[top_idx] = True
    return mask, n_selected


# Kept as an alias for any existing internal call sites / imports.
_top_k_mask = top_k_mask


def recall_at_k(y_true, y_score, k_fraction: float) -> float:
    """
    Of all actual fraud cases, what fraction are captured in the top
    `k_fraction` highest-risk transactions? Returns NaN if there is no
    fraud in `y_true` (recall is undefined, not zero, in that case).
    """
    y_true = np.asarray(y_true)
    total_fraud = int(y_true.sum())
    if total_fraud == 0:
        return float("nan")
    mask, _ = _top_k_mask(y_score, k_fraction)
    captured = int(y_true[mask].sum())
    return captured / total_fraud


def precision_at_k(y_true, y_score, k_fraction: float) -> float:
    """
    Of the top `k_fraction` highest-risk transactions, what fraction are
    actually fraudulent?
    """
    y_true = np.asarray(y_true)
    mask, n_selected = _top_k_mask(y_score, k_fraction)
    if n_selected == 0:
        return float("nan")
    captured = int(y_true[mask].sum())
    return captured / n_selected


def multi_budget_table(y_true, y_score, budgets: list) -> list:
    """
    Recall@K and Precision@K across several review-capacity budgets in one
    call, e.g. budgets=[0.01, 0.02, 0.05] for a 1% / 2% / 5% review queue.
    Not hard-coded to any single budget — any list of fractions works.
    """
    rows = []
    for k in budgets:
        rows.append({
            "budget": k,
            "recall_at_k": recall_at_k(y_true, y_score, k),
            "precision_at_k": precision_at_k(y_true, y_score, k),
        })
    return rows


def fixed_budget_report(y_true, y_score, review_budget: float) -> dict:
    """
    Full operational report for one review budget (e.g. review_budget=0.02
    means "operations reviews the top 2% highest-risk transactions").

    Reports: number reviewed, fraud captured, total fraud, fraud recall,
    and precision among reviewed transactions. This is the reusable
    interface Phase 2A calls for — pass any `review_budget` in (0, 1].
    """
    y_true = np.asarray(y_true)
    total_fraud = int(y_true.sum())
    mask, n_reviewed = _top_k_mask(y_score, review_budget)
    fraud_captured = int(y_true[mask].sum())

    fraud_recall = (fraud_captured / total_fraud) if total_fraud > 0 else float("nan")
    precision_reviewed = (fraud_captured / n_reviewed) if n_reviewed > 0 else float("nan")

    return {
        "review_budget": review_budget,
        "n_transactions_total": int(len(y_true)),
        "n_transactions_reviewed": int(n_reviewed),
        "fraud_cases_captured": fraud_captured,
        "total_fraud_cases": total_fraud,
        "fraud_recall": fraud_recall,
        "precision_among_reviewed": precision_reviewed,
    }
