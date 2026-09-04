"""
Phase 2A: Reusable ranking and threshold metrics.

These functions are model-agnostic — they operate on (y_true, y_score)
pairs, so they will be reused unchanged for every future model (LightGBM,
Isolation Forest anomaly scores, the fused risk score, etc.), which is the
whole point of a shared evaluation framework: any "model X outperforms
model Y" claim later in the project goes through this same code.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def compute_ranking_metrics(y_true, y_score) -> dict:
    """
    Threshold-independent ranking quality. PR-AUC (average precision) is
    the primary metric for this project given class imbalance — ROC-AUC is
    reported alongside it because it's a familiar reference point, but it
    is not used as the primary criterion since it can look optimistic under
    heavy imbalance (dominated by the large true-negative volume).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        raise ValueError(
            "compute_ranking_metrics requires both classes present in y_true; "
            f"got unique values {np.unique(y_true)}"
        )
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }


def compute_threshold_metrics(y_true, y_score, threshold: float) -> dict:
    """
    Precision / recall / F1 / false-positive-rate at one explicit,
    documented threshold. Does not select the threshold — the caller must
    supply it (see `select_threshold_by_f1` for a train/val-safe way to
    choose one).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan")

    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": fpr,
        "n_flagged": int(y_pred.sum()),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }


def select_threshold_by_f1(y_val_true, y_val_score) -> dict:
    """
    Choose the classification threshold that maximizes F1 on the
    VALIDATION set only. This is the one place threshold *selection*
    happens in Phase 2A — the resulting threshold is then applied
    (unchanged) to the test set for final reporting. Never call this with
    test-set scores.
    """
    y_val_true = np.asarray(y_val_true)
    y_val_score = np.asarray(y_val_score)
    precisions, recalls, thresholds = precision_recall_curve(y_val_true, y_val_score)
    # precision_recall_curve returns len(thresholds) == len(precisions) - 1
    f1s = np.where(
        (precisions[:-1] + recalls[:-1]) > 0,
        2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-12),
        0.0,
    )
    if len(thresholds) == 0:
        # Degenerate case (e.g. constant scores) — fall back to 0.5.
        return {"threshold": 0.5, "val_f1_at_threshold": 0.0}
    best_idx = int(np.argmax(f1s))
    return {
        "threshold": float(thresholds[best_idx]),
        "val_f1_at_threshold": float(f1s[best_idx]),
    }


def get_pr_curve(y_true, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    return precision, recall, thresholds


def get_roc_curve(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return fpr, tpr, thresholds
