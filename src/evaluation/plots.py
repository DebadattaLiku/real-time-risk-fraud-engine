"""
Phase 2A: Evaluation plots. Kept to the three that are actually decision-
relevant for comparing models under this project's goals — no unnecessary
visualizations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_pr_curves(models: dict, out_path: Path, title: str = "Precision-Recall Curve") -> Path:
    """
    `models`: {label: (y_true, y_score)} — one curve per entry, overlaid,
    so multiple models can be compared on one plot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score

    fig, ax = plt.subplots(figsize=(7, 6))
    for label, (y_true, y_score) in models.items():
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax.plot(recall, precision, label=f"{label} (PR-AUC={ap:.4f})")
    baseline_rate = np.mean(list(models.values())[0][0])
    ax.axhline(baseline_rate, color="gray", linestyle="--", linewidth=1,
               label=f"No-skill baseline ({baseline_rate:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_roc_curves(models: dict, out_path: Path, title: str = "ROC Curve") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score

    fig, ax = plt.subplots(figsize=(7, 6))
    for label, (y_true, y_score) in models.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, label=f"{label} (ROC-AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_recall_at_k_curve(
    models: dict, out_path: Path,
    budgets=None, title: str = "Recall@K (Review Budget) Curve",
) -> Path:
    """
    `models`: {label: (y_true, y_score)}. Plots fraud recall as a function
    of review-budget fraction, across a fine grid of budgets so the curve
    is smooth rather than just the 3 headline points (1%/2%/5%).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.evaluation.operational_eval import recall_at_k

    if budgets is None:
        budgets = np.linspace(0.005, 0.20, 40)

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, (y_true, y_score) in models.items():
        recalls = [recall_at_k(y_true, y_score, k) for k in budgets]
        ax.plot(np.array(budgets) * 100, recalls, label=label)
    for headline in (0.01, 0.02, 0.05):
        ax.axvline(headline * 100, color="lightgray", linestyle=":", linewidth=1)
    ax.set_xlabel("Review budget (top K% of transactions, by score)")
    ax.set_ylabel("Fraud recall")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
