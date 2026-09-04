"""
Phase 2A test suite: evaluation framework (`src/evaluation/metrics.py` and
`src/evaluation/operational_eval.py`) on small, hand-constructed examples
where the correct answer can be verified by hand.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import (
    compute_ranking_metrics, compute_threshold_metrics, select_threshold_by_f1,
)
from src.evaluation.operational_eval import (
    recall_at_k, precision_at_k, fixed_budget_report, multi_budget_table,
)


# ---------------------------------------------------------------------------
# metrics.py
# ---------------------------------------------------------------------------

def test_compute_ranking_metrics_perfect_separation():
    y_true = [0, 0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.3, 0.8, 0.9]  # perfectly ranks fraud above non-fraud
    result = compute_ranking_metrics(y_true, y_score)
    assert result["pr_auc"] == pytest.approx(1.0)
    assert result["roc_auc"] == pytest.approx(1.0)


def test_compute_ranking_metrics_requires_both_classes():
    with pytest.raises(ValueError):
        compute_ranking_metrics([0, 0, 0], [0.1, 0.2, 0.3])


def test_compute_threshold_metrics_known_confusion_matrix():
    # 2 true positives, 1 false positive, 1 false negative, 2 true negatives
    y_true = [1, 1, 1, 0, 0, 0]
    y_score = [0.9, 0.8, 0.1, 0.7, 0.2, 0.3]
    result = compute_threshold_metrics(y_true, y_score, threshold=0.5)
    # predicted positive: indices 0,1,3 (scores 0.9, 0.8, 0.7)
    # y_true at those indices: 1, 1, 0 -> TP=2, FP=1
    # remaining: idx2 (true=1, pred=0) -> FN=1; idx4,5 (true=0,pred=0) -> TN=2
    assert result["true_positives"] == 2
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["true_negatives"] == 2
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["false_positive_rate"] == pytest.approx(1 / 3)


def test_select_threshold_by_f1_returns_reasonable_threshold():
    y_val_true = [0, 0, 0, 0, 1, 1, 1]
    y_val_score = [0.05, 0.1, 0.2, 0.3, 0.6, 0.7, 0.9]
    result = select_threshold_by_f1(y_val_true, y_val_score)
    assert 0.0 <= result["threshold"] <= 1.0
    assert 0.0 <= result["val_f1_at_threshold"] <= 1.0
    # With clean separation, best F1 threshold should perfectly separate.
    assert result["val_f1_at_threshold"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# operational_eval.py
# ---------------------------------------------------------------------------

def test_recall_at_k_hand_verified():
    # 10 transactions, 2 fraud (indices 8, 9 = highest scores).
    y_true = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.15, 0.05, 0.3, 0.25, 0.35, 0.4, 0.9, 0.95]
    # top 20% = top 2 transactions -> indices 8,9 (scores .9, .95) both fraud
    recall = recall_at_k(y_true, y_score, k_fraction=0.2)
    assert recall == pytest.approx(1.0)


def test_recall_at_k_partial_capture():
    # 10 transactions, fraud at idx 8 (score .9) and idx 0 (score .05, LOW).
    y_true = [1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    y_score = [0.05, 0.2, 0.15, 0.05, 0.3, 0.25, 0.35, 0.4, 0.9, 0.5]
    # top 10% = top 1 transaction -> idx 8 (score .9) is fraud -> captures 1/2 fraud
    recall = recall_at_k(y_true, y_score, k_fraction=0.1)
    assert recall == pytest.approx(0.5)


def test_recall_at_k_no_fraud_returns_nan():
    y_true = [0, 0, 0, 0]
    y_score = [0.1, 0.2, 0.3, 0.4]
    result = recall_at_k(y_true, y_score, k_fraction=0.5)
    assert np.isnan(result)


def test_precision_at_k_hand_verified():
    # top 30% of 10 = 3 transactions. Highest scores: idx 9(.95), 8(.9), 7(.4)
    y_true = [0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
    y_score = [0.1, 0.2, 0.15, 0.05, 0.3, 0.25, 0.35, 0.4, 0.9, 0.95]
    # top 3 by score: idx9(.95,fraud=0), idx8(.9,fraud=1), idx7(.4,fraud=1)
    precision = precision_at_k(y_true, y_score, k_fraction=0.3)
    assert precision == pytest.approx(2 / 3)


def test_fixed_budget_report_correct_count_and_fields():
    n = 100
    y_true = np.zeros(n, dtype=int)
    y_true[:5] = 1  # 5 fraud cases total
    y_score = np.arange(n)  # monotonically increasing -> top scores are last indices
    # Put fraud cases among the highest-scored transactions.
    y_true = np.zeros(n, dtype=int)
    y_true[-5:] = 1  # last 5 (highest scores) are fraud

    report = fixed_budget_report(y_true, y_score, review_budget=0.05)
    assert report["n_transactions_total"] == 100
    assert report["n_transactions_reviewed"] == 5  # top 5% of 100 = 5
    assert report["total_fraud_cases"] == 5
    assert report["fraud_cases_captured"] == 5
    assert report["fraud_recall"] == pytest.approx(1.0)
    assert report["precision_among_reviewed"] == pytest.approx(1.0)


def test_fixed_budget_report_review_count_scales_with_budget():
    n = 1000
    rng = np.random.default_rng(0)
    y_true = (rng.random(n) < 0.05).astype(int)
    y_score = rng.random(n)
    for budget in (0.01, 0.02, 0.05, 0.1):
        report = fixed_budget_report(y_true, y_score, review_budget=budget)
        expected_n = max(1, int(np.ceil(n * budget)))
        assert report["n_transactions_reviewed"] == expected_n


def test_multi_budget_table_not_hardcoded_to_one_budget():
    n = 500
    rng = np.random.default_rng(1)
    y_true = (rng.random(n) < 0.05).astype(int)
    y_score = rng.random(n)
    budgets = [0.01, 0.02, 0.05, 0.10]
    table = multi_budget_table(y_true, y_score, budgets)
    assert len(table) == len(budgets)
    assert [row["budget"] for row in table] == budgets
    # Recall should be monotonically non-decreasing as budget increases.
    recalls = [row["recall_at_k"] for row in table]
    assert all(r2 >= r1 - 1e-9 for r1, r2 in zip(recalls, recalls[1:]))


def test_precision_at_k_no_positives_selected_edge_case():
    y_true = [0, 0, 0]
    y_score = [0.1, 0.2, 0.3]
    # k_fraction must be > 0; smallest valid budget still selects >=1 row.
    result = precision_at_k(y_true, y_score, k_fraction=0.01)
    assert result == 0.0  # 0 fraud in the 1 selected transaction


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
