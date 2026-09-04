"""
Phase 2B: LightGBM training.

Keeps the same PR-AUC-primary evaluation philosophy as the rest of this
project by using a custom `feval` for early stopping that calls the exact
same `sklearn.metrics.average_precision_score` function
`src/evaluation/metrics.py` uses for reporting — so "best iteration" is
chosen by the same yardstick the project reports results in, not by
LightGBM's own internal metric implementation, which could disagree at the
margins.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score

DEFAULT_PARAMS = dict(
    objective="binary",
    metric="None",  # disable LightGBM's default metric (binary_logloss) so
                     # early stopping is driven ONLY by our custom PR-AUC
                     # feval below. Without this, LightGBM tracks
                     # binary_logloss alongside the custom metric, and
                     # first_metric_only=True in the early-stopping
                     # callback locks onto whichever metric is registered
                     # first — which turned out to be binary_logloss, not
                     # our PR-AUC feval. This was caught during a real run:
                     # the scale_pos_weight variant stopped after a single
                     # boosting round even though its validation PR-AUC was
                     # visibly still climbing every round when inspected
                     # manually, because binary_logloss (which increases
                     # under heavy scale_pos_weight, as the model becomes
                     # deliberately less well-calibrated) was silently
                     # driving the stopping decision instead.
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    min_child_samples=50,
    reg_alpha=0.1,
    reg_lambda=0.1,
    n_estimators=1000,
    n_jobs=1,       # this execution environment has a single CPU core
    random_state=42,
    verbose=-1,
)


def _pr_auc_feval(y_true, y_pred):
    """
    Custom eval metric for LightGBM's sklearn API. In the installed
    LightGBM version (4.7.0), a callable passed as `eval_metric` receives
    (y_true, y_pred) arrays directly — NOT a `Dataset` object with a
    `.get_label()` method (that's the native/Booster-level `feval`
    signature, which differs from the sklearn-wrapper signature). Verified
    empirically: an earlier version of this function assumed the
    Dataset-style signature and raised
    `AttributeError: 'numpy.ndarray' object has no attribute 'get_label'`
    on the very first real training run.
    """
    return "pr_auc", float(average_precision_score(y_true, y_pred)), True  # higher is better


def train_lightgbm(
    Z_train, y_train, Z_val, y_val,
    categorical_features: list,
    scale_pos_weight: float | None = None,
    params: dict | None = None,
    early_stopping_rounds: int = 50,
):
    """
    Train one LightGBM model. Early stopping and best-iteration selection
    use VALIDATION ONLY (`Z_val`/`y_val`) — never test data. Returns the
    fitted `LGBMClassifier` and a small dict of training diagnostics
    (best_iteration, best validation PR-AUC at that iteration).

    `scale_pos_weight=None` means standard/unweighted training. Passing a
    numeric value (e.g. n_negative/n_positive computed from TRAINING labels
    only) enables class-imbalance weighting — the caller decides whether to
    use it and with what value; this function does not choose one on its
    own, per the brief's instruction not to blindly assume weighting helps.
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    if scale_pos_weight is not None:
        p["scale_pos_weight"] = scale_pos_weight

    model = lgb.LGBMClassifier(**p)
    model.fit(
        Z_train, y_train,
        eval_set=[(Z_val, y_val)],
        eval_metric=_pr_auc_feval,
        categorical_feature=categorical_features,
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, first_metric_only=True, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    best_iteration = model.best_iteration_
    # best_score_ is keyed by eval-set name ("valid_0") then metric name.
    best_val_pr_auc = float(model.best_score_["valid_0"]["pr_auc"])

    info = {
        "best_iteration": int(best_iteration) if best_iteration else int(p["n_estimators"]),
        "best_val_pr_auc_during_training": best_val_pr_auc,
        "params": p,
    }
    return model, info


def get_feature_importance(model, feature_names: list, importance_type: str = "gain") -> list:
    """
    Returns [(feature_name, importance), ...] sorted descending. `gain`
    (total split gain contributed by a feature) is used by default rather
    than `split` (raw split count), since gain better reflects how much a
    feature actually reduced loss, though both are legitimate and neither
    implies a causal relationship — see the Phase 2B report for the caveat.
    """
    importances = model.booster_.feature_importance(importance_type=importance_type)
    pairs = list(zip(feature_names, importances.tolist()))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs
