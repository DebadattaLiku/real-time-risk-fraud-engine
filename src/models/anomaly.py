"""
Phase 3: Isolation Forest anomaly detection.

## Score direction (documented explicitly, per the Phase 3 brief)

scikit-learn's `IsolationForest.decision_function(X)` returns values where
HIGHER = MORE NORMAL (inlier-like) and LOWER (more negative) = MORE
ANOMALOUS. This is the OPPOSITE convention from a fraud probability, where
higher = more suspicious.

This module defines:

    anomaly_score = -decision_function(X)

so that, consistently with every other score in this project (LightGBM's
`predict_proba`, Logistic Regression's `predict_proba`),
**HIGHER anomaly_score = MORE UNUSUAL / MORE SUSPICIOUS**. All downstream
Phase 3 code (evaluation, complementarity analysis, figures) uses this
sign-flipped `anomaly_score`, never the raw `decision_function` output
directly — mixing the two conventions without this explicit flip would
silently invert every ranking metric.

`contamination` is accepted as a parameter (per the brief's suggested
hyperparameter list) but note: it only shifts `decision_function`'s
internal offset by a constant (`score_samples(X) - offset_`, where
`offset_` is set from the contamination-quantile of TRAINING scores); it
does not change the relative ORDER of scores. Since every metric this
project uses (PR-AUC, ROC-AUC, Recall@K, Precision@K, rank correlation) is
invariant to a constant score shift, `contamination` is expected to have
no effect on any reported ranking metric here — this is verified
empirically in the Phase 3 report rather than assumed.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

DEFAULT_PARAMS = dict(
    n_estimators=100,
    max_samples="auto",
    contamination="auto",
    max_features=1.0,
    random_state=42,
    n_jobs=1,  # this execution environment has a single CPU core
)


class AnomalyDetector:
    """
    Thin wrapper around `sklearn.ensemble.IsolationForest` with an explicit
    fit/transform-style contract (mirrors the other model wrappers in this
    project) and the sign-flip documented above baked in, so callers never
    have to remember to negate the score themselves.
    """

    def __init__(self, **params):
        p = dict(DEFAULT_PARAMS)
        p.update(params)
        self.params = p
        self.model = IsolationForest(**p)
        self._fitted = False

    def fit(self, Z_train: np.ndarray) -> "AnomalyDetector":
        """
        Fits on TRAINING DATA ONLY. Isolation Forest is unsupervised — no
        label array is accepted or used here, by design; passing `y` would
        not even be meaningful to `IsolationForest.fit`, but the point is
        made explicit in this signature (no `y` parameter exists at all)
        rather than merely documented.
        """
        self.model.fit(Z_train)
        self._fitted = True
        return self

    def anomaly_score(self, Z: np.ndarray) -> np.ndarray:
        """Higher = more anomalous/suspicious. See module docstring."""
        if not self._fitted:
            raise RuntimeError(
                "AnomalyDetector.anomaly_score called before fit(). Call "
                "fit(Z_train) first, using only the training partition."
            )
        return -self.model.decision_function(Z)
