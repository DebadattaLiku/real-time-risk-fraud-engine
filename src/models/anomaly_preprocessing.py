"""
Phase 3: Anomaly-detection-specific preprocessing.

Sits on the same Phase 1 `FeaturePipeline` output as Phase 2A/2B's
preprocessors, but with a different treatment again — Isolation Forest
needs fully numeric, NaN-free input, and (per the Phase 3 brief) should NOT
blindly one-hot every categorical column.

## Feature subset: V columns excluded, everything else kept

Isolation Forest isolates points via random axis-aligned splits; its
effectiveness degrades as irrelevant/noisy dimensions dominate the split
budget (a well-known curse-of-dimensionality issue for isolation-based
methods, distinct from a supervised tree model like LightGBM, which can
learn to ignore uninformative features via loss-driven gain — an
unsupervised isolation forest has no such signal to lean on).

The `V1`-`V339` columns are Vesta's own opaque, pre-engineered features
(339 of the model's 391 available columns — Phase 0 found much of this
group undocumented and highly variable in missingness). Including all 339
would let this single, unexplained feature group dominate the isolation
splits by sheer dimensional weight. They are EXCLUDED from the anomaly
feature set for this reason, documented here rather than silently dropped.
Every other feature group is kept: `TransactionAmt`, `card1-6`, `addr1-2`,
`dist1-2`, `P_emaildomain`, `R_emaildomain`, `ProductCD`, `C1-14`, `D1-15`,
`M1-9` — 52 features total (38 numeric + 14 categorical).

This is a documented judgment call, not a validated ablation — the
alternative (include V columns) was not tested in Phase 3 due to compute
budget prioritization; see the Phase 3 report's limitations section.

## Numeric handling

Median imputation, fit on TRAINING DATA ONLY (same rationale as Phase 2A's
LogReg path: several of these numeric columns, e.g. `D` columns, are
skewed and have real missingness, and Isolation Forest cannot accept NaN
at all). No scaling — Isolation Forest's random-split partitioning is
scale-invariant per feature (each feature's split threshold is chosen
within that feature's own observed range), so scaling would not change
results and was correctly omitted.

## Categorical handling: frequency encoding, not one-hot

Each of the 14 categorical columns is replaced with a single numeric
column: the TRAINING-set frequency (proportion of training rows) of that
category value. This avoids a one-hot dimensionality explosion (kept out
per the brief's explicit instruction not to blindly one-hot high-cardinality
columns like `P_emaildomain`/`R_emaildomain`, ~59-60 distinct values each),
keeps the total feature count low and interpretable, and is a natural fit
for anomaly detection specifically: a rare category value gets a low
frequency value, which is itself a weak anomaly signal Isolation Forest can
use directly. A category value never seen in training (including in
validation/test) is assigned frequency 0.0 — the natural "as rare as
possible relative to training" value, not an arbitrary placeholder.

No target encoding, no `isFraud` anywhere in this module, no encoder or
imputer fit on anything but the training partition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.schema import FeatureSpec

EXCLUDED_FEATURE_GROUPS = {"V"}


def get_anomaly_feature_columns(schema: list[FeatureSpec]) -> tuple[list, list]:
    """Returns (numeric_cols, categorical_cols) for the anomaly feature subset."""
    numeric = [
        s.name for s in schema
        if s.used and s.value_type == "numeric" and s.feature_group not in EXCLUDED_FEATURE_GROUPS
    ]
    categorical = [
        s.name for s in schema
        if s.used and s.value_type == "categorical" and s.feature_group not in EXCLUDED_FEATURE_GROUPS
    ]
    return numeric, categorical


class AnomalyPreprocessor:
    """
    Explicit fit/transform contract, matching the other model-specific
    preprocessors in this project. Output is a fully numeric, NaN-free
    numpy array suitable for `sklearn.ensemble.IsolationForest` (or any
    other numeric-input estimator).
    """

    def __init__(self, schema: list[FeatureSpec]):
        self.schema = schema
        self.numeric_cols, self.categorical_cols = get_anomaly_feature_columns(schema)
        self._medians: dict[str, float] = {}
        self._frequencies: dict[str, dict] = {}
        self._fitted = False

    def fit(self, X_train: pd.DataFrame) -> "AnomalyPreprocessor":
        for col in self.numeric_cols:
            self._medians[col] = float(X_train[col].median())

        n_train = len(X_train)
        for col in self.categorical_cols:
            counts = X_train[col].astype(str).value_counts()
            self._frequencies[col] = (counts / n_train).to_dict()

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(
                "AnomalyPreprocessor.transform called before fit(). Call "
                "fit(X_train) first, using only the training partition."
            )
        blocks = []

        if self.numeric_cols:
            numeric_block = X[self.numeric_cols].copy()
            for col in self.numeric_cols:
                numeric_block[col] = numeric_block[col].fillna(self._medians[col])
            blocks.append(numeric_block.to_numpy(dtype=np.float32))

        if self.categorical_cols:
            cat_block = np.zeros((len(X), len(self.categorical_cols)), dtype=np.float32)
            for j, col in enumerate(self.categorical_cols):
                freq_map = self._frequencies[col]
                # Unseen category values (never observed in training,
                # including values only seen in val/test) get frequency 0.0.
                cat_block[:, j] = X[col].astype(str).map(freq_map).fillna(0.0).to_numpy(dtype=np.float32)
            blocks.append(cat_block)

        out = np.concatenate(blocks, axis=1)
        assert not np.isnan(out).any(), "AnomalyPreprocessor output must never contain NaN"
        return out

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        self.fit(X_train)
        return self.transform(X_train)

    def get_feature_names(self) -> list:
        return list(self.numeric_cols) + list(self.categorical_cols)
