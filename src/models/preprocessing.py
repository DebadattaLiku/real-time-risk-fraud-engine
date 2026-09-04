"""
Phase 2A: Logistic-Regression-specific preprocessing.

This sits ON TOP of the Phase 1 `FeaturePipeline` output (391 leakage-safe,
transaction-time-only columns; categorical NaNs already replaced with an
explicit `"__missing__"` placeholder; numeric NaNs still raw). Phase 1's
output is model-agnostic. Logistic Regression specifically needs:

    - numeric: no NaNs allowed at all -> median imputation (a LEARNED
      transformation, fit on train only), then scaling, since LogReg's
      L2-penalized solver is scale-sensitive (unscaled wide-range columns
      like V-features vs. TransactionAmt would dominate the penalty term).
    - categorical: one-hot encoding, since LogReg has no native handling
      of categorical variables. Encoded using only categories observed in
      TRAINING data; unseen categories in val/test are safely ignored
      (encoded as all-zero) rather than crashing or silently guessing.

None of this is target encoding, and nothing here is fit on anything but
the training partition, enforced structurally (see `LogRegPreprocessor`
below) the same way Phase 1's `FeaturePipeline` enforces its own contract.

A future model (e.g. LightGBM) will need a DIFFERENT preprocessing path —
tree models don't need scaling and often handle categoricals natively —
which is exactly why this lives in `src/models/` as a model-specific step,
not folded into the shared Phase 1 pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.schema import FeatureSpec


def get_numeric_and_categorical_columns(schema: list[FeatureSpec]) -> tuple[list, list]:
    numeric = [s.name for s in schema if s.used and s.value_type == "numeric"]
    categorical = [s.name for s in schema if s.used and s.value_type == "categorical"]
    return numeric, categorical


class LogRegPreprocessor:
    """
    Wraps a scikit-learn ColumnTransformer with an explicit fit/transform
    contract (mirroring Phase 1's `FeaturePipeline`): `transform()` raises
    if called before `fit()`, and `fit()` accepts exactly one dataframe —
    the caller is responsible for passing only the training partition.
    """

    def __init__(self, schema: list[FeatureSpec]):
        self.schema = schema
        self.numeric_cols, self.categorical_cols = get_numeric_and_categorical_columns(schema)

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        # Phase 1 already replaced categorical NaN with the constant
        # "__missing__" placeholder (not a learned value), so no imputer
        # step is needed here — only the (learned-from-train-categories)
        # one-hot encoding. handle_unknown="ignore" is what makes unseen
        # validation/test categories safe rather than fatal.
        categorical_pipeline = Pipeline([
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)),
        ])

        self._column_transformer = ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, self.numeric_cols),
                ("categorical", categorical_pipeline, self.categorical_cols),
            ],
            remainder="drop",
            sparse_threshold=0.0,  # force dense float output; see module docstring on width
        )
        self._fitted = False

    def fit(self, X_train: pd.DataFrame) -> "LogRegPreprocessor":
        self._column_transformer.fit(X_train)
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(
                "LogRegPreprocessor.transform called before fit(). Call "
                "fit(X_train) first, using only the training partition."
            )
        Z = self._column_transformer.transform(X)
        return np.asarray(Z, dtype=np.float32)

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        self.fit(X_train)
        return self.transform(X_train)

    def get_output_feature_names(self) -> list:
        return list(self._column_transformer.get_feature_names_out())
