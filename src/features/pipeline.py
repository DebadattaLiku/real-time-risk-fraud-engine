"""
Phase 1: Feature preparation pipeline.

Deliberately minimal, matching the Phase 1 scope: this pipeline selects the
"used" columns from the feature schema, preserves numeric values and their
missingness as-is (no imputation, no scaling — those are learned
transformations reserved for a later phase if/when they're introduced), and
gives categorical columns an explicit missing category instead of leaving
raw NaN.

No transformation here is "fit" in the statistical sense (no mean, mode,
frequency, or target statistic is learned from any partition), so the
FIT -> transform(val) -> transform(test) contract is trivially satisfied.
The `fit`/`transform` split is still implemented explicitly (rather than a
single stateless function) so that later phases which DO need learned
transformations (scalers, encoders) can extend this class without changing
its calling contract, and so the "fit uses train only" rule is structurally
enforced rather than just documented.
"""

from __future__ import annotations

import pandas as pd

from src.features.schema import FeatureSpec

MISSING_CATEGORY = "__missing__"


class FeaturePipeline:
    def __init__(self, schema: list[FeatureSpec]):
        self.schema = schema
        self._feature_names = [s.name for s in schema if s.used]
        self._categorical_names = [
            s.name for s in schema if s.used and s.value_type == "categorical"
        ]
        self._numeric_names = [
            s.name for s in schema if s.used and s.value_type == "numeric"
        ]
        self._fitted = False

    def fit(self, train_df: pd.DataFrame) -> "FeaturePipeline":
        """
        Confirms the pipeline's expected columns are present in the training
        data. Learns nothing statistical from `train_df` in Phase 1 (no
        transformation here requires a learned parameter) — this method
        exists to establish the fit/transform contract for later phases and
        to fail fast if the schema and the data have diverged.
        """
        missing = [c for c in self._feature_names if c not in train_df.columns]
        if missing:
            raise ValueError(
                f"FeaturePipeline.fit: columns in schema but not in "
                f"train_df: {missing}"
            )
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError(
                "FeaturePipeline.transform called before fit(). Call "
                "fit(train_df) first, using only the training partition."
            )
        missing = [c for c in self._feature_names if c not in df.columns]
        if missing:
            raise ValueError(f"transform: columns missing from input df: {missing}")

        X = df[self._feature_names].copy()

        for col in self._categorical_names:
            if not isinstance(X[col].dtype, pd.CategoricalDtype):
                X[col] = X[col].astype("category")
            if MISSING_CATEGORY not in X[col].cat.categories:
                X[col] = X[col].cat.add_categories([MISSING_CATEGORY])
            X[col] = X[col].fillna(MISSING_CATEGORY)

        # Numeric columns: preserve raw values and NaNs as-is. No imputation,
        # no scaling in Phase 1.
        return X

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        self.fit(train_df)
        return self.transform(train_df)

    def get_feature_names(self) -> list:
        return list(self._feature_names)

    def get_target(self, df: pd.DataFrame, target_col: str) -> pd.Series:
        return df[target_col].astype("int32")
