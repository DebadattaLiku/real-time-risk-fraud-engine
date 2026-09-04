"""
Phase 2B: LightGBM-specific preprocessing.

Sits on top of the same Phase 1 `FeaturePipeline` output (391 leakage-safe,
transaction-time-only columns) that Phase 2A's `LogRegPreprocessor` uses —
same upstream contract, different downstream treatment, because LightGBM
needs fundamentally different preprocessing from a linear model:

    - numeric: NO imputation. LightGBM has native missing-value handling
      (it learns, per split, which branch a missing value should default
      to), so imputing here would throw away real information and is
      explicitly NOT done, per the Phase 2B brief.
    - categorical: NO one-hot encoding (Phase 2A's approach would be
      wasteful and unnecessary for a tree model, and loses the fact that
      LightGBM can split directly on categorical groupings). Instead, each
      categorical column is mapped to a fixed, TRAIN-ONLY vocabulary of
      integer category codes via pandas `Categorical`. Two special buckets
      are kept explicit and distinct:
        - "__missing__"  — Phase 1 already assigned this for values that
          were genuinely NaN in the raw data. It is a normal, trained-on
          category like any other (LightGBM can learn "missing here means
          X").
        - "__unseen__"   — reserved for any category value that appears in
          validation/test but was NEVER observed in training. This is
          deliberately kept as its own bucket, separate from
          "__missing__", so a genuinely-missing value and a
          never-before-seen value are not silently conflated — they mean
          different things and a model interpreting feature importance or
          a person reading this code should be able to tell them apart.

No target encoding, no encoder/vocabulary built from anything but the
training partition.
"""

from __future__ import annotations

import pandas as pd

from src.features.schema import FeatureSpec
from src.models.preprocessing import get_numeric_and_categorical_columns

UNSEEN_CATEGORY = "__unseen__"


class LightGBMPreprocessor:
    """
    Explicit fit/transform contract, mirroring `FeaturePipeline` and
    `LogRegPreprocessor`: `transform()` raises if called before `fit()`,
    and `fit()` accepts exactly one dataframe (the training partition).
    """

    def __init__(self, schema: list[FeatureSpec]):
        self.schema = schema
        self.numeric_cols, self.categorical_cols = get_numeric_and_categorical_columns(schema)
        self._categories: dict[str, list] = {}
        self._fitted = False

    def fit(self, X_train: pd.DataFrame) -> "LightGBMPreprocessor":
        for col in self.categorical_cols:
            # Train-only vocabulary. Sorted for a deterministic, reproducible
            # code assignment (same categories -> same codes on every run).
            train_categories = sorted(X_train[col].astype(str).unique().tolist())
            if UNSEEN_CATEGORY not in train_categories:
                train_categories = train_categories + [UNSEEN_CATEGORY]
            self._categories[col] = train_categories
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError(
                "LightGBMPreprocessor.transform called before fit(). Call "
                "fit(X_train) first, using only the training partition."
            )
        cols = self.numeric_cols + self.categorical_cols
        out = X[cols].copy()

        for col in self.categorical_cols:
            cats = self._categories[col]
            values = out[col].astype(str)
            # Any value not in the train-fit vocabulary (including
            # "__missing__" values only ever seen in val/test, which
            # shouldn't normally happen since Phase 1 fills missing the
            # same way everywhere, but is handled safely regardless) is
            # mapped to the explicit UNSEEN_CATEGORY bucket rather than
            # becoming a silent NaN or crashing.
            values = values.where(values.isin(cats), UNSEEN_CATEGORY)
            out[col] = pd.Categorical(values, categories=cats)

        # Numeric columns: left untouched, raw NaNs preserved for
        # LightGBM's native missing-value handling.
        return out

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        self.fit(X_train)
        return self.transform(X_train)

    def get_categorical_feature_names(self) -> list:
        return list(self.categorical_cols)
