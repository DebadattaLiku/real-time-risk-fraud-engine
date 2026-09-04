"""
Phase 1: Feature schema — a reproducible record of every column's role,
type, group, and inclusion decision.

This module builds metadata ONLY. It does not transform data (see
`pipeline.py` for that) and it does not learn any parameter from data other
than descriptive missingness percentages, which are computed from the
TRAINING split alone to avoid any peeking at validation/test distributions
when making feature-inclusion decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import re

import pandas as pd

# Columns explicitly excluded from the model input matrix in Phase 1, and why.
# This is a documented, deliberate list — not an accidental drop.
NON_FEATURE_COLUMNS = {
    "TransactionID": "identifier — not predictive, used only as a row key",
    "isFraud": "target variable — must never appear in X",
    "TransactionDT": (
        "raw absolute time delta deliberately excluded as a raw predictive "
        "feature in Phase 1. It IS used for temporal ordering and the "
        "train/validation/test split. Reason for exclusion from X: (1) its "
        "scale is strictly non-overlapping across train/val/test by "
        "construction (train < val < test), so a model could learn a "
        "spurious direct time->fraud mapping that does not represent a "
        "generalizable pattern and would not transfer to new data outside "
        "the observed calendar range; (2) any genuine temporal signal "
        "worth capturing (recency, velocity, time-since-last-event) belongs "
        "in derived behavioral features, planned for a later phase, not as "
        "the raw delta itself. This is a modeling-safety decision, not a "
        "leakage removal — TransactionDT is technically available at "
        "scoring time and is not future information."
    ),
}


def infer_feature_group(col_name: str) -> str:
    """Assign each column to a documented feature group by name pattern."""
    if col_name in ("TransactionID",):
        return "id"
    if col_name in ("isFraud",):
        return "target"
    if col_name in ("TransactionDT",):
        return "time"
    if col_name == "TransactionAmt":
        return "amount"
    if col_name == "ProductCD":
        return "product"
    if col_name.startswith("card") and col_name[4:].isdigit():
        return "card"
    if col_name.startswith("addr") and col_name[4:].isdigit():
        return "addr"
    if col_name in ("dist1", "dist2"):
        return "distance"
    if "emaildomain" in col_name:
        return "email"
    if re.fullmatch(r"C\d+", col_name):
        return "C"
    if re.fullmatch(r"D\d+", col_name):
        return "D"
    if re.fullmatch(r"M\d+", col_name):
        return "M"
    if re.fullmatch(r"V\d+", col_name):
        return "V"
    if re.fullmatch(r"id_\d+", col_name):
        return "identity"  # present only if identity data is ever merged
    if col_name == "DeviceType" or col_name == "DeviceInfo":
        return "device"
    return "other"


@dataclass
class FeatureSpec:
    name: str
    pandas_dtype: str
    value_type: str          # "numeric" | "categorical"
    feature_group: str
    missing_pct_train: float
    used: bool
    exclusion_reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def build_feature_schema(train_df: pd.DataFrame) -> list[FeatureSpec]:
    """
    Build the feature schema using ONLY the training partition. Missingness
    percentages, dtype classification, and group assignment are all
    descriptive of `train_df` — validation/test are never inspected here,
    so no inclusion/exclusion decision can be informed by them.
    """
    specs = []
    for col in train_df.columns:
        dt = train_df[col].dtype
        # Robust to pandas version differences: pandas 3.x introduced a
        # default 'str' dtype for string columns that is NOT `object` and
        # would be silently misclassified as numeric by an `== object`
        # check, which would then skip the missing-category handling in
        # the pipeline for string columns entirely. Using
        # `is_numeric_dtype` as the authoritative test avoids that.
        is_numeric = pd.api.types.is_numeric_dtype(dt)
        value_type = "numeric" if is_numeric else "categorical"
        missing_pct = float(train_df[col].isna().mean())
        group = infer_feature_group(col)

        if col in NON_FEATURE_COLUMNS:
            specs.append(FeatureSpec(
                name=col,
                pandas_dtype=str(dt),
                value_type=value_type,
                feature_group=group,
                missing_pct_train=missing_pct,
                used=False,
                exclusion_reason=NON_FEATURE_COLUMNS[col],
            ))
        else:
            specs.append(FeatureSpec(
                name=col,
                pandas_dtype=str(dt),
                value_type=value_type,
                feature_group=group,
                missing_pct_train=missing_pct,
                used=True,
                exclusion_reason=None,
            ))
    return specs


def schema_to_dataframe(schema: list[FeatureSpec]) -> pd.DataFrame:
    return pd.DataFrame([s.to_dict() for s in schema])


def schema_summary(schema: list[FeatureSpec]) -> dict:
    used = [s for s in schema if s.used]
    excluded = [s for s in schema if not s.used]
    numeric_used = [s for s in used if s.value_type == "numeric"]
    categorical_used = [s for s in used if s.value_type == "categorical"]
    by_group = {}
    for s in used:
        by_group.setdefault(s.feature_group, 0)
        by_group[s.feature_group] += 1
    return {
        "n_total_columns": len(schema),
        "n_used": len(used),
        "n_excluded": len(excluded),
        "excluded_columns": [(s.name, s.exclusion_reason) for s in excluded],
        "n_numeric_used": len(numeric_used),
        "n_categorical_used": len(categorical_used),
        "used_by_group": by_group,
    }
