"""
Phase 1: Clean, validated data loading for the IEEE-CIS transaction table.

This module is the single entry point for loading `train_transaction.csv`
into the pipeline. It intentionally does NOT merge `train_identity.csv` —
Phase 0's approved architecture scoped Phase 0 EDA to the transaction table
alone (all fields analyzed there — card/addr/D/C/V — live in
train_transaction.csv), and Phase 1's task description says not to merge
identity data unless already required. It isn't, so we don't.

Responsibilities:
    - Confirm the file exists and give a clear, actionable error if not.
    - Load with a memory-efficient dtype map (reusing the same strategy
      proven in Phase 0's `inspect_data.py`, re-derived here independently
      since this module has a different contract: it must raise on
      validation failure rather than print-and-continue).
    - Validate: required columns present, `TransactionID` uniqueness,
      target column present and binary, `TransactionDT` present/non-null
      and numeric.

This module does NOT split, engineer features, or fit any transformation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"

# Columns that MUST be present for the pipeline to function at all.
# (The full 394-column schema is validated implicitly by downstream feature
# grouping in Phase 1's feature module; this is the hard minimum contract.)
REQUIRED_COLUMNS = ["TransactionID", "isFraud", "TransactionDT"]


class DataValidationError(Exception):
    """Raised when the loaded dataset fails a required integrity check."""


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@dataclass
class LoadReport:
    path: Path
    n_rows: int
    n_cols: int
    required_columns_present: bool
    missing_required_columns: list
    transaction_id_unique: bool
    n_duplicate_transaction_ids: int
    target_is_binary: bool
    target_unique_values: list
    transaction_dt_valid: bool
    transaction_dt_null_count: int


def _build_efficient_dtypes(path: Path, nrows: int = 5000) -> dict:
    """
    Same rationale as Phase 0's `inspect_data.build_efficient_dtypes`:
    object -> category, int64 -> int32, float64 -> float32. Re-derived here
    (not imported) because `src/data/load.py` is meant to be usable
    independently of the Phase 0 EDA script, which is frozen/approved and
    not a dependency other modules should rely on.
    """
    sample = pd.read_csv(path, nrows=nrows)
    dtypes = {}
    for col in sample.columns:
        dt = sample[col].dtype
        if dt == object:
            dtypes[col] = "category"
        elif dt == "int64":
            dtypes[col] = "int32"
        elif dt == "float64":
            dtypes[col] = "float32"
        else:
            dtypes[col] = dt
    return dtypes


def load_train_transaction(
    path: Path | str | None = None,
    config: dict | None = None,
    validate: bool = True,
) -> tuple[pd.DataFrame, LoadReport]:
    """
    Load `train_transaction.csv` with a memory-efficient dtype map and
    validate it. Raises `FileNotFoundError` if the file is missing and
    `DataValidationError` if a required integrity check fails (when
    `validate=True`).

    Returns (dataframe, LoadReport).
    """
    if path is None:
        if config is None:
            config = load_config()
        path = REPO_ROOT / config["paths"]["train_transaction"]
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Expected the IEEE-CIS `train_transaction.csv` at this path. "
            "See README.md for how to obtain and place the dataset."
        )

    dtypes = _build_efficient_dtypes(path)
    df = pd.read_csv(path, dtype=dtypes)

    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    required_ok = len(missing_required) == 0

    if not required_ok:
        report = LoadReport(
            path=path, n_rows=len(df), n_cols=df.shape[1],
            required_columns_present=False,
            missing_required_columns=missing_required,
            transaction_id_unique=False, n_duplicate_transaction_ids=-1,
            target_is_binary=False, target_unique_values=[],
            transaction_dt_valid=False, transaction_dt_null_count=-1,
        )
        if validate:
            raise DataValidationError(
                f"Missing required column(s) in {path}: {missing_required}"
            )
        return df, report

    n_dupe_ids = int(df["TransactionID"].duplicated().sum())
    id_unique = n_dupe_ids == 0

    target_values = sorted(df["isFraud"].dropna().unique().tolist())
    target_is_binary = set(target_values).issubset({0, 1})

    dt_null_count = int(df["TransactionDT"].isna().sum())
    dt_is_numeric = pd.api.types.is_numeric_dtype(df["TransactionDT"])
    dt_valid = dt_is_numeric and dt_null_count == 0

    report = LoadReport(
        path=path, n_rows=len(df), n_cols=df.shape[1],
        required_columns_present=required_ok,
        missing_required_columns=missing_required,
        transaction_id_unique=id_unique,
        n_duplicate_transaction_ids=n_dupe_ids,
        target_is_binary=target_is_binary,
        target_unique_values=target_values,
        transaction_dt_valid=dt_valid,
        transaction_dt_null_count=dt_null_count,
    )

    if validate:
        problems = []
        if not id_unique:
            problems.append(
                f"TransactionID is not unique: {n_dupe_ids} duplicate(s)."
            )
        if not target_is_binary:
            problems.append(
                f"isFraud is not strictly binary: unique values = {target_values}"
            )
        if not dt_valid:
            problems.append(
                f"TransactionDT invalid: numeric={dt_is_numeric}, "
                f"null_count={dt_null_count}"
            )
        if problems:
            raise DataValidationError(
                f"Validation failed for {path}:\n" + "\n".join(f"  - {p}" for p in problems)
            )

    return df, report


if __name__ == "__main__":
    df, report = load_train_transaction()
    print(report)
    print(f"Loaded shape: {df.shape}")
