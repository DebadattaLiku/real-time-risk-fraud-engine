"""
Phase 1: Reproducible, leakage-safe temporal train/validation/test split.

Strict chronological split by `TransactionDT`. No shuffling, no randomness
involved in row assignment (the `random_seed` in config is retained for
downstream stochastic steps in later phases — e.g. model initialization —
and is NOT used here, since there is nothing random about this split).

Guarantee enforced and verified:
    max(TransactionDT_train) <= min(TransactionDT_validation)
    max(TransactionDT_validation) <= min(TransactionDT_test)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd


@dataclass
class SplitMetadata:
    time_column: str
    id_column: str
    train_ratio: float
    val_ratio: float
    test_ratio: float

    n_rows_total: int
    n_rows_train: int
    n_rows_val: int
    n_rows_test: int

    train_row_start: int
    train_row_end: int   # exclusive, in TransactionDT-sorted order
    val_row_start: int
    val_row_end: int      # exclusive
    test_row_start: int
    test_row_end: int     # exclusive

    train_dt_min: int
    train_dt_max: int
    val_dt_min: int
    val_dt_max: int
    test_dt_min: int
    test_dt_max: int

    method: str = "time_based"  # never "random"

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: Path) -> None:
        import json
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def compute_temporal_split(
    df: pd.DataFrame,
    time_col: str,
    id_col: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitMetadata]:
    """
    Sort `df` by `time_col` (stable sort; ties broken by original row order,
    never by shuffling) and cut it into three chronological partitions.

    Returns (train_df, val_df, test_df, metadata). The three returned frames
    are disjoint, non-overlapping-by-row, and ordered
    train < validation < test in time.
    """
    if not abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must sum to 1.0, got "
            f"{train_ratio + val_ratio + test_ratio}"
        )

    df_sorted = df.sort_values(time_col, kind="mergesort").reset_index(drop=True)
    n = len(df_sorted)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df_sorted.iloc[:train_end].reset_index(drop=True)
    val_df = df_sorted.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df_sorted.iloc[val_end:].reset_index(drop=True)

    metadata = SplitMetadata(
        time_column=time_col,
        id_column=id_col,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        n_rows_total=n,
        n_rows_train=len(train_df),
        n_rows_val=len(val_df),
        n_rows_test=len(test_df),
        train_row_start=0,
        train_row_end=train_end,
        val_row_start=train_end,
        val_row_end=val_end,
        test_row_start=val_end,
        test_row_end=n,
        train_dt_min=int(train_df[time_col].min()),
        train_dt_max=int(train_df[time_col].max()),
        val_dt_min=int(val_df[time_col].min()),
        val_dt_max=int(val_df[time_col].max()),
        test_dt_min=int(test_df[time_col].min()),
        test_dt_max=int(test_df[time_col].max()),
    )

    return train_df, val_df, test_df, metadata


def verify_split(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
    time_col: str, id_col: str,
) -> dict:
    """
    Independent verification of the split guarantees. Does not trust the
    metadata object returned by `compute_temporal_split` — recomputes
    everything directly from the three dataframes so a bug in metadata
    construction can't hide a real violation.
    """
    results = {}

    ids_train = set(train_df[id_col])
    ids_val = set(val_df[id_col])
    ids_test = set(test_df[id_col])

    results["no_row_overlap_train_val"] = len(ids_train & ids_val) == 0
    results["no_row_overlap_val_test"] = len(ids_val & ids_test) == 0
    results["no_row_overlap_train_test"] = len(ids_train & ids_test) == 0

    max_train_dt = train_df[time_col].max()
    min_val_dt = val_df[time_col].min()
    max_val_dt = val_df[time_col].max()
    min_test_dt = test_df[time_col].min()

    results["max_train_dt_lte_min_val_dt"] = bool(max_train_dt <= min_val_dt)
    results["max_val_dt_lte_min_test_dt"] = bool(max_val_dt <= min_test_dt)

    n_total = len(train_df) + len(val_df) + len(test_df)
    results["train_proportion"] = len(train_df) / n_total
    results["val_proportion"] = len(val_df) / n_total
    results["test_proportion"] = len(test_df) / n_total

    results["all_ids_unique_across_splits"] = (
        len(ids_train) + len(ids_val) + len(ids_test)
        == len(ids_train | ids_val | ids_test)
    )

    return results
