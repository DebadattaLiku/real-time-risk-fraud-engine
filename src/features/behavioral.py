"""
Phase 4: Leakage-safe historical behavioral features, keyed on the `card1`
pseudo-entity (Phase 0's recommended candidate — NOT a verified customer
ID; see the Phase 4 report for the caveats that still apply).

## Critical rule

For every transaction, a behavioral feature may use ONLY transactions that
occurred STRICTLY BEFORE it in the deterministic ordering below. The
current transaction never contributes to its own historical aggregates,
and no future transaction — from any split — ever contributes to a past
one. `isFraud` is never read anywhere in this module (asserted, not just
documented — see `compute_behavioral_features`).

## Deterministic ordering

Sort by `(TransactionDT, TransactionID)`, ascending, stable. `TransactionDT`
alone has ties (Phase 0 found transactions sharing an identical value);
`TransactionID` is added as a documented, reproducible tiebreaker so the
"strictly before" relation is a total order with no ambiguity.

## How train/validation/test boundaries are respected

This module does not know or care about split boundaries — it is called
once on the concatenation of train+validation+test (already, by
construction, in chronological order — see Phase 1's `compute_temporal_split`),
sorted again explicitly here per the ordering policy above. This
automatically gives every validation-partition row access to all prior
TRAIN-partition history, and every test-partition row access to all prior
TRAIN+VALIDATION history, exactly as the brief's diagram specifies, without
this module needing any split-aware branching. The caller
(`src/run_phase4_behavioral.py`) is responsible for ensuring `isFraud` is
never part of the input passed here — enforced by an assertion, not trust.

## Implementation approach

Naive per-row Python loops over ~590K rows / up to ~13.5K entities would be
slow. Instead, every feature is computed via vectorized pandas
group-cumulative operations (`cumcount`, `cumsum`, `shift`, `cummin`/
`cummax`, and a groupby+time-rolling count for the velocity windows) — all
O(n) or O(n log n), operating on the entity-sorted frame in one pass.

## Cold-start / insufficient-history policy (see also the Phase 4 report)

- `bhv_prev_txn_count = 0` for an entity's first-ever transaction — a real,
  meaningful value, not a placeholder for missing data.
- Amount statistics (`bhv_hist_mean_amt`, `_min_amt`, `_max_amt`) are `NaN`
  when `bhv_prev_txn_count == 0` (no history exists to summarize).
- `bhv_hist_std_amt` is additionally `NaN` when `bhv_prev_txn_count < 2`
  (a single data point has no defined sample variance) — this is a
  strictly stronger condition than "no history," so entities with exactly
  one prior transaction get valid mean/min/max but a `NaN` std, which is
  the mathematically correct distinction, not an inconsistency.
- `bhv_time_since_prev_txn` is `NaN` for a first transaction (no previous
  timestamp to subtract).
- Ratio/diff/z-score features (`bhv_amt_to_hist_mean_ratio`,
  `bhv_amt_diff_from_hist_mean`, `bhv_amt_zscore`) are `NaN` whenever their
  denominator is zero, undefined, or itself `NaN` — never silently
  defaulted to 0 or 1, which would fabricate a "no deviation from history"
  signal that isn't actually known.
- No imputation is performed on any of the above — consistent with the
  already-approved Phase 2B LightGBM convention of relying on LightGBM's
  native missing-value handling rather than inventing values (see
  `src/models/lightgbm_preprocessing.py`). This is a deliberate
  "preserve baseline compatibility" choice, not an oversight.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BEHAVIORAL_FEATURE_PREFIX = "bhv_"

# A small, deliberately short list of velocity windows, per the brief's
# "a small number of useful windows is sufficient." 1 hour captures rapid
# burst-like activity; 24 hours captures same-day velocity. Both are
# common, interpretable choices in fraud-velocity literature.
VELOCITY_WINDOWS_SECONDS = {
    "1h": 3600,
    "24h": 86400,
}


def compute_behavioral_features(
    df: pd.DataFrame,
    entity_col: str = "card1",
    time_col: str = "TransactionDT",
    id_col: str = "TransactionID",
    amount_col: str = "TransactionAmt",
) -> pd.DataFrame:
    """
    `df` must contain at least [id_col, time_col, entity_col, amount_col]
    and must NOT contain `isFraud` (asserted below). Typically called on
    the concatenation of train+validation+test transaction-level frames.

    Returns a DataFrame with `id_col` plus every `bhv_*` column, ONE ROW
    PER INPUT ROW, safe to left-merge back onto any of the original
    train/validation/test frames via `id_col` (deliberately merge-based
    rather than positional, so there is no fragile assumption about row
    order surviving the round trip).
    """
    assert "isFraud" not in df.columns, (
        "isFraud must never be passed into compute_behavioral_features — "
        "behavioral features must be constructible without the target."
    )
    required = {id_col, time_col, entity_col, amount_col}
    missing = required - set(df.columns)
    assert not missing, f"compute_behavioral_features missing required columns: {missing}"

    d = df[[id_col, time_col, entity_col, amount_col]].copy()
    # Deterministic ordering policy: (TransactionDT, TransactionID).
    d = d.sort_values([time_col, id_col], kind="mergesort").reset_index(drop=True)

    amt = d[amount_col].astype("float64")
    grp_key = d[entity_col]

    # --- Historical transaction count (strictly prior) ---
    prev_count = d.groupby(entity_col, sort=False).cumcount()  # already excludes current row
    prev_count_log1p = np.log1p(prev_count)

    # --- Historical amount mean/std (vectorized cumulative formulas) ---
    # NOTE: pandas' cumsum() treats NaN as contributing 0 to the running
    # total (it does not propagate NaN forward), so the denominator for
    # mean/std must be a count of VALID (non-NaN) prior amounts specifically
    # — not the raw prior-transaction count — or a missing historical
    # amount would silently be treated as if it were 0, producing a
    # fabricated hist_mean of 0.0 instead of the correct "unknown" (NaN).
    # Caught by test_missing_amount_does_not_crash_and_propagates_as_nan.
    amt_is_valid = amt.notna().astype("float64")
    d["_amt_valid_flag"] = amt_is_valid
    cum_valid_count_incl = d.groupby(entity_col, sort=False)["_amt_valid_flag"].cumsum()
    cum_valid_count_prior = cum_valid_count_incl - amt_is_valid

    cum_sum_incl = d.groupby(entity_col, sort=False)[amount_col].cumsum()
    cum_sum_prior = cum_sum_incl - amt.fillna(0.0)

    amt_sq = amt ** 2
    d["_amt_sq"] = amt_sq
    cum_sumsq_incl = d.groupby(entity_col, sort=False)["_amt_sq"].cumsum()
    cum_sumsq_prior = cum_sumsq_incl - amt_sq.fillna(0.0)

    valid_count_safe = cum_valid_count_prior.replace(0, np.nan)
    hist_mean = cum_sum_prior / valid_count_safe  # NaN when 0 VALID prior amounts

    valid_count_minus1 = (cum_valid_count_prior - 1)
    valid_count_minus1_safe = valid_count_minus1.where(valid_count_minus1 > 0, np.nan)  # NaN when <2 valid prior amounts
    var_numerator = cum_sumsq_prior - cum_valid_count_prior * (hist_mean ** 2)
    hist_var = var_numerator / valid_count_minus1_safe
    hist_var = hist_var.clip(lower=0)  # guard tiny negative values from floating-point error
    hist_std = np.sqrt(hist_var)

    # --- Historical min/max (strictly prior, via shift + cummin/cummax) ---
    d["_prev_amt"] = d.groupby(entity_col, sort=False)[amount_col].shift(1)
    hist_min = d.groupby(entity_col, sort=False)["_prev_amt"].cummin()
    hist_max = d.groupby(entity_col, sort=False)["_prev_amt"].cummax()

    # --- Time since previous transaction (strictly prior) ---
    prev_time = d.groupby(entity_col, sort=False)[time_col].shift(1)
    time_since_prev = d[time_col] - prev_time  # NaN for first transaction in entity

    # --- Current amount relative to history ---
    hist_mean_safe = hist_mean.replace(0, np.nan)
    amt_to_hist_mean_ratio = amt / hist_mean_safe
    amt_diff_from_hist_mean = amt - hist_mean
    hist_std_safe = hist_std.replace(0, np.nan)
    amt_zscore = (amt - hist_mean) / hist_std_safe

    out = pd.DataFrame({
        id_col: d[id_col].values,
        f"{BEHAVIORAL_FEATURE_PREFIX}prev_txn_count": prev_count.astype("float64").values,
        f"{BEHAVIORAL_FEATURE_PREFIX}prev_txn_count_log1p": prev_count_log1p.values,
        f"{BEHAVIORAL_FEATURE_PREFIX}hist_mean_amt": hist_mean.values,
        f"{BEHAVIORAL_FEATURE_PREFIX}hist_std_amt": hist_std.values,
        f"{BEHAVIORAL_FEATURE_PREFIX}hist_min_amt": hist_min.values,
        f"{BEHAVIORAL_FEATURE_PREFIX}hist_max_amt": hist_max.values,
        f"{BEHAVIORAL_FEATURE_PREFIX}time_since_prev_txn": time_since_prev.values,
        f"{BEHAVIORAL_FEATURE_PREFIX}amt_to_hist_mean_ratio": amt_to_hist_mean_ratio.values,
        f"{BEHAVIORAL_FEATURE_PREFIX}amt_diff_from_hist_mean": amt_diff_from_hist_mean.values,
        f"{BEHAVIORAL_FEATURE_PREFIX}amt_zscore": amt_zscore.values,
    })

    # --- Velocity windows: prior-transaction counts in recent windows ---
    velocity_cols = _compute_velocity_counts(d, entity_col, time_col, id_col)
    out = out.merge(velocity_cols, on=id_col, how="left", validate="one_to_one")

    return out


def _compute_velocity_counts(
    d: pd.DataFrame, entity_col: str, time_col: str, id_col: str,
) -> pd.DataFrame:
    """
    Prior-transaction counts within each configured window, per entity,
    strictly excluding the current transaction.

    Implementation: for each entity group (whose rows are already in
    ascending time order, since `d` is globally sorted by
    `(TransactionDT, TransactionID)` before this is called), use
    `numpy.searchsorted` on that group's timestamp array to find, for every
    row `i`, the earliest prior row whose timestamp falls within the
    window — the count of prior transactions in-window is then simply
    `i - that index`. This is a direct, easily-verified O(n log n)
    computation.

    (An earlier implementation used `groupby(...).rolling(window, on=...)`,
    which turned out to have fragile/surprising row-alignment behavior on
    the installed pandas version — verified by a real test failure
    `test_velocity_counts_non_negative_and_le_prev_count` catching
    silently-wrong, not crashing, results. Replaced with this explicit,
    transparently-correct approach instead of trying to coerce the rolling
    API into the right alignment.)
    """
    d = d.reset_index(drop=True)
    times = d[time_col].to_numpy(dtype="float64")
    n = len(d)

    counts = {label: np.zeros(n, dtype="int64") for label in VELOCITY_WINDOWS_SECONDS}

    groups = d.groupby(entity_col, sort=False).indices  # {entity_value: array of positions in d}
    for _, positions in groups.items():
        positions = np.sort(positions)  # defensive; already ascending given d's global sort
        t = times[positions]
        for label, seconds in VELOCITY_WINDOWS_SECONDS.items():
            lower_idx = np.searchsorted(t, t - seconds, side="left")
            prior_in_window = np.arange(len(t)) - lower_idx
            counts[label][positions] = prior_in_window

    result = pd.DataFrame({id_col: d[id_col].values})
    for label in VELOCITY_WINDOWS_SECONDS:
        result[f"{BEHAVIORAL_FEATURE_PREFIX}prior_count_{label}"] = counts[label]

    return result


def get_behavioral_feature_names(features_df: pd.DataFrame, id_col: str = "TransactionID") -> list:
    return [c for c in features_df.columns if c != id_col and c.startswith(BEHAVIORAL_FEATURE_PREFIX)]


def behavioral_diagnostics(features_df: pd.DataFrame, entity_col_values: pd.Series | None = None) -> dict:
    """
    Summary diagnostics for a behavioral feature dataframe (the direct
    output of `compute_behavioral_features`), used for the Phase 4 report's
    diagnostics section. Does not require labels.
    """
    n = len(features_df)
    prev_count = features_df["bhv_prev_txn_count"]
    n_cold_start = int((prev_count == 0).sum())

    diagnostics = {
        "n_transactions": n,
        "pct_cold_start_no_prior_history": float(n_cold_start / n) if n > 0 else float("nan"),
        "prev_txn_count_distribution": {
            "min": float(prev_count.min()), "p25": float(prev_count.quantile(0.25)),
            "median": float(prev_count.median()), "mean": float(prev_count.mean()),
            "p75": float(prev_count.quantile(0.75)), "p95": float(prev_count.quantile(0.95)),
            "max": float(prev_count.max()),
        },
        "time_since_prev_txn_distribution_seconds": {
            "pct_missing_first_txn": float(features_df["bhv_time_since_prev_txn"].isna().mean()),
            "median_when_present": float(features_df["bhv_time_since_prev_txn"].median()),
            "p95_when_present": float(features_df["bhv_time_since_prev_txn"].quantile(0.95)),
        },
        "missingness_by_feature": {
            col: float(features_df[col].isna().mean())
            for col in get_behavioral_feature_names(features_df)
        },
        "extreme_zscore_pct_abs_gt_5": float((features_df["bhv_amt_zscore"].abs() > 5).mean()),
    }

    if entity_col_values is not None:
        diagnostics["n_unique_entities"] = int(entity_col_values.nunique())
        diagnostics["n_total_transactions"] = int(len(entity_col_values))
        diagnostics["mean_txns_per_entity"] = float(len(entity_col_values) / entity_col_values.nunique())

    return diagnostics
