"""
Phase 10: Reference data profile construction.

## Reference partition choice: VALIDATION, not TRAIN, not TEST

- The untouched TEST partition is never used as a reference — the brief
  explicitly forbids it, and it would also break the project's
  long-standing "test set is only for final, one-time evaluation" rule
  from every prior phase.
- Raw TRAIN is not used either: it spans a much longer, less
  deployment-representative period (Phase 0 found ~70% of the ~182-day
  dataset), and was also used to FIT preprocessing objects — using it as
  the drift reference would conflate "what the model was fit on" with
  "what recent, representative traffic looks like."
- VALIDATION is used: it's the partition Phase 5 already treated as the
  best available proxy for "approved, representative, near-deployment"
  data (policy thresholds were calibrated on it), and Phase 1-9 never fit
  anything to it beyond model/threshold *selection* (never parameter
  fitting). This makes it the most defensible "developer-approved
  historical reference" available without needing new data collection.

## What gets monitored (documented, not exhaustive)

Selected from ACTUAL repository feature names, not invented:

- Numeric (transaction-level): `TransactionAmt`, `C1`, `C13`, `C14`, `D2`,
  `V258` — the first is the obvious business-meaningful signal; the rest
  are Phase 2B's own top-10 LightGBM features by gain (see
  `reports/phase2b_lightgbm_summary.md`), so drift monitoring targets what
  the model actually relies on, not an arbitrary column list.
- Categorical: `ProductCD` (low cardinality, business-meaningful),
  `R_emaildomain` (also in Phase 2B's top-10 by gain).
- Behavioral (Phase 4, exact existing names): `bhv_prev_txn_count`
  (historical count), `bhv_hist_mean_amt` (historical amount stats),
  `bhv_time_since_prev_txn` (recency), `bhv_prior_count_24h` (velocity) —
  one representative feature per category the Phase 4 report organizes
  behavioral features into.
- Model outputs: risk score (continuous) and decision (categorical,
  APPROVE/REVIEW/BLOCK).

## Artifact contents

Each numeric/behavioral feature stores: quantile-based bin edges + counts
(for PSI), plus a bounded random sample (<=5,000 values, for KS) — NOT the
full reference column, so the artifact is self-contained and small without
requiring the raw dataset at drift-analysis time. Each categorical feature
stores its normalized category-frequency table.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

MONITORED_NUMERIC_FEATURES = ["TransactionAmt", "C1", "C13", "C14", "D2", "V258"]
MONITORED_CATEGORICAL_FEATURES = ["ProductCD", "R_emaildomain"]
MONITORED_BEHAVIORAL_FEATURES = [
    "bhv_prev_txn_count", "bhv_hist_mean_amt", "bhv_time_since_prev_txn", "bhv_prior_count_24h",
]

N_BINS = 10
MAX_REFERENCE_SAMPLE = 5000
REFERENCE_PROFILE_VERSION = "v1"

# Categorical PSI is highly sensitive to rare, long-tail categories: a
# reference category with e.g. 0.3% frequency will very often show 0
# occurrences in a modest current batch purely from sampling noise, and
# summing that "surprise" across dozens of such rare categories can
# inflate PSI dramatically without reflecting any real distributional
# change. This is a well-known categorical-PSI pitfall, not specific to
# this project — the standard fix (used here) is to group reference
# categories below a minimum frequency into a single "__OTHER__" bucket
# before computing PSI, while still tracking the FULL original category
# set separately so genuinely new/unseen categories are still surfaced
# honestly (see `compute_categorical_drift` in `src/drift/detectors.py`).
CATEGORICAL_MIN_FREQ = 0.01
OTHER_BUCKET_LABEL = "__OTHER__"


def _quantile_bin_edges(values: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """
    Reference-quantile-based bin edges (standard PSI practice: each
    reference bin starts with ~equal count). Duplicate edges (from
    skewed/near-constant features, e.g. many zeros) are deduplicated;
    if fewer than 2 distinct edges remain, falls back to a single
    min/max-bounded bin rather than raising. Outer edges are extended to
    +/-inf so any future out-of-range value still lands in a real bin
    instead of being silently dropped.
    """
    if len(values) == 0:
        return np.array([-np.inf, np.inf])
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(values, quantiles)
    edges = np.unique(edges)
    if len(edges) < 2:
        lo, hi = float(values.min()), float(values.max())
        edges = np.array([lo - 1e-9, hi + 1e-9])
    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def numeric_feature_profile(values, seed: int = 42) -> dict:
    series = pd.Series(values).astype(float)
    valid = series.dropna()
    valid_arr = valid.to_numpy()
    edges = _quantile_bin_edges(valid_arr)
    counts, _ = np.histogram(valid_arr, bins=edges)

    rng = np.random.default_rng(seed)
    sample = valid_arr
    if len(sample) > MAX_REFERENCE_SAMPLE:
        idx = rng.choice(len(sample), size=MAX_REFERENCE_SAMPLE, replace=False)
        sample = sample[idx]

    return {
        "bin_edges": edges.tolist(),
        "bin_counts": counts.tolist(),
        "mean": float(valid.mean()) if len(valid) else None,
        "median": float(valid.median()) if len(valid) else None,
        "std": float(valid.std()) if len(valid) else None,
        "min": float(valid.min()) if len(valid) else None,
        "max": float(valid.max()) if len(valid) else None,
        "missing_pct": float(series.isna().mean()) if len(series) else None,
        "n": int(len(valid)),
        "sample": sample.tolist(),
    }


def categorical_feature_profile(values, min_freq: float = CATEGORICAL_MIN_FREQ) -> dict:
    series = pd.Series(values).astype(str)
    raw_freq = series.value_counts(normalize=True).to_dict()

    grouped_freq: dict = {}
    other_total = 0.0
    for category, freq in raw_freq.items():
        if freq >= min_freq:
            grouped_freq[category] = freq
        else:
            other_total += freq
    if other_total > 0:
        grouped_freq[OTHER_BUCKET_LABEL] = other_total

    return {
        "category_frequencies": grouped_freq,
        "all_reference_categories": sorted(raw_freq.keys()),  # full set, for honest new-category surfacing
        "n": int(len(series)),
        "n_categories": len(raw_freq),
        "n_categories_grouped": len(grouped_freq),
        "min_freq_threshold": min_freq,
    }


def build_reference_profile(
    reference_df: pd.DataFrame,
    risk_scores,
    decisions,
    reference_partition_name: str = "validation",
) -> dict:
    """
    `reference_df` must contain the monitored numeric/categorical/behavioral
    columns (raw transaction columns + already-computed `bhv_*` columns —
    NOT recomputed here; see `src/drift/monitor.py`'s module docstring for
    why). `risk_scores`/`decisions` are the model's own outputs on this
    same reference set (e.g. from the approved Phase 5 validation
    evaluation), never recomputed by a second model call here.
    """
    profile = {
        "metadata": {
            "reference_partition": reference_partition_name,
            "n_reference_transactions": int(len(reference_df)),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "reference_version": REFERENCE_PROFILE_VERSION,
            "monitored_numeric_features": MONITORED_NUMERIC_FEATURES,
            "monitored_categorical_features": MONITORED_CATEGORICAL_FEATURES,
            "monitored_behavioral_features": MONITORED_BEHAVIORAL_FEATURES,
        },
        "numeric": {}, "categorical": {}, "behavioral": {},
    }

    for f in MONITORED_NUMERIC_FEATURES:
        if f in reference_df.columns:
            profile["numeric"][f] = numeric_feature_profile(reference_df[f])
    for f in MONITORED_CATEGORICAL_FEATURES:
        if f in reference_df.columns:
            profile["categorical"][f] = categorical_feature_profile(reference_df[f])
    for f in MONITORED_BEHAVIORAL_FEATURES:
        if f in reference_df.columns:
            profile["behavioral"][f] = numeric_feature_profile(reference_df[f])

    profile["risk_score"] = numeric_feature_profile(pd.Series(risk_scores))
    decision_counts = pd.Series(decisions).astype(str).value_counts(normalize=True)
    profile["decision"] = {
        d: float(decision_counts.get(d, 0.0)) for d in ("APPROVE", "REVIEW", "BLOCK")
    }

    return profile


def save_reference_profile(profile: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(profile, f, indent=2)


def load_reference_profile(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)
