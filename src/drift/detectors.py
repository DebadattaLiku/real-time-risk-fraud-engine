"""
Phase 10: Statistical drift detectors.

Two focused techniques, not an exhaustive battery:

- **Population Stability Index (PSI)** — the primary measure for both
  numeric (binned) and categorical features, chosen because it is the
  standard, widely-understood measure for this exact use case (comparing
  a reference distribution to a current one) and produces a single
  interpretable number with conventional severity bands.
- **Kolmogorov-Smirnov (KS) statistic** — a complementary numeric-only
  check, added because it answers a genuinely different question than PSI
  (KS is sensitive to any distributional difference, including shape
  changes that might not show up strongly in a coarse 10-bin PSI), using
  `scipy.stats.ks_2samp` against the reference's bounded sample.

## Severity thresholds are PROJECT-CONFIGURABLE, not universal truth

`PSI_SEVERITY_THRESHOLDS` below reflects a commonly-cited industry
convention for PSI interpretation (roughly: <0.1 stable, 0.1-0.25
moderate, >0.25 major) adapted into four bands for this project. These are
explicitly configurable constants, not a claim that e.g. PSI=0.11 is
"objectively" drift for every possible feature or business context.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.drift.reference import OTHER_BUCKET_LABEL

PSI_EPS = 1e-4  # smoothing floor for zero-frequency bins/categories

# Ordered weakest -> strongest; used both for single-metric interpretation
# and for aggregating many results into one overall batch severity
# (src/drift/report.py).
SEVERITY_ORDER = ["NO_SIGNIFICANT_DRIFT", "LOW_DRIFT", "MODERATE_DRIFT", "HIGH_DRIFT"]
PSI_SEVERITY_THRESHOLDS = {"low": 0.10, "moderate": 0.20, "high": 0.30}


def severity_from_psi(psi: float) -> str:
    if psi < PSI_SEVERITY_THRESHOLDS["low"]:
        return "NO_SIGNIFICANT_DRIFT"
    if psi < PSI_SEVERITY_THRESHOLDS["moderate"]:
        return "LOW_DRIFT"
    if psi < PSI_SEVERITY_THRESHOLDS["high"]:
        return "MODERATE_DRIFT"
    return "HIGH_DRIFT"


def _psi_from_percentages(ref_pct: np.ndarray, cur_pct: np.ndarray) -> float:
    """
    Core PSI formula: sum((cur% - ref%) * ln(cur% / ref%)) over aligned
    bins/categories. Both inputs are clipped away from exactly zero
    (`PSI_EPS`) BEFORE the log — a zero-frequency reference or current bin
    would otherwise produce a division-by-zero or ln(0). Clipping (not
    dropping the bin, not renormalizing afterward) is the standard,
    numerically stable way to handle this: the epsilon floor is tiny
    (1e-4) relative to any bin that has real data in it, so its effect on
    genuinely populated bins is negligible.
    """
    ref = np.clip(np.asarray(ref_pct, dtype=float), PSI_EPS, None)
    cur = np.clip(np.asarray(cur_pct, dtype=float), PSI_EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def bin_current_numeric(current_values, bin_edges: list) -> np.ndarray:
    values = pd.Series(current_values).dropna().astype(float).to_numpy()
    if len(values) == 0:
        return np.zeros(len(bin_edges) - 1, dtype=int)
    counts, _ = np.histogram(values, bins=np.asarray(bin_edges, dtype=float))
    return counts


def compute_numeric_drift(reference_profile: dict, current_values) -> dict:
    """
    `reference_profile` is one feature's entry from a reference profile's
    `numeric`/`behavioral`/`risk_score` section (has `bin_edges`,
    `bin_counts`, `sample`, `mean`, etc. — see `src/drift/reference.py`).
    """
    ref_counts = np.asarray(reference_profile["bin_counts"], dtype=float)
    cur_counts = bin_current_numeric(current_values, reference_profile["bin_edges"])

    ref_total = ref_counts.sum()
    cur_total = cur_counts.sum()
    ref_pct = ref_counts / ref_total if ref_total > 0 else np.zeros_like(ref_counts)
    cur_pct = cur_counts / cur_total if cur_total > 0 else np.zeros_like(cur_counts)
    psi = _psi_from_percentages(ref_pct, cur_pct)

    cur_valid = pd.Series(current_values).dropna().astype(float).to_numpy()
    ref_sample = np.asarray(reference_profile.get("sample", []), dtype=float)
    if len(ref_sample) > 1 and len(cur_valid) > 1:
        ks_result = ks_2samp(ref_sample, cur_valid)
        ks_statistic, ks_p_value = float(ks_result.statistic), float(ks_result.pvalue)
    else:
        ks_statistic, ks_p_value = None, None

    return {
        "psi": psi,
        "ks_statistic": ks_statistic,
        "ks_p_value": ks_p_value,
        "reference_mean": reference_profile.get("mean"),
        "current_mean": float(np.mean(cur_valid)) if len(cur_valid) else None,
        "reference_median": reference_profile.get("median"),
        "current_median": float(np.median(cur_valid)) if len(cur_valid) else None,
        "n_current": int(cur_total),
        "severity": severity_from_psi(psi),
    }


def compute_categorical_drift(reference_profile: dict, current_values) -> dict:
    """
    `reference_profile` is one feature's entry from a reference profile's
    `categorical` section (has `category_frequencies` — already grouped
    into an `__OTHER__` bucket for rare categories, see
    `src/drift/reference.py` — and `all_reference_categories`, the FULL
    original set, used only for honest new/missing-category surfacing
    below, never for the PSI computation itself).

    The current batch is grouped the SAME way (any category not one of
    the reference's individually-kept categories collapses into
    `__OTHER__`) before computing PSI — comparing like with like, and
    avoiding the well-known instability of naive categorical PSI on
    long-tailed distributions (see `src/drift/reference.py`'s module
    docstring for the concrete example that motivated this).
    """
    ref_freq: dict = reference_profile["category_frequencies"]
    known_categories = set(reference_profile.get("all_reference_categories", ref_freq.keys()))
    kept_categories = set(ref_freq.keys()) - {OTHER_BUCKET_LABEL}

    current_series = pd.Series(current_values).astype(str)
    raw_cur_freq = current_series.value_counts(normalize=True).to_dict()

    grouped_cur_freq: dict = {}
    other_total = 0.0
    for category, freq in raw_cur_freq.items():
        if category in kept_categories:
            grouped_cur_freq[category] = freq
        else:
            other_total += freq
    if other_total > 0:
        grouped_cur_freq[OTHER_BUCKET_LABEL] = other_total

    all_categories = sorted(set(ref_freq) | set(grouped_cur_freq))
    ref_pct = np.array([ref_freq.get(c, 0.0) for c in all_categories])
    cur_pct = np.array([grouped_cur_freq.get(c, 0.0) for c in all_categories])
    psi = _psi_from_percentages(ref_pct, cur_pct)

    # New/missing categories are surfaced using the FULL original
    # reference category set (not the grouped one) — grouping is purely a
    # PSI-stability technique, it must never hide a genuinely new category
    # from being reported.
    new_categories = sorted(set(raw_cur_freq) - known_categories)
    missing_categories = sorted(known_categories - set(raw_cur_freq))

    return {
        "psi": psi,
        "new_categories": new_categories,        # surfaced explicitly, never silently dropped
        "missing_categories": missing_categories,  # present in reference, absent from this batch
        "n_current": int(len(current_series)),
        "current_frequencies": {k: float(v) for k, v in raw_cur_freq.items()},
        "severity": severity_from_psi(psi),
    }


def compute_decision_drift(reference_decision_profile: dict, current_decisions) -> dict:
    """Same PSI mechanics as `compute_categorical_drift`, specialized for
    the fixed APPROVE/REVIEW/BLOCK category set (no target labels used —
    decisions are the policy's own output, already computed)."""
    result = compute_categorical_drift(
        {"category_frequencies": reference_decision_profile}, current_decisions,
    )
    result["reference_frequencies"] = {k: float(v) for k, v in reference_decision_profile.items()}
    return result
