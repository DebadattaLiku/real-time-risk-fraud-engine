"""
Phase 5: Deriving decision-policy thresholds FROM validation data, rather
than inventing threshold values.

Methodology (documented here, not just in the report):

1. **`approve_threshold`** is set from a chosen "total intervention budget"
   — the combined fraction of transactions that should land in REVIEW or
   BLOCK (everything else is APPROVEd automatically). This is the same
   review-capacity concept Phase 2A-4 already used for top-K evaluation
   (`review_capacity` = 1%/2%/5% etc.), reframed as a threshold: the score
   at the boundary of the top `review_capacity` fraction of VALIDATION
   scores becomes `approve_threshold`.

2. **`block_threshold`** is set by scanning increasingly large "top slice"
   fractions of VALIDATION scores (from small to the full intervention
   budget) and finding the LARGEST such slice whose measured precision
   (fraud rate within that slice, using VALIDATION labels) still meets a
   chosen minimum-precision target. This directly operationalizes the
   brief's instruction to avoid auto-blocking large numbers of legitimate
   transactions: a stricter (higher) precision target shrinks the BLOCK
   bucket; a looser one grows it, but never beyond what validation
   evidence actually supports.

Both derivations use VALIDATION data exclusively. Nothing here touches
test data or `isFraud` outside of the validation labels explicitly passed
in.
"""

from __future__ import annotations

import numpy as np

from src.evaluation.operational_eval import top_k_mask, precision_at_k
from src.decision.policy import DecisionPolicy


def score_at_top_fraction(scores, fraction: float) -> float:
    """
    The score value at the boundary of the top `fraction` of `scores`
    (the lowest score among the selected top slice) — i.e. the value a
    `< threshold` / `>= threshold` split should use to select exactly that
    top fraction (subject to the same tie-breaking convention as
    `top_k_mask`).
    """
    scores = np.asarray(scores, dtype="float64")
    mask, n_selected = top_k_mask(scores, fraction)
    return float(scores[mask].min())


def derive_block_fraction_for_precision_target(
    y_true, scores, max_fraction: float, min_precision: float, step: float = 0.0005,
) -> dict:
    """
    Scan block-bucket fractions from `step` up to `max_fraction` (in
    `step` increments) and return the LARGEST fraction whose precision
    (fraud rate within that top slice, measured on the given labels) is
    still >= `min_precision`. Precision-vs-fraction is not perfectly
    monotonic at the margins (individual transactions can be added or
    dropped from the slice in ways that locally shift precision), so the
    full grid is scanned rather than stopping at the first violation.

    If NO fraction in the grid meets `min_precision`, returns the SMALLEST
    tested fraction along with its (sub-target) achieved precision —
    surfaced explicitly rather than silently defaulting to something else,
    so the caller/report can see the target wasn't achievable at this
    scale.
    """
    candidate_fractions = np.arange(step, max_fraction + 1e-9, step)
    results = []
    for f in candidate_fractions:
        p = precision_at_k(y_true, scores, float(f))
        results.append((float(f), float(p)))

    meeting_target = [(f, p) for f, p in results if p >= min_precision]
    if meeting_target:
        best_fraction, best_precision = max(meeting_target, key=lambda x: x[0])
        target_achieved = True
    else:
        best_fraction, best_precision = results[0]
        target_achieved = False

    return {
        "block_fraction": best_fraction,
        "achieved_precision": best_precision,
        "min_precision_target": min_precision,
        "target_achieved": target_achieved,
        "all_candidates": results,
    }


def build_policy_from_validation(
    y_val, val_scores,
    review_capacity: float,
    block_precision_target: float,
    name: str,
    block_scan_step: float = 0.0005,
) -> tuple[DecisionPolicy, dict]:
    """
    Build one `DecisionPolicy` entirely from validation data:
        approve_threshold <- score at the top `review_capacity` boundary
        block_threshold   <- score at the top `block_fraction` boundary,
                              where block_fraction is derived to hit
                              `block_precision_target` (capped at
                              `review_capacity`, since BLOCK can never be
                              larger than the total intervention budget)

    Returns (policy, diagnostics) — diagnostics records exactly how the
    thresholds were derived, for the report and for reproducibility.
    """
    approve_threshold = score_at_top_fraction(val_scores, review_capacity)

    block_search = derive_block_fraction_for_precision_target(
        y_val, val_scores, max_fraction=review_capacity, min_precision=block_precision_target,
        step=block_scan_step,
    )
    block_threshold = score_at_top_fraction(val_scores, block_search["block_fraction"])

    # Guard: block_threshold must be >= approve_threshold by construction
    # (block_fraction <= review_capacity implies this), but assert it
    # explicitly rather than trusting the arithmetic silently.
    assert block_threshold >= approve_threshold, (
        f"Derived block_threshold ({block_threshold}) < approve_threshold "
        f"({approve_threshold}) — threshold derivation invariant violated."
    )

    policy = DecisionPolicy(approve_threshold=approve_threshold, block_threshold=block_threshold, name=name)

    diagnostics = {
        "name": name,
        "review_capacity_target": review_capacity,
        "block_precision_target": block_precision_target,
        "approve_threshold": approve_threshold,
        "block_threshold": block_threshold,
        "block_fraction_selected": block_search["block_fraction"],
        "block_precision_achieved": block_search["achieved_precision"],
        "block_precision_target_achieved": block_search["target_achieved"],
    }
    return policy, diagnostics
