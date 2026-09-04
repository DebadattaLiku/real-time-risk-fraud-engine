"""
Phase 6: Online behavioral state manager.

Maintains, per `card1` pseudo-entity, exactly the summary statistics
needed to reproduce Phase 4's `bhv_*` feature formulas
(`src/features/behavioral.py`) transaction-by-transaction, without storing
full transaction history.

## State kept per entity (sufficient statistics, not raw history)

    count            -> bhv_prev_txn_count
    mean_amt, m2      -> bhv_hist_mean_amt, bhv_hist_std_amt (Welford's
                         online algorithm — see "Numerical stability" below)
    min_amt, max_amt -> bhv_hist_min_amt / bhv_hist_max_amt
    last_time        -> bhv_time_since_prev_txn
    recent_times      -> bhv_prior_count_1h / bhv_prior_count_24h (a bounded
                         deque of timestamps within the last 24h only —
                         older entries are pruned, so memory per entity is
                         bounded by that entity's 24h transaction volume,
                         not its full lifetime count)

## Numerical stability (Phase 13 audit fix)

Earlier versions of this module tracked `sum_amt`/`sum_sq_amt` and derived
variance as `(sum_sq_amt - count * mean**2) / (count - 1)` — the same
formula Phase 4's offline `src/features/behavioral.py` uses. Phase 6's
real-data validation found this formula numerically unstable for a
specific real case (an entity with a very low true variance and a very
large prior transaction count): `sum_sq_amt` and `count * mean**2` become
two very large, very close numbers, and their difference loses precision
to catastrophic cancellation (documented in
`reports/phase6_realtime_engine_summary.md`, transaction `3491193`, a
0.113 absolute error with no downstream decision impact).

This module now tracks `mean_amt` and `m2` (Welford's algorithm — the sum
of squared deviations from the RUNNING mean, updated incrementally) and
computes `variance = m2 / (count - 1)` directly — mathematically
equivalent to the old formula in exact arithmetic, but immune to that
cancellation because it never computes or subtracts two large near-equal
sums. `bulk_initialize()` uses a stable two-pass computation per historical
block (mean first, then sum of squared deviations from that mean) merged
into any existing state via Chan et al.'s parallel-variance combination
formula — so the incremental (`update()`) and bulk (`bulk_initialize()`)
paths remain exactly consistent with each other, verified by
`test_bulk_initialize_equivalent_to_sequential_updates`.

Phase 4's offline `src/features/behavioral.py` is deliberately left
UNCHANGED by this fix: its formula is fully vectorized across the entire
dataset in one pass (not a natural fit for Welford's inherently
incremental algorithm without a much larger rewrite), and the real-world
impact there was already shown to be negligible (p99 absolute difference
7.4e-6 across 35,845 real comparisons, one outlier with no decision
impact) — not worth the risk of touching validated, frozen offline
feature-generation code for Phase 13's audit-and-polish scope. This
online module is the one that matters for live serving, so it's the one
that was fixed.

## Two ways state is populated

- `update()`: the real-time path — one transaction at a time, called AFTER
  that transaction's features/prediction/decision are already computed
  (see `src/engine/risk_engine.py` for why prediction-before-update is
  enforced).
- `bulk_initialize()`: a vectorized "warm start" from a block of historical
  data (e.g. train+validation, before an online simulation begins on a
  test slice) — mathematically equivalent to calling `update()` once per
  historical row in chronological order, but computed via pandas
  aggregation instead of a slow Python loop over potentially hundreds of
  thousands of rows. Equivalence between the two paths is verified by a
  dedicated test (`test_bulk_initialize_equivalent_to_sequential_updates`).

`isFraud` never appears anywhere in this module — there is no parameter
for it, structurally, not just by convention.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

VELOCITY_WINDOWS_SECONDS = {"1h": 3600, "24h": 86400}
_MAX_WINDOW_SECONDS = max(VELOCITY_WINDOWS_SECONDS.values())


@dataclass
class EntityState:
    count: int = 0
    mean_amt: float = 0.0   # Welford running mean of prior amounts
    m2: float = 0.0          # Welford sum of squared deviations from mean_amt (variance = m2 / (count - 1))
    min_amt: float | None = None
    max_amt: float | None = None
    last_time: float | None = None
    recent_times: deque = field(default_factory=deque)  # ascending, pruned to last 24h

    @property
    def sum_amt(self) -> float:
        """Derived (= mean_amt * count), not separately stored. Kept as a
        property so existing snapshot consumers/tests that read
        `state["sum_amt"]` keep working unchanged — summing amounts was
        never the numerically unstable operation (see module docstring);
        only the variance formula needed to change."""
        return self.mean_amt * self.count

    def to_dict(self) -> dict:
        return {
            "count": self.count, "sum_amt": self.sum_amt, "mean_amt": self.mean_amt, "m2": self.m2,
            "min_amt": self.min_amt, "max_amt": self.max_amt, "last_time": self.last_time,
            "recent_times": list(self.recent_times),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EntityState":
        return cls(
            count=d["count"], mean_amt=d["mean_amt"], m2=d["m2"],
            min_amt=d["min_amt"], max_amt=d["max_amt"], last_time=d["last_time"],
            recent_times=deque(d["recent_times"]),
        )


class BehavioralStateManager:
    """
    Online counterpart to `src/features/behavioral.py`. One instance holds
    state for every `card1` entity seen so far.
    """

    def __init__(self, entity_col: str = "card1"):
        self.entity_col = entity_col
        self._states: dict = {}

    def reset(self) -> None:
        """Clears all entity state — a fresh engine with no history."""
        self._states = {}

    def _get_or_create(self, entity_id) -> EntityState:
        if entity_id not in self._states:
            self._states[entity_id] = EntityState()
        return self._states[entity_id]

    def compute_features(self, entity_id, current_time: float, current_amount: float) -> dict:
        """
        STEP 4 of the online processing sequence: generate behavioral
        features using ONLY the entity's EXISTING state (never the current
        transaction — this method never mutates state).
        """
        state = self._states.get(entity_id)  # do NOT create/mutate on read

        if state is None or state.count == 0:
            prev_count = 0
            hist_mean = hist_std = hist_min = hist_max = float("nan")
            time_since_prev = float("nan")
        else:
            prev_count = state.count
            hist_mean = state.mean_amt
            if state.count >= 2:
                variance = state.m2 / (state.count - 1)
                hist_std = math.sqrt(max(variance, 0.0))
            else:
                hist_std = float("nan")
            hist_min = state.min_amt
            hist_max = state.max_amt
            time_since_prev = (
                current_time - state.last_time if state.last_time is not None else float("nan")
            )

        prev_count_log1p = math.log1p(prev_count)

        hist_mean_safe = hist_mean if (hist_mean == hist_mean and hist_mean != 0) else float("nan")
        amt_to_hist_mean_ratio = current_amount / hist_mean_safe if hist_mean_safe == hist_mean_safe else float("nan")
        amt_diff_from_hist_mean = current_amount - hist_mean if hist_mean == hist_mean else float("nan")
        hist_std_safe = hist_std if (hist_std == hist_std and hist_std != 0) else float("nan")
        amt_zscore = (
            (current_amount - hist_mean) / hist_std_safe
            if (hist_std_safe == hist_std_safe and hist_mean == hist_mean) else float("nan")
        )

        prior_count_1h, prior_count_24h = self._velocity_counts(state, current_time)

        return {
            "bhv_prev_txn_count": float(prev_count),
            "bhv_prev_txn_count_log1p": prev_count_log1p,
            "bhv_hist_mean_amt": hist_mean,
            "bhv_hist_std_amt": hist_std,
            "bhv_hist_min_amt": hist_min if hist_min is not None else float("nan"),
            "bhv_hist_max_amt": hist_max if hist_max is not None else float("nan"),
            "bhv_time_since_prev_txn": time_since_prev,
            "bhv_amt_to_hist_mean_ratio": amt_to_hist_mean_ratio,
            "bhv_amt_diff_from_hist_mean": amt_diff_from_hist_mean,
            "bhv_amt_zscore": amt_zscore,
            "bhv_prior_count_1h": float(prior_count_1h),
            "bhv_prior_count_24h": float(prior_count_24h),
        }

    def _velocity_counts(self, state: EntityState | None, current_time: float) -> tuple:
        if state is None or not state.recent_times:
            return 0, 0
        # Prune entries older than the 24h window relative to NOW (read-time
        # pruning guarantees correctness even if update-time pruning lagged).
        cutoff_24h = current_time - VELOCITY_WINDOWS_SECONDS["24h"]
        while state.recent_times and state.recent_times[0] < cutoff_24h:
            state.recent_times.popleft()
        count_24h = len(state.recent_times)
        cutoff_1h = current_time - VELOCITY_WINDOWS_SECONDS["1h"]
        count_1h = sum(1 for t in state.recent_times if t >= cutoff_1h)
        return count_1h, count_24h

    def update(self, entity_id, current_time: float, current_amount: float) -> None:
        """
        STEP 10 of the online processing sequence: incorporate the CURRENT
        transaction into entity state. Must only be called AFTER that
        transaction's features/prediction/decision have already been
        produced from `compute_features()` — enforced by the calling
        convention in `RiskDecisionEngine`, not by this method itself
        (this method has no way to know whether a prediction happened;
        the ordering guarantee lives one layer up).
        """
        state = self._get_or_create(entity_id)
        state.count += 1
        # Welford's online update: numerically stable running mean/M2 —
        # see the module docstring for why this replaced the earlier
        # sum_amt/sum_sq_amt formula.
        delta = current_amount - state.mean_amt
        state.mean_amt += delta / state.count
        delta2 = current_amount - state.mean_amt
        state.m2 += delta * delta2
        state.min_amt = current_amount if state.min_amt is None else min(state.min_amt, current_amount)
        state.max_amt = current_amount if state.max_amt is None else max(state.max_amt, current_amount)
        state.last_time = current_time
        state.recent_times.append(current_time)
        cutoff_24h = current_time - VELOCITY_WINDOWS_SECONDS["24h"]
        while state.recent_times and state.recent_times[0] < cutoff_24h:
            state.recent_times.popleft()

    def bulk_initialize(
        self, historical_df: pd.DataFrame,
        time_col: str = "TransactionDT", amount_col: str = "TransactionAmt",
        as_of_time: float | None = None,
    ) -> None:
        """
        Vectorized "warm start" equivalent to calling `update()` once per
        row of `historical_df`, in `(time_col, TransactionID)`-sorted
        order — but computed via pandas aggregation, not a Python loop.
        `historical_df` must be entirely in the past relative to whatever
        transactions will be processed next.
        """
        if len(historical_df) == 0:
            return
        d = historical_df.sort_values([time_col, "TransactionID"], kind="mergesort")
        if as_of_time is None:
            as_of_time = float(d[time_col].max())
        cutoff_24h = as_of_time - VELOCITY_WINDOWS_SECONDS["24h"]

        grouped = d.groupby(self.entity_col, sort=False)
        for entity_id, g in grouped:
            amts = g[amount_col].to_numpy(dtype="float64")
            times = g[time_col].to_numpy(dtype="float64")
            state = self._get_or_create(entity_id)

            # Stable two-pass computation for this historical block: mean
            # first, then sum of squared deviations FROM that mean (never
            # a large-sum-minus-large-sum subtraction) — then merged into
            # any existing state via Chan et al.'s parallel-variance
            # combination formula, so this stays exactly consistent with
            # update()'s incremental Welford tracking regardless of how
            # many times bulk_initialize()/update() are interleaved (see
            # module docstring, and test_bulk_initialize_equivalent_to_sequential_updates).
            n_b = len(amts)
            mean_b = float(amts.mean())
            m2_b = float(np.sum((amts - mean_b) ** 2))

            if state.count == 0:
                state.count, state.mean_amt, state.m2 = n_b, mean_b, m2_b
            else:
                n_a, mean_a, m2_a = state.count, state.mean_amt, state.m2
                n = n_a + n_b
                delta = mean_b - mean_a
                state.mean_amt = mean_a + delta * n_b / n
                state.m2 = m2_a + m2_b + delta ** 2 * n_a * n_b / n
                state.count = n

            g_min, g_max = float(amts.min()), float(amts.max())
            state.min_amt = g_min if state.min_amt is None else min(state.min_amt, g_min)
            state.max_amt = g_max if state.max_amt is None else max(state.max_amt, g_max)
            state.last_time = float(times.max())
            recent = times[times >= cutoff_24h]
            for t in np.sort(recent):
                state.recent_times.append(float(t))

    def get_state_snapshot(self, entity_id) -> dict | None:
        state = self._states.get(entity_id)
        return state.to_dict() if state is not None else None

    def serialize(self) -> dict:
        return {str(k): v.to_dict() for k, v in self._states.items()}

    def load_serialized(self, data: dict, key_type=int) -> None:
        self._states = {key_type(k): EntityState.from_dict(v) for k, v in data.items()}
