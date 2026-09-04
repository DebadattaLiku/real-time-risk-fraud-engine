# Phase 6 Summary — Real-Time Risk Decision Engine Simulation

This report documents a local, stateful simulation of the approved offline
pipeline (Phase 1 features + Phase 4 behavioral features + Phase 4/5
LightGBM model + Phase 5 frozen decision policy) processing transactions
one at a time, and validates that this online simulation reproduces the
offline batch pipeline to a documented numerical tolerance. All numbers
are from a real run against IEEE-CIS test data.

**This is a local simulation, not a deployed system** — see §11-12.

## 1. Real-time architecture

```
Incoming Transaction (dict)
        |
RiskDecisionEngine.process_transaction()
        |
   [STEP 1-10, see §2]
        |
   {transaction_id, risk_score, decision,
    behavioral_features, state_updated, processing_metadata}
```

`RiskDecisionEngine` (`src/engine/risk_engine.py`) wraps, unchanged:
Phase 1's `FeaturePipeline`, Phase 4's `LightGBMPreprocessor`, the
selected transaction+behavioral LightGBM model, and Phase 5's frozen
`DecisionPolicy` — plus one new component, `BehavioralStateManager`
(`src/engine/state.py`), which maintains the running historical summary
statistics needed to compute Phase 4's `bhv_*` features online.

## 2. Transaction processing sequence

Implemented exactly as specified, in this order, inside
`process_transaction()`:

1. Receive transaction (dict)
2. Validate required fields (`_validate_transaction`)
3. Read current historical state (implicit — `compute_features` reads without mutating)
4. Generate behavioral features from EXISTING state only
5. Combine transaction + behavioral features
6. Apply preprocessing (Phase 1 → Phase 4, unchanged)
7. Generate fraud risk prediction
8. Apply frozen Phase 5 decision policy
9. Return prediction and decision
10. Update behavioral state with the CURRENT transaction — **last**

Step 10 is the final statement in `process_transaction()`'s control flow —
the update call physically cannot execute before the prediction/decision
are already computed and packaged into the result, and any exception
raised in steps 2-9 (e.g. failed validation) prevents step 10 from running
at all, so state is never partially updated.

## 3. Behavioral state design

`BehavioralStateManager` keeps, per `card1` entity, **sufficient
statistics** rather than raw transaction history:

| State field | Feeds |
|---|---|
| `count` | `bhv_prev_txn_count` |
| `sum_amt`, `sum_sq_amt` | `bhv_hist_mean_amt`, `bhv_hist_std_amt` |
| `min_amt`, `max_amt` | `bhv_hist_min_amt`, `bhv_hist_max_amt` |
| `last_time` | `bhv_time_since_prev_txn` |
| `recent_times` (deque, pruned to last 24h) | `bhv_prior_count_1h`, `bhv_prior_count_24h` |

Memory per entity is bounded by that entity's 24-hour transaction volume
(the `recent_times` deque), not its full lifetime transaction count — a
long-lived, high-volume card does not require unbounded memory.

Two ways state is populated: `update()` (one transaction at a time, the
real-time path) and `bulk_initialize()` (a vectorized pandas-based "warm
start" from a block of historical data, mathematically equivalent to
calling `update()` once per row in chronological order but computed via
aggregation instead of a slow Python loop — verified equivalent by
`test_bulk_initialize_equivalent_to_sequential_updates`).

## 4. Why prediction occurs before state update

If the current transaction's amount/timestamp were folded into state
*before* computing its own behavioral features, the transaction would see
itself in its own history — reintroducing exactly the leakage Phase 4's
offline pipeline was built to prevent (self-exclusion). Enforcing
update-after-prediction in the online engine is the direct real-time
analogue of Phase 4's "current transaction excluded from its own
historical aggregates" rule, verified here by
`test_state_not_updated_before_prediction` and the self-history/future
leakage tests in §7 below.

## 5. State lifecycle

- `state_manager.reset()` — clears all entity state (verified: subsequent
  transactions are treated as cold-start).
- `engine.reset_state()` — convenience wrapper for the same.
- `state_manager.serialize()` / `load_serialized()` — JSON-safe dict
  round-trip (deques converted to lists and back), verified to reproduce
  identical `compute_features()` output before and after a round trip.
  Distributed/persistent storage is explicitly out of scope for this
  phase, per the brief.

## 6. Input validation

`_validate_transaction()` rejects, before touching state:
- `isFraud` present in the input dict at all (labels must never enter
  real-time prediction — enforced structurally, not by convention)
- any required schema column missing as a **key** (not just a null value)
- `TransactionID`, `TransactionDT`, `TransactionAmt`, or `card1` being `None`
- `TransactionDT`/`TransactionAmt` not numeric or not finite
- negative `TransactionAmt`

All validation failures raise `TransactionValidationError` before any
state mutation — verified directly by
`test_invalid_transaction_does_not_partially_update_state` (state snapshot
identical before/after a rejected transaction).

## 7. Offline-online behavioral feature parity results

Simulated 3,000 chronologically-ordered test transactions (see §11 for
why this subset size, not the full test set), state warm-started from all
501,959 train+validation transactions via `bulk_initialize`.

| | Value |
|---|---|
| Transactions compared | 3,000 |
| Individual feature comparisons | 36,000 (12 features × 3,000 txns) |
| Transactions fully matching | **2,997 / 3,000 (99.9%)** |
| Max absolute difference | 0.1132 (one feature, one transaction — see below) |
| 99th percentile absolute difference (all 35,845 non-NaN comparisons) | 7.4e-6 |
| Median absolute difference | 0.0 |

**Tolerance was investigated, not assumed.** An initial run at a strict
1e-6 tolerance showed only 702/3,000 transactions "matching" — investigated
before accepting or dismissing this. The full difference distribution
showed the 99th percentile at 7.4e-6 (ordinary floating-point
summation-order noise between pandas' vectorized `cumsum` (offline) and
numpy's pairwise-summation `.sum()` (used in `bulk_initialize`)) and
exactly **one genuine outlier**: transaction `3491193`'s `bhv_hist_std_amt`
differed by 0.113 (100% relative error) — traced to catastrophic
cancellation in the "sum-of-squares minus n·mean²" variance formula (used
identically by both the offline and online code paths), which both
implementations share and which becomes numerically unstable specifically
for an entity with a very low true variance and a very large prior
transaction count. This is a known floating-point stability limitation of
this variance formula, not a logic discrepancy between the two pipelines.
Tolerance was set to **1e-4** — comfortably above the 7.4e-6 p99 noise
floor, well below the one outlier (so it is still visibly reported, not
hidden by a loose tolerance).

## 8. Offline-online risk-score consistency

| | Value |
|---|---|
| Compared | 3,000 |
| Mismatches (tolerance 1e-6) | 1 |
| Max absolute difference | 0.000455 |
| Mean absolute difference | 1.5e-7 |
| Pearson correlation | 0.999999996 |

The single score mismatch corresponds to the same transaction
(`3491193`) whose behavioral feature showed the variance-formula
instability in §7 — its small upstream feature difference propagated to a
correspondingly small (0.0044 vs 0.0040, both far from either decision
threshold) score difference, and to **no** decision difference (see §9).

## 9. Offline-online decision consistency

**3,000 / 3,000 (100.0%) identical decisions** between the online
simulation and the offline batch pipeline for the same transactions,
including the one transaction with a measurable score difference — its
score difference (0.0044 vs 0.0040) was far from both the approve
(0.3206) and block (0.6311) thresholds, so it did not change the decision.

## 10. Simulation fraud performance (post-hoc, labels used only after prediction)

| Metric | Online Simulation | Offline (same 3,000-txn subset) |
|---|---|---|
| PR-AUC | 0.5057 | 0.5057 |
| ROC-AUC | 0.8930 | 0.8930 |
| Recall@1% | 0.2727 | — |
| Recall@2% | 0.4318 | — |
| Recall@5% | 0.5795 | — |

Decision distribution (online simulation, n=3,000):

| Bucket | Count | % | Fraud Count | Fraud Rate |
|---|---|---|---|---|
| APPROVE | 2,951 | 98.37% | 56 | 1.90% |
| REVIEW | 25 | 0.83% | 11 | 44.00% |
| BLOCK | 24 | 0.80% | 21 | 87.50% |

Fraud captured (REVIEW+BLOCK): 32/88 (36.36%); BLOCK alone: 21; missed in
APPROVE: 56.

**This 3,000-transaction subset's metrics are not directly comparable to
Phase 5's full-test-set numbers** (PR-AUC 0.5483, Recall@2% 40.9% on all
88,581 test transactions) — it is the chronologically *first* slice of
the test period, not a representative sample, and naturally shows some
variation (lower recall, lower total fraud rate: 2.93% here vs. 3.48%
full-test). The comparison that matters for this phase is online-vs-
offline **on the identical subset** (§7-9 above), which showed near-perfect
agreement — not simulation-subset-vs-full-test-set, which is expected to
differ simply from being a different (and much smaller) population.

## 11. Limitations of the local simulation

- **Subset size**: 3,000 of 88,581 test transactions (3.4%) were
  simulated one-by-one, not the full test set. At ~24-28ms/transaction
  (dominated by per-call pandas/LightGBM overhead on this single-core
  sandbox), the full test set would take ~35-41 minutes — impractical for
  this environment. 3,000 was chosen as large enough for a statistically
  meaningful sample (88 fraud cases) while keeping runtime tractable.
- **Single-threaded, single-process**: no concurrency, no multi-entity
  parallelism, no throughput/latency stress testing under load.
- **No true streaming infrastructure**: transactions are pulled from a
  pre-loaded, pre-sorted pandas DataFrame and fed to the engine in a
  Python loop — not received from an actual message queue, socket, or
  API endpoint.
- **State is in-process, in-memory only**: no persistence across process
  restarts is exercised in the simulation (though `serialize`/
  `load_serialized` exist and are tested), no distributed state store.
- **The one numerical outlier (§7)** reflects a known instability in the
  naive variance formula shared by both pipelines; a numerically more
  stable algorithm (e.g. Welford's online variance) would likely eliminate
  it, but was not implemented in this phase (would count as a design
  change beyond "reuse existing logic" for this phase).
- **Model/feature/policy are frozen inputs**, not re-validated here beyond
  the reproducibility check (exact match confirmed against Phase 4/5
  saved metrics before the simulation began) — this phase does not
  re-litigate whether the model or policy themselves are good, only
  whether the online reproduction of them is faithful.

## 12. Future path toward deployment

This simulation demonstrates the online processing pattern is sound and
faithful to the offline pipeline. A genuine production deployment would
still need, at minimum: a real message queue or API layer (explicitly out
of scope here — Kafka, FastAPI, etc. are Phase 6 exclusions), persistent
and possibly distributed behavioral state (a database or cache, not an
in-process Python dict), concurrency and throughput testing under
realistic load, monitoring/alerting for state or model drift, a plan for
periodic model retraining and state re-warming, and operational review of
the frozen policy thresholds against real (not historical-validation)
outcomes over time. None of this is claimed to exist yet.

## What was intentionally NOT implemented

Kafka, RabbitMQ, distributed streaming, cloud infrastructure, FastAPI
deployment, Docker deployment, dashboard redesign, new fraud models, model
retraining, threshold redesign, score fusion — all reserved for later
phases, per the Phase 6 scope restrictions.
