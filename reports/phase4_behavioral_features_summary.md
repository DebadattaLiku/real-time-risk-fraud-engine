# Phase 4 Summary — Leakage-Safe Behavioral Features and Ablation Study

This report documents leakage-safe historical behavioral features built on
the `card1` pseudo-entity, and a controlled ablation experiment testing
whether they improve on the Phase 2B LightGBM baseline. All numbers are
from real runs against the IEEE-CIS data, using the identical temporal
split and evaluation framework as every prior phase.

**Headline finding: behavioral features produced a small but real,
validation-confirmed improvement.** Full evidence below.

## 1. Behavioral feature motivation

Phase 2B's LightGBM baseline uses only transaction-level features — each
transaction is scored in isolation, with no notion of "is this unusual
*for this card*." Phase 0 identified `card1` as a usable (if imperfect)
pseudo-entity for grouping repeat activity; Phase 4 tests whether adding
*strictly historical* summaries of a card's prior behavior — how many
times it's been used before, its typical spend, how this transaction
compares — gives the model signal the transaction-level features alone
don't capture.

## 2. Why `card1` was selected

Carried forward from Phase 0's pseudo-entity analysis: `card1` has zero
missing values (every transaction has a usable key) and the lowest
cardinality-to-coverage tradeoff among the candidates tested there. It was
already the explicitly recommended candidate, and Phase 4's instructions
fixed it as the starting entity — not re-derived here.

## 3. Why it is only a pseudo-entity (not re-litigated, but binding)

`card1` is Vesta's anonymized card identifier — it is not a verified
customer ID. The same card value could plausibly represent a shared card,
a re-issued card, or (rarely) a hash collision, none of which is
verifiable from this data. Every behavioral feature below is therefore a
summary of "prior transactions sharing this `card1` value," not "this
customer's prior behavior" in a strict sense. This caveat applies to every
number in this report.

## 4. Deterministic ordering policy

Sort by `(TransactionDT, TransactionID)`, ascending, stable (`kind="mergesort"`).
`TransactionDT` alone has ties (Phase 0 found transactions sharing an
identical value); `TransactionID` is the documented tiebreaker, verified by
a dedicated test (`test_same_timestamp_ordered_by_transaction_id`) and
confirmed independent of input row order
(`test_ordering_is_independent_of_input_row_order`).

## 5. Historical-state architecture

Implemented in `src/features/behavioral.py`, called once on the
concatenation of train+validation+test (already chronologically ordered by
Phase 1's `compute_temporal_split`), re-sorted explicitly by the ordering
policy above. This automatically gives every validation row access to all
prior TRAIN history, and every test row access to all prior TRAIN+VALIDATION
history — without the module needing any split-aware branching — exactly
matching the brief's required diagram. `isFraud` is never read: the
function asserts its absence rather than merely omitting it from the
column list (`test_isfraud_column_raises_assertion`).

All features are computed via **vectorized pandas group-cumulative
operations** (`cumcount`, `cumsum`, `shift`+`cummin`/`cummax`, and a
`numpy.searchsorted`-based per-entity velocity count) rather than per-row
Python loops — the full 590,540-row computation runs in **under 1
second**.

## 6. Exact behavioral features created (12 total, `bhv_` prefix)

| Feature | Definition |
|---|---|
| `bhv_prev_txn_count` | Count of prior transactions for this `card1` |
| `bhv_prev_txn_count_log1p` | `log(1 + prev_txn_count)` |
| `bhv_hist_mean_amt` | Mean amount of prior transactions |
| `bhv_hist_std_amt` | Sample std (ddof=1) of prior transaction amounts |
| `bhv_hist_min_amt` | Min amount among prior transactions |
| `bhv_hist_max_amt` | Max amount among prior transactions |
| `bhv_time_since_prev_txn` | Seconds since the immediately preceding transaction |
| `bhv_amt_to_hist_mean_ratio` | Current amount ÷ historical mean |
| `bhv_amt_diff_from_hist_mean` | Current amount − historical mean |
| `bhv_amt_zscore` | (Current amount − historical mean) ÷ historical std |
| `bhv_prior_count_1h` | Count of prior transactions within the last 1 hour |
| `bhv_prior_count_24h` | Count of prior transactions within the last 24 hours |

Historical median was **not implemented** — a deliberate, documented
compute-practicality decision per the brief's own hedge ("if
computationally practical"); the cumulative mean/std/min/max above are all
O(n) vectorizable, while an exact expanding median is not, and the marginal
value over mean+std+min+max was judged not to justify the added complexity
at this stage.

## 7. Cold-start policy

- `bhv_prev_txn_count = 0` for an entity's first transaction — a real,
  meaningful value, not a placeholder.
- Amount statistics are `NaN` when `prev_txn_count == 0` (no history to
  summarize).
- `bhv_hist_std_amt` is additionally `NaN` when `prev_txn_count < 2` — a
  single prior point has no defined sample variance, a strictly stronger
  and mathematically correct condition distinct from "no history at all"
  (verified: `test_single_historical_transaction_std_is_nan_not_zero`).
- Ratio/diff/z-score features are `NaN` whenever their denominator is zero
  or undefined — never silently defaulted to 0 or 1
  (`test_zero_historical_mean_ratio_is_nan_not_inf`,
  `test_zero_historical_std_when_prior_amounts_identical`).
- **No imputation** — consistent with the already-approved Phase 2B
  convention of relying on LightGBM's native missing-value handling, kept
  identical here so Model A and Model B differ in exactly one respect
  (presence of behavioral columns), not in how missingness is treated.

## 8. Leakage prevention strategy and safety-test coverage

20 dedicated tests in `tests/test_phase4_behavioral_features.py`, including:
strictly-historical behavior at each step of a 3-transaction sequence; an
explicit large-outlier self-exclusion check; adding a future transaction
provably does not change any earlier row's features (bit-for-bit);
far-future transactions don't leak into velocity windows; same-timestamp
determinism; order-independence; validation-sees-train-history and
test-sees-train+validation-history continuity checks; a future
test-partition row provably does not affect an earlier validation row;
`isFraud` presence raises; and edge cases (single history point, zero
variance, zero/NaN denominators, missing amounts, sparse entities).

**Two genuine bugs were found and fixed during this process, before any
real-data run:**

1. `pandas.cumsum()` silently treats `NaN` as contributing 0 to a running
   total rather than propagating `NaN` forward. The original implementation
   divided by the raw prior-transaction count, so a missing historical
   amount would have produced a fabricated `hist_mean` of `0.0` instead of
   the correct "unknown" (`NaN`). Caught by
   `test_missing_amount_does_not_crash_and_propagates_as_nan`. Fixed by
   tracking a separate *valid-amount* cumulative count as the denominator.
2. The initial velocity-window implementation used
   `groupby(...).rolling(window, on=...)` and `reset_index(drop=True)` to
   flatten the result — which silently discards the correct row alignment
   (the grouped-rolling result is ordered by *entity group*, not original
   row order), producing wrong-but-not-crashing numbers. Caught by
   `test_velocity_counts_non_negative_and_le_prev_count` (a row had a
   nonzero 1-hour prior count despite zero total prior transactions —
   logically impossible). Replaced with an explicit, transparently-correct
   `numpy.searchsorted`-based per-entity computation.

Both bugs were caught by tests *before* touching real data — exactly the
scenario this test suite exists for.

## 9. Behavioral feature diagnostics (real data, 590,540 transactions)

- **Cold start**: 2.30% of transactions are their `card1`'s first-ever
  appearance in the dataset.
- **`card1` coverage**: 13,553 unique entities, mean 43.6 transactions/entity.
- **Prior transaction count distribution**: median 312, mean 1,264 (heavily
  right-skewed — a small number of very high-volume `card1` values pull the
  mean well above the median), P95 = 6,110, max = 14,931.
- **Time since previous transaction**: median 5,923 seconds (~1.6 hours)
  when present; P95 = 775,346 seconds (~9 days).
- **Missingness**: 2.30% for every amount-derived feature (matches the
  cold-start rate exactly, as expected); 4.01-4.33% for `bhv_hist_std_amt`
  and `bhv_amt_zscore` (entities with exactly one prior transaction, where
  std is mathematically undefined, in addition to zero-history cases).
  `bhv_prev_txn_count`, `_log1p`, and both velocity counts have 0%
  missingness (always computable, even at 0).
- **Extreme values**: 1.48% of transactions have `|bhv_amt_zscore| > 5` — a
  modest, plausible rate of "this amount is unusual for this card,"
  not evidence of a broken computation.

## 10. Ablation experiment design

| | Model A (baseline) | Model B (+behavioral) |
|---|---|---|
| Features | 391 (Phase 1/2B, unchanged) | 391 + 12 behavioral = 403 |
| Hyperparameters | Phase 2B selected config | **identical** |
| `scale_pos_weight` | None (Phase 2B selection) | **identical** (None) |
| Categorical features | 14 (unchanged) | **identical** 14 |
| Early stopping | 50 rounds, PR-AUC feval | **identical** |
| Train/val/test split | Phase 1 boundaries | **identical** |

The only variable that differs between A and B is the presence of the 12
`bhv_*` columns — a clean, controlled comparison. Model A was retrained
(not reused from a saved file, since Phase 2B/3 did not persist the model
object) and checked for **exact reproducibility** against the saved Phase
2B benchmark before anything else was trusted:

| | Original (Phase 2B) | Reproduced (Phase 4, Model A) | Match |
|---|---|---|---|
| Val PR-AUC | 0.60918 | 0.60918 | exact |
| Test PR-AUC | 0.54279 | 0.54279 | exact |

## 11. Validation results (the decision basis)

| Model | Val PR-AUC | Val ROC-AUC | Val Recall@1% | Val Recall@2% | Val Recall@5% |
|---|---|---|---|---|---|
| A (baseline) | 0.6092 | 0.9250 | 0.2686 | 0.4464 | 0.6456 |
| B (+behavioral) | 0.6147 | 0.9253 | 0.2702 | 0.4504 | 0.6492 |

**Validation PR-AUC delta (B minus A): +0.0055.** Model B is ahead on every
validation metric measured, though by a modest margin. Per the protocol,
this validation-only comparison is what determines the ablation's
conclusion — test results (below) are reported but were not consulted to
reach it.

Note: Model A's early stopping triggered at iteration 923/1000. Model B
used the **full 1000-iteration budget without early stopping** — its
validation PR-AUC may not have fully plateaued within the shared
`n_estimators=1000` cap. This is flagged in Limitations (§16) as a
possible source of (likely modest) additional headroom not captured here,
since increasing the iteration budget for Model B alone would have broken
the "keep hyperparameters identical" control.

## 12. Final untouched test results

| Model | Test PR-AUC | Test ROC-AUC | Test Recall@1% | Test Recall@2% | Test Recall@5% |
|---|---|---|---|---|---|
| A (baseline) | 0.5428 | 0.9024 | 0.2530 | 0.4084 | 0.5793 |
| B (+behavioral) | 0.5483 | 0.9058 | 0.2530 (tied) | 0.4090 | 0.6014 |

Test results are directionally consistent with the validation-based
decision (B ahead on PR-AUC, ROC-AUC, and Recall@2%/5%; exactly tied at
Recall@1%) — reported here as confirmation, not as the basis for the
conclusion.

## 13. Operational review-budget comparison (test, 2% budget = 1,772 reviewed)

| Model | Fraud Captured | Total Fraud | Recall | Precision |
|---|---|---|---|---|
| A (baseline) | 1,259 | 3,083 | 0.4084 | 0.7105 |
| B (+behavioral) | 1,261 | 3,083 | 0.4090 | 0.7116 |

At the 5% budget the gap widens somewhat: A captures 1,786/3,083 (57.9%
recall) vs. B's 1,854/3,083 (60.1% recall) — a difference of 68 additional
fraud cases caught at the same review volume.

## 14. Feature importance: did the model actually use the behavioral information?

Yes — measurably, not just nominally. Of Model B's 403 features, ranked by
total split gain:

| Rank | Feature | Gain |
|---|---|---|
| 15 | `bhv_hist_max_amt` | 13,148 |
| 16 | `bhv_hist_mean_amt` | 12,427 |
| 17 | `bhv_hist_std_amt` | 12,034 |
| 20 | `bhv_prev_txn_count` | 10,324 |
| 23 | `bhv_hist_min_amt` | 9,076 |
| 31 | `bhv_amt_diff_from_hist_mean` | 7,304 |
| 32 | `bhv_prior_count_24h` | 7,126 |
| 35 | `bhv_amt_zscore` | 6,494 |
| 40 | `bhv_amt_to_hist_mean_ratio` | 5,634 |
| 45 | `bhv_time_since_prev_txn` | 4,981 |
| 74 | `bhv_prev_txn_count_log1p` | 2,136 |
| 110 | `bhv_prior_count_1h` | 902 |

All 12 behavioral features rank within the top ~110 of 403 — none were
ignored entirely — and 8 of them land in the top 45, immediately below the
transaction-level features (`V258`, `C1`, `C14`, etc.) that already
dominated Phase 2B's importance ranking. The amount-history features
(`hist_max_amt`, `hist_mean_amt`, `hist_std_amt`) are the strongest of the
group, ahead of the pure count/velocity features. As with Phase 2B, gain
reflects the model's *reliance* on a feature, not a causal claim, and does
not indicate the direction of the relationship — not overinterpreted here
beyond "the model used this information."

## 15. Assumptions, limitations, and interpretation (kept separate from measured results)

**Measured**: the numbers in §11-14 are direct outputs of the evaluation
framework, unedited.

**Assumptions**:
- `card1` groups repeat activity meaningfully — see §3; not independently
  verifiable.
- The 1-hour/24-hour velocity windows are a reasonable, not
  exhaustively-searched, choice.
- Excluding the historical median was a compute-practicality judgment
  call, not a tested ablation of "does median help."

**Limitations**:
- Model B did not trigger early stopping (§11) — the reported gain may
  understate what a larger iteration budget could produce, though whether
  that gap is meaningful was not tested (doing so would have broken the
  controlled-comparison design).
- Only one pseudo-entity (`card1`) and one feature set were tested; richer
  entities (e.g. `card1`+`addr1`, previously found in Phase 0 to have much
  higher cardinality and lower per-entity coverage) were out of scope.
- The improvement is real but modest (+0.0055 val PR-AUC, +0.0055 test
  PR-AUC) — smaller than the LightGBM-vs-LogReg gap found in Phase 2B, and
  smaller than what a stronger behavioral feature set (e.g. richer
  windows, additional entities, interaction terms) might plausibly achieve.
- The 2.3% cold-start rate and the heavy right-skew of `prev_txn_count`
  (median 312 vs. mean 1,264) mean behavioral feature quality varies a lot
  across the transaction population — no per-segment breakdown of the
  improvement (e.g., "does it help high-volume cards more than low-volume
  ones") was performed in this phase.

**Interpretation**: the ablation is a fair, controlled, validation-decided
comparison, and it shows a small but consistent improvement from adding
`card1`-based historical behavioral features. This is not a dramatic
result, and it should not be oversold as one — but it is a genuine,
reproducible signal, corroborated by the feature-importance evidence that
the model is actually using the new information rather than ignoring it.

## 16. Final conclusion

**Behavioral features improved performance, based on the validation
comparison this experiment was designed to make the decision on.**
Validation PR-AUC improved from 0.6092 to 0.6147 (+0.0055); the untouched
test set showed a consistent, though similarly modest, improvement
(0.5428 to 0.5483 PR-AUC; +68 fraud cases captured at a 5% review budget).
Feature importance confirms the model meaningfully used the new
information rather than treating it as noise. The improvement is real but
small — this supports continuing to build on the `card1` pseudo-entity
approach in future phases, while tempering expectations about the size of
gains a single, minimally-engineered behavioral feature set can deliver on
its own.

## What was intentionally NOT implemented

Risk fusion, Isolation Forest integration, score stacking, autoencoders,
graph analysis, the APPROVE/REVIEW/BLOCK policy, Kafka/streaming
infrastructure, dashboard changes — all reserved for later phases, per the
Phase 4 scope restrictions.
