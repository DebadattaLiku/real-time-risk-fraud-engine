# Phase 5 Summary — Validation-Calibrated Risk Decision Policy

This report documents the conversion of the Phase 4 selected model's fraud
probability into an operational three-way APPROVE/REVIEW/BLOCK decision
policy, designed entirely on validation data and evaluated once on the
untouched test set. All numbers are from a real run against the IEEE-CIS
data, using the identical temporal split and model as Phase 4.

## 1. Why a probability score is not itself an operational decision

Phase 4's model outputs a continuous fraud probability per transaction —
useful for ranking, but not directly actionable. Operations needs one of
three concrete outcomes per transaction (let it through, hold it for a
human, or block it automatically), and someone has to decide where the
cut points go. That decision trades off three things against each other:
fraud losses (missed fraud in APPROVE), finite human review capacity
(REVIEW queue size), and legitimate-customer friction (legitimate
transactions blocked or delayed). This phase makes that trade-off
explicit and calibrates it against historical validation evidence rather
than leaving it implicit or arbitrary.

## 2. Decision-policy architecture

`src/decision/policy.py` — a `DecisionPolicy` dataclass with
`approve_threshold` and `block_threshold`, enforcing the rule:

```
if risk_score < approve_threshold:  APPROVE
elif risk_score < block_threshold:  REVIEW
else:                                BLOCK
```

Both thresholds are validated at construction (must be in `[0, 1]`,
`approve_threshold <= block_threshold`, invalid ordering raises
`InvalidThresholdError`). `decide()` requires only scores (no labels — the
frozen policy is applicable at real prediction time, before a true label
would ever exist) and raises on `NaN` input rather than silently producing
an undefined decision. `src/decision/evaluation.py` turns
`(y_true, decisions)` into the full operational report (decision
distribution, per-bucket fraud capture, friction metrics) — kept separate
from `src/decision/policy.py`'s decision logic since evaluation *needs*
labels and policy *decision* never does.

## 3. Definitions of APPROVE, REVIEW, and BLOCK

- **APPROVE**: low predicted fraud risk; transaction proceeds automatically.
- **REVIEW**: intermediate predicted fraud risk; sent to a limited manual-review queue.
- **BLOCK**: high predicted fraud risk; automatically blocked/declined.

These are **modeling-policy decisions calibrated against historical data**,
not guaranteed real-world financial or legal rules — a real deployment
would need additional review (regulatory, business, fraud-ops
input) before any threshold here is used operationally.

## 4. Threshold-selection methodology

Implemented in `src/decision/threshold_selection.py`:

1. **`approve_threshold`** = the validation score at the boundary of the
   top `review_capacity` fraction of transactions (e.g. top 2%) — i.e.
   `review_capacity` is the total fraction of transactions routed to
   REVIEW+BLOCK combined; everything else is APPROVEd.
2. **`block_threshold`** is derived by scanning increasing "top slice"
   fractions of validation scores (from small, in 0.05-percentage-point
   steps, up to `review_capacity`) and selecting the **largest** slice
   whose measured precision (fraud rate within that slice, on validation
   labels) still meets a chosen `block_precision_target`. This directly
   operationalizes "avoid blocking large numbers of legitimate
   transactions" — a higher target shrinks BLOCK, a lower one grows it,
   but only as far as validation evidence actually supports.

Both derivations use **validation data exclusively**; the function's
signature has no test-data parameter at all (checked structurally by a
test, not just by convention).

## 5. Validation-only policy design

```
TRAIN      -> fit model (Phase 4, reproduced — see §9)
VALIDATION -> ALL threshold derivation, all policy comparison, selection
TEST       -> frozen policy applied once, unchanged
```

No test label or test score was inspected before the policy was frozen and
saved to `config/decision_policy.yaml`.

## 6. Operating policies considered

Three candidates, each an explicit experimental scenario (not a universal
recommendation), with thresholds **derived** from validation data per §4 —
not invented:

| Policy | `review_capacity` | `block_precision_target` | Rationale |
|---|---|---|---|
| Conservative | 1% | 98% | Minimal intervention; BLOCK only when validation shows near-certainty |
| Balanced | 2% | 90% | Moderate intervention; BLOCK requires high but not extreme confidence |
| Aggressive | 5% | 70% | High intervention; accepts more false positives for greater fraud capture |

## 7. Validation results (all three candidates)

| Policy | Approve Thr | Block Thr | APPROVE% | REVIEW% | BLOCK% | Recall (R+B) | Block Precision | Review Precision | Legit Blocked% | Legit Reviewed% |
|---|---|---|---|---|---|---|---|---|---|---|
| Conservative | 0.8078 | 0.9677 | 98.9998% | 0.5001% | 0.5001% | 0.2702 | 0.9842 | 0.8713 | 0.0082% | 0.0667% |
| Balanced (selected) | 0.3206 | 0.6311 | 97.9996% | 0.6999% | 1.3005% | 0.4504 | 0.9019 | 0.5339 | 0.1321% | 0.3379% |
| Aggressive | 0.0887 | 0.2305 | 94.9989% | 2.5502% | 2.4509% | 0.6492 | 0.7029 | 0.1988 | 0.7540% | 2.1160% |

All three achieved their `block_precision_target` on validation
(confirmed in `diagnostics.block_precision_target_achieved = True` for
each). As expected, Aggressive captures far more fraud (64.9% vs.
Conservative's 27.0%) but at substantially more legitimate-customer
friction (2.87% of legitimate transactions intervened vs. Conservative's
0.075%).

## 8. Selected policy and rationale

**Balanced** was selected, on validation evidence only:

- Conservative's fraud recall (27.0%) is low relative to what's clearly
  achievable at only slightly more intervention — Balanced nearly doubles
  recall (45.0%) for a modest increase in legitimate-transaction friction
  (0.13% blocked vs. 0.008%).
- Aggressive's incremental recall over Balanced (64.9% vs. 45.0%, +19.9
  points) comes at a disproportionate friction cost (2.87% of legitimate
  transactions intervened vs. Balanced's 0.47% — roughly 6x more
  legitimate-customer friction for 1.4x the recall) and a much lower BLOCK
  precision (70.3% vs. 90.2%), meaning a meaningfully higher share of
  automatic blocks would be wrong.
- Balanced keeps BLOCK-bucket precision high (90.2% on validation) while
  still capturing a meaningfully larger share of fraud than Conservative.

This is a documented judgment call among the three measured options, not a
formula — a different organization's cost structure (e.g. much higher
per-fraud-dollar losses, or much cheaper manual review) could reasonably
prefer Aggressive or Conservative instead. The full validation comparison
table (§7) is provided so that judgment can be revisited.

## 9. Frozen threshold configuration

Saved to `config/decision_policy.yaml`:

```yaml
policy_name: balanced
approve_threshold: 0.32061562877852007
block_threshold: 0.631136794838037
review_capacity_assumption: 0.02
block_precision_target: 0.9
```

The underlying model was **reproduced** (Phase 4 did not persist the
trained model object) with identical hyperparameters/features/split/seed,
and checked for exact match against the saved Phase 4 benchmark before
anything else was trusted:

| | Original (Phase 4) | Reproduced (Phase 5) | Match |
|---|---|---|---|
| Val PR-AUC | 0.61469 | 0.61469 | exact |
| Test PR-AUC | 0.54825 | 0.54825 | exact |

## 10. Final untouched test results

| Bucket | Count | % of Total | Fraud Count | Legit Count | Fraud Rate |
|---|---|---|---|---|---|
| APPROVE | 86,730 | 97.9104% | 1,792 | 84,938 | 2.07% |
| REVIEW | 722 | 0.8151% | 333 | 389 | 46.12% |
| BLOCK | 1,129 | 1.2745% | 958 | 171 | 84.85% |

- **Fraud captured (REVIEW+BLOCK)**: 1,291 / 3,083 total fraud cases (41.87% recall)
- **Fraud captured by BLOCK alone**: 958
- **Fraud missed in APPROVE**: 1,792 (58.13% of all fraud — the largest
  single number in this report, and the plainest statement of this
  policy's limitation: at this operating point, the majority of fraud
  cases are still approved automatically)
- **Precision among BLOCKED**: 84.85% (test) vs. 90.19% (validation) — a
  real but modest drop, consistent with the PR-AUC generalization gap
  already observed in Phase 2B/4
- **Precision among REVIEWED**: 46.12% (test) vs. 53.39% (validation)

## 11. Customer-friction analysis

| | Count | % of Legitimate Transactions |
|---|---|---|
| Legitimate transactions BLOCKED | 171 | 0.2000% |
| Legitimate transactions REVIEWED | 389 | 0.4550% |
| Total legitimate transactions intervened upon | 560 | 0.6550% |

Framed differently: of 85,498 legitimate test transactions, 84,938
(99.35%) are approved automatically with no friction at all. The 171
legitimate transactions automatically blocked (0.20%) are the policy's
direct cost in false positives — each represents a real customer whose
transaction would be declined without human review under this policy.

## 12. Comparison with simple fixed-budget ranking

At the SAME matched intervention volume (test: 1,851 transactions, ~2.09%
of the test set — the Balanced policy's actual REVIEW+BLOCK total):

| | Three-Way Policy | Simple Ranking (Top-K to REVIEW) |
|---|---|---|
| Transactions flagged | 1,851 | 1,851 |
| Fraud captured | 1,291 | 1,291 |
| Fraud recall | 0.4187 | 0.4187 |
| Precision (BLOCK tier / overall) | 0.8485 | 0.6975 |

**Fraud recall is identical between the two approaches** — this is
expected and not a meaningful finding on its own: both flag the exact same
number of transactions ranked by the exact same score, so they capture the
exact same fraud cases (the top 1,851 by score, regardless of how that set
is subdivided). **The real difference is structural, not in recall**: the
three-way policy identifies a 1,129-transaction high-confidence subset
(BLOCK) with 84.85% precision — meaningfully higher than the flat
ranked list's blended 69.75% precision across the same total volume — that
can plausibly be automated without human review, while the simple ranking
gives operations one undifferentiated queue of 1,851 transactions with no
signal about which ones are safe to act on without a human. This is the
concrete value the three-way structure adds: it doesn't catch more fraud
at the same budget, but it tells you which fraction of that budget can be
acted on with high confidence versus which needs a person.

## 13. Limitations and assumptions

**Measured**: all numbers in Sections 7-12 are direct evaluation-framework outputs.

**Assumptions**:
- The three `review_capacity` / `block_precision_target` scenarios are a
  reasonable but non-exhaustive grid — other combinations were not tested.
- The block-fraction search uses a fixed 0.05-percentage-point step size;
  a finer grid could find marginally different thresholds.
- Policy selection (Section 8) uses documented but ultimately judgment-based
  reasoning; it is not a claimed optimum under any single formal objective.

**Limitations**:
- 58.1% of test fraud is still missed in APPROVE — this policy is a
  meaningful improvement over doing nothing, not a solution that catches
  most fraud.
- Precision drops from validation to test in both the BLOCK (90.2% to
  84.9%) and REVIEW (53.4% to 46.1%) buckets — consistent with the
  generalization gap documented since Phase 2B; a production deployment
  should expect real-world precision closer to test than validation.
- This policy assumes a fixed, static operating point; it does not adapt
  to time-varying fraud patterns, seasonal effects, or model drift, none
  of which were assessed in this phase.
- The "manual review capacity" and "legitimate-customer friction" figures
  are model-derived estimates, not validated against an actual operations
  team's real throughput or a real cost-of-friction estimate — both would
  be necessary before treating any of these thresholds as production-ready.

## What was intentionally NOT implemented

New fraud models, Isolation Forest integration, score fusion, model
stacking, autoencoders, graph analysis, Kafka/streaming infrastructure,
real-time API deployment, dashboard changes — all reserved for later
phases, per the Phase 5 scope restrictions.
