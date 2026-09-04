# Phase 3 Summary — Anomaly Detection and Complementarity Analysis

This report documents an Isolation Forest anomaly detector and a rigorous
test of whether it provides fraud-detection signal that the Phase 2B
LightGBM model does not already capture. All numbers are from a real run
against the IEEE-CIS data, using the identical temporal split and
evaluation framework as every prior phase.

**Headline finding, stated up front per the brief's instruction not to bury
a negative result: Isolation Forest did NOT demonstrate meaningful
complementary value over LightGBM in this analysis.** The evidence for
this is laid out in full below.

## 1. Isolation Forest design

`sklearn.ensemble.IsolationForest`, wrapped in `AnomalyDetector`
(`src/models/anomaly.py`). Fit on the training partition only — the class's
`fit()` method has no `y` parameter at all (checked structurally by a test,
not just documented), so it is not possible to accidentally pass labels.

**Score direction**: `anomaly_score = -decision_function(X)`. scikit-learn's
raw `decision_function` returns *higher = more normal*; this project needs
the opposite convention (consistent with every other score used so far —
higher = more suspicious), so the sign is flipped once, in one place, and
documented in the module so no other code has to remember to do it.

## 2. Anomaly feature subset and rationale

**52 features** (38 numeric + 14 categorical) — the full Phase 1 schema
**except the 339 `V` columns**, which are excluded. Rationale: Isolation
Forest isolates points via random axis-aligned splits, and its
effectiveness degrades as irrelevant/noisy dimensions dominate the split
budget — unlike a supervised model, it has no loss signal to learn which
features to ignore. The `V` columns are Vesta's own opaque, largely
undocumented engineered features (a Phase 0 finding) and would have
outnumbered every other feature group combined (339 vs. 52). This is a
**documented judgment call, not a validated ablation** — see §14 for a
specific, important consequence of this choice discovered during analysis.

## 3. Preprocessing strategy

`AnomalyPreprocessor` (`src/models/anomaly_preprocessing.py`):
- **Numeric**: median imputation, fit on training data only (Isolation
  Forest cannot accept NaN). No scaling — axis-aligned random splits are
  scale-invariant per feature, so scaling would not change results.
- **Categorical**: frequency encoding (not one-hot), fit on training data
  only. Each category value is replaced by its training-set frequency —
  avoids a one-hot dimensionality explosion from `P_emaildomain`/
  `R_emaildomain` (~59-60 distinct values each) and gives the detector a
  natural, monotonic "how rare is this value" signal directly. Category
  values never seen in training get frequency 0.0. No target encoding
  anywhere.

## 4. Hyperparameters tested

Three configurations (a modest sweep, per the brief):

| Config | n_estimators | max_samples | max_features |
|---|---|---|---|
| default | 100 | auto (256) | 1.0 |
| more_trees_larger_subsample | 200 | 1024 | 1.0 |
| more_trees_half_features | 200 | auto (256) | 0.5 |

`contamination` was deliberately **not** varied: it only shifts
`decision_function`'s internal offset by a constant derived from training
data, which cannot change the relative order of scores — and every metric
in this project (PR-AUC, ROC-AUC, Recall@K, correlation) is invariant to a
constant shift. Varying it would not have produced a meaningful comparison,
so compute was spent on parameters that actually affect the score ranking
instead.

## 5. Selected configuration and validation-based rationale

| Config | Val PR-AUC | Val ROC-AUC | Val Recall@2% |
|---|---|---|---|
| default | 0.0489 | 0.5832 | 0.0569 |
| more_trees_larger_subsample | 0.0392 | 0.5441 | 0.0207 |
| **more_trees_half_features (selected)** | **0.0529** | 0.5839 | **0.0782** |

Selected on validation PR-AUC (primary) and Recall@2% (both best for this
config) — `max_features=0.5` forces each tree to consider a random half of
the 52 features per split, which apparently reduces some noise. **All three
configurations perform close to random** (PR-AUC near the ~3.4% base fraud
rate; ROC-AUC barely above 0.5) — the configuration choice matters far less
than the fact that none of them work well.

## 6. Validation metrics (selected configuration)

| Metric | Value |
|---|---|
| PR-AUC | 0.0529 |
| ROC-AUC | 0.5839 |
| Recall@1% | 0.0309 |
| Recall@2% | 0.0782 |
| Recall@5% | 0.1108 |
| Precision@1% | 0.1061 |
| Precision@2% | 0.1343 |
| Precision@5% | 0.0761 |

## 7. Final test metrics (selected configuration, untouched until now)

| Metric | Value |
|---|---|
| PR-AUC | **0.0449** |
| ROC-AUC | 0.5448 |
| Recall@1% | 0.0237 |
| Recall@2% | 0.0383 |
| Recall@5% | 0.0843 |
| Precision@1% | 0.0824 |
| Precision@2% | 0.0666 |
| Precision@5% | 0.0587 |

For reference, the no-skill baseline (flagging transactions at random) has
an expected PR-AUC equal to the base fraud rate: **0.0348** on this test
set. Isolation Forest's test PR-AUC of 0.0449 is only marginally above
that — visible directly in the PR curve figure, which hugs the no-skill
line almost its entire length.

## 8. Comparison with LightGBM

LightGBM was **retrained with identical hyperparameters, split, and fixed
random_state=42** solely to obtain its scores for this comparison (Phase 2B
did not persist the model object to disk). Before trusting any comparison,
the reproduction was checked against the original Phase 2B metrics:

| | Original (Phase 2B) | Reproduced (Phase 3) | Match |
|---|---|---|---|
| Val PR-AUC | 0.60918 | 0.60918 | ✅ exact |
| Test PR-AUC | 0.54279 | 0.54279 | ✅ exact |

Confirmed bit-for-bit identical — the Phase 2B benchmark is untouched.

| Metric | LightGBM (Phase 2B) | Isolation Forest (Phase 3) |
|---|---|---|
| Test PR-AUC | **0.5428** | 0.0449 |
| Test ROC-AUC | **0.9024** | 0.5448 |

LightGBM outperforms Isolation Forest by roughly **12x on PR-AUC**. This
gap is expected in principle (LightGBM is supervised; Isolation Forest is
not) — but the size of the gap, combined with the near-chance absolute
performance, is itself informative for the complementarity question.

## 9. Fixed review-budget results (test)

| Model | Budget | Reviewed | Fraud Captured | Total Fraud | Recall | Precision |
|---|---|---|---|---|---|---|
| Isolation Forest | 1% | 886 | 73 | 3,083 | 0.0237 | 0.0824 |
| Isolation Forest | 2% | 1,772 | 118 | 3,083 | 0.0383 | 0.0666 |
| Isolation Forest | 5% | 4,430 | 260 | 3,083 | 0.0843 | 0.0587 |
| LightGBM (reference) | 2% | 1,772 | 1,259 | 3,083 | 0.4084 | 0.7105 |

## 10. Score correlation

| | Pearson r | Spearman r |
|---|---|---|
| Validation | 0.1022 | **-0.2550** |
| Test | 0.0630 | **-0.2322** |

Both correlations are weak in magnitude. The Spearman (rank) correlation is
**negative** — transactions LightGBM ranks as higher fraud risk tend to be
ranked as *slightly less* anomalous by Isolation Forest, not more. Low
correlation is a *necessary* condition for potential complementary value,
but — as the brief warns — **not sufficient**: the scatter plot and overlap
analysis below show this weak/negative correlation does not translate into
Isolation Forest catching fraud LightGBM misses.

## 11. High-risk overlap analysis (test, matched review budgets)

| Budget | Row overlap | Fraud: both | Fraud: only LightGBM | Fraud: only Isolation Forest | Total fraud |
|---|---|---|---|---|---|
| 1% | 60/886 rows (6.8%) | 55 | 725 | **18** | 3,083 |
| 2% | 109/1,772 rows (6.2%) | 94 | 1,165 | **24** | 3,083 |
| 5% | 292/4,430 rows (6.6%) | 217 | 1,569 | **43** | 3,083 |

At every budget, Isolation Forest's top-K selects an almost entirely
different set of transactions from LightGBM's top-K (only ~6-7% row
overlap) — genuinely low redundancy in *what gets flagged*. But the fraud
cases Isolation Forest uniquely catches are a small fraction of total fraud
(18/3,083 = 0.6% at the 1% budget; 43/3,083 = 1.4% at the 5% budget) —
low overlap in *selection* did not translate into meaningfully more *fraud
coverage*.

## 12. Diagnostic union analysis (NOT an operational proposal)

**Labeled diagnostic only, per the brief** — the union exceeds the fixed
review budget (up to ~2x) and is not evaluated as a deployable policy:

| Budget (each model) | Union size | Effective combined budget | Union fraud recall | Incremental fraud vs. LightGBM alone |
|---|---|---|---|---|
| 1% | 1,712 | 1.93% | 0.2588 | **+18** (780 → 798) |
| 2% | 3,435 | 3.88% | 0.4162 | **+24** (1,259 → 1,283) |
| 5% | 8,568 | 9.67% | 0.5933 | **+43** (1,786 → 1,829) |

Even under this generous (budget-violating) diagnostic framing, adding
Isolation Forest's top-K to LightGBM's top-K increases fraud recall by
roughly 1.4-2.3 percentage points at the cost of reviewing an *additional*
~0.9-4.7 percentage points of transaction volume — a poor exchange rate,
and this is the theoretical *ceiling* of the benefit, not a deployable
number.

## 13. Conditional analysis: does Isolation Forest catch what LightGBM misses?

Thresholds derived from **validation only** (never test): `lgbm_low` = P50
of validation LightGBM scores (0.00520), `anomaly_high` = P90 of validation
anomaly scores (-0.01475).

| Split | n matching (low-LGBM, high-anomaly) | Fraud in this group | Fraud rate | Overall fraud rate | **Lift** |
|---|---|---|---|---|---|
| Validation | 6,408 | 11 | 0.17% | 3.43% | **0.050x** |
| Test | 6,524 | 33 | 0.51% | 3.48% | **0.145x** |

This is the most direct answer to the Phase 3 hypothesis, and it is
**negative**: transactions that LightGBM scores as low-risk *and*
Isolation Forest scores as highly anomalous are **less** likely to be
fraud than the overall base rate (14.5% and 5.0% of the base rate,
respectively) — the opposite of what "Isolation Forest finds missed fraud"
would predict. In this quadrant, Isolation Forest's "anomalous" signal is
overwhelmingly flagging **unusual-but-legitimate** transactions, not
fraud LightGBM overlooked.

## 14. Limitations

- **The excluded `V` columns carry substantial supervised signal that
  Isolation Forest never sees.** Phase 2B's feature importance analysis
  found `V258` was LightGBM's single most important feature by gain — and
  the entire `V` group is excluded from Isolation Forest's input by design
  (§2). This is a plausible, direct contributor to Isolation Forest's weak
  performance and was not tested against an alternative (e.g., including
  `V` columns, or a dimensionality-reduced version of them) in Phase 3 —
  flagged here as a specific, concrete direction for future investigation
  rather than left as a vague caveat.
- **Only one feature subset was tested.** Per §2, this is a documented
  choice, not a validated ablation.
- **Isolation Forest's absolute performance is close to random** (test
  PR-AUC 0.045 vs. a 0.035 no-skill baseline), so any complementarity
  conclusion here describes a weak detector, not a strong one operating in
  a genuinely different feature space.
- **The conditional analysis uses one specific threshold pair** (P50/P90 on
  validation); a different pair might show a different (though, given how
  weak the underlying signal is, unlikely to be qualitatively different)
  result.
- Reproducing the Phase 2B LightGBM model added compute cost purely for
  this comparison; it is deterministic (confirmed exact match, §8) and
  changes nothing about the Phase 2B benchmark itself.

## Conclusion

**Isolation Forest, as implemented and tested here, does not provide
sufficient complementary signal over LightGBM to justify inclusion in the
Version 1 hybrid risk engine.** The evidence is consistent across every
angle examined: near-chance absolute performance (PR-AUC barely above the
no-skill baseline), weak/negative rank correlation with LightGBM, minimal
unique fraud capture at matched review budgets (0.6-1.4% of total fraud),
a poor cost/benefit ratio even under a budget-violating diagnostic union,
and — most directly — a *negative* fraud lift in the exact quadrant
("LightGBM says low risk, Isolation Forest says highly anomalous") where
complementary value would need to show up. This is not a forced negative
conclusion — it is what the measured numbers show. The most promising
concrete lead for a different outcome, if anomaly detection is revisited
later, is including the `V` feature group despite its opacity (§14),
which was out of scope for this phase's compute budget.

## What was intentionally NOT implemented

Risk fusion, weighted score combination, stacking, a logistic meta-model,
the APPROVE/REVIEW/BLOCK policy, behavioral historical aggregation
features, pseudo-entity behavioral features, autoencoder anomaly detection,
graph analysis, streaming infrastructure — all reserved for later phases,
per the Phase 3 scope restrictions.
