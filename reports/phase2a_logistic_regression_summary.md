# Phase 2A Summary — Evaluation Framework and Logistic Regression Baseline

This report documents the reusable evaluation framework and two Logistic
Regression baselines built on the approved Phase 0/Phase 1 pipeline. All
numbers are from a real run against the IEEE-CIS training data, using the
exact Phase 1 chronological split (train 413,378 / val 88,581 / test 88,581
rows).

## 1. Preprocessing design

Phase 1's `FeaturePipeline` already produces a leakage-safe, model-agnostic
feature matrix (391 columns: 377 numeric, 14 categorical; categorical NaNs
replaced with an explicit `"__missing__"` placeholder; numeric NaNs left
raw). Logistic Regression needs more than that, so `src/models/preprocessing.py`
adds a model-specific layer on top, implemented as a scikit-learn
`ColumnTransformer` wrapped in a `LogRegPreprocessor` class with an explicit
`fit`/`transform` contract (mirrors Phase 1's pattern — `transform()` raises
if called before `fit()`):

- **Numeric** (377 cols): `SimpleImputer(strategy="median")` → `StandardScaler()`.
  Both fit on the training partition only. Median imputation (not mean) was
  chosen because several numeric groups (`V`, `D`, `C`) are heavily skewed
  and Phase 0 found large missingness in some of them (up to 93%) — median
  is more robust to that. Scaling is necessary because LogReg's L2 penalty
  is scale-sensitive, and the raw numeric columns span very different
  ranges (e.g. `TransactionAmt` in dollars vs. `C`/`D`/`V` columns in
  smaller or differently-scaled units).
- **Categorical** (14 cols): `OneHotEncoder(handle_unknown="ignore")`, fit
  on training categories only. No separate imputer is needed here — Phase 1
  already replaced categorical NaNs with the constant `"__missing__"`
  category before this layer ever sees the data. `handle_unknown="ignore"`
  means a category value that appears only in validation/test (never in
  training) is encoded as all-zero for that feature, rather than crashing —
  verified by a real injected-unseen-category test.
- **No target encoding**, **no encoder fit on validation/test** — enforced
  structurally, not just by convention.
- Output: dense float32 array, 541 columns after one-hot expansion (377
  numeric + 164 one-hot columns from 14 categorical fields).

## 2. Class imbalance strategy

Two baselines were trained and compared, exactly as scoped:

- **Baseline A — standard**: no class weighting.
- **Baseline B — `class_weight="balanced"`**.

**SMOTE was not used.** Rationale: (1) this project's primary metrics —
PR-AUC and Recall@K — are ranking metrics computed directly from predicted
probabilities; resampling the training set mainly changes threshold-based
decisions, not ranking quality; (2) `class_weight="balanced"` achieves the
same cost-sensitivity as oversampling without duplicating or synthesizing
transaction rows, at negligible extra cost; (3) synthetic interpolation
(SMOTE) in a 541-dimensional space with ~164 sparse one-hot dimensions is
harder to justify as producing realistic synthetic fraud examples than a
simple reweighting.

## 3. Solver note (implementation detail, not a modeling-approach change)

This execution environment has a **single CPU core**. Standard
`LogisticRegression(solver="lbfgs")` took >0.5s/iteration on the 413,378 ×
541 matrix and did not converge within a practical time budget (estimated
20–40 minutes per model). Both baselines were instead fit with
`SGDClassifier(loss="log_loss")` — scikit-learn's own documented
large-scale-equivalent optimizer for logistic regression (identical
L2-penalized log-loss objective; same log-odds/coefficient interpretation),
which fits via efficient single-pass gradient updates. This is a solver
choice, not a change to the model family.

## 4. A real calibration problem, found and fixed before reporting results

The first full run produced a Recall@1% on the test set that was *below*
the base fraud rate — worse than randomly selecting 1% of transactions.
Before writing this up, it was investigated rather than reported at face
value:

1. **Ruled out a tie-breaking artifact**: the raw `SGDClassifier`
   `predict_proba` output was saturating to exactly `1.0` for over 1,000
   test transactions (float precision saturation of the sigmoid on large
   decision-function values), and the top-1% budget cutoff (886
   transactions) was landing inside that tied block, with only 25/886
   happening to be fraud.
2. **Fixed the saturation**: wrapped both baselines in
   `CalibratedClassifierCV(method="sigmoid", cv=3)`, fit on the training
   partition only (its internal cross-validation never touches validation
   or test). This produces properly bounded, non-saturated probabilities.
3. **The extreme Recall@1% did not fully go away after fixing the
   artifact** — it dropped to ~0.001 (essentially still collapsed). This
   was traced to a genuine pattern, not a bug: 553 test-set transactions
   share an (almost) identical feature signature (`TransactionAmt=106.0`,
   `ProductCD='S'`, `card1=15775`, same `card4`/`card6`/`C1`/`C2`), each
   with a **distinct `TransactionID` and distinct `TransactionDT`** —
   confirmed these are genuinely different, legitimate transactions (a
   recurring-billing pattern), not a data-duplication bug. **Zero** training
   rows match this exact `card1`/amount signature — it starts appearing
   only in the test time period. Meanwhile, `ProductCD='S'` as a whole has
   a consistently elevated fraud rate across all three splits (train 6.04%,
   val 7.38%, test 4.84%, vs. ~3.5% overall) — a real, stable signal the
   model correctly picked up. The model appears to have over-generalized
   that real "ProductCD='S' is riskier" signal into near-certainty for this
   specific large recurring-transaction cohort it never saw in training,
   which is entirely legitimate. This is reported as a genuine, investigated
   limitation of the Logistic Regression baseline (see §9) — not asserted
   with more confidence than the investigation supports, since a full causal
   attribution would need further feature-attribution analysis outside
   Phase 2A's scope.

The PR curve and Recall@K plots (`reports/figures/phase2a_pr_curve_test.png`,
`reports/figures/phase2a_recall_at_k_test.png`) visibly show this: a sharp
precision spike/crash right at the lowest-recall end of the curve, then
recovery. This was left in the results and figures rather than hidden or
excluded from the report.

## 5. Evaluation methodology

`src/evaluation/metrics.py` (ranking + single-threshold metrics),
`src/evaluation/operational_eval.py` (Recall@K / Precision@K / fixed review
budget), and `src/evaluation/plots.py` are model-agnostic — reused unchanged
for every model in this project going forward.

- **Primary ranking metric: PR-AUC** (average precision) — appropriate
  given the ~3.5% fraud rate; ROC-AUC reported alongside as a secondary,
  more familiar reference.
- **Threshold selection uses validation only.** `select_threshold_by_f1`
  picks the F1-maximizing threshold on validation predictions; that
  threshold is then applied unchanged to test. Test is never used to pick
  a threshold or choose between the two baselines.
- **Operational metrics**: Recall@K / Precision@K at budgets 1%, 2%, 5%
  (not hard-coded — `multi_budget_table` takes any list of budgets), plus a
  full fixed-review-budget report (transactions reviewed, fraud captured,
  total fraud, fraud recall, precision among reviewed) at a headline 2%
  budget via `fixed_budget_report`.

## 6. Validation/test protocol

```
Train      -> fit LogRegPreprocessor, fit both models
Validation -> compare baselines, select threshold (F1-optimal)
Test       -> final evaluation only, reported once
```

No hyperparameter or threshold decision in this report was made by looking
at test metrics.

## 7. Measured results

### Ranking metrics

| Model | Val PR-AUC | Val ROC-AUC | Test PR-AUC | Test ROC-AUC |
|---|---|---|---|---|
| Standard LogReg | 0.3635 | 0.8063 | 0.1604 | 0.7940 |
| Balanced LogReg | 0.3519 | 0.8406 | 0.1481 | 0.8006 |

### Threshold metrics (val-selected F1-optimal threshold, applied to test)

| Model | Threshold | Val Precision | Val Recall | Val F1 | Test Precision | Test Recall | Test F1 |
|---|---|---|---|---|---|---|---|
| Standard LogReg | 0.1102 | 0.533 | 0.321 | 0.401 | 0.282 | 0.330 | 0.304 |
| Balanced LogReg | 0.0670 | 0.381 | 0.385 | 0.383 | 0.213 | 0.371 | 0.270 |

### Threshold metrics at fixed threshold = 0.5 (reference point)

| Model | Val Precision | Val Recall | Val FPR | Test Precision | Test Recall | Test FPR |
|---|---|---|---|---|---|---|
| Standard LogReg | 0.864 | 0.119 | 0.0007 | 0.253 | 0.126 | 0.0135 |
| Balanced LogReg | 0.829 | 0.092 | 0.0007 | 0.207 | 0.092 | 0.0127 |

### Recall@K / Precision@K

| Model | Split | Recall@1% | Precision@1% | Recall@2% | Precision@2% | Recall@5% | Precision@5% |
|---|---|---|---|---|---|---|---|
| Standard | Val | 0.212 | 0.729 | 0.317 | 0.543 | 0.431 | 0.296 |
| Standard | Test | 0.0013 | 0.0045 | 0.161 | 0.280 | 0.373 | 0.260 |
| Balanced | Val | 0.186 | 0.638 | 0.290 | 0.497 | 0.451 | 0.310 |
| Balanced | Test | 0.000 | 0.000 | 0.135 | 0.235 | 0.330 | 0.229 |

Test Recall@1% is anomalously low for the investigated reason in §4 — this
is a real limitation of this linear baseline, not withheld or smoothed over.

### Fixed review-budget report (2% budget = 1,772 transactions reviewed)

| Model | Split | Reviewed | Fraud Captured | Total Fraud | Fraud Recall | Precision |
|---|---|---|---|---|---|---|
| Standard | Val | 1,772 | 963 | 3,042 | 0.317 | 0.543 |
| Standard | Test | 1,772 | 496 | 3,083 | 0.161 | 0.280 |
| Balanced | Val | 1,772 | 881 | 3,042 | 0.290 | 0.497 |
| Balanced | Test | 1,772 | 416 | 3,083 | 0.135 | 0.235 |

## 8. Selected variant

**Standard Logistic Regression** is selected, based on validation PR-AUC
(0.3635 vs. 0.3519) — the only comparison basis used, per the validation
protocol in §6. It is also ahead on validation Recall@2% (0.317 vs. 0.290)
and Precision@2% (0.543 vs. 0.497). `class_weight="balanced"` did improve
validation ROC-AUC (0.8406 vs. 0.8063) and F1-at-selected-threshold
(0.383 vs. 0.401 — actually slightly lower) but did not improve the primary
metric or the operational metrics that matter for this project's review-
budget framing. This selection is provisional and specific to Logistic
Regression — it does not predict which imbalance strategy will work best
for LightGBM in Phase 2B, which has different sensitivity to reweighting.

## 9. Limitations of Logistic Regression (measured, not generic)

- **PR-AUC drops substantially from validation to test** (0.36 → 0.16 for
  the standard model) — a real generalization gap under the strict
  chronological split, not an artifact of the split itself (integrity
  checks pass; see Phase 1). This is the kind of gap a purely random
  train/test split would have hidden.
- **Severe, investigated overconfidence on out-of-training-distribution
  recurring-transaction patterns** (§4) — the model assigns near-certainty
  fraud scores to a legitimate, large, recurring-billing cohort it never
  saw during training, driving Recall@1% to near zero on test despite
  reasonable overall ranking quality (test ROC-AUC ≈ 0.79–0.80). A linear
  model has no mechanism to express "I haven't seen this segment before,
  lower my confidence" — this is a structural limitation of the model
  family, not just this dataset.
- **One-hot encoding assumes categorical values seen in training are the
  relevant vocabulary** — new category values are safely handled
  (all-zero), but this also means the model has literally zero information
  about them beyond whatever correlated numeric features suggest.
- **Linear decision boundary** — cannot capture interaction effects
  between features without them being manually engineered; tree-based
  models (Phase 2B) can capture these natively, which is a large part of
  the motivation for testing LightGBM next.

## 10. What will be implemented next

Per the approved scope restrictions, Phase 2A does **not** include
LightGBM/XGBoost/Random Forest, Isolation Forest, behavioral/historical
aggregation features, pseudo-entity aggregation, risk fusion, the
APPROVE/REVIEW/BLOCK policy, graph analysis, or streaming infrastructure —
all reserved for later phases. Phase 2B (not started) will introduce a
tree-based model using this same evaluation framework (`src/evaluation/`)
unchanged, enabling a fair, same-protocol comparison against these
Logistic Regression baselines.
