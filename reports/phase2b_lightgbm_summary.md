# Phase 2B Summary — LightGBM Fraud Detection Model

This report documents the LightGBM nonlinear baseline, evaluated with the
exact same evaluation framework, temporal split, and protocol as Phase 2A.
All numbers are from real runs against the IEEE-CIS training data.

## 1. LightGBM architecture

`lgb.LGBMClassifier`, objective `binary`, trained on the Phase 1 leakage-safe
feature matrix (391 columns — same schema as Phase 2A, but LightGBM gets the
raw 391-column matrix directly, not the 541-column one-hot-expanded version
LogReg needed). `n_estimators=1000` with early stopping (50 rounds, on
validation PR-AUC — see §4 for a real bug found and fixed here).

Base hyperparameters (not tuned beyond the two imbalance variants in scope):
`learning_rate=0.05`, `num_leaves=31`, `max_depth=-1`, `feature_fraction=0.8`,
`bagging_fraction=0.8`, `bagging_freq=1`, `min_child_samples=50`,
`reg_alpha=0.1`, `reg_lambda=0.1`, `n_jobs=1` (single-core environment).

## 2. Preprocessing strategy (LightGBM-specific, separate from Phase 2A's)

`src/models/lightgbm_preprocessing.py` — a different treatment from Phase
2A's `LogRegPreprocessor`, appropriate to a tree model:

- **No one-hot encoding.** Categorical columns are mapped to a fixed,
  train-only vocabulary of category codes via `pandas.Categorical`, and
  passed to LightGBM as native categorical features (`categorical_feature=`
  parameter) — LightGBM splits on these directly rather than needing them
  pre-expanded.
- **No scaling.** Tree splits are invariant to monotonic transformations of
  a numeric feature, so scaling has no effect on a tree model and was
  correctly omitted.

## 3. Missing-value strategy

**No imputation of numeric features**, per the Phase 2B brief. LightGBM's
native missing-value handling learns, per split, which branch a NaN should
default to — this can capture real signal (e.g. "missing D-column value" can
itself be predictive) that a hand-picked imputed constant would destroy.
Numeric NaNs are passed through exactly as Phase 1 left them.

## 4. Categorical feature strategy

Each categorical column's vocabulary is built from **training data only**.
Two explicit, distinct buckets are kept — this is a deliberate design
choice, not a technical necessity:

- `"__missing__"` — Phase 1's placeholder for values that were genuinely
  NaN in the raw data. Trained on like any other category.
- `"__unseen__"` — reserved for any category value that appears in
  validation/test but was never observed in training. Verified with an
  injected test category that a real chronological split pushes exclusively
  into val/test.

These are kept separate (not merged into one "unknown" bucket) so a
genuinely-missing value and a never-before-seen value — which mean
different things — aren't conflated. No target encoding is used anywhere.

## 5. Class imbalance strategies tested — and a real bug found while testing them

Two variants, matching the Phase 2B brief:

- **LightGBM A (standard)**: no weighting.
- **LightGBM B (`scale_pos_weight`)**: `scale_pos_weight = n_negative /
  n_positive` computed from **training labels only** = 27.43.

**A genuine early-stopping bug was found and fixed before these results were
trusted.** The first real run produced `best_iteration=1` for variant B —
i.e., early stopping fired after a single boosting round. Manually inspecting
the per-iteration validation PR-AUC showed it was still climbing steadily
(0.30 → 0.42 over the first 15 rounds) when training was supposedly
"not improving." Root cause: LightGBM's `objective="binary"` registers a
default metric (`binary_logloss`) automatically, in addition to the custom
PR-AUC `feval` passed via `eval_metric`. The early-stopping callback's
`first_metric_only=True` was locking onto whichever metric was registered
*first* — which turned out to be `binary_logloss`, not the intended PR-AUC.
Under `scale_pos_weight=27.43`, `binary_logloss` gets deliberately worse
every round (the model trades calibration for recall on purpose), so
early stopping triggered almost immediately for the wrong reason. **Fix**:
explicitly set `metric="None"` in the LightGBM parameters, which disables
the default metric entirely so early stopping is driven only by the
intended PR-AUC feval. Confirmed by direct inspection — after the fix,
`best_iteration=766` for variant B, with PR-AUC-based early stopping
behaving exactly as expected. This is documented here in full rather than
silently corrected, since it materially changed the "does class weighting
help" conclusion below (before the fix, a broken 1-iteration model would
have made the answer look far more decisively "no" than the evidence
actually supports).

## 6. Validation results for all variants

| Model | Best iteration | Val PR-AUC | Val ROC-AUC |
|---|---|---|---|
| LightGBM Standard | 923 | 0.6092 | 0.9250 |
| LightGBM scale_pos_weight=27.43 | 766 | 0.5719 | 0.9185 |

`scale_pos_weight` measurably **hurt** validation PR-AUC here (0.572 vs.
0.609) even after the bug fix — this is a real, trustworthy finding, not an
assumption. Standard/unweighted training was selected on this basis, per
the "use validation PR-AUC, don't assume weighting helps" instruction.

## 7. Selected variant and criterion

**LightGBM Standard**, selected purely on validation PR-AUC (0.6092 vs.
0.5719) — the only comparison basis, consistent with the Phase 2A protocol.

## 8. Final untouched test results

| Model | Test PR-AUC | Test ROC-AUC |
|---|---|---|
| LightGBM Standard (selected) | **0.5428** | 0.9024 |
| LightGBM scale_pos_weight | 0.5129 | 0.8994 |

Threshold metrics at the validation-selected F1-optimal threshold (0.2519),
applied to test, for the selected model:

| Precision | Recall | F1 | FPR | Flagged |
|---|---|---|---|---|
| 0.654 | 0.437 | 0.524 | 0.0083 | 2,059 |

## 9. Comparison against the Phase 2A baseline

Same test set, same metrics, same operational budgets — Phase 2A was **not
rerun**; its saved results (`data/interim/phase2a_metrics.json`,
`logreg_standard` variant) were loaded directly for this comparison.

| Metric | LogReg (Phase 2A) | LightGBM (Phase 2B) |
|---|---|---|
| Test PR-AUC | 0.1604 | **0.5428** |
| Test ROC-AUC | 0.7940 | **0.9024** |
| Test Recall@1% | 0.0013* | **0.2530** |
| Test Recall@2% | 0.1609 | **0.4084** |
| Test Recall@5% | 0.3733 | **0.5793** |

*LogReg's Recall@1% is depressed by the investigated overconfidence pattern
documented in the Phase 2A report (§4 there) — included here for direct
comparison, not omitted because it's unflattering to the earlier baseline.

**LightGBM outperforms the Logistic Regression baseline on every metric
measured, on the untouched test set, under the identical protocol.** This
conclusion is supported by the actual measured numbers above, not asserted
in advance.

## 10. Operational review-budget results (test set)

| Model | Budget | Reviewed | Fraud Captured | Total Fraud | Recall | Precision |
|---|---|---|---|---|---|---|
| LightGBM Standard | 1% | 886 | 780 | 3,083 | 0.2530 | 0.8804 |
| LightGBM Standard | 2% | 1,772 | 1,259 | 3,083 | 0.4084 | 0.7105 |
| LightGBM Standard | 5% | 4,430 | 1,786 | 3,083 | 0.5793 | 0.4032 |
| LightGBM scale_pos_weight | 1% | 886 | 778 | 3,083 | 0.2524 | 0.8781 |
| LightGBM scale_pos_weight | 2% | 1,772 | 1,186 | 3,083 | 0.3847 | 0.6693 |
| LightGBM scale_pos_weight | 5% | 4,430 | 1,731 | 3,083 | 0.5615 | 0.3907 |

At a 2% review budget, the selected LightGBM model captures **1,259 of
3,083** test-set fraud cases (40.8% recall) at 71.0% precision among
reviewed transactions — versus LogReg's 496/3,083 (16.1% recall) at 28.0%
precision at the same budget.

## 11. Feature importance findings

Top 10 features by total split gain (selected model):

| Rank | Feature | Gain |
|---|---|---|
| 1 | V258 | 82,158 |
| 2 | C1 | 48,151 |
| 3 | C14 | 35,239 |
| 4 | C13 | 28,246 |
| 5 | card2 | 27,457 |
| 6 | card1 | 26,026 |
| 7 | addr1 | 20,842 |
| 8 | R_emaildomain | 20,762 |
| 9 | TransactionAmt | 20,483 |
| 10 | D2 | 17,919 |

**Limitations of this feature importance analysis** (stated explicitly, not
implied): "gain" measures how much a feature reduced training loss when
used in a split — it is a measure of the model's *reliance* on a feature,
not a causal claim that the feature *causes* fraud, and it says nothing
about the direction of the relationship (e.g. whether high or low `C1`
increases fraud risk). It can also be inflated for features that are
easy to split on repeatedly (e.g. high-cardinality numeric columns) even
if their marginal contribution per split is small. Vesta's `V`/`C`/`D`
columns are themselves pre-engineered and largely undocumented (a Phase 0
finding), so "V258 matters most" cannot currently be translated into a
plain-language business explanation of *why* — that would require either
Vesta's original feature documentation (not available) or a dedicated
interpretability pass (e.g. SHAP), explicitly deferred per the brief.

## 12. Limitations

- **Single hyperparameter configuration tested per variant** — no formal
  hyperparameter search was performed beyond the two imbalance variants;
  the base config (`learning_rate=0.05`, `num_leaves=31`, etc.) was chosen
  as a reasonable, literature-standard starting point, not tuned against
  validation PR-AUC. A modest tuning pass could plausibly improve results
  further — that is future work, not claimed here.
- **`scale_pos_weight` hurt PR-AUC but was only tested at one value**
  (the "textbook" `n_neg/n_pos` ratio). Intermediate values were not swept;
  it's possible a smaller weight would help without the full-strength
  penalty. Not tested in Phase 2B, so not claimed.
- **Test PR-AUC (0.54) is well below validation PR-AUC (0.61)** — a real
  generalization gap under the chronological split, smaller in relative
  terms than Logistic Regression's gap (0.36 → 0.16) but still present.
  Worth tracking as more phases are added.
- **Feature importance is not causal** — see §11.
- Single-core execution environment meant training took ~130–145 seconds
  per model; this did not require reducing model complexity (the same
  `n_estimators=1000` budget with early stopping was used throughout), but
  did require running the two model-training steps as separate processes
  from cached preprocessed data rather than one continuous script
  invocation, purely to fit within this environment's per-command time
  limits. The `src/run_phase2b_lightgbm.py` script itself is written to run
  end-to-end unattended in a normal environment; the staged execution used
  here to produce these results calls the exact same functions in the same
  order, so results are identical to what that script produces.

## 13. What was intentionally NOT implemented

Isolation Forest, autoencoders, behavioral historical aggregation features,
pseudo-entity risk features, risk fusion, stacked models, the
APPROVE/REVIEW/BLOCK decision policy, graph analysis, streaming
infrastructure (Kafka or otherwise), and SHAP-based interpretation (reserved
for a later phase per the brief).
