# Phase 1 Pipeline Summary — Real-Time Risk Decision & Fraud Intelligence Engine

This report documents the Phase 1 leakage-safe data pipeline: loading,
temporal splitting, feature schema, and transaction-time-only feature
preparation. All numbers below are from a real run against
`data/raw/train_transaction.csv` (590,540 rows × 394 columns).

## 1. Pipeline architecture

```
train_transaction.csv
        │
        ▼
src/data/load.py           — load + validate (columns, ID uniqueness,
                              binary target, TransactionDT integrity)
        │
        ▼
src/data/split.py          — strict chronological split (70/15/15),
                              sorted by TransactionDT, no shuffling
        │
        ├─────────────► train_df
        ├─────────────► val_df
        └─────────────► test_df
        │
        ▼
src/features/schema.py     — feature schema built from train_df ONLY
                              (name, dtype, group, missingness, used/excluded)
        │
        ▼
src/features/pipeline.py   — FeaturePipeline: fit(train_df) → transform(any_df)
        │
        ▼
X_train / y_train, X_val / y_val, X_test / y_test   (in-memory, no CSV duplication)
```

Orchestrated end-to-end by `src/run_phase1_pipeline.py`.

## 2. Data flow

1. `load_train_transaction()` reads the CSV with a memory-efficient dtype
   map (category/int32/float32 — the same strategy validated in Phase 0)
   and validates it before anything downstream touches it.
2. `compute_temporal_split()` sorts by `TransactionDT` and cuts three
   contiguous, non-overlapping partitions — no row is ever shuffled.
3. `build_feature_schema()` inspects **`train_df` only** to classify every
   column and decide what's used vs. excluded, so no inclusion decision is
   informed by validation or test data.
4. `FeaturePipeline.fit(train_df)` establishes the fit contract (validates
   schema-vs-data column consistency; there is no learned statistical
   parameter yet — see §5). `transform()` is then applied independently to
   train, validation, and test.

## 3. Temporal split design

Reused the exact boundaries approved in Phase 0 EDA — split is fully
reproducible because it's a deterministic sort-and-cut, not a random
process.

| Split | Rows | % | `TransactionDT` range |
|---|---|---|---|
| Train | 0 – 413,377 (413,378 rows) | 70.00% | 86,400 → 10,437,996 |
| Validation | 413,378 – 501,958 (88,581 rows) | 15.00% | 10,438,003 → 13,151,840 |
| Test | 501,959 – 590,539 (88,581 rows) | 15.00% | 13,151,880 → 15,811,131 |

Verified directly against the real split output (not just the proposed
Phase 0 numbers): `max(train_dt) ≤ min(val_dt)` and `max(val_dt) ≤
min(test_dt)` both hold, and `TransactionID` sets across the three
partitions are pairwise disjoint. See `data/interim/phase1_split_metadata.json`
for the machine-readable record.

## 4. Feature schema approach

`build_feature_schema()` produces one `FeatureSpec` per column with:
`name`, `pandas_dtype`, `value_type` (numeric/categorical), `feature_group`,
`missing_pct_train` (computed from train only), `used`, `exclusion_reason`.

Column classification uses `pandas.api.types.is_numeric_dtype` rather than
an `== object` check — the latter would silently misclassify pandas' newer
string dtype as numeric (see §9, bug #3), which would skip missing-category
handling for those columns entirely.

Feature groups (391 used columns): `V` (339), `D` (15), `C` (14), `M` (9),
`card` (6), `addr` (2), `distance` (2), `email` (2), `amount` (1),
`product` (1). 377 numeric, 14 categorical.

Full per-column record: `reports/phase1_feature_schema.csv`.

## 5. Leakage prevention strategy

- **No random splitting** — `compute_temporal_split` sorts by
  `TransactionDT` with a stable mergesort and slices contiguously; there is
  no `sample`, `shuffle`, or RNG call anywhere in the split path. Verified
  by `test_split_never_shuffles_never_randomizes`, which asserts identical
  output across repeated calls.
- **Feature decisions from train only** — `build_feature_schema(train_df)`
  never receives val/test data, so missingness-based or type-based
  decisions can't be informed by them. Verified by
  `test_schema_missingness_computed_only_from_given_df`.
- **Fit/transform contract structurally enforced** — `FeaturePipeline`
  raises `RuntimeError` if `transform()` is called before `fit()`, and
  `fit()` only accepts one dataframe (the training partition). No learned
  statistical parameter (mean, mode, frequency, target statistic) is
  computed in Phase 1 — numeric values and their missingness are passed
  through unchanged, and categorical missing values get a fixed placeholder
  string, not a data-derived one. This means the fit/transform split is
  currently trivial in effect, but the contract is real: any later phase
  that adds a learned transformation (e.g. a scaler or an encoder with
  frequencies) plugs into `fit()`/`transform()` and automatically inherits
  the same train-only guarantee, checked by
  `test_pipeline_fit_uses_only_the_dataframe_passed_in`.
- **Target/ID/raw-time excluded from X** — `isFraud`, `TransactionID`, and
  `TransactionDT` are never present in the feature matrix, checked by
  assertions in the runner script and by
  `test_pipeline_transform_excludes_target_and_id`.
- **`TransactionDT` explicitly reviewed, not silently included or excluded**
  — documented decision (see feature schema `exclusion_reason` and §9): used
  for ordering/splitting only, not as a raw feature, because its
  distribution is strictly non-overlapping across train/val/test by
  construction and would let a model learn a spurious direct time-to-fraud
  mapping. This is a modeling-safety choice, not a claim that `TransactionDT`
  is leakage — it is available at scoring time.
- **High missingness was not treated as automatic exclusion** — `D` columns
  (up to 93% missing) and `V` columns (up to 86% missing) are still included
  as candidate features, per Phase 0's documented finding that missingness
  is not itself leakage.

## 6. What is intentionally NOT yet implemented

- No ML models (Logistic Regression, LightGBM, etc.)
- No anomaly detection (Isolation Forest)
- No behavioral/historical aggregation features (e.g. `card1`-based velocity)
  — Phase 0 recommended `card1` as the pseudo-entity candidate for this,
  but building those features is out of scope for Phase 1
- No risk fusion layer
- No final APPROVE/REVIEW/BLOCK decision policy
- No scaling, encoding (one-hot/target/frequency), or imputation — numeric
  NaNs are preserved as-is; categorical NaNs get an explicit placeholder
  category only
- No merge of `train_identity.csv` into the modeling table
- No writing of large duplicate CSV copies of X/y to disk — only lightweight
  JSON/CSV metadata artifacts are persisted
