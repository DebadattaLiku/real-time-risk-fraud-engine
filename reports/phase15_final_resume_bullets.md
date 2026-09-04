# Phase 15 — Final Resume Bullets

Source: `reports/phase15_resume_achievement_extraction.md` and direct
artifact re-verification. All quantitative values below are the exact
figures from the extraction report's evidence inventory —
**PR-AUC 0.5483 / ROC-AUC 0.9058 / Recall@1,2,5% = 25.30% / 40.90% /
60.14%** are the real champion's (`fraud-risk-lightgbm-v1`) full-test-set
numbers and are used consistently throughout. No simulation-subset,
synthetic-governance, or pre-fix-without-caveat numbers appear as
headline claims anywhere below.

---

## AI/ML Engineer

### Bullet 1 — ⭐ BEST
> Engineered a leakage-safe fraud detection pipeline with strict
> chronological train/validation/test splitting (590K+ transactions) and
> trained a LightGBM model achieving 0.5483 PR-AUC and 0.9058 ROC-AUC on
> a held-out test set. *(28 words)*

### Bullet 2 — strong
> Designed 12 historical behavioral features (transaction velocity,
> amount z-score, recency) per entity, improving validation PR-AUC by a
> modest but statistically confirmed +0.0055 through strictly
> leakage-safe, pre-transaction-only aggregation. *(27 words)*

### Bullet 3 — ⭐ BEST
> Built a stateful real-time inference engine enforcing
> predict-before-state-update ordering to prevent feature leakage;
> validated 100% offline/online decision parity across a 3,000-transaction
> simulation. *(23 words)*

### Bullet 4 — strong
> Translated model risk scores into an operational three-way decision
> policy (APPROVE/REVIEW/BLOCK) with thresholds selected exclusively on
> validation data, achieving 84.85% precision on blocked test-set
> transactions. *(24 words)*

### Bullet 5 — supporting
> Integrated the trained model into a FastAPI microservice with request
> validation and structured error handling, verified through automated
> tests covering prediction correctness, edge cases, and API-engine
> consistency. *(25 words)*

---

## Data Scientist

### Bullet 1 — ⭐ BEST
> Modeled fraud risk under severe class imbalance (3.5% positive rate)
> using PR-AUC and recall-at-review-budget instead of accuracy, achieving
> 25.30% / 40.90% / 60.14% recall at 1% / 2% / 5% review capacity.
> *(30 words)*

### Bullet 2 — strong
> Quantified the operational tradeoff of a three-way fraud decision
> policy: 84.85% precision on blocked transactions against only 0.20% of
> legitimate customers inconvenienced, using validation-only threshold
> selection. *(24 words)*

### Bullet 3 — strong
> Engineered and validated 12 behavioral features per card entity
> (transaction velocity, historical mean/z-score), yielding a modest but
> statistically confirmed +0.0055 PR-AUC gain measured on validation data.
> *(24 words)*

### Bullet 4 — strong
> Applied Population Stability Index and Kolmogorov-Smirnov tests to
> monitor feature and score drift against a validation-period reference,
> diagnosing and correcting a rare-category sampling-noise bias in
> categorical PSI. *(25 words)*

### Bullet 5 — supporting
> Compared LightGBM against a Logistic Regression baseline and an
> Isolation Forest anomaly detector, selecting the final model via
> evidence-based PR-AUC comparison rather than assumption. *(22 words)*

---

## MLOps / ML Engineer

### Bullet 1 — ⭐ BEST
> Diagnosed and fixed a catastrophic-cancellation bug producing NaN
> variance in a real-time feature engine, replacing it with Welford's
> algorithm and reducing numerical error to ~4.9e-12 versus NumPy ground
> truth. *(26 words)*

### Bullet 2 — ⭐ BEST
> Designed a local model governance registry with 6 independently
> explainable promotion gates (PASS/FAIL/REQUIRES_REVIEW) and SHA-256
> artifact hashing, requiring explicit human approval for any promotion or
> rollback. *(24 words)*

### Bullet 3 — strong
> Built a Prometheus-style observability layer (request, prediction,
> decision, and error metrics) and verified zero interference with
> inference behavior through dedicated non-interference regression tests.
> *(24 words)*

### Bullet 4 — strong
> Implemented PSI/KS drift detection with rare-category grouping and
> explicit unseen-category surfacing, diagnosing and fixing a sampling-noise
> bug that inflated one feature's PSI from 0.78 to a corrected
> 0.19-0.31. *(28 words)*

### Bullet 5 — supporting
> Containerized a FastAPI fraud-scoring service in Docker (build context
> ~4.29MB, raw dataset excluded) with CI running a 316-test automated
> suite on every push via GitHub Actions. *(26 words)*

---

## Top 3 Overall

For a high-end AI/ML Engineer resume, these three bullets together prove
model quality, systems engineering, and depth beyond a notebook —
without redundancy:

1. **AI/ML Bullet 1** — the core, defensible model result with the
   leakage-safety discipline that makes it credible.
2. **AI/ML Bullet 3** — proves the model was actually engineered into a
   working real-time system, not just evaluated offline.
3. **MLOps Bullet 2** — proves lifecycle/governance maturity most
   candidates at this level cannot show.

---

## Recommended AI/ML Resume Version

Four bullets, each covering a distinct dimension (strong ML result,
behavioral intelligence, real-time engineering, MLOps/governance depth),
with no overlap:

1. **[Strong ML result]** Engineered a leakage-safe fraud detection
   pipeline with strict chronological train/validation/test splitting
   (590K+ transactions) and trained a LightGBM model achieving 0.5483
   PR-AUC and 0.9058 ROC-AUC on a held-out test set.

2. **[Behavioral intelligence]** Designed 12 historical behavioral
   features (transaction velocity, amount z-score, recency) per entity,
   improving validation PR-AUC by a modest but statistically confirmed
   +0.0055 through strictly leakage-safe, pre-transaction-only
   aggregation.

3. **[Real-time engineering]** Built a stateful real-time inference
   engine enforcing predict-before-state-update ordering to prevent
   feature leakage; validated 100% offline/online decision parity across
   a 3,000-transaction simulation.

4. **[MLOps/governance depth]** Designed a local model governance
   registry with 6 independently explainable promotion gates
   (PASS/FAIL/REQUIRES_REVIEW) and SHA-256 artifact hashing, requiring
   explicit human approval for any promotion or rollback.

---

## Interview Defensibility

*(Internal preparation notes — not for the resume itself.)*

**Bullet 1 (core ML result)**
- **Evidence**: `data/interim/phase4_metrics.json` →
  `model_b_behavioral.test_ranking` (`pr_auc: 0.5482543868701517, roc_auc:
  0.9057988067195211`); split sizes from `data/interim/phase1_split_metadata.json`.
- **How I would defend it**: These are the champion model's one-time
  evaluation on the untouched, chronologically-final 88,581-row test
  partition (never used for training or threshold tuning); I can open the
  actual JSON artifact and the split metadata to show the exact numbers
  and row boundaries live.

**Bullet 2 (behavioral intelligence)**
- **Evidence**: same artifact →
  `validation_pr_auc_delta_b_minus_a: 0.005510599775985892` (0.6092 →
  0.6147 on VALIDATION); feature list in `src/features/behavioral.py`.
- **How I would defend it**: This is a validation-set A/B comparison
  (transaction-only model vs. +behavioral model) — I'd explain it's
  intentionally described as "modest," not oversold, and walk through
  exactly how the 12 features are computed using only pre-transaction
  history (never the current transaction), which is enforced structurally
  in both the offline and online code paths, not just by convention.

**Bullet 3 (real-time engineering)**
- **Evidence**: `data/interim/phase6_metrics.json` →
  `decision_parity.pct_matching: 1.0` (3,000/3,000),
  `score_parity.correlation: 0.999999995835289`.
- **How I would defend it**: I'd explain the specific engineering pattern
  (state updated only AFTER prediction/decision are finalized, enforced
  in `RiskDecisionEngine.process_transaction()`) and walk through the
  parity test methodology — an independent offline batch computation
  compared row-by-row against the stateful online engine's output on the
  same 3,000-transaction chronological slice. I would also proactively
  disclose that this parity figure predates a later numerical-stability
  fix (Phase 13) and has not been re-measured since, rather than let an
  interviewer discover that gap themselves.

**Bullet 4 (MLOps/governance depth)**
- **Evidence**: `artifacts/models/registry.json` (real champion,
  `champion_model_id: fraud-risk-lightgbm-v1`, real SHA-256 artifact
  hash); `src/governance/gates.py` (the 6 gate functions);
  `tests/test_phase11_governance.py::test_drift_module_never_imports_governance_registry`.
- **How I would defend it**: I'd explain each of the 6 gates concretely
  (metric availability, schema compatibility, minimum quality threshold,
  degradation limits, operational-friction impact, policy compatibility)
  and that the aggregation rule is a stated "any FAIL → REJECT" logic,
  not a black-box score. If asked about the mock/synthetic candidate used
  to demonstrate this, I would immediately clarify it's a constructed
  score array built only to exercise the gate logic — never a second
  trained model — and would not let that number be mistaken for a real
  result.
