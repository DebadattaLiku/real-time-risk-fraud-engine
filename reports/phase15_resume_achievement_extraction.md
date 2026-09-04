# Phase 15 — Resume Achievement Extraction

Evidence-only extraction pass over the completed Phases 0-14 repository.
No resume bullets are finalized here — this is raw material plus
positioning analysis. Every number below was re-verified against a real
artifact in this session (not carried over from memory of earlier
reports).

## Executive Summary

This project's strongest resume material is not "built a fraud
classifier" — it's the **full lifecycle**: a leakage-safe evaluation
protocol, a modest-but-proven feature-engineering win, an operational
decision policy with quantified tradeoffs, a stateful real-time engine
with verified offline/online parity, and a genuine MLOps layer
(monitoring, drift detection, model governance) built and tested end to
end, including a real bug found and fixed with a quantified proof. The
project supports distinct, honest angles for AI/ML Engineer, Data
Scientist, and MLOps Engineer resumes; a Quant/Finance angle is weak and
should not be forced (see Section 7). The single most differentiating
material — most candidates show a trained model; this project shows a
trained model PLUS the governance/observability discipline around it,
demonstrated with real numbers rather than asserted.

---

## Quantitative Achievement Inventory

| Category | Metric / Achievement | Exact Result | Context | Evidence |
|---|---|---:|---|---|
| Model performance | LightGBM (+behavioral) Test PR-AUC | 0.5483 | Full chronological test set, one-time | `data/interim/phase4_metrics.json` |
| Model performance | LightGBM (+behavioral) Test ROC-AUC | 0.9058 | Full test set | same |
| Model performance | Logistic Regression baseline Test PR-AUC | 0.1604 | Full test set | `reports/phase2a_logistic_regression_summary.md` |
| Model performance | Logistic Regression baseline Test ROC-AUC | 0.7940 | Full test set | same |
| Model performance | Transaction-only LightGBM Test PR-AUC (pre-behavioral) | 0.5428 | Full test set | `data/interim/phase4_metrics.json` |
| Model performance | Behavioral-feature PR-AUC improvement | +0.0055 (0.6092 → 0.6147) | Validation set, decision basis | same |
| Model performance | Recall@1% review budget | 25.30% | Full test set | same |
| Model performance | Recall@2% review budget | 40.90% | Full test set | same |
| Model performance | Recall@5% review budget | 60.14% | Full test set | same |
| Model performance | Precision@1% / @2% / @5% | 88.04% / 71.16% / 41.85% | Full test set | same |
| Model selection | Isolation Forest evaluated + rejected | Test PR-AUC 0.0449 vs. 0.0348 no-skill | Full test set | `reports/phase3_anomaly_detection_summary.md` |
| Operational decisioning | Fraud captured (REVIEW+BLOCK) | 1,291 / 3,083 (41.9%) | Full test set, frozen policy | `data/interim/phase5_metrics.json` |
| Operational decisioning | Fraud captured by BLOCK alone | 958 / 3,083 (31.1%) | Full test set | same |
| Operational decisioning | Precision among BLOCKed | 84.85% | Full test set | same |
| Operational decisioning | Legitimate customers blocked | 171 / 85,498 (0.20%) | Full test set | same |
| Operational decisioning | Decision distribution | APPROVE 97.91% / REVIEW 0.82% / BLOCK 1.27% | Full test set | same |
| Operational decisioning | Fraud missed in APPROVE | 1,792 / 3,083 (58.1%) | Full test set — honest limitation | same |
| Real-time engineering | Offline/online decision parity | 3,000 / 3,000 (100%) | Real-time simulation | `data/interim/phase6_metrics.json` |
| Real-time engineering | Offline/online score correlation | 0.999999996 | Same simulation | same |
| Real-time engineering | Offline/online feature parity | 2,997 / 3,000 (99.9%) exact match | Same simulation, predates Phase 13 fix | same |
| MLOps — monitoring | Passive observability layer | Request/prediction/decision/risk-score/error counters, Prometheus-text `/metrics` | Real, tested | `src/monitoring/`, `reports/phase9_monitoring_observability_summary.md` |
| MLOps — drift | PSI + KS batch drift detection | 6 numeric + 2 categorical + 4 behavioral signals + outputs | VALIDATION-partition reference (88,581 txns) | `src/drift/`, `artifacts/drift/reference_profile.json` |
| MLOps — drift | Categorical PSI stability fix | Naive PSI 0.78 → 0.19-0.31 after rare-category grouping | Real finding during Phase 10 validation | `reports/phase10_drift_detection_summary.md` |
| MLOps — governance | Local model registry with 6 explainable promotion gates | PASS/FAIL/REQUIRES_REVIEW, never a black-box score | `src/governance/gates.py` | `reports/phase11_model_lifecycle_governance_summary.md` |
| MLOps — governance | Real champion artifact SHA-256 hashing | `bb5de876...` | Real file hash | `artifacts/models/registry.json` |
| MLOps — governance | Structural drift/governance isolation | Verified via AST-parsing every `src/drift/*.py` import statement | Test-enforced, not just documented | `tests/test_phase11_governance.py::test_drift_module_never_imports_governance_registry` |
| Engineering quality | Automated test count | 316 / 316 passing | Live `pytest -q` run | this session |
| Engineering quality | Numerical-stability bug found + fixed | Old formula: negative variance → NaN; Welford's: error ≈ 4.9e-12 vs. numpy ground truth | Real adversarial synthetic test | `reports/phase13_engineering_audit_summary.md`, `tests/test_phase6_state.py` |
| Engineering quality | Docker build context | ~4.29MB (raw ~1.3GB dataset excluded) | Measured Phase 8 | `reports/phase8_containerization_summary.md` |
| Engineering quality | CI | GitHub Actions, runs full suite on every push | Added Phase 13 | `.github/workflows/tests.yml` |
| Engineering quality | Dependency-declaration audit | Found + fixed an undeclared direct import (`starlette`) | Real audit finding | `reports/phase13_engineering_audit_summary.md` |
| Presentation | Streamlit dashboard | 7 pages, zero duplicated inference/feature/drift/governance logic | Live-smoke-tested (HTTP 200 on all pages, no exceptions) | `reports/phase12_dashboard_summary.md` |

---

## Top 10 Resume Achievements

**1. End-to-end fraud risk platform spanning modeling → governance**
- *Why it matters*: shows systems thinking beyond a Jupyter notebook —
  most candidates show a model; this shows the discipline around one.
- *Technical depth*: 14 phases, each independently tested (316 tests
  total), from EDA through model governance.
- *Quantitative evidence*: 316/316 tests passing across 22 test modules
  covering every layer.
- *Resume potential*: very strong as a single headline bullet or as the
  project title/framing itself.

**2. Leakage-safe chronological evaluation discipline**
- *Why it matters*: temporal leakage is one of the most common,
  highest-impact mistakes in applied fraud/risk ML — catching it (or
  never introducing it) is a real signal of rigor.
- *Technical depth*: strict `TransactionDT`-ordered 70/15/15 split (never
  shuffled), no preprocessing object fit outside TRAIN, no threshold
  selected on TEST.
- *Quantitative evidence*: 413,378 / 88,581 / 88,581 row split; TEST
  touched exactly once.
- *Resume potential*: strong, especially framed as "designed a
  leakage-safe evaluation protocol," which reads as senior-level judgment.

**3. Behavioral feature engineering with a measured, honest lift**
- *Why it matters*: shows the ability to design a feature engineering
  approach AND to evaluate it rigorously rather than just assume it
  helped.
- *Technical depth*: 12 `bhv_*` features per `card1` pseudo-entity,
  strictly computed from pre-current-transaction history only.
- *Quantitative evidence*: +0.0055 validation PR-AUC (0.6092 → 0.6147) —
  real, modest, correctly-sized claim.
- *Resume potential*: moderate-strong; best paired with the real-time
  engineering achievement below, since the interesting part is serving
  these features correctly in real time, not just computing them offline.

**4. Real-time stateful inference engine with verified parity**
- *Why it matters*: this is the piece that separates "trained a model"
  from "built a system that could actually serve it correctly" —
  predict-before-update ordering is a genuine anti-leakage engineering
  pattern, not just an ML concept.
- *Technical depth*: `RiskDecisionEngine` maintains per-entity behavioral
  state, updated only after prediction/decision are final; validated
  against an independent offline computation.
- *Quantitative evidence*: 100% decision parity, 0.999999996 score
  correlation across a 3,000-transaction simulation.
- *Resume potential*: very strong for ML Engineer/MLOps roles —
  "real-time" claims are common; a verified parity number backing it up
  is not.

**5. Operational decision policy connecting a score to a business action**
- *Why it matters*: shows the ability to translate a probability into an
  actual operational workflow with quantified tradeoffs — exactly what a
  hiring manager wants to see beyond "I trained a model."
- *Technical depth*: 3-way APPROVE/REVIEW/BLOCK policy, thresholds
  selected on VALIDATION only, evaluated once on TEST.
- *Quantitative evidence*: 84.85% precision among BLOCKed transactions,
  41.9% fraud capture at REVIEW+BLOCK, only 0.20% of legitimate customers
  blocked.
- *Resume potential*: strong for Data Science angle — this is where
  business-tradeoff fluency shows.

**6. Model governance framework with explainable promotion gates**
- *Why it matters*: MLOps interviewers specifically probe "how do you
  decide whether to ship a new model" — most candidates have no concrete
  answer; this project has a working one.
- *Technical depth*: 6 independently-explainable gates
  (PASS/FAIL/REQUIRES_REVIEW), real artifact SHA-256 hashing, explicit
  human-approved promotion/rollback only.
- *Quantitative evidence*: real champion hash
  `bb5de8767ebaffae90a8ca634380524e2002f67d38fb87528ea5911479686342`;
  structural test proving the drift module cannot import the governance
  module.
- *Resume potential*: very strong, differentiating material for MLOps
  roles specifically.

**7. Passive drift detection with a genuinely diagnosed methodology bug**
- *Why it matters*: shows statistical maturity — not just "I ran PSI" but
  "I found PSI was giving a misleading signal and fixed the methodology."
- *Technical depth*: PSI + KS, reference-quantile binning, rare-category
  grouping to avoid long-tail sampling-noise inflation.
- *Quantitative evidence*: a real finding — naive categorical PSI of 0.78
  (falsely flagged as HIGH_DRIFT on stable data) traced to a 55-category
  long tail, fixed to a representative 0.19-0.31 via standard rare-bucket
  grouping.
- *Resume potential*: strong, and a great interview story (see Interview
  Hooks).

**8. Found and fixed a real numerical-stability bug with quantified proof**
- *Why it matters*: this is concrete, senior-level engineering — most
  candidates can't point to a specific numerical bug they found and
  fixed with a before/after measurement.
- *Technical depth*: replaced a catastrophic-cancellation-prone variance
  formula with Welford's online algorithm + Chan's parallel-merge.
- *Quantitative evidence*: old formula produced a **negative variance →
  NaN** on an adversarial (but realistic) input; the fix achieves error
  ≈4.9e-12 against `numpy`'s ground truth.
- *Resume potential*: excellent as an interview story; moderate as a
  standalone bullet (needs a sentence of setup to land).

**9. Production-style API + Docker + CI, fully tested**
- *Why it matters*: shows the ability to ship, not just prototype.
- *Technical depth*: 9-endpoint FastAPI service, Docker image with
  dataset explicitly excluded from the build context, GitHub Actions CI.
- *Quantitative evidence*: ~4.29MB build context (vs. ~1.3GB raw
  dataset), 316 tests running in CI on every push.
- *Resume potential*: strong supporting material, especially for
  MLOps/ML Engineer angles.

**10. Full-stack presentation layer (dashboard) with zero logic duplication**
- *Why it matters*: shows the discipline to build a UI that calls the
  real system rather than reimplementing it — a common anti-pattern this
  project explicitly avoided and tested for.
- *Technical depth*: 7-page Streamlit dashboard, every page backed by the
  real API or a direct call to the real engine.
- *Quantitative evidence*: real Docker + Streamlit smoke test — all pages
  return HTTP 200, zero exceptions in server logs.
- *Resume potential*: moderate; good supporting bullet, not a headline.

---

## Strongest Project Narrative

```text
Fraud Classification
       │
       ▼
Leakage-Aware Evaluation        ← establishes the evaluation is trustworthy
       │
       ▼
Behavioral Intelligence          ← shows feature engineering + honest evaluation
       │
       ▼
Operational Decision Policy       ← connects ML output to a business action
       │
       ▼
Stateful Real-Time Inference       ← proves the system, not just the model, works
       │
       ▼
API + Docker                        ← ships it
       │
       ▼
Monitoring                           ← observes it running
       │
       ▼
Drift Detection                       ← knows when the world changes
       │
       ▼
Model Governance                       ← controls how it changes safely
```

**Why this is stronger than "built a fraud classifier"**: any candidate
can claim to have trained a fraud model. Far fewer can show the full
progression from "is this evaluation even trustworthy" through "does this
system update its own state safely in real time" to "how would a new
model get promoted, and how would you know if the current one degraded."
This progression is also literally how a real fraud-risk team's project
would be scoped over its first year — it reads as an accurate mental
model of the problem domain, not just an ML exercise with extra steps
tacked on. The narrative should be told in that order in any
project-summary paragraph, not reordered to lead with the flashiest
piece (governance/dashboard) at the expense of the foundation
(leakage-safety) that makes everything after it credible.

---

## AI/ML Engineer Resume Angle

Prioritize: modeling, feature engineering, behavioral intelligence,
real-time inference.

- LightGBM fraud model (Test PR-AUC 0.5483, ROC-AUC 0.9058) trained on a
  leakage-safe, chronologically-split 590K-row dataset
- Designed and validated 12 historical behavioral features per entity,
  measurably improving validation PR-AUC (+0.0055)
- Built a stateful real-time inference engine enforcing predict-before-
  state-update ordering, verified via 100% offline/online decision parity
  across a 3,000-transaction simulation
- Evaluated and rejected Isolation Forest as a complementary signal based
  on evidence (PR-AUC 0.0449 vs. 0.0348 no-skill) rather than by
  assumption

## Data Science Resume Angle

Prioritize: business problem, PR-AUC, recall-at-budget, decision
tradeoffs, drift analysis.

- Framed fraud detection as a severe class-imbalance problem (3.499% base
  rate) and selected PR-AUC/recall-at-review-budget over accuracy/ROC-AUC
  accordingly
- Recall@1%/2%/5% review budgets of 25.30%/40.90%/60.14%, translating
  model output into staffing-relevant operational metrics
- Designed a 3-way decision policy (APPROVE/REVIEW/BLOCK) balancing
  84.85% block precision against a 0.20% legitimate-customer intervention
  rate, with thresholds selected on validation data only
- Applied PSI/KS distributional drift analysis and diagnosed a real
  methodology flaw (naive categorical PSI inflated by long-tail rare
  categories), fixing it via rare-category grouping

## MLOps Resume Angle

Prioritize: FastAPI, Docker, monitoring, drift, governance, CI, lifecycle.

- Shipped a 9-endpoint FastAPI fraud-scoring service in a Docker image
  with the ~1.3GB raw dataset explicitly excluded from the ~4.29MB build
  context
- Built a Prometheus-style observability layer (request/prediction/
  decision/error metrics) verified to have zero interference with
  inference behavior via dedicated non-interference tests
- Designed a local model governance registry with 6 independently-
  explainable promotion gates, real artifact SHA-256 hashing, and
  human-approved-only promotion/rollback — structurally isolated from the
  drift subsystem (verified via AST-based import analysis, not just code
  review)
- Diagnosed and fixed a real numerical-stability bug (catastrophic
  cancellation causing NaN variance) using Welford's algorithm, backed by
  a quantified before/after regression test (error reduced to ~4.9e-12)
- Maintained 316 passing automated tests with CI running the full suite
  on every push

## Quant / Finance Resume Angle

**Not recommended as a primary framing — include only with explicit
caveats, if at all.** This project is a fraud-detection engineering
system, not a quantitative finance project: there is no pricing,
portfolio construction, risk-factor modeling, market microstructure, or
P&L-linked decisioning anywhere in this codebase. The word "risk" in the
project's title refers to fraud-transaction risk scoring, not financial/
market risk — a Quant resume reviewer would likely (correctly) see this
distinction immediately, and forcing the connection risks reading as
either confused about the target role or as padding. **If** a candidate
is applying to a fraud/credit-risk-adjacent quant role specifically (e.g.
a bank's fraud analytics or credit-risk-adjacent team, not a trading
desk), the operational decision-policy tradeoff work (Section: Data
Science angle) is the only genuinely relevant material — the model
governance and statistical rigor (chronological evaluation, drift
methodology) could be mentioned as supporting evidence of quantitative
discipline, but should not be framed as "quant finance" work.

---

## Claims to Avoid

- **Synthetic governance candidate PR-AUC ≈ 0.8700**: this number comes
  from a constructed, label-informed blend of the champion's own real
  scores — built solely to exercise the promotion-gate code path, never a
  trained model. Using it as a headline number (even implicitly, e.g. "improved
  model performance to 0.87 PR-AUC") would be a materially false claim.
  If the governance system is mentioned at all, cite the mechanism (6
  explainable gates, real hashing, human-gated promotion) and the REAL
  champion's real PR-AUC (0.5483), never the mock number.
- **Simulation-subset PR-AUC ≈ 0.5057**: this is a 3,000-row subset used
  only to validate real-time parity, not a model-quality metric — the
  real, full-test-set PR-AUC (0.5483, on 88,581 rows) is the only number
  that should ever be quoted as "model performance."
- **Offline/online parity without its timing caveat**: the 99.9%
  feature-parity / 100% decision-parity numbers were measured BEFORE the
  Phase 13 numerical-stability fix. It is accurate to cite them as "Phase
  6 validated," but claiming the CURRENT system achieves exactly these
  numbers would be an unverified claim — no fresh parity run has been
  performed since the fix.
- **Demo-scale monitoring traffic**: every "requests observed" number in
  this project comes from test/demo runs of tens of requests. Do not
  imply the monitoring system has been exercised at production volume or
  under production load.
- **Claims implying actual production deployment**: this system has never
  received live transaction traffic. "Production-style," "production-
  grade engineering," and "deployment-ready" are accurate and safe;
  "deployed to production," "in production," or "processes live
  transactions" are not.
- **`card1` as a customer ID**: it is Vesta's internal, anonymized
  card-related identifier, not a verified one-to-one customer mapping.
  Any resume language about "customer behavioral profiles" should instead
  say "entity-level" or "card-level" behavioral features.
- **Isolation Forest as part of the final architecture**: it was
  evaluated and explicitly rejected (Section 2). It belongs in resume
  material as evidence of rigorous model selection, never as a component
  of "the system I built" in present tense.
- **Unsupported latency/scalability claims**: this project measured
  individual request processing time in demonstration/test runs (e.g.
  ~155ms for one real, logged Docker prediction) but never ran a load
  test, concurrency benchmark, or throughput measurement. Do not claim a
  requests-per-second figure, a P99 latency under load, or "scales to X
  transactions" — none of that was measured.

---

## Technology Stack

| Layer | Technologies |
|---|---|
| Language | Python 3.12 |
| ML | LightGBM 4.7, scikit-learn 1.8 (Logistic Regression, Isolation Forest, preprocessing), pandas 3.0, NumPy 2.4, SciPy 1.17 (KS test) |
| API | FastAPI 0.141, Starlette 1.6, Uvicorn 0.52, Pydantic 2.13 |
| Dashboard | Streamlit 1.63 |
| Monitoring | Custom Prometheus-text-format exporter (stdlib-based, no external metrics library) |
| Drift | Custom PSI/KS implementation (SciPy-backed) |
| Governance | Custom JSON-file-backed local registry (no external MLOps platform) |
| Deployment | Docker |
| Testing/CI | pytest 9.1, `responses` (HTTP mocking), GitHub Actions |

Only technologies with a direct, verified import or configuration file in
this repository are listed — no framework or platform not actually used
appears here.

---

## Interview Hooks

**1. Chronological split**
Decision → split by `TransactionDT`, never shuffled. Why → random
splitting would let the model implicitly learn from future transactions.
Tradeoff → smaller effective training set per fold if ever cross-
validating (not done here — single split only). Result → 413,378/88,581/
88,581 row split, TEST touched exactly once.

**2. PR-AUC over ROC-AUC/accuracy**
Decision → PR-AUC as the primary ranking metric. Why → ROC-AUC is
misleadingly optimistic under 3.499% fraud prevalence; accuracy is
useless under this imbalance. Tradeoff → PR-AUC is harder to interpret
intuitively than accuracy for non-technical stakeholders. Result → Test
PR-AUC 0.5483 vs. ROC-AUC 0.9058 on the same model — illustrates exactly
why the distinction matters.

**3. LightGBM over deep learning**
Decision → gradient-boosted trees, not a neural network. Why →
established strong performance on structured/tabular data with mixed
numeric/categorical features at this dataset scale; faster to train and
iterate. Tradeoff → less capacity for automatic feature interaction
discovery than some deep architectures. Result → 0.9058 ROC-AUC with
interpretable feature importances (`reports/figures/phase2b_feature_importance.png`).

**4. Isolation Forest rejection**
Decision → evaluated, then explicitly excluded from the final system.
Why → measured, not assumed: Test PR-AUC 0.0449, barely above the 0.0348
no-skill baseline, and its flagged transactions were overwhelmingly
already caught by LightGBM. Tradeoff → gave up a hoped-for
"complementary" signal. Result → a documented, evidence-based negative
result — arguably a stronger engineering signal than a positive one.

**5. Historical behavioral features + the pseudo-entity caveat**
Decision → 12 features aggregated per `card1`, explicitly labeled a
pseudo-entity, not a verified customer ID. Why → the dataset provides no
true customer identifier; overclaiming here would be a real integrity
issue. Tradeoff → behavioral aggregation may mix multiple real
customers/cards under one `card1` value, adding noise. Result → still a
real, measurable +0.0055 validation PR-AUC improvement despite that
noise.

**6. Threshold selection on validation only**
Decision → decision-policy thresholds derived exclusively from
VALIDATION, evaluated once on TEST. Why → repeatedly tuning against TEST
results is a subtle form of leakage (implicit overfitting to the
"held-out" set). Tradeoff → validation-selected thresholds may be
slightly suboptimal on the true test distribution. Result → 84.85%
block precision confirmed on a TEST set the thresholds never saw.

**7. Predict-before-state-update ordering**
Decision → `RiskDecisionEngine` updates behavioral state only AFTER
prediction/decision are finalized. Why → the alternative (update-then-
predict) would let a transaction leak into its own features. Tradeoff →
requires careful method sequencing/API discipline in the engine — an easy
mistake to introduce accidentally. Result → verified via a dedicated
parity test suite, not just asserted in a docstring.

**8. Welford's algorithm for online variance**
Decision → replaced `sum_sq_amt - count*mean²` with Welford's
incremental algorithm in the real-time engine. Why → the naive formula
subtracts two large, nearly-equal numbers for high-volume, low-variance
entities, losing precision catastrophically. Tradeoff → the offline batch
computation (`src/features/behavioral.py`) was deliberately left
unchanged — Welford's is inherently incremental and a full vectorized
rewrite there was judged out of proportion for the audit's scope. Result
→ old formula produced a NEGATIVE variance (NaN std) on a realistic
adversarial case; the fix achieves ~4.9e-12 error vs. `numpy` ground
truth.

**9. PSI vs. KS as complementary, not redundant, drift measures**
Decision → run both, not just one. Why → PSI is the standard,
interpretable severity-banded measure but is coarse (10 bins); KS is
sensitive to distributional shape differences PSI's binning can miss.
Tradeoff → two numbers to reconcile per feature instead of one. Result →
both used in every drift report, each contributing a genuinely different
signal.

**10. Governance gates as explainable, not a black-box score**
Decision → 6 independent PASS/FAIL/REQUIRES_REVIEW gates, aggregated by a
stated "worst-gate-wins" rule, not a weighted composite score. Why → a
promotion/rejection decision needs to be auditable — "why was this
rejected" must have a specific, nameable answer. Tradeoff → less
flexible than a single tunable score; can't easily express "great in one
dimension, compensates for weak in another." Result → every real
governance report in this project names the exact gate(s) responsible for
its recommendation (e.g. "Rejected: failed gate(s):
no_unacceptable_degradation").

---

## Recommended Resume Strategy

1. **Lead with the full-lifecycle framing**, not just the model — the
   differentiation is the governance/observability/real-time discipline
   around a fraud model, not the fraud model alone. A single strong
   project bullet block should touch: leakage-safe evaluation → real
   champion metrics → real-time engine with verified parity → MLOps
   layer (monitoring/drift/governance) → test/CI rigor.
2. **Use the real champion's numbers only** (PR-AUC 0.5483, ROC-AUC
   0.9058, Recall@1/2/5% 25.30/40.90/60.14%) as the headline quantitative
   proof — never the mock governance number, never the simulation-subset
   number.
3. **Tailor by resume type** using the four angle sections above —
   the same underlying project, different emphasis, all still accurate.
4. **Reserve 2-3 bullets, not more, for MLOps-specific material** (drift/
   governance/monitoring) even on an MLOps-focused resume — depth in the
   interview conversation matters more than bullet count, and the
   Interview Hooks section above gives ample material to expand on
   verbally once a conversation starts.
5. **Keep the numerical-stability-bug story in reserve for interviews**
   specifically, not necessarily as a standalone bullet — it's a strong
   verbal story ("tell me about a bug you found") but needs a sentence of
   setup that doesn't fit cleanly into bullet-point format.
6. **Never let a bullet's phrasing outrun the evidence** — every caution
   in Section "Claims to Avoid" should be treated as a hard constraint
   when Part 2 drafts actual bullets, not a soft suggestion.
