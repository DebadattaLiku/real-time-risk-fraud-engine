# Real-Time Risk Decision & Fraud Intelligence Engine

A production-style ML platform for real-time fraud risk scoring, historical
behavioral intelligence, operational decisioning, monitoring, drift
detection, and model governance — built end-to-end on the IEEE-CIS Fraud
Detection dataset.

**This is a local, portfolio-grade engineering system, not a deployed
production service.** Every claim in this document is traceable to a real,
executed artifact in this repository — see `reports/phase14_documentation_plan.md`
for the full evidence audit behind this README.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.141-teal)
![LightGBM](https://img.shields.io/badge/LightGBM-4.7-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.63-red)
![Docker](https://img.shields.io/badge/Docker-containerized-2496ED)
![Tests](https://img.shields.io/badge/tests-316%20passing-brightgreen)

## Status

**Phases 0-14 complete.** See `reports/phase*_summary.md` for each phase's
full write-up and real, executed validation results, and
`reports/phase13_engineering_audit_summary.md` / `phase14_documentation_plan.md`
for the two most recent ones.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Key Results](#key-results)
4. [Dataset](#dataset)
5. [Leakage-Aware Evaluation](#leakage-aware-evaluation)
6. [Machine Learning Methodology](#machine-learning-methodology)
7. [Behavioral Intelligence](#behavioral-intelligence)
8. [Decision Policy](#decision-policy)
9. [Real-Time Stateful Engine](#real-time-stateful-engine)
10. [API](#api)
11. [Monitoring](#monitoring)
12. [Drift Detection](#drift-detection)
13. [Model Governance](#model-governance)
14. [Dashboard](#dashboard)
15. [Docker](#docker)
16. [Testing & CI](#testing--ci)
17. [Project Structure](#project-structure)
18. [Reproducibility / Quick Start](#reproducibility--quick-start)
19. [Engineering Decisions](#engineering-decisions)
20. [Limitations & Responsible Interpretation](#limitations--responsible-interpretation)
21. [Phase / Research Log](#phase--research-log)

---

## Project Overview

Fraud detection is a genuinely hard applied ML problem, for reasons that
shape almost every design decision in this repository:

- **Severe class imbalance.** Only 3.499% of the 590,540 transactions in
  this dataset are fraudulent. A model optimized for accuracy would
  happily predict "not fraud" every time and be right 96.5% of the time
  while catching zero fraud — this project evaluates with PR-AUC and
  recall-at-review-budget instead, which are the metrics that actually
  matter under this kind of imbalance.
- **Temporal leakage is easy to introduce by accident.** A random
  train/test split would let the model implicitly learn from the future
  to predict the past — every split, every feature, and every
  preprocessing step in this project is strictly chronological (see
  [Leakage-Aware Evaluation](#leakage-aware-evaluation)).
- **A fraud probability alone isn't an operational decision.** A risk
  score has to become an action — approve, send to a human reviewer, or
  block — and that mapping has real business tradeoffs (analyst workload
  vs. fraud caught vs. legitimate customers inconvenienced). See
  [Decision Policy](#decision-policy).
- **Fraud patterns are historical, not just transactional.** A single
  transaction in isolation is less informative than "how does this
  compare to this same card's own recent history?" — which requires
  maintaining state across transactions in real time, not just scoring
  each one independently. See [Real-Time Stateful Engine](#real-time-stateful-engine).
- **A model that works today can silently stop working.** Data
  distributions shift, and a model deployed once and never checked again
  is a liability — which is why this project includes passive drift
  monitoring and an explicit, human-gated model governance workflow, not
  just a trained model.

This repository builds all of that: a leakage-safe evaluation pipeline, a
trained LightGBM fraud model augmented with historical behavioral
features, a frozen three-way decision policy, a stateful real-time
serving engine, a FastAPI service, a Docker image, Prometheus-style
observability, PSI/KS drift detection, a local model governance registry
with promotion gates, and a Streamlit dashboard that presents all of it —
each phase independently tested and documented.

---

## System Architecture

```text
IEEE-CIS Fraud Transactions
          │
          ▼
 Chronological Data Pipeline        (Phases 0-1: EDA, leakage-safe split/pipeline)
          │
          ▼
Historical Behavioral Features      (Phase 4: bhv_* features, card1 pseudo-entity)
          │
          ▼
      LightGBM                      (Phase 2B/4: the trained champion model)
          │
          ▼
      Risk Score
          │
          ▼
    Decision Policy                 (Phase 5: frozen APPROVE/REVIEW/BLOCK thresholds)
     ┌────┼────┐
     ▼    ▼    ▼
 APPROVE REVIEW BLOCK
          │
          ▼
 ┌─────────────────────────────┐
 │ Real-Time Stateful Engine   │    (Phase 6: RiskDecisionEngine)
 │ FastAPI                     │    (Phase 7-8: API + Docker)
 │ Monitoring                  │    (Phase 9: Prometheus-style /metrics)
 │ Drift Detection             │    (Phase 10: PSI + KS vs. reference)
 │ Model Governance            │    (Phase 11: registry + promotion gates)
 │ Streamlit Dashboard         │    (Phase 12: presentation layer)
 └─────────────────────────────┘
```

**A note on Isolation Forest.** An earlier design considered fusing a
supervised model with an unsupervised anomaly detector. Phase 3 evaluated
Isolation Forest exactly this way and found it did **not** provide
meaningful complementary fraud signal (Test PR-AUC 0.0449 — only
marginally above the 0.0348 no-skill baseline for this dataset's fraud
rate, and its flagged transactions were >99% already caught by LightGBM
alone). It was explicitly **rejected** as a modeling decision, backed by
evidence, and is not part of the shipped architecture — see
[Machine Learning Methodology](#machine-learning-methodology) and
`reports/phase3_anomaly_detection_summary.md`.

### Offline vs. online

| | Offline | Online |
|---|---|---|
| **What** | Training, validation, evaluation, drift reference construction, governance evaluation | Request validation, historical state lookup, behavioral feature generation, prediction, decision, state update, monitoring |
| **When** | Once per model/policy/reference version | Every request, in real time |
| **Where** | `src/models/`, `src/decision/threshold_selection.py`, `src/run_phase10_build_reference.py`, `src/governance/evaluation.py` | `src/engine/risk_engine.py`, `src/api/main.py` |

**The current transaction is scored BEFORE its own data is added to that
entity's historical behavioral state.** This ordering — predict, decide,
*then* update state — is enforced in `RiskDecisionEngine.process_transaction()`
and is the concrete mechanism that prevents a transaction from ever
leaking into its own features. It's verified by a dedicated offline/online
parity test suite (see [Real-Time Stateful Engine](#real-time-stateful-engine)).

---

## Key Results

All figures below are the champion model's (`fraud-risk-lightgbm-v1`)
**final, one-time evaluation on the held-out chronological test set**
(88,581 transactions, never touched during training or threshold
selection) unless a row explicitly says otherwise.

| Area | Metric | Result | Evaluation Context |
|---|---|---:|---|
| Model | Test PR-AUC | **0.5483** | Full chronological test set (one-time) |
| Model | Test ROC-AUC | **0.9058** | Full chronological test set (one-time) |
| Ranking | Recall@1% review budget | **25.30%** | Full test set |
| Ranking | Recall@2% review budget | **40.90%** | Full test set |
| Ranking | Recall@5% review budget | **60.14%** | Full test set |
| Behavioral features | PR-AUC improvement vs. transaction-only model | **+0.0055** (modest but measurable) | Validation set — the basis for the go/no-go decision to keep behavioral features |
| Decisioning | Fraud captured (REVIEW + BLOCK) | **1,291 / 3,083** (41.9%) | Full test set, frozen policy |
| Decisioning | Precision among BLOCKed transactions | **84.85%** | Full test set |
| Decisioning | Legitimate customers blocked | **171 / 85,498** (0.20%) | Full test set |
| Serving | Offline/online decision parity | **3,000 / 3,000 (100%)** | Phase 6 real-time simulation, 3,000-transaction slice |
| Serving | Offline/online risk-score correlation | **0.999999996** | Same simulation |
| Testing | Automated tests passing | **316 / 316** | Latest local run (`pytest -q`) |

**On the offline/online score/decision parity above**: decisions matched
100% and scores correlated at 0.999999996, but the underlying *feature*
parity check (2,997/3,000 exact matches) surfaced one numerically unstable
outlier — documented in `reports/phase6_realtime_engine_summary.md`. That
finding was later diagnosed and fixed in Phase 13 (the real-time engine's
variance computation now uses Welford's algorithm instead of a
cancellation-prone formula — see [Testing & CI](#testing--ci) and
`reports/phase13_engineering_audit_summary.md`). **The parity numbers
above are from the original Phase 6 run and were not re-measured after
that fix** (doing so requires a fresh multi-minute real-data simulation,
not performed as part of this documentation phase) — presented here with
that context intact, not silently updated.

**On the Phase 6 simulation's own PR-AUC (0.5057)**: this is a *different,
much smaller* number from a 3,000-row subset used to validate real-time
parity — it is not, and should not be read as, the model's real
performance. The 0.5483 figure above (the full 88,581-row test set) is
the actual champion evaluation.

---

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection)
(Kaggle competition, Vesta Corporation transaction data) — anonymized,
real-world e-commerce transaction and identity data.

- **590,540 transactions**, 394 columns, **3.499% fraud rate**
- `isFraud`: the binary target
- `TransactionDT`: a relative timestamp (seconds from a reference point,
  not a calendar date) — this is what makes the data genuinely temporal
  and is why random splitting would be misleading (see next section)
- Transaction-level fields (amount, product code, card network) plus a
  large block of anonymized numeric/categorical engineered features
  (`C1-C14`, `D1-D15`, `V1-V339`, etc.) whose exact real-world meaning
  Vesta did not disclose

**`card1` is used as a pseudo-entity for historical behavioral
aggregation because the dataset does not provide a verified customer
identifier.** It is Vesta's internal card-related identifier, not a
confirmed one-to-one mapping to a real person or account — every phase
report and every behavioral-feature docstring in this codebase is
explicit about that distinction, and this README preserves it.

The raw CSVs are not committed to this repository (~1.3GB, and Kaggle
requires authenticated access) — download them from the competition page
and place them under `data/raw/` as `train_transaction.csv`,
`train_identity.csv`, `test_transaction.csv`, `test_identity.csv` before
running the Phase 0-6 pipeline from scratch. Running the API, dashboard,
or test suite does **not** require this — the trained model bundle and
all evaluation artifacts are already committed (see
[Reproducibility / Quick Start](#reproducibility--quick-start)).

---

## Leakage-Aware Evaluation

```text
Past ──────────────────────────────────────────────► Future
  │                │                │                  │
  └── TRAIN ───────┘── VALIDATION ──┘──── TEST ─────────┘
    413,378 rows      88,581 rows      88,581 rows
      (70%)             (15%)             (15%)
```

The split is by `TransactionDT`, strictly chronological — never shuffled,
never random. This matters because random splitting would let a model
implicitly "see the future" relative to any given prediction (e.g. a
later transaction's behavioral pattern informing an earlier one's
features), which does not reflect how the model would actually be used at
deployment time: scoring a transaction using only what happened *before*
it. TRAIN is used to fit the model; VALIDATION is used for all threshold
and hyperparameter selection; TEST is touched exactly once, for final
reporting.

---

## Machine Learning Methodology

**Baseline — Logistic Regression** (Phase 2A): established a simple,
interpretable baseline on transaction-level features to calibrate
expectations before investing in a more complex model.

**Champion — LightGBM** (Phase 2B/4): gradient-boosted trees, the
established strong performer for structured/tabular data with mixed
numeric and categorical features and meaningful class imbalance. Trained
on transaction-level features, then augmented with behavioral features
(below) after those were validated to help.

**Anomaly detection experiment — Isolation Forest** (Phase 3): evaluated
as a potential complementary, unsupervised signal — the hypothesis being
that an anomaly detector might catch fraud patterns the supervised model
misses. Measured directly: Test PR-AUC of 0.0449, barely above the 0.0348
no-skill baseline, with its own flagged transactions overwhelmingly
already caught by LightGBM. **Rejected** based on this evidence — not
used anywhere in the shipped system. This is a model-selection decision
demonstrated with data, not an omission.

---

## Behavioral Intelligence

Twelve `bhv_*` features computed per `card1` pseudo-entity, using **only**
that entity's transactions strictly before the current one:

| Feature | What it captures |
|---|---|
| `bhv_prev_txn_count` | How many prior transactions this entity has |
| `bhv_hist_mean_amt` | Historical mean transaction amount |
| `bhv_hist_std_amt` | Historical amount standard deviation |
| `bhv_hist_min_amt` / `bhv_hist_max_amt` | Historical amount range |
| `bhv_time_since_prev_txn` | Recency — seconds since the last transaction |
| `bhv_amt_to_hist_mean_ratio` | Current amount vs. this entity's typical amount |
| `bhv_amt_diff_from_hist_mean` | Current amount minus the historical mean |
| `bhv_amt_zscore` | How many standard deviations this amount is from history |
| `bhv_prior_count_1h` / `bhv_prior_count_24h` | Short-term transaction velocity |

**Features are computed strictly from transactions that occurred before
the current transaction being scored — never from the current transaction
itself.** This is enforced structurally (not just as a convention) in
both the offline batch computation (`src/features/behavioral.py`) and the
online stateful engine (`src/engine/state.py`, which updates state only
*after* prediction). Adding these features produced a modest but
measurable validation PR-AUC improvement of **+0.0055** over the
transaction-only model (0.6147 vs. 0.6092) — real, reproducible, and
correctly sized: this is a genuine but incremental gain, not a
transformative one, and this README does not claim otherwise.

---

## Decision Policy

```text
Risk Score
    │
    ├── < approve_threshold        → APPROVE
    ├── between thresholds          → REVIEW  (sent to a human analyst)
    └── ≥ block_threshold           → BLOCK
```

Thresholds (`approve_threshold = 0.3206`, `block_threshold = 0.6311`)
were selected on the **validation** partition only, then frozen and
evaluated exactly once on the test set — never tuned against test-set
results. On the full test set, this policy produces:

| Decision | % of traffic | Fraud rate within bucket |
|---|---:|---:|
| APPROVE | 97.91% | 2.07% |
| REVIEW | 0.82% | 46.12% |
| BLOCK | 1.27% | 84.85% |

This reflects a real, explicit business tradeoff: BLOCK is highly
precise (84.85% of blocked transactions are genuinely fraudulent) but
still lets **58.1% of all fraud through as APPROVE** (1,792 of 3,083
fraud cases) — a limitation this project states plainly rather than
hides (see [Limitations](#limitations--responsible-interpretation)).
REVIEW exists specifically to route ambiguous cases to human judgment
rather than forcing a binary automated call on every transaction.

---

## Real-Time Stateful Engine

```text
Validate
   │
   ▼
Read historical state          (existing behavioral state for this card1, read-only)
   │
   ▼
Generate behavioral features    (from that state, NOT the current transaction)
   │
   ▼
Preprocess                       (frozen Phase 1 pipeline + Phase 4 preprocessor)
   │
   ▼
Predict                            (LightGBM)
   │
   ▼
Decision                             (frozen Phase 5 policy)
   │
   ▼
Update state                          ← ONLY NOW, after prediction/decision are final
```

This ordering is the concrete anti-leakage mechanism: a transaction can
never influence the very features used to score it, because state is
read *before* any computation and written *after* the decision is already
made. Verified by a dedicated offline/online parity test suite comparing
the stateful engine's output against an independently-computed offline
batch calculation across a 3,000-transaction chronological simulation
(see [Key Results](#key-results) for the numbers, with the parity-dataset
timing caveat noted there).

---

## API

FastAPI service (`src/api/main.py`), 9 endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/predict` | POST | Score one transaction → risk score + decision |
| `/health` | GET | Service/model/policy operational status |
| `/metadata` | GET | Model type/version, policy thresholds, supported decisions |
| `/metrics` | GET | Prometheus-text-format counters/gauges |
| `/monitoring/summary` | GET | The same monitoring data as JSON |
| `/drift/summary` | GET | Latest known drift status (read-only) |
| `/drift/analyze` | POST | Run a new batch drift analysis |
| `/model-governance/summary` | GET | Current champion + registry status (read-only) |
| `/dev/reset-state` | POST | **Development-only** — clears in-memory behavioral state |

Interactive docs at `/docs` (Swagger UI) once the service is running.

**Real, executed example** (from this project's own Docker validation —
see `reports/phase13_engineering_audit_summary.md`):

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"TransactionID": 9900001, "TransactionDT": 100000, "TransactionAmt": 55.0, "card1": 77777, "ProductCD": "W"}'
```

```json
{"transaction_id":9900001,"risk_score":0.05003272790754602,"decision":"APPROVE","model_version":"phase4_lightgbm_transaction_plus_behavioral_v1","policy_version":"balanced","processing_status":"success","processing_time_ms":155.47386999998025}
```

---

## Monitoring

Prometheus-style passive observability (Phase 9) — request counts,
prediction success/failure, per-endpoint latency, decision distribution,
risk-score statistics, and validation/data-quality error counts, all
served from `GET /metrics` (text) and `GET /monitoring/summary` (JSON).
It only *observes* — it never changes a risk score, decision, or
behavioral state, verified by a dedicated non-interference test.

**These numbers reflect demonstration and test-run traffic (tens of
requests), not production-scale volume** — this project has never
received real production traffic, and this README does not imply
otherwise.

```bash
curl http://localhost:8000/metrics
curl http://localhost:8000/monitoring/summary
python scripts/demo_monitoring.py   # end-to-end walkthrough with real predictions
```

---

## Drift Detection

Passive, batch-only comparison of incoming data and model-output
distributions against an approved historical reference (Phase 10) — PSI
(Population Stability Index) as the primary measure, KS (Kolmogorov-
Smirnov) as a complementary check, both on the **VALIDATION** partition
as reference (never the untouched test set).

**Monitored**: 6 numeric transaction features, 2 categorical features
(with rare categories grouped into an `__OTHER__` bucket, and genuinely
new/unseen categories always surfaced explicitly, never silently
dropped), 4 behavioral features, plus the model's own risk-score and
decision-distribution outputs.

**Important methodological finding**: several behavioral features
(`bhv_prev_txn_count`, `bhv_hist_mean_amt`, etc.) are *cumulative*
counters that structurally grow over the dataset's timeline — an entity
seen later in the chronological order has, by construction, more prior
history than one seen earlier. A batch drawn from a later period can
therefore show real, measurable PSI movement in these features **that
reflects this structural design, not a data-quality problem.** Drift in a
behavioral feature is not automatically evidence of a broken pipeline —
see `reports/phase10_drift_detection_summary.md` for the concrete example
this project found and investigated.

```bash
python -m src.run_phase10_build_reference   # build the reference profile (once)
python scripts/demo_drift_monitoring.py      # 4 controlled scenarios, real + synthetic

curl http://localhost:8000/drift/summary
curl -X POST http://localhost:8000/drift/analyze -d '{"records": [...]}'
```

---

## Model Governance

A lightweight local framework (Phase 11) for deciding whether a candidate
model should replace the current champion — **not** an automated
deployment system.

```text
Candidate → Evaluation → Promotion Gates → Recommendation (PROMOTE / REJECT / REQUIRES_REVIEW) → Explicit, human-approved Promotion
```

The registry (`artifacts/models/registry.json`) tracks champion + candidate
metadata: model type, evaluation metrics, feature-schema/policy version,
and a **real SHA-256 hash** of the actual trained model artifact
(`bb5de8767ebaffae90a8ca634380524e2002f67d38fb87528ea5911479686342`).
Six explicit gates (metric availability, schema compatibility, minimum
quality, degradation limits, operational/friction impact, policy
compatibility) each return PASS/FAIL/REQUIRES_REVIEW with a stated
explanation — never a black-box score. No candidate is ever promoted
automatically: `promote()`/`rollback()` require an explicit
`approved_by`/`reason`, and it's structurally impossible for Phase 10's
drift detector or any API route to call them (verified by a test that
parses every drift-module source file and confirms none import the
governance module).

**Current real champion**: `fraud-risk-lightgbm-v1` — the actual approved
model documented throughout this README.

The governance *mechanism* was demonstrated end-to-end using two
constructed candidates — a deliberately weaker one and a deliberately
stronger one, both built by perturbing the champion's own real scores,
never by training a second real model. **The "stronger" demonstration
candidate's ≈0.8700 PR-AUC is a synthetic governance demonstration — not
a production/model performance result** — it exists purely to exercise
the promotion-gate code path and is not a claim about any real
model's capability. Full detail, correctly labeled, in
`reports/phase11_model_lifecycle_governance_summary.md`.

```bash
python -m src.run_phase11_register_champion   # register the real champion (once)
curl http://localhost:8000/model-governance/summary   # read-only
```

---

## Dashboard

A Streamlit dashboard (`dashboard/`, Phase 12) presenting the whole system
end-to-end — **presentation layer only**, no inference/feature/drift/
governance logic lives inside it; every page calls the real FastAPI
service or the real `RiskDecisionEngine` directly, or reads real,
already-computed artifacts.

**Pages**: Executive Overview · Live Scoring · Analytics · Monitoring ·
Drift · Governance · Architecture.

No screenshots are included here — this environment doesn't support
browser automation, and rather than fabricate images, this README points
you to run it yourself (two commands, below) or to read the actual page
source under `dashboard/pages/`.

```bash
uvicorn src.api.main:app --reload      # terminal 1 (optional but recommended)
streamlit run dashboard/app.py          # terminal 2
```

---

## Docker

```bash
docker build -t fraud-risk-api:local .
docker run -d --name fraud-risk-api -p 8000:8000 fraud-risk-api:local
docker inspect --format='{{.State.Health.Status}}' fraud-risk-api
curl http://localhost:8000/health
```

The image copies only `src/`, `config/`, `scripts/`, the trained model
bundle, and the small drift/governance artifacts — **the raw ~1.3GB
dataset is explicitly excluded** (`.dockerignore`), keeping the build
context small (~4.29MB, as measured in Phase 8). Behavioral state is
in-memory only and resets on container restart — no persistent storage in
this phase, documented plainly rather than glossed over. This is a local
containerized deployment for development/demonstration, **not a cloud or
production deployment**.

---

## Testing & CI

**316 automated tests, 0 failures** (latest local `pytest -q` run) across
data validation, leakage guards, feature correctness, model training,
decision policy, the real-time engine, the API, monitoring, drift
detection, governance, and the dashboard. A GitHub Actions workflow
(`.github/workflows/tests.yml`, added in Phase 13) runs the full suite on
every push — verified to require only what's committed to the repository,
not the raw dataset.

Phase 13 also found and fixed a real numerical-stability bug: the
real-time engine's online variance formula could produce a **negative
variance → `NaN` standard deviation** for real-shaped, high-volume,
low-variance entities (a catastrophic-cancellation failure mode). Fixed
with Welford's online algorithm; a dedicated regression test confirms the
fix's accuracy against `numpy`'s ground truth to within `~5e-12`, versus
the old formula's outright failure on the same adversarial input. Full
detail in `reports/phase13_engineering_audit_summary.md`.

This is not a claim of 100% code coverage — it's an honest count of what
actually passes, today, in this repository.

---

## Project Structure

```text
fraud-risk-engine/
├── config/                 # run configuration, frozen decision policy
├── data/
│   ├── raw/                # untouched source CSVs (not committed)
│   └── interim/            # trained model bundle, saved evaluation metrics
├── artifacts/               # drift reference profile, model governance registry
├── src/
│   ├── data/                 # loading, inspection, chronological splitting
│   ├── features/               # leakage-safe pipeline + behavioral features
│   ├── models/                  # Logistic Regression, LightGBM, Isolation Forest
│   ├── decision/                  # decision policy + threshold selection
│   ├── engine/                     # real-time RiskDecisionEngine + state manager
│   ├── api/                         # FastAPI service
│   ├── monitoring/                   # observability
│   ├── drift/                         # PSI/KS drift detection
│   └── governance/                     # model lifecycle governance
├── dashboard/                # Streamlit dashboard (presentation layer only)
├── tests/                    # 316 automated tests
├── scripts/                  # demonstration and utility scripts
├── reports/                  # every phase's full write-up + real validation logs
└── .github/workflows/        # CI (test suite on every push)
```

---

## Reproducibility / Quick Start

```bash
git clone <this-repo>
cd fraud-risk-engine

pip install -r requirements.txt   # exact pinned versions

pytest -q                          # 316 tests — no raw dataset required

uvicorn src.api.main:app --reload   # start the API (terminal 1)
streamlit run dashboard/app.py       # start the dashboard (terminal 2)
```

The trained model bundle, evaluation metrics, drift reference profile,
and governance registry are all committed (small, a few MB total) —
running the tests, API, or dashboard needs nothing beyond the two
commands above. **Only re-running the full Phase 0-6 pipeline from
scratch** (to retrain from raw data) requires downloading the Kaggle
dataset yourself into `data/raw/` (see [Dataset](#dataset)).

---

## Engineering Decisions

| Decision | Why |
|---|---|
| Chronological split (never random) | Prevents temporal leakage; approximates real deployment conditions |
| PR-AUC as the primary ranking metric | ROC-AUC is misleadingly optimistic under 3.5% fraud prevalence |
| LightGBM as the champion model | Strong, well-established performance on structured/tabular data with mixed feature types |
| Behavioral features (`card1` pseudo-entity) | Captures historical transaction patterns a single transaction alone can't show |
| Isolation Forest evaluated and rejected | Measured, not assumed, to add negligible complementary fraud value |
| Three-way APPROVE/REVIEW/BLOCK policy | Connects a continuous risk score to a real operational action, with a human-review path |
| Stateful real-time engine (predict-then-update) | Structurally prevents a transaction from leaking into its own features |
| PSI + KS drift detection | Two different, complementary lenses on distributional change |
| Explicit, human-gated governance promotion | Prevents any code path — automated or accidental — from silently swapping the production model |
| Local JSON registry, not a full MLOps platform | Proportionate to this project's actual scale; documented as a lightweight, not enterprise, tool |

---

## Limitations & Responsible Interpretation

- **Anonymized historical dataset, not live production traffic.** Every
  result in this README comes from chronological backtesting on a static,
  already-labeled Kaggle dataset — the system has never scored real,
  live transactions.
- **`card1` is a pseudo-entity, not a verified customer identity** — see
  [Dataset](#dataset).
- **58.1% of all test-set fraud is still approved**, not caught by
  REVIEW/BLOCK — a real, stated limitation of the current policy, not
  hidden behind the strong blocked-precision number.
- **No ground-truth-based online accuracy monitoring.** Drift detection
  compares distributions, not true fraud outcomes (which aren't available
  at prediction time) — it cannot by itself confirm the model is "still
  accurate."
- **Monitoring and drift numbers reflect demo/test-scale traffic**, not
  production volume.
- **The dashboard is a demonstration/presentation interface**, not a
  production operations console, and has no authentication.
- **The governance ≈0.8700 PR-AUC candidate is a synthetic
  demonstration** — see [Model Governance](#model-governance). It is not,
  and must never be read as, a real model result.
- **No authentication anywhere in the service.** Every endpoint is
  reachable by anyone who can reach the process — acceptable for a local
  portfolio system, a real gap for any public deployment.
- **Offline/online parity figures predate a later numerical-stability
  fix** — see [Key Results](#key-results).

---

## Phase / Research Log

| Phase | Focus | Report |
|---|---|---|
| 0 | Leakage-aware EDA | `reports/phase0_eda_summary.md` |
| 1 | Chronological split, leakage-safe pipeline | `reports/phase1_pipeline_summary.md` |
| 2A | Logistic Regression baseline | `reports/phase2a_logistic_regression_summary.md` |
| 2B | LightGBM champion | `reports/phase2b_lightgbm_summary.md` |
| 3 | Isolation Forest complementarity (rejected) | `reports/phase3_anomaly_detection_summary.md` |
| 4 | Behavioral intelligence features | `reports/phase4_behavioral_features_summary.md` |
| 5 | Decision policy | `reports/phase5_decision_policy_summary.md` |
| 6 | Stateful real-time engine | `reports/phase6_realtime_engine_summary.md` |
| 7 | FastAPI service | `reports/phase7_api_service_summary.md` |
| 8 | Docker containerization | `reports/phase8_containerization_summary.md` |
| 9 | Monitoring & observability | `reports/phase9_monitoring_observability_summary.md` |
| 10 | Drift detection | `reports/phase10_drift_detection_summary.md` |
| 11 | Model lifecycle governance | `reports/phase11_model_lifecycle_governance_summary.md` |
| 12 | Streamlit dashboard | `reports/phase12_dashboard_summary.md` |
| 13 | Engineering/reproducibility audit | `reports/phase13_engineering_audit_summary.md` |
| 14 | Portfolio documentation | `reports/phase14_documentation_plan.md` (this README) |

---

## Principles this project follows

- No random shuffling for train/val/test splits — strictly time-ordered.
- No behavioral/aggregate feature is computed using information from
  transactions that occur after the transaction being scored.
- No preprocessing object (scaler, encoder, imputer) is fit on anything but
  the training partition.
- No modeling decision is informed by the test partition.
- Any claim that the hybrid system outperforms a baseline must be backed by
  same-split, same-protocol, reproducible experiments — not asserted in advance.
