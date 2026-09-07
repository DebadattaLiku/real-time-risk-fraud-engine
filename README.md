# Real-Time Risk Decision & Fraud Intelligence Engine

A production-style machine learning platform for **real-time fraud risk scoring, historical behavioral intelligence, operational decisioning, explainability, monitoring, drift detection, and model governance**, built end-to-end on the IEEE-CIS Fraud Detection dataset.

> **Portfolio scope:** This is a local, portfolio-grade engineering system rather than a deployed production service. Every reported metric is backed by an executed experiment, benchmark, test, or artifact in this repository.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.141-teal)
![LightGBM](https://img.shields.io/badge/LightGBM-4.7-green)
![Redis](https://img.shields.io/badge/Redis-stateful-red)
![Kafka](https://img.shields.io/badge/Kafka-streaming-black)
![SHAP](https://img.shields.io/badge/SHAP-explainability-purple)
![MLflow](https://img.shields.io/badge/MLflow-MLOps-blue)
![Docker](https://img.shields.io/badge/Docker-containerized-2496ED)
![Kubernetes](https://img.shields.io/badge/Kubernetes-manifests-326CE5)
![Tests](https://img.shields.io/badge/tests-357%20passing-brightgreen)

## Status

**Phases 0–15 complete**, including the production-infrastructure upgrade:

- Leakage-aware ML evaluation
- LightGBM fraud-risk modeling
- Historical behavioral intelligence
- APPROVE / REVIEW / BLOCK decisioning
- Stateful real-time inference
- Redis-backed state
- Kafka streaming architecture
- SHAP explainability
- FastAPI serving
- Docker containerization
- Prometheus-style monitoring
- PSI / KS drift detection
- Model governance and promotion gates
- MLflow experiment tracking and model registry
- Kubernetes deployment manifests
- AWS deployment architecture mapping
- Failure testing
- Latency and throughput benchmarking
- Streamlit operational dashboard
- 357 automated tests passing, 1 correctly skipped

See `reports/production_readiness.md` for the consolidated production-upgrade assessment.

## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Key Results](#key-results)
- [Production Infrastructure](#production-infrastructure)
- [Dataset](#dataset)
- [Leakage-Aware Evaluation](#leakage-aware-evaluation)
- [Machine Learning](#machine-learning)
- [Behavioral Intelligence](#behavioral-intelligence)
- [Decision Engine](#decision-engine)
- [Real-Time Stateful Inference](#real-time-stateful-inference)
- [Kafka Streaming](#kafka-streaming)
- [Redis State](#redis-state)
- [SHAP Explainability](#shap-explainability)
- [API](#api)
- [Monitoring](#monitoring)
- [Drift Detection](#drift-detection)
- [MLflow & Model Governance](#mlflow--model-governance)
- [Docker](#docker)
- [Kubernetes](#kubernetes)
- [Dashboard](#dashboard)
- [Performance Benchmarks](#performance-benchmarks)
- [Failure Testing](#failure-testing)
- [Testing & CI](#testing--ci)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Engineering Decisions](#engineering-decisions)
- [Limitations](#limitations)
- [Reproducibility](#reproducibility)

## Project Overview

Fraud detection is not simply a classification problem.

A useful fraud-risk system must address:

1. **Severe class imbalance** — only 3.499% of the 590,540 transactions are fraudulent.
2. **Temporal leakage** — future information must never influence historical predictions.
3. **Behavioral context** — transaction risk depends on an entity's historical activity.
4. **Operational decisioning** — a probability must become an action.
5. **State management** — behavioral features must be available during online inference.
6. **Explainability** — high-risk decisions should have interpretable contributing factors.
7. **Observability** — production systems need request, latency, decision, and error monitoring.
8. **Distribution shift** — changing transaction behavior can degrade model reliability.
9. **Model governance** — model replacement should require explicit evaluation and human approval.
10. **Failure handling** — state stores, duplicate messages, and streaming infrastructure can fail.

This repository therefore goes beyond training a classifier and implements an end-to-end fraud-risk decision platform.

## System Architecture

```text
                         TRANSACTION STREAM
                                │
                                ▼
                       ┌─────────────────┐
                       │      Kafka      │
                       │ Producer/Topic  │
                       └────────┬────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Fraud Processing      │
                    │ Consumer              │
                    │ Retry + DLQ +         │
                    │ Idempotency            │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │    Redis State        │
                    │ Historical Behavioral │
                    │ State / Features      │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Behavioral Features   │
                    │ Leakage-Safe History   │
                    └───────────┬───────────┘
                                │
                                ▼
                       ┌────────────────┐
                       │    LightGBM    │
                       │ Fraud Risk     │
                       │ Probability    │
                       └───────┬────────┘
                               │
                               ▼
                    ┌───────────────────────┐
                    │ APPROVE / REVIEW /    │
                    │ BLOCK Decision Engine │
                    └───────────┬───────────┘
                                │
                 ┌──────────────┴──────────────┐
                 │                             │
                 ▼                             ▼
        ┌─────────────────┐          ┌──────────────────┐
        │ FastAPI Serving │          │ SHAP Explainability│
        └────────┬────────┘          └──────────────────┘
                 │
                 ▼
        ┌──────────────────────────────┐
        │ Prometheus-style Monitoring  │
        │ Latency / Errors / Decisions │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │ PSI / KS Drift Detection     │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │ MLflow + Governance Registry │
        │ Evaluation + Promotion Gates │
        └──────────────┬───────────────┘
                       │
                       ▼
             Docker → Kubernetes
                       │
                       ▼
                AWS Architecture
                  Mapping
```

The system separates **offline model development** from **online stateful inference**, while sharing the same frozen model, preprocessing, behavioral definitions, and decision policy.

## Key Results

All primary model results below come from the untouched chronological test set of **88,581 transactions**.

| Area | Metric | Result |
|---|---|---:|
| Model | Test PR-AUC | **0.5483** |
| Model | Test ROC-AUC | **0.9058** |
| Ranking | Recall @ 1% budget | **25.30%** |
| Ranking | Recall @ 2% budget | **40.90%** |
| Ranking | Recall @ 5% budget | **60.14%** |
| Behavioral features | Validation PR-AUC improvement | **+0.0055** |
| Decisioning | Fraud captured by REVIEW + BLOCK | **41.87%** |
| Decisioning | BLOCK precision | **84.85%** |
| Decisioning | Legitimate customers blocked | **0.20%** |
| Serving | Offline/online decision parity | **3,000 / 3,000** |
| Serving | Risk-score correlation | **0.999999996** |
| Testing | Automated tests | **357 passing + 1 skipped** |

The behavioral features improved validation PR-AUC from **0.6092 → 0.6147**, a measurable but intentionally modest gain.

The final champion remains:

```text
fraud-risk-lightgbm-v1
```

## Production Infrastructure

The production-oriented upgrade adds:

| Layer | Implementation | Validation |
|---|---|---|
| Streaming | Kafka / `kafka-python` | Client + consumer logic tested |
| Stateful features | Redis | Real local Redis benchmarks |
| Model | LightGBM | Full chronological evaluation |
| Explainability | SHAP | Real champion-model explanations |
| Decisioning | APPROVE / REVIEW / BLOCK | Frozen validation-derived thresholds |
| Serving | FastAPI | API integration + parity tests |
| Monitoring | Prometheus-style metrics | Non-interference validated |
| Drift | PSI + KS | Real reference-profile analysis |
| MLOps | MLflow | Real champion registration |
| Governance | Local promotion registry | Promotion / rollback gates tested |
| Containerization | Docker | Image build + health validation |
| Orchestration | Kubernetes | YAML syntax validated |
| Cloud | AWS architecture | Design mapping only |

## Dataset

**IEEE-CIS Fraud Detection**

The dataset contains:

- 590,540 transactions
- 394 columns
- 3.499% fraud prevalence
- Transaction-level features
- Anonymized numeric and categorical features
- Transaction identity information

`TransactionDT` provides the temporal ordering used throughout the project.

The dataset does not provide a verified customer identifier. Therefore, `card1` is used only as a **pseudo-entity** for historical behavioral aggregation.

It must not be interpreted as a confirmed one-to-one customer or account identifier.

Raw Kaggle CSV files are approximately 1.3 GB and are not required to run the API, dashboard, or test suite.

## Leakage-Aware Evaluation

The dataset is split strictly by time:

```text
Past ───────────────────────────────────────────────► Future

TRAIN              VALIDATION                 TEST
413,378             88,581                    88,581
 70%                  15%                       15%
```

No random shuffling is used.

The protocol is:

```text
TRAIN
  ↓
Model + preprocessing fitting

VALIDATION
  ↓
Hyperparameter selection
Threshold selection
Reference-profile construction

TEST
  ↓
Final one-time evaluation
```

The test partition is never used for:

- Training
- Feature fitting
- Threshold selection
- Hyperparameter tuning
- Governance decisions

## Machine Learning

### Baseline

Logistic Regression was implemented as an interpretable baseline.

Standard test performance:

```text
PR-AUC  = 0.1604
ROC-AUC = 0.7940
```

### Champion

LightGBM was selected for its suitability for structured tabular data with mixed feature types and class imbalance.

The champion combines:

```text
Transaction Features
        +
Historical Behavioral Features
        ↓
LightGBM
        ↓
Fraud Probability
```

Final test performance:

```text
PR-AUC  = 0.5483
ROC-AUC = 0.9058
```

### Anomaly Detection Experiment

Isolation Forest was evaluated as a complementary unsupervised signal.

Results:

```text
Test PR-AUC = 0.0449
```

The result was only marginally above the no-skill fraud baseline of approximately 0.0348, and its flagged transactions were overwhelmingly already captured by LightGBM.

Therefore, Isolation Forest was **experimentally rejected** rather than included for the sake of architectural complexity.

## Behavioral Intelligence

Twelve historical `bhv_*` features are generated using only information available **before the current transaction**.

Examples include:

```text
bhv_prev_txn_count
bhv_hist_mean_amt
bhv_hist_std_amt
bhv_hist_min_amt
bhv_hist_max_amt
bhv_time_since_prev_txn
bhv_amt_to_hist_mean_ratio
bhv_amt_diff_from_hist_mean
bhv_amt_zscore
bhv_prior_count_1h
bhv_prior_count_24h
```

These features capture:

- Historical transaction frequency
- Typical transaction amount
- Amount deviation from historical behavior
- Transaction recency
- Short-term velocity
- Historical variability

The feature computation is deliberately ordered as:

```text
Read historical state
        ↓
Generate behavioral features
        ↓
Predict
        ↓
Decide
        ↓
Update state
```

This prevents the current transaction from leaking into its own behavioral features.

Behavioral intelligence produced:

```text
Validation PR-AUC:
0.6092 → 0.6147

Improvement:
+0.0055
```

The gain is real but incremental.

## Decision Engine

A fraud probability alone is not an operational decision.

The system maps model risk into:

```text
                Risk Score
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
     APPROVE      REVIEW       BLOCK
```

Thresholds were selected using the validation partition and then frozen:

```text
approve_threshold = 0.3206
block_threshold   = 0.6311
```

Final test traffic distribution:

| Decision | Traffic | Fraud rate |
|---|---:|---:|
| APPROVE | 97.91% | 2.07% |
| REVIEW | 0.82% | 46.12% |
| BLOCK | 1.27% | 84.85% |

Overall:

```text
Fraud captured by REVIEW + BLOCK:
1,291 / 3,083 = 41.87%

BLOCK precision:
84.85%

Legitimate customers blocked:
0.20%
```

The policy explicitly exposes the tradeoff between fraud capture and customer friction.

## Real-Time Stateful Inference

The `RiskDecisionEngine` implements the complete online transaction lifecycle:

```text
1. Validate request
       ↓
2. Read historical state
       ↓
3. Generate behavioral features
       ↓
4. Apply frozen preprocessing
       ↓
5. Generate LightGBM probability
       ↓
6. Apply frozen decision policy
       ↓
7. Update historical state
       ↓
8. Record monitoring metrics
```

The state update occurs **only after prediction and decisioning**.

A dedicated 3,000-transaction chronological simulation validated:

```text
Decision parity:
3,000 / 3,000 = 100%

Risk-score correlation:
0.999999996
```

A numerical-stability issue discovered during the original parity work was later fixed using **Welford's online variance algorithm**.

The adversarial regression test reduced variance-computation error to approximately:

```text
5 × 10⁻¹²
```

## Kafka Streaming

The streaming layer implements real Kafka clients using `kafka-python`.

Components include:

```text
TransactionProducer
TransactionConsumer
InMemoryBroker
FraudProcessingConsumer
```

The consumer includes:

- Retry handling
- Dead-letter routing
- Graceful shutdown
- Duplicate-message idempotency
- Integration with the real `RiskDecisionEngine`

### Validation scope

The Kafka client and consumer logic were implemented and tested.

However, a real Kafka broker could not be provisioned in the development environment because of network/environment restrictions.

Therefore:

> **No real Kafka throughput or end-to-end Kafka latency numbers are claimed.**

Streaming behavior was instead tested using the explicit `InMemoryBroker` test double.

For environments with normal Docker access:

```bash
docker compose -f docker-compose.kafka.yml up -d
```

## Redis State

The state layer uses a pluggable backend abstraction:

```text
StateBackend
├── InMemoryStateBackend
└── RedisStateBackend
```

This allows the same behavioral feature logic to operate against either in-memory state or Redis-backed state.

Configuration:

```bash
STATE_BACKEND=redis
REDIS_URL=redis://localhost:6379/0
```

Real local Redis benchmark:

```text
Redis lookup latency

p50  = 0.0525 ms
p95  = 0.0891 ms
p99  = 0.1230 ms
```

This was measured using a real local Redis server rather than a mocked client.

## SHAP Explainability

SHAP explanations are computed against the same feature matrix used for the actual champion-model prediction.

Explainability is exposed separately so normal scoring does not incur unnecessary explanation overhead.

```text
Normal request
/predict
    ↓
risk_score + decision

Explanation request
/predict/explain
    ↓
risk_score + decision + feature contributions
```

Measured SHAP latency:

```text
~14–18 ms p50
```

A dedicated test verifies that enabling explainability does not alter:

- Risk score
- Decision
- Feature matrix used for prediction

## API

FastAPI provides the serving layer.

### Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/predict` | POST | Score a transaction |
| `/predict/explain` | POST | Score + SHAP explanation |
| `/health` | GET | Service/model health |
| `/metadata` | GET | Model and policy metadata |
| `/metrics` | GET | Prometheus-style metrics |
| `/monitoring/summary` | GET | Monitoring summary |
| `/drift/summary` | GET | Current drift state |
| `/drift/analyze` | POST | Run drift analysis |
| `/model-governance/summary` | GET | Governance/champion state |
| `/dev/reset-state` | POST | Development state reset |

Interactive Swagger documentation:

```text
http://localhost:8000/docs
```

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"TransactionID": 9900001, "TransactionDT": 100000, "TransactionAmt": 55.0, "card1": 77777, "ProductCD": "W"}'
```

## Monitoring

The monitoring layer exposes Prometheus-compatible text metrics and JSON summaries.

Tracked signals include:

- Request counts
- HTTP status counts
- Endpoint latency
- Prediction success/failure
- APPROVE / REVIEW / BLOCK counts
- Risk-score statistics
- Validation failures
- Data-quality errors

Monitoring is intentionally **passive**.

A non-interference test verifies that enabling monitoring does not change:

- Risk scores
- Decisions
- Behavioral state

Example:

```bash
curl http://localhost:8000/metrics
curl http://localhost:8000/monitoring/summary
```

## Drift Detection

The system uses:

```text
PSI — Primary
KS  — Complementary
```

The reference distribution is built from the **validation partition**, never the untouched test partition.

Monitored signals include:

- Numeric transaction features
- Categorical features
- Behavioral features
- Model risk scores
- Decision distributions

PSI interpretation:

```text
< 0.10       No significant drift
0.10–0.20    Low drift
0.20–0.30    Moderate drift
≥ 0.30       High drift
```

Behavioral features require special interpretation.

Features such as historical transaction counts naturally increase over chronological time, so measurable PSI movement does not automatically indicate a broken pipeline.

The system therefore treats drift as a **monitoring signal requiring investigation**, not automatic proof of model degradation.

## MLflow & Model Governance

### MLflow

MLflow is used for:

- Experiment tracking
- Champion metric logging
- Hyperparameter logging
- Model registration

The existing champion is registered explicitly as:

```text
existing_champion_registration
```

It is **not** represented as a newly trained model.

This prevents misleading experiment history.

### Governance

The project also maintains a lightweight local governance registry.

The current champion is:

```text
fraud-risk-lightgbm-v1
```

The actual model artifact is tracked using SHA-256:

```text
bb5de8767ebaffae90a8ca634380524e2002f67d38fb87528ea5911479686342
```

Promotion requires six explicit gates:

```text
1. Metric availability
2. Feature-schema compatibility
3. Minimum candidate quality
4. Degradation limits
5. Operational/friction impact
6. Decision-policy compatibility
```

Possible outcomes:

```text
PROMOTE
REJECT
REQUIRES_REVIEW
```

Promotion and rollback require explicit human approval.

The governance demonstration includes a synthetic stronger candidate with approximately 0.8700 PR-AUC. This is **not a real model-performance result** and is used only to exercise the governance code path.

## Docker

The service is containerized with Docker.

Build:

```bash
docker build -t fraud-risk-api:local .
```

Run:

```bash
docker run -d \
  --name fraud-risk-api \
  -p 8000:8000 \
  fraud-risk-api:local
```

Health check:

```bash
curl http://localhost:8000/health
```

The Docker build context was measured at approximately:

```text
4.29 MB
```

The raw ~1.3 GB dataset is excluded from the image.

The production-upgrade Docker image includes the Redis, Kafka, and SHAP runtime dependencies.

MLflow is intentionally kept outside the main runtime image because of its heavier dependency tree; the application degrades gracefully when MLflow is unavailable.

## Kubernetes

The repository includes Kubernetes manifests for:

```text
k8s/
├── namespace.yaml
├── configmap.yaml
├── secret.example.yaml
├── redis-deployment.yaml
├── api-deployment.yaml
├── api-service.yaml
└── README.md
```

The manifests were YAML-syntax validated.

They were **not applied to a real Kubernetes cluster** because no cluster was available in the development environment.

Therefore, this repository does not claim a live Kubernetes deployment.

## AWS Architecture

The AWS layer is an architecture mapping rather than a deployed environment.

The intended mapping is:

```text
Kafka              → Amazon MSK
Redis              → Amazon ElastiCache
Containers         → ECS / EKS
Container Registry → Amazon ECR
Secrets            → AWS Secrets Manager
Monitoring         → CloudWatch + Prometheus/Grafana
```

No AWS resources were created for this portfolio project.

## Dashboard

The Streamlit dashboard provides a presentation layer over the real system.

Pages:

```text
Executive Overview
Live Scoring
Analytics
Monitoring
Drift
Governance
Architecture
```

The dashboard does not duplicate:

- Model inference
- Behavioral feature logic
- Drift calculations
- Governance logic

Instead, it presents real API outputs and project artifacts.

Run:

```bash
streamlit run dashboard/app.py
```

## Performance Benchmarks

Real local measurements are documented in:

```text
reports/streaming_benchmark.md
```

### Model inference

Measured using the real champion model:

```text
p50  = 5.114 ms
p95  = 6.052 ms
p99  = 6.841 ms

Throughput ≈ 187.5 predictions/sec
```

### Direct engine — in-memory state

```text
p50  = 66.968 ms
p95  = 89.366 ms
p99  = 136.796 ms

Throughput ≈ 14.2 transactions/sec
```

### Direct engine — Redis state

```text
p50  = 66.319 ms
p95  = 87.005 ms
p99  = 140.724 ms

Throughput ≈ 14.2 transactions/sec
```

### Redis operations

```text
Lookup:
p50  = 0.0525 ms
p95  = 0.0891 ms
p99  = 0.1230 ms

Update:
p50  = 0.0602 ms
p95  = 0.1026 ms
p99  = 0.1400 ms
```

These are **local development measurements**, not production-scale SLA claims.

Kafka broker throughput and end-to-end Kafka latency remain **not measured** because a real broker was unavailable in the development environment.

## Failure Testing

The production upgrade includes explicit failure-scenario tests covering:

- Redis restart/outage behavior
- Duplicate Kafka messages
- Idempotency
- Streaming failure handling
- Error propagation
- State-backend behavior

Two real production-style defects were discovered and fixed:

### Duplicate-message idempotency

A duplicate transaction ID could previously be processed more than once.

The consumer now detects duplicate transaction IDs and skips them.

### Redis outage handling

A missing clean failure path for Redis outages was identified and fixed.

Both fixes are covered by executed tests.

See:

```text
reports/failure_testing.md
```

## Testing & CI

Final local test result:

```text
357 automated tests passing
1 skipped
0 failures
```

Breakdown:

```text
316 tests  → original project phases
41 tests   → production infrastructure upgrade
1 skipped  → real Kafka broker test
```

The Kafka test is correctly skipped because no real broker was available in the development environment.

The suite covers:

- Data validation
- Leakage prevention
- Feature correctness
- Model training
- Decision policy
- Stateful inference
- API
- Monitoring
- Drift detection
- Governance
- Dashboard
- Redis backend
- Kafka streaming
- SHAP
- MLflow
- Failure scenarios

Run:

```bash
pytest -q
```

GitHub Actions runs the test suite on pushes.

## Project Structure

```text
fraud-risk-engine/
│
├── config/
│   └── decision_policy.yaml
│
├── data/
│   ├── raw/
│   └── interim/
│
├── artifacts/
│   ├── models/
│   └── drift/
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── decision/
│   ├── engine/
│   │   ├── risk_engine.py
│   │   ├── state.py
│   │   └── state_backend.py
│   ├── api/
│   ├── monitoring/
│   ├── drift/
│   ├── governance/
│   ├── explainability/
│   │   └── shap_explainer.py
│   ├── mlops/
│   │   └── mlflow_tracking.py
│   └── streaming/
│       ├── kafka_client.py
│       ├── fraud_processing_consumer.py
│       └── fake_broker.py
│
├── dashboard/
│
├── scripts/
│   ├── benchmark_end_to_end.py
│   ├── benchmark_redis_state.py
│   └── log_champion_to_mlflow.py
│
├── tests/
│   ├── test_production_failure_scenarios.py
│   ├── test_production_mlflow.py
│   ├── test_production_redis_state.py
│   ├── test_production_shap.py
│   └── test_production_streaming.py
│
├── reports/
│   ├── production_architecture_audit.md
│   ├── production_readiness.md
│   ├── streaming_benchmark.md
│   └── failure_testing.md
│
├── k8s/
├── Dockerfile
├── docker-compose.yml
├── docker-compose.kafka.yml
├── requirements.txt
└── requirements-docker.txt
```

## Quick Start

### 1. Clone

```bash
git clone https://github.com/DebadattaLiku/real-time-risk-fraud-engine.git
cd real-time-risk-fraud-engine
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run tests

```bash
pytest -q
```

Expected:

```text
357 passed
1 skipped
```

### 4. Start the API

```bash
uvicorn src.api.main:app --reload
```

Open:

```text
http://localhost:8000/docs
```

### 5. Start the dashboard

```bash
streamlit run dashboard/app.py
```

### 6. Optional Redis-backed state

Start Redis locally and configure:

```bash
STATE_BACKEND=redis
REDIS_URL=redis://localhost:6379/0
```

### 7. Optional Kafka stack

```bash
docker compose -f docker-compose.kafka.yml up -d
```

Then configure:

```bash
KAFKA_BROKERS=localhost:9092
```

### 8. MLflow

```bash
python scripts/log_champion_to_mlflow.py
```

### 9. Benchmarks

```bash
python scripts/benchmark_redis_state.py
python scripts/benchmark_end_to_end.py
```

## Engineering Decisions

| Decision | Rationale |
|---|---|
| Chronological train/validation/test split | Prevents temporal leakage |
| PR-AUC as primary metric | Appropriate for severe class imbalance |
| Recall at review budget | Measures operational fraud capture |
| LightGBM champion | Strong tabular-data performance |
| Historical behavioral features | Captures entity-level transaction patterns |
| Isolation Forest rejected | Measured complementary value was negligible |
| APPROVE / REVIEW / BLOCK | Converts probability into operational action |
| Predict-then-update state order | Prevents current-transaction leakage |
| Redis backend abstraction | Separates state infrastructure from feature logic |
| Kafka consumer idempotency | Prevents duplicate transaction processing |
| SHAP as opt-in | Preserves low-latency normal inference |
| PSI + KS | Complementary distribution-shift signals |
| Validation reference profile | Avoids contaminating the final test evaluation |
| Human-gated governance | Prevents silent model replacement |
| MLflow + local governance | Separates experiment tracking from deployment approval |
| Kubernetes manifests | Demonstrates orchestration readiness without claiming deployment |
| AWS architecture mapping | Shows cloud deployment design without fabricating infrastructure |

## Limitations

This project deliberately documents its boundaries.

### Dataset

The model was evaluated on the anonymized IEEE-CIS Kaggle dataset rather than live production transactions.

### Entity identity

`card1` is a pseudo-entity identifier, not a verified customer identity.

### Fraud capture

The final policy captures:

```text
41.87% of fraud
```

through REVIEW + BLOCK.

Approximately **58.1% of test-set fraud remains APPROVED**.

The strong BLOCK precision therefore does not imply complete fraud prevention.

### Online monitoring

The system does not have real-time ground-truth fraud labels.

PSI/KS monitoring identifies distributional changes but cannot independently prove that the model remains accurate.

### Kafka

Kafka clients and consumer behavior are implemented and tested, but no real Kafka broker was available in the development environment.

Therefore:

```text
Kafka throughput = NOT MEASURED
Kafka end-to-end latency = NOT MEASURED
```

### Kubernetes

Manifests are authored and validated but were not deployed to a real cluster.

### AWS

AWS infrastructure is architectural design only.

No AWS resources were created.

### Authentication

The FastAPI service has no authentication.

This is acceptable for a local portfolio system but would require security controls before public production deployment.

### Production traffic

Latency, throughput, monitoring, and drift measurements are local development/test measurements rather than production-scale SLAs.

## Reproducibility

The project intentionally separates reproducibility from the large raw dataset.

The repository contains:

- Trained model artifacts
- Evaluation metrics
- Decision policy
- Drift reference profile
- Governance registry
- Production test suite
- Benchmark scripts
- Infrastructure definitions
- Documentation and validation reports

The raw IEEE-CIS CSVs are approximately 1.3 GB and require authenticated Kaggle access.

They should be placed under:

```text
data/raw/
```

as:

```text
train_transaction.csv
train_identity.csv
test_transaction.csv
test_identity.csv
sample_submission.csv
```

The complete API, dashboard, and automated test suite do not require downloading the raw dataset.

## Validation Reports

Detailed evidence is available in:

```text
reports/
├── production_architecture_audit.md
├── production_readiness.md
├── streaming_benchmark.md
└── failure_testing.md
```

Earlier project phases are documented through:

```text
reports/phase*_summary.md
```

These reports contain the detailed experimental methodology, benchmarks, failure investigations, and production-readiness assessment.

## Project Principles

This project follows several non-negotiable principles:

```text
No random train/test shuffling.

No behavioral feature may use future information.

No preprocessing object is fitted on validation or test data.

No model-selection decision is based on the final test set.

No fabricated production metrics.

No fabricated cloud deployment claims.

No automatic model promotion.

No hiding failed experiments.

No hiding system limitations.
```

The result is intended to demonstrate not only that a fraud model can achieve strong offline metrics, but that it can be surrounded by the engineering infrastructure required to turn a model into a **stateful, observable, explainable, governed real-time decision system**.
