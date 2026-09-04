# Real-Time Risk Decision & Fraud Intelligence Engine

> **Production-oriented fraud intelligence platform for real-time transaction risk scoring, behavioral feature engineering, operational decisioning, monitoring, drift detection, and model governance.**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Deployment-Docker-2496ED.svg)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/Tests-316%20passed-success.svg)](#testing)
[![Git LFS](https://img.shields.io/badge/Dataset-Git%20LFS-orange.svg)](https://git-lfs.com/)

---

## 1. Overview

Fraud detection is not simply a binary classification problem.

A production fraud system must answer four questions for every transaction:

1. **How risky is this transaction?**
2. **What historical behavior does it exhibit?**
3. **Should the transaction be approved, reviewed, or blocked?**
4. **Can the system continue operating reliably as the data and model environment change?**

This project implements an end-to-end **Real-Time Risk Decision & Fraud Intelligence Engine** that addresses these requirements.

The system combines:

- Leakage-aware chronological model development
- LightGBM fraud-risk modeling
- Strictly historical behavioral features
- Stateful online feature computation
- Three-way operational decisioning
- FastAPI real-time serving
- Prometheus-compatible monitoring
- PSI/KS drift detection
- Model governance and promotion gates
- Dockerized deployment
- Streamlit operational dashboard
- Automated testing and CI

The project uses the **IEEE-CIS Fraud Detection** dataset, consisting of anonymized e-commerce transactions provided through the Kaggle competition.

---

## 2. System Architecture

```text
                         ┌──────────────────────┐
                         │      Transaction      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Input Validation &   │
                         │   Data Quality       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Historical State     │
                         │      Lookup          │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Behavioral Feature   │
                         │     Generation       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Preprocessing      │
                         │ Numeric + Categorical│
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ LightGBM Fraud Risk  │
                         │       Model          │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Risk Score        │
                         │       [0, 1]          │
                         └──────────┬───────────┘
                                    │
                                    ▼
                     ┌─────────────────────────────┐
                     │     Decision Policy         │
                     └─────────────┬───────────────┘
                                   │
                  ┌────────────────┼────────────────┐
                  ▼                ▼                ▼
             ┌────────┐      ┌────────┐      ┌────────┐
             │APPROVE │      │ REVIEW │      │ BLOCK  │
             └────────┘      └────────┘      └────────┘
                  │                │                │
                  └────────────────┼────────────────┘
                                   ▼
                     ┌─────────────────────────────┐
                     │ State Update After Prediction│
                     └─────────────────────────────┘

              ┌─────────────────────────────────────────┐
              │ Monitoring │ Drift │ Governance │ Logs │
              └─────────────────────────────────────────┘
```

---

## 3. Quantitative Results & Experimental Evidence

The final LightGBM + behavioral-feature champion was evaluated on a completely held-out chronological test partition.

### 3.1 Model Performance

| Model / Experiment | Test PR-AUC | Test ROC-AUC |
|---|---:|---:|
| Logistic Regression — standard | 0.1604 | 0.7940 |
| Logistic Regression — balanced | 0.1481 | 0.8006 |
| LightGBM — pre-behavioral | **0.5428** | **0.9024** |
| LightGBM + behavioral features | **0.5483** | **0.9058** |

The behavioral layer improved test PR-AUC by **+0.0055** and ROC-AUC by **+0.0034**.

The improvement is deliberately reported as modest: the behavioral features provide measurable incremental value, but the project does not overstate their impact.

### 3.2 Operational Ranking Performance

| Metric | Result |
|---|---:|
| Recall @ 1% intervention budget | **25.30%** |
| Recall @ 2% intervention budget | **40.90%** |
| Recall @ 5% intervention budget | **60.14%** |
| Test fraud prevalence | **3.48%** |

At a **2% intervention budget**, the champion captures **1,259 / 3,083 fraudulent transactions = 40.90% recall**.

### 3.3 Decision Policy Performance

The production policy converts model risk scores into `APPROVE`, `REVIEW`, and `BLOCK`.

| Decision / Metric | Test Result |
|---|---:|
| APPROVE | **97.9104%** |
| REVIEW | **0.8151%** |
| BLOCK | **1.2745%** |
| Fraud captured by REVIEW + BLOCK | **1,291 / 3,083 = 41.87%** |
| BLOCK precision | **84.85%** |
| Legitimate transactions blocked | **0.20%** |
| Legitimate transactions reviewed | **0.455%** |
| Total legitimate intervention rate | **0.655%** |

This demonstrates the distinction between **model ranking** and **operational decisioning**: the system is evaluated under realistic intervention constraints rather than accuracy alone.

### 3.4 Alternative Anomaly Detection Experiment

Isolation Forest was evaluated as a complementary unsupervised component.

| Metric | Isolation Forest |
|---|---:|
| Test PR-AUC | **0.0449** |
| Test ROC-AUC | **0.5448** |
| Fraud prevalence baseline | **0.0348** |
| Additional fraud cases uniquely captured beyond LightGBM @ 2% budget | **24** |

Because its performance was close to the fraud-rate baseline and it added limited incremental detection value, Isolation Forest was **rejected as a production component**.

This experiment demonstrates that additional algorithms were evaluated based on measurable incremental value rather than added for architectural complexity.

### 3.5 Behavioral Feature Engineering

The behavioral pipeline computes **12 strictly historical features**.

| Engineering Result | Value |
|---|---:|
| Rows processed for full behavioral computation | **590,540** |
| Computation time | **< 1 second** |
| Current transaction included before prediction | **No** |
| Historical state updated | **After prediction** |

Behavioral features include historical transaction counts, amount statistics, recency, amount deviation, and prior 1-hour / 24-hour activity.

### 3.6 Offline / Online Parity

A chronological real-time simulation compared offline feature generation against the stateful online engine.

| Parity Metric | Result |
|---|---:|
| Fully matched feature rows | **2,997 / 3,000** |
| Median feature difference | **0** |
| 99th-percentile feature difference | **7.4 × 10⁻⁶** |
| Risk-score Pearson correlation | **0.999999996** |
| Risk-score MAE | **~1.5 × 10⁻⁷** |
| Decision agreement | **3,000 / 3,000** |
| Direct engine ↔ API mismatches | **0 / 10** |

These parity measurements were obtained before the later numerical-stability fix and are retained as the historical parity validation result.

### 3.7 Numerical Stability

The stateful engine was upgraded from naive variance accumulation to **Welford's online algorithm**.

For an adversarial large-base, tightly clustered numerical test:

| Numerical Check | Result |
|---|---:|
| NumPy ground-truth std | **0.0009924527975957473** |
| Welford std | **0.000992452792689105** |
| Absolute error | **~4.91 × 10⁻¹²** |

The previous naive formulation could produce a negative variance and `NaN` under this adversarial condition; the Welford implementation remained numerically stable.

### 3.8 Docker & Serving Validation

| Engineering Metric | Result |
|---|---:|
| Docker build context | **~4.29 MB** |
| Container startup / health | **~6 seconds** |
| Model loaded once at startup | **Yes** |
| Stateful engine reused by API | **Yes** |

### 3.9 Automated Testing

The project contains tests spanning model behavior, state management, API serving, monitoring, drift, governance, dashboard, Docker configuration, and numerical stability.

**Latest full test result: 316 passed.**

### 3.10 Model Governance

The champion model is:

```text
Model:    fraud-risk-lightgbm-v1
Version:  v1
Status:   CHAMPION
```

Artifact integrity is verified using SHA-256:

```text
bb5de8767ebaffae90a8ca634380524e2002f67d38fb87528ea5911479686342
```

Governance evaluates:

- Metric availability
- Feature-schema compatibility
- Candidate quality
- Performance degradation
- Operational compatibility
- Decision-policy compatibility

A candidate that fails any hard gate is rejected; candidates requiring review do not automatically replace the champion.

> **Note:** A synthetic label-informed candidate used during governance testing achieved PR-AUC 0.8700, but this was a governance test artifact and is **not** a real model-performance result. It is intentionally excluded from the headline performance metrics.

---

### Headline Numbers

If you only remember a few numbers from this project:

**0.5483 PR-AUC** · **0.9058 ROC-AUC** · **60.14% Recall@5%** · **84.85% BLOCK precision** · **0.20% legitimate blocked** · **999999996 correlation** · **316 tests passed**


---

## 4. Dataset

The project uses the **IEEE-CIS Fraud Detection** dataset.

The dataset contains anonymized e-commerce transaction and identity information.

The raw dataset is maintained locally and tracked through **Git LFS** because of its large size.

### Dataset files

```text
data/raw/
├── train_transaction.csv
├── train_identity.csv
├── test_transaction.csv
├── test_identity.csv
└── sample_submission.csv
```

The training transaction dataset contains approximately **590K transactions**.

---

## 5. Leakage-Aware Machine Learning Pipeline

Fraud data is inherently temporal.

Randomly splitting transactions can allow information from the future to influence historical predictions.

To avoid this, the project uses a strict chronological split based on `TransactionDT`.

```text
590,540 transactions

        ┌──────────────────────┐
        │       TRAIN          │
        │      413,378         │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │    VALIDATION        │
        │       88,581         │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │        TEST          │
        │       88,581         │
        └──────────────────────┘
```

No future transaction information is used when generating behavioral features for an earlier transaction.

---

## 6. Baseline → Champion Evolution

The project intentionally develops the system incrementally instead of jumping directly to a complex model.

### Logistic Regression Baseline

The first baseline established the difficulty of the problem.

| Model | Test PR-AUC | Test ROC-AUC |
|---|---:|---:|
| Logistic Regression | 0.1604 | 0.7940 |
| LightGBM | 0.5428 | 0.9024 |
| LightGBM + Behavioral Features | **0.5483** | **0.9058** |

The LightGBM model substantially improved ranking quality over the linear baseline.

---

## 7. Behavioral Intelligence

A transaction-level fraud model becomes more informative when it understands the historical behavior associated with an entity.

The project uses `card1` as a **pseudo-entity identifier** for historical behavioral aggregation. It is not treated as a verified customer identity.

The behavioral layer generates strictly historical features such as:

- Previous transaction count
- Historical mean transaction amount
- Historical standard deviation
- Historical minimum/maximum amount
- Time since previous transaction
- Amount-to-history-mean ratio
- Amount difference from historical mean
- Historical amount z-score
- Prior transaction count in the previous hour
- Prior transaction count in the previous 24 hours

### Critical design rule

```text
READ HISTORICAL STATE
        ↓
GENERATE FEATURES
        ↓
PREDICT
        ↓
MAKE DECISION
        ↓
UPDATE STATE
```

The current transaction is **never added to the state before prediction**.

This prevents target/temporal leakage in the online feature pipeline.

---

## 8. Behavioral Feature Impact

The behavioral layer produced a measurable, validation-confirmed improvement:

```text
LightGBM
PR-AUC = 0.5428

        ↓

LightGBM + Behavioral Features
PR-AUC = 0.5483
```

The improvement is intentionally reported as **modest rather than exaggerated**.

This demonstrates an important engineering principle:

> A feature engineering layer should be retained because it provides measurable incremental value—not simply because it makes the architecture more complicated.

---

## 9. Fraud Risk → Operational Decision

A fraud probability alone is not enough for an operational system.

The engine converts the risk score into three actions:

```text
                 Risk Score
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
     APPROVE       REVIEW        BLOCK
```

Thresholds are selected using the validation partition rather than tuned directly on the final test set.

### Test decision distribution

| Decision | Test Distribution |
|---|---:|
| APPROVE | 97.9104% |
| REVIEW | 0.8151% |
| BLOCK | 1.2745% |

The combined **REVIEW + BLOCK** actions captured:

**1,291 / 3,083 fraudulent transactions = 41.87%**

The BLOCK decision achieved:

**84.85% precision**

while only blocking approximately:

**0.20% of legitimate transactions.**

---

## 10. Why PR-AUC?

Fraud is highly imbalanced.

The test fraud rate is approximately **3.48%**.

In such a setting, accuracy can be misleading.

For example, a classifier predicting every transaction as legitimate can achieve very high accuracy while detecting no fraud.

Therefore the project emphasizes:

- PR-AUC
- ROC-AUC
- Recall@K
- Precision at operational thresholds
- Legitimate intervention rate
- Decision distribution

These metrics better reflect the actual operating constraints of fraud detection.

---

## 11. Real-Time Stateful Inference

The offline feature-generation pipeline and online serving pipeline use the same behavioral logic.

The real-time engine follows:

```text
Transaction
    ↓
Validate
    ↓
Read historical state
    ↓
Generate behavioral features
    ↓
Predict risk
    ↓
Apply decision policy
    ↓
Return response
    ↓
Update historical state
```

The model bundle is loaded once at application startup.

The stateful risk engine is reused by the API rather than duplicating inference logic inside the HTTP layer.

---

## 12. Offline / Online Parity

A chronological real-time simulation was performed to verify that online behavioral feature generation agrees with offline feature computation.

On a 3,000-transaction chronological simulation:

- **2,997 / 3,000** feature rows fully matched under the original strict comparison
- Median feature difference: **0**
- 99th-percentile feature difference: **7.4e-6**
- Risk-score Pearson correlation: **0.999999996**
- Risk-score MAE: approximately **1.5e-7**
- Decision agreement: **3,000 / 3,000**

These results were obtained before the later numerical-stability engineering fix and are retained as the historical parity validation result.

---

## 13. FastAPI Serving

The model is exposed through a FastAPI service.

### Main endpoints

```text
GET  /health
GET  /metadata
GET  /metrics

POST /predict

GET  /monitoring/summary

GET  /drift/summary
POST /drift/analyze

GET  /model-governance/summary

POST /dev/reset-state
```

Interactive API documentation is available through FastAPI's `/docs` endpoint when the service is running.

### Example

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d @example_transaction.json
```

---

## 14. Monitoring

The serving layer includes application and model monitoring without modifying the underlying inference logic.

Tracked signals include:

### API health

- Request counts
- HTTP status codes
- Latency
- Prediction success/failure

### Model behavior

- Prediction counts
- Decision distribution
- Risk-score statistics
- Risk-score buckets

### Data quality

- Validation failures
- Missing/invalid inputs
- Data-quality counters

Metrics are exposed in a Prometheus-compatible text format through:

```text
/metrics
```

---

## 15. Drift Detection

The system maintains a reference profile derived from the **validation partition**, keeping the final test set independent from drift-reference construction.

Drift monitoring uses:

### Primary

**Population Stability Index (PSI)**

### Complementary

**Kolmogorov-Smirnov (KS) statistic**

Monitored signals include:

- `TransactionAmt`
- `C1`
- `C13`
- `C14`
- `D2`
- `V258`
- `ProductCD`
- `R_emaildomain`
- Behavioral features
- Risk score
- Decision distribution

### PSI interpretation

| PSI | Interpretation |
|---:|---|
| < 0.10 | No significant shift |
| 0.10–0.20 | Low |
| 0.20–0.30 | Moderate |
| ≥ 0.30 | High |

The implementation also handles:

- Previously unseen categorical values
- Rare-category grouping
- `__OTHER__` buckets
- Numeric distribution changes

### Important interpretation

Behavioral features naturally accumulate historical information.

Therefore, chronological drift in variables such as historical transaction counts or historical mean amount does **not automatically imply a data-quality problem**.

---

## 16. Model Governance

The project includes a lightweight model-governance framework designed around controlled model promotion.

Each candidate model is evaluated against multiple gates:

```text
Candidate Model
      │
      ▼
Metric Availability
      │
      ▼
Feature Schema Compatibility
      │
      ▼
Candidate Quality
      │
      ▼
Performance Degradation
      │
      ▼
Operational Compatibility
      │
      ▼
Decision Policy Compatibility
      │
      ▼
PROMOTE / REVIEW / REJECT
```

### Governance capabilities

- Model metadata
- Model registry
- Artifact SHA-256 verification
- Feature-schema compatibility checks
- Performance gates
- Operational degradation gates
- Decision-policy compatibility
- Explicit promotion approval
- Rollback support

Current champion:

```text
Model: fraud-risk-lightgbm-v1
Version: v1
Status: CHAMPION
```

Artifact SHA-256:

```text
bb5de8767ebaffae90a8ca634380524e2002f67d38fb87528ea5911479686342
```

---

## 17. Numerical Stability Engineering

The stateful behavioral engine uses **Welford's online algorithm** for numerically stable running variance estimation.

A naive formulation based on:

```text
E[X²] - E[X]²
```

can suffer catastrophic cancellation for large-valued, tightly clustered observations.

The engineering audit exposed this failure mode and replaced the calculation with Welford's stable online update.

For the adversarial numerical test:

```text
NumPy ground-truth std:
0.0009924527975957473

Welford std:
0.000992452792689105

Absolute error:
≈ 4.91 × 10⁻¹²
```

This was accompanied by a regression test to prevent recurrence.

---

## 18. Dockerized Deployment

The service can be packaged into a Docker image.

```text
Dockerfile
docker-compose.yml
.dockerignore
.env.example
```

The model bundle is included explicitly while the large raw dataset remains managed through Git LFS.

The Docker build was also used to expose and resolve dependency compatibility issues, including exact version pinning for the serving environment.

---

## 19. Streamlit Dashboard

A Streamlit dashboard provides a presentation layer over the underlying engine.

Pages include:

```text
Executive Overview
Live Scoring
Analytics
Monitoring
Drift
Governance
Architecture
```

The dashboard intentionally avoids duplicating:

- Inference logic
- Behavioral feature generation
- Drift calculations
- Governance decisions

Instead, it consumes the same engine/API functionality used by the production path.

---

## 20. Testing

The repository contains an extensive automated test suite covering:

- Data processing
- Feature engineering
- Model inference
- Behavioral state
- Online/offline parity
- Decision policy
- FastAPI endpoints
- Monitoring
- Drift detection
- Governance
- Dashboard
- Docker configuration
- Numerical stability

Current test result:

```text
316 passed
```

The project also includes CI configuration for automated testing.

---

## 21. Repository Structure

```text
real-time-risk-fraud-engine/
│
├── data/
│   └── raw/
│       ├── train_transaction.csv
│       ├── train_identity.csv
│       ├── test_transaction.csv
│       ├── test_identity.csv
│       └── sample_submission.csv
│
├── src/
│   ├── engine/
│   ├── features/
│   ├── models/
│   ├── monitoring/
│   ├── drift/
│   ├── governance/
│   └── ...
│
├── dashboard/
│
├── tests/
│
├── reports/
│
├── configs/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── environment.yml
├── .gitattributes
├── .gitignore
└── README.md
```

---

## 22. Reproducibility

The project separates:

```text
Raw Data
   ↓
Feature Engineering
   ↓
Training
   ↓
Validation
   ↓
Model Artifact
   ↓
Serving
   ↓
Monitoring
```

The final test partition remains isolated from model-selection decisions.

Model artifacts are versioned and protected through SHA-256 verification.

---

## 23. Engineering Principles

The project was designed around several principles:

### 1. Prevent leakage before optimizing the model

Temporal correctness is more important than squeezing out another fraction of a validation point.

### 2. Optimize for the operating environment

Fraud detection is constrained by review capacity and false-positive costs.

### 3. Separate scoring from decisioning

A risk score and an operational action are different concepts.

### 4. Keep online and offline logic consistent

Production feature computation should reproduce the behavior validated offline.

### 5. Monitor the system after deployment

A model can degrade even when its code has not changed.

### 6. Govern model changes

A new model should not automatically replace the existing champion.

### 7. Treat numerical stability as a production concern

Correct mathematical formulas are not always sufficient for reliable floating-point computation.

---

## 24. Limitations

This project intentionally documents several limitations.

### Pseudo-entity identity

`card1` is used as a pseudo-entity for behavioral aggregation. It is not a verified customer identifier.

### Historical dataset

The system is validated using historical Kaggle data rather than a live production transaction stream.

### Authentication

The current FastAPI service does not implement production-grade authentication/authorization.

### Behavioral drift

Cumulative behavioral variables naturally evolve over chronological time and therefore require contextual interpretation when drift is detected.

### Production infrastructure

The project demonstrates production-oriented ML engineering patterns but is not intended to represent a complete enterprise fraud platform with distributed state stores, message queues, feature stores, or multi-region deployment.

---

## 25. Why This Project Is Different

Many fraud-detection projects stop at:

```text
Dataset → Model → Accuracy
```

This project extends the problem into a complete decision system:

```text
                    ┌─────────────────┐
                    │ Fraud Modeling  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   Behavioral    │
                    │   Intelligence  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Risk Decisioning│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Real-Time API   │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
          Monitoring       Drift        Governance
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                       Production Loop
```

The emphasis is therefore not only on **model performance**, but on:

**ML + feature engineering + state + serving + monitoring + drift + governance + reliability.**

---

## 26. Tech Stack

### Machine Learning

- Python
- LightGBM
- Scikit-learn
- NumPy
- Pandas

### API / Serving

- FastAPI
- Uvicorn
- Pydantic

### Monitoring

- Prometheus-compatible metrics
- Structured logging
- Data-quality monitoring

### Drift

- PSI
- KS statistics

### Dashboard

- Streamlit

### Engineering

- Docker
- Git
- Git LFS
- GitHub Actions
- Pytest

---

## 27. Running Locally

Clone the repository:

```bash
git clone https://github.com/DebadattaLiku/real-time-risk-fraud-engine.git
cd real-time-risk-fraud-engine
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000/docs
```

Run tests:

```bash
pytest -q
```

Run the Streamlit dashboard using the project's dashboard entry point.

---

## 28. Future Improvements

Potential extensions include:

- Redis/Kafka-backed distributed state
- Feature-store integration
- Model calibration
- Cost-sensitive threshold optimization
- Champion/challenger experimentation
- Authentication and authorization
- API rate limiting
- Automated retraining pipelines
- Cloud deployment
- Real-time event streaming
- Explainable fraud decisions
- SHAP-based investigation workflows
- Automated drift-triggered retraining
- Distributed observability
- Online model experimentation

---

## 29. Author

**Debadatta Panda**  
MTech — Industrial Mathematics & Scientific Computing  
Indian Institute of Technology Madras

This project was developed as a production-oriented machine-learning engineering portfolio project, with emphasis on **fraud detection, real-time decision systems, MLOps, model governance, and reliable ML serving**.

---

## License & Dataset Notice

The source code of this repository is provided for educational and portfolio purposes.

The IEEE-CIS Fraud Detection dataset is subject to the competition's own terms and conditions. Users obtaining or using the dataset should review and comply with the applicable Kaggle competition rules and dataset terms.
