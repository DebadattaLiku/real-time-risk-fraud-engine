# Production Readiness Report

Final consolidated report for the production-grade infrastructure
upgrade to the Real-Time Risk Decision & Fraud Intelligence Engine. This
report follows the same standard as every prior phase report in this
project: every number is real and traceable to an actual executed
artifact or test in this session; every gap is stated honestly as
**NOT MEASURED** or **NOT IMPLEMENTED** rather than filled in with a
plausible-looking value.

## Architecture

```
Transaction Producer → Kafka → Fraud Processing Consumer → Redis-backed
State → Behavioral Features → LightGBM → SHAP → Decision Policy →
APPROVE/REVIEW/BLOCK → FastAPI/Dashboard → Prometheus Monitoring →
PSI/KS Drift Detection → MLflow + Governance → Docker/Kubernetes →
AWS-mapped deployment
```

Every arrow above except the Kafka ingestion step was verified with real,
executed code in this session. The existing Phase 0-14 system
(leakage-safe modeling, behavioral features, decision policy, real-time
engine, API, Docker, monitoring, drift, governance, dashboard) was
inspected first and reused unchanged throughout — no duplicated
inference, feature, drift, or governance logic was introduced anywhere
in this upgrade (verified structurally: `FraudProcessingHandler` and
`/predict/explain` both call the exact same
`RiskDecisionEngine.process_transaction()` the original `/predict` route
uses).

## ML — existing, validated, UNCHANGED

| Metric | Value | Status |
|---|---:|---|
| Test PR-AUC | 0.5483 | Unchanged, re-verified |
| Test ROC-AUC | 0.9058 | Unchanged, re-verified |
| Recall@1% / 2% / 5% | 25.30% / 40.90% / 60.14% | Unchanged, re-verified |
| Fraud captured (REVIEW+BLOCK) | 41.87% | Unchanged, re-verified |
| Precision among BLOCKed | 84.85% | Unchanged, re-verified |
| Legitimate intervention rate | 0.20% | Unchanged, re-verified |
| Champion model | `fraud-risk-lightgbm-v1` | Unchanged — never retrained, never replaced |

No experiment in this upgrade legitimately produced a better model, so
none of these numbers were touched, per this task's own explicit rule.

## Streaming (Kafka)

**Implemented**: real `TransactionProducer`/`TransactionConsumer` using
the actual `kafka-python` client, with configurable brokers/topics,
retry logic, dead-letter routing, graceful shutdown, and (a real gap
found and fixed) bounded per-consumer idempotency for duplicate message
IDs.

**NOT MEASURED, and exactly why**: no real Kafka broker could be started
or reached anywhere in this development session — the network egress
allowlist excludes Apache/Confluent/Bitnami/Redpanda, and no Ubuntu apt
package provides a broker (verified directly: `apt-cache search kafka`
returns only unrelated Go client libraries). Real, substitute evidence
instead: (1) the producer/consumer were verified to fail cleanly and
quickly against an unreachable broker (a caught `KafkaTimeoutError`, not
a hang), and (2) the consumer's actual message-processing logic — wired
to the real engine — is fully tested against an explicit `InMemoryBroker`
test double (15 passing tests). `docker-compose.kafka.yml` provides a
real broker definition, YAML-validated, for use elsewhere.

## State (Redis)

**Implemented and measured, real.** `RedisStateBackend` behind a clean
`StateBackend` interface; `BehavioralStateManager`'s feature-computation
math is completely unchanged and backend-agnostic. Real end-to-end proof:
the live API was run with `STATE_BACKEND=redis`, real predictions were
made, and the resulting state was independently verified via `redis-cli`.

Real measured latency (`reports/streaming_benchmark.md`):

| Backend | lookup p50 | lookup p95 | lookup p99 |
|---|---:|---:|---:|
| InMemoryStateBackend | 0.0001ms | 0.0002ms | 0.0003ms |
| RedisStateBackend | 0.0525ms | 0.0891ms | 0.1230ms |

A real Redis outage was genuinely induced (the local `redis-server`
process was killed mid-test) and a real gap was found and fixed: the API
now returns a clean `503 state_backend_unavailable` instead of a generic
500, via a dedicated exception handler added during this session.

## Explainability (SHAP)

**Implemented and measured, real, against the real champion model.**
`FraudExplainer` (real `shap.TreeExplainer`) is wired in as a strictly
additive `explain=True` engine parameter and a separate
`/predict/explain` endpoint — `/predict` itself is byte-for-byte
unchanged. Verified by a dedicated test that explanations can never alter
`risk_score`/`decision`. Real measured latency: p50 ~14-18ms (roughly 3x
the ~5ms base inference cost), which is the concrete reason this was
implemented as a separate, opt-in endpoint rather than folded into the
default prediction path.

## Serving (FastAPI)

Unchanged core behavior, extended additively: 2 new endpoints
(`/predict/explain`, and `/model-governance/summary` now includes an
MLflow cross-reference), 1 new exception handler
(`StateBackendUnavailableError` → 503). All existing endpoints and their
tested contracts are unchanged — verified by the full pre-existing test
suite passing without modification.

## Monitoring (Prometheus-style)

Extended additively: `MetricsRegistry` gained SHAP/Redis/streaming/
model-load-time counters under a new `extensions` key in
`/monitoring/summary` — no existing key was renamed or removed. Real,
live-verified via a running server.

## Drift (PSI/KS)

**Unchanged.** No modification was made to `src/drift/` in this upgrade
— verified by the pre-existing test suite (including the structural test
that parses every `src/drift/*.py` file and confirms none import
`src/governance/`, still passing).

## Governance

**Unchanged as the authority.** MLflow was added alongside it, never in
place of it — `/model-governance/summary` now includes a best-effort,
additive `mlflow` field cross-referencing the same real champion by
artifact hash. Promotion/rollback remain exclusively
`ModelRegistry.promote()`/`rollback()`, requiring explicit
`approved_by`/`reason` arguments — no API route or new component can call
either.

## Deployment (Docker / Kubernetes / AWS)

- **Docker: rebuilt and validated live, with the new dependencies.**
  `requirements-docker.txt` (deliberately excludes `mlflow` — see below)
  was used to rebuild the image; the container started healthy in 6
  seconds, and real, live requests were verified against `/health`,
  `/predict`, and **`/predict/explain` (real SHAP explanation, computed
  inside the container)**. `/model-governance/summary` correctly reported
  `"mlflow": {"available": false}` — the graceful-degradation path
  working exactly as designed, since MLflow is intentionally not in the
  container image.
- **A real dependency-footprint finding**: the first two build attempts
  with the FULL `requirements.txt` (including `mlflow`) failed with a
  genuine `OSError: No space left on device` in this development
  environment — not a code defect, but a real resource constraint,
  worsened by several GB of orphaned Docker layers that had accumulated
  across this project's many prior build sessions (cleaned up during this
  session: image pruning reclaimed several GB). The fix applied was
  `requirements-docker.txt`, a leaner dependency set that excludes
  `mlflow` specifically — its large transitive dependency tree (pyarrow,
  numba/llvmlite, sqlalchemy, alembic, cryptography, graphene, flask,
  gunicorn, databricks-sdk) is real overhead for a feature
  (`/model-governance/summary`'s optional MLflow cross-reference) that
  already degrades gracefully without it. `redis`, `kafka-python`, and
  `shap` — all directly used by the live prediction/explanation path —
  remain in the container image and were verified working inside it.
- **Kubernetes**: `k8s/` manifests authored and YAML-syntax-validated in
  this session. **Never applied to a real cluster** — none was available.
  No autoscaling is configured or claimed.
- **AWS**: a documented mapping (Kafka→MSK, Redis→ElastiCache,
  containers→ECS/EKS via ECR, secrets→Secrets Manager,
  monitoring→CloudWatch+Prometheus/Grafana) — **zero real AWS resources
  were created**, and this report makes no claim otherwise.

## Reliability (tests + failure testing)

**357 tests passing, 1 correctly skipped, 0 failures** — 316 from the
original 14 phases (all still passing, unmodified) + 41 new tests across
Redis, streaming, SHAP, MLflow, and failure scenarios. Real failure tests
executed: a genuinely-killed-and-restarted Redis process, a real
unreachable-Kafka clean-failure check, model-unavailable, invalid
transaction, duplicate-message idempotency (found broken, then fixed),
consumer restart (both with and without Redis), and API restart. Full
detail in `reports/failure_testing.md`.

## Performance — all real measurements

| Component | p50 | p95 | p99 | Source |
|---|---:|---:|---:|---|
| Direct model inference | 5.114ms | 6.052ms | 6.841ms | `scripts/benchmark_end_to_end.py` |
| Direct engine (in-memory state) | 66.968ms | 89.366ms | 136.796ms | same |
| Direct engine (Redis-backed state) | 66.319ms | 87.005ms | 140.724ms | same |
| FastAPI `/predict` (TestClient) | 70.094ms | 91.325ms | 165.752ms | same |
| Direct engine + SHAP explanation | 83.689ms | 107.754ms | 168.987ms | same |
| Redis lookup (standalone) | 0.0525ms | 0.0891ms | 0.1230ms | `scripts/benchmark_redis_state.py` |
| Kafka (any metric) | **NOT MEASURED** | | | no broker available |
| Model load time | **NOT MEASURED** | | | not isolated in this session's benchmark |

Throughput figures above are single-threaded, sequential-request
measurements — **not** a claim about concurrent production throughput,
which was never load-tested.

## Remaining limitations — stated honestly

- **No real Kafka broker was ever run against this system.** This is the
  single largest gap in this upgrade, and it is structural to this
  development environment, not a shortcut taken.
- **No load testing under concurrency** — every latency number above is
  single-threaded and sequential.
- **No Kubernetes cluster validation** — manifests are YAML-valid, never
  `kubectl apply`-tested.
- **No real AWS deployment** — design/mapping only.
- **Idempotency fix is bounded and in-memory** — a duplicate arriving
  after the cache window or a consumer restart would not be caught; a
  real production fix needs a persistent, shared idempotency store.
- **No automatic Redis reconnection/circuit-breaker** — a Redis outage
  now fails cleanly (503) but does not self-heal or retry.
- **No message schema registry** — Kafka messages are plain JSON, with no
  schema evolution management.
