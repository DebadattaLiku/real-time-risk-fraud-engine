# Production Architecture Audit

Real audit of the repository as it stood before this upgrade, performed
before any code was written. This report is the Phase 1 deliverable of
the production-grade upgrade; the changes it recommends are implemented
and tested in the phases that follow (see `reports/production_readiness.md`
for the consolidated final state).

## Current architecture (as found)

```
IEEE-CIS transactions -> chronological split -> LightGBM (+behavioral features)
    -> frozen 3-way decision policy -> RiskDecisionEngine (in-memory state)
    -> FastAPI -> Docker -> Prometheus-style /metrics -> PSI/KS drift -> local
    JSON model governance -> Streamlit dashboard
```

316 tests passing, real champion `fraud-risk-lightgbm-v1`, real measured
metrics (PR-AUC 0.5483, ROC-AUC 0.9058 on the full chronological test
set). All of this was verified accurate by re-running the existing test
suite and re-reading the real saved evaluation artifacts before any
change was made.

## Existing data flow

```
Transaction -> validate -> read state (read-only) -> generate behavioral
features from that state -> preprocess -> predict -> decide -> update
state (last)
```

## Existing state flow

`BehavioralStateManager` held a single Python dict (`self._states`),
process-local, non-persistent, reset on process restart. This was the
single most consequential finding of this audit: it is what makes
horizontal scaling structurally impossible without a change (see
"Components that need adapters" below) — every API process would have an
independent, inconsistent view of any given entity's history.

## Current bottlenecks (as measured, not assumed)

- **State store**: in-memory only — bounds the system to a single
  process, and loses all history on restart. Real measured cost of
  externalizing it (Redis): lookup p50 goes from ~0.0001ms (in-memory) to
  ~0.05ms (local Redis) — see `reports/streaming_benchmark.md`. Small in
  absolute terms, but non-zero, and this project now measures it rather
  than assuming it away.
- **Model inference**: real measured p50 ~5.1ms for a single
  `predict_proba` call on the 403-feature real model (see
  `reports/streaming_benchmark.md`) — not a bottleneck at the volumes this
  project has actually tested.
- **SHAP explanation** (new in this upgrade): real measured p50 ~14-18ms —
  meaningful relative to the ~5ms base inference cost, which is exactly
  why it was implemented as a separate, opt-in endpoint (`/predict/explain`)
  rather than folded into the default `/predict` path.

## Components that could be reused directly

`RiskDecisionEngine`, `FeaturePipeline`, `LightGBMPreprocessor`, the
frozen `DecisionPolicy`, the FastAPI app structure, the Prometheus-style
`MetricsRegistry`, the drift and governance subsystems — all reused
unchanged. This upgrade's own discipline (verified by tests) is that
**none of these were duplicated or reimplemented** for streaming/Redis/
SHAP/MLflow; every new component is a thin adapter around them.

## Components that needed adapters

- **State storage**: extracted a `StateBackend` interface (`get`/`put`/
  `items`/`clear`/`count`) so `BehavioralStateManager`'s feature-computation
  math is completely unchanged and storage-agnostic. `InMemoryStateBackend`
  (default) and `RedisStateBackend` (new) both implement it.
- **Transaction ingestion**: a new `FraudProcessingHandler` adapts a
  transaction dict (from Kafka, or from an in-memory test double) into
  exactly the same `RiskDecisionEngine.process_transaction()` call
  `/predict` makes — verified identical by a dedicated test.

## Risks introduced by streaming (identified, and how each is handled)

- **At-least-once delivery -> duplicate processing**: a REAL gap was
  found (a duplicate transaction ID processed twice double-counts
  behavioral state) and FIXED at the `FraudProcessingHandler` layer with
  a bounded idempotency cache — NOT fixed inside `RiskDecisionEngine`
  itself, since delivery semantics are a streaming concern, not an
  inference concern. See `reports/failure_testing.md`.
- **Consumer failure mid-message**: handled via bounded retry +
  dead-letter routing (`TransactionConsumer`/`FakeTransactionConsumer`),
  tested against both a real handler exception and a real engine-level
  validation rejection.
- **No real Kafka broker available in this development environment**:
  network egress restrictions block Apache/Confluent/Bitnami/Redpanda,
  and no Ubuntu apt package provides a broker (confirmed by direct
  inspection: `apt-cache search kafka` returns only unrelated Go client
  libraries). This is the single largest, most consequential finding of
  this audit for the streaming component specifically — it means genuine
  Kafka throughput/latency numbers do not exist anywhere in this project,
  by necessity, not by omission. `docker-compose.kafka.yml` provides a
  real, standard broker definition for use on a machine with normal
  Docker Hub access.

## State consistency concerns

Running more than one API replica with `STATE_BACKEND=memory` (the
Phase-8 Docker default) would give each replica an independent,
inconsistent view of behavioral state — silently wrong, not just slow.
This is now explicitly documented in `k8s/api-deployment.yaml`'s
comments and is the reason `STATE_BACKEND=redis` exists: Redis makes
state genuinely shared across processes, verified by
`test_consumer_restart_with_redis_backend_preserves_state`.

## Failure modes (see `reports/failure_testing.md` for the full, executed test-by-test account)

Redis failure, Kafka unavailability, model unavailability, invalid
transactions, duplicate messages, consumer restart, and API restart were
all tested with REAL induced failures where practical (the local
`redis-server` process was genuinely killed and restarted during test
execution) or realistic simulation where a real induction wasn't
practical (API process restart, simulated as a fresh engine object).

## Serialization concerns

`EntityState.to_dict()`/`from_dict()` (already existing, from Phase 13's
numerical-stability fix) round-trip cleanly to JSON — reused directly for
Redis storage, no new serialization format invented. Kafka messages use
plain JSON via `json.dumps`/`json.loads` — no schema registry, no Avro/
Protobuf; a real limitation for a large-scale production system (schema
evolution would be unmanaged) but proportionate to this project's actual
scope, and documented as such.

## Configuration concerns

New environment variables (`STATE_BACKEND`, `REDIS_URL`, `KAFKA_BROKERS`,
`ENABLE_SHAP_EXPLAINABILITY`) all have sensible local defaults and are
documented in `.env.example` and `k8s/configmap.yaml`. No credentials are
hard-coded anywhere (verified by a repository-wide grep in this session).

## Security concerns

No new attack surface beyond what Phase 13 already reviewed: the new
`/predict/explain` endpoint has the same (lack of) authentication as
every other endpoint — a pre-existing, already-documented limitation, not
a new one. Redis and Kafka connection strings come only from environment
variables. `k8s/secret.example.yaml` contains only placeholder values.

## Deployment concerns

Kubernetes manifests were authored and YAML-syntax-validated but never
applied to a real cluster (none was available). AWS is documented as a
mapping (Kafka -> MSK, Redis -> ElastiCache, etc.) with zero real AWS
resources created or claimed. See `reports/production_readiness.md` for
the full, itemized "implemented locally" vs. "design only" accounting.
