# Phase 9 Summary — Monitoring & Observability

This report documents a passive observability layer added to the Phase 7/8
FastAPI service. All numbers are from real runs — a fast synthetic-data
test suite, a real demonstration against the actual model and real data,
and a real Docker container build/run with the new endpoints validated
live.

**Phase 9 is service observability. It is NOT automated ML drift
detection** — no drift alerts, no retraining triggers, no statistical
comparison against training-time distributions exist in this phase. See
"Future path toward drift detection" below.

## 1. Phase objective

Answer "can we understand how the service is behaving while it processes
transactions?" by adding visibility into request activity, latency,
failures, predictions, decisions, and basic data quality — without
changing any approved inference behavior.

## 2. Observability architecture

```
src/monitoring/
    __init__.py
    metrics.py            # MetricsRegistry: thread-safe counters/gauges
    middleware.py          # MetricsMiddleware: passive per-request observer
    data_quality.py          # safe, aggregate-only signal extraction
    prometheus_export.py      # Prometheus text-format rendering
    logging_config.py          # structured key=value logging
```

Cleanly separated from inference logic: nothing in `src/monitoring/`
imports or calls the model, the feature pipeline, or the decision policy.
`src/api/main.py`'s routes call into both `src/engine/risk_engine.py` (for
inference, unchanged) and `src/monitoring/` (for observation) as two
independent concerns — the monitoring calls in `/predict` are placed
strictly AFTER `engine.process_transaction()` already returned its final
result (see the non-interference section for direct verification).

## 3. Monitoring design principles (and how they're enforced, not just stated)

- **Passive**: `MetricsMiddleware.dispatch()` awaits `call_next(request)`
  and returns its result completely unchanged — it reads the response's
  status code and elapsed time only, never its body. `/predict` calls
  `metrics.record_prediction_success(...)` using values already present in
  `result` (the dict `engine.process_transaction()` returned) — nothing
  computed by the monitoring layer feeds back into the response.
- **Fail-safe**: every metrics call site uses a `None`-returning accessor
  (`get_metrics_registry_or_none()`) or catches the initialization
  `RuntimeError` — if metrics are ever unavailable, `/predict` still works
  (verified directly: `test_monitoring_absent_does_not_break_predict`).
- **In-memory, resettable**: `MetricsRegistry` is a plain Python object
  installed once at startup; a fresh instance means a clean state, and
  nothing here persists across a restart.

## 4. Request metrics

`MetricsMiddleware` (installed via `app.add_middleware`, applies to every
route automatically) records, per request: total count, count by
endpoint, count by HTTP status class (`2xx`/`4xx`/`5xx`), and elapsed
wall-clock latency (both overall average and per-endpoint average).
Latency is a simple running mean — not a claim of production-grade
percentiles (p50/p95/p99), which this lightweight in-memory design does
not implement; documented explicitly rather than implied.

## 5. Prediction metrics

Three counters distinguish exactly what the brief asked for:
`prediction_attempts` (reached the engine, i.e. passed request-schema
validation), `prediction_successes`, `prediction_failures` (engine-level
validation/processing failures). A request rejected by Pydantic's own
schema validation (missing field, wrong type, `isFraud` present) never
increments `prediction_attempts` at all — it's counted separately as a
`request_validation_failure`, so **a failed validation request is never
counted as a successful prediction**, verified directly
(`test_invalid_request_counted_as_failure_not_success`).

## 6. Decision metrics

`decision_counts` (`APPROVE`/`REVIEW`/`BLOCK`), `total`, and a computed
`distribution` (each count divided by the total). No fraud labels are
used or needed — decisions are recorded directly from the policy's own
output, exactly as `/predict` already returns them.

## 7. Risk-score monitoring

Count, mean, min, max, plus a 10-bucket decile histogram (`[0.0-0.1)`,
`[0.1-0.2)`, ..., `[0.9-1.0]`) — chosen for simplicity/interpretability
and deliberately NOT tied to the decision policy's approve/block
thresholds, so this histogram stays a neutral distribution summary
independent of whatever policy happens to be loaded. Nothing here
recalibrates or redesigns the model's output.

## 8. Data-quality monitoring

Transaction amount summary (count/mean/min/max) extracted via
`extract_transaction_amount()` — a single safe number per prediction, not
the full payload. `request_validation_failures` and
`engine_validation_failures` are tracked as data-quality signals as well
as error signals (they answer both "is the input data healthy" and "what
kind of failure occurred"). Full transaction payloads are never logged or
stored anywhere in the monitoring layer.

## 9. Error monitoring

Four distinguished categories: `request_validation_failures` (schema-level,
never reached the engine), `engine_validation_failures` (engine rejected
otherwise-well-formed input), `engine_not_initialized_errors` (503 —
distinguished from a client error), `internal_errors` (genuinely
unexpected exceptions, via a new catch-all `Exception` handler that
returns only a generic message — verified it does not intercept more
specific handlers like `HTTPException`/`RequestValidationError`, which
Starlette dispatches to their own more-specific handlers first). No
traceback, internal path, or raw payload is ever included in any HTTP
response — unchanged from Phase 7's already-tested guarantee.

## 10. Structured logging

`key=value`-style lines via a dedicated `fraud_api` logger:
`event=request_received`, `event=prediction_processed
transaction_id=... decision=... risk_score=... duration_ms=...`,
`event=validation_failure source=... detail=...`. Never logs the full
transaction payload, individual feature values, or `isFraud` (which the
engine rejects outright before logging could even see it). Real captured
example (from `scripts/demo_monitoring.py`):

```
2026-09-02 13:05:53,161 level=INFO logger=fraud_api event=prediction_processed transaction_id=3488959 decision=APPROVE risk_score=0.000735 duration_ms=68.56
```

## 11. `/metrics` design

Prometheus **text exposition format** (`# HELP`/`# TYPE` comments, plain
counters/gauges, label syntax for per-endpoint/per-decision breakdowns),
hand-rendered with the standard library (`src/monitoring/prometheus_export.py`)
— no `prometheus_client` dependency added. **This means the format is
scrape-compatible with a real Prometheus server; it does NOT mean a
Prometheus server or Grafana is deployed anywhere in this project** —
neither is, and none was added this phase.

## 12. `/monitoring/summary` design

The same underlying `MetricsRegistry.snapshot()` data as `/metrics`, in
nested JSON — `requests`, `predictions`, `decisions`, `risk_scores`,
`data_quality`, `errors`, plus `uptime_seconds`. Metrics reset when the
process restarts, documented explicitly in both the endpoint description
and this report (no persistence in this phase).

## 13. Non-interference validation — the critical result

Two dedicated tests directly compare a plain `RiskDecisionEngine` call
against the SAME transaction processed through the fully-monitored API
(middleware + metrics recording + structured logging all active):

- `test_monitoring_does_not_change_risk_scores_or_decisions`: 5 real
  transactions, risk scores match to `<1e-9`, decisions identical.
- `test_monitoring_does_not_change_behavioral_state_updates`: a 4-transaction
  sequence for one entity, processed once directly and once through the
  monitored API — final behavioral state (`count`, `sum_amt`, `last_time`)
  identical between the two.

Both passed. Monitoring genuinely does not touch the inference path.

## 14. Docker compatibility — REAL, EXECUTED

Rebuilt the Phase 8 image with Phase 9's changes (same base-image
substitution and CA-trust workarounds documented in the Phase 8 report —
this sandbox's Docker/registry situation is unchanged) and validated live:

```
$ docker build --build-arg BASE_IMAGE=ubuntu-noble-local:latest -t fraud-risk-api:local .
...
Successfully built 62d032e40ad6
Successfully tagged fraud-risk-api:local

$ docker run -d --name fraud-risk-api-test -p 8123:8000 fraud-risk-api:local
$ docker inspect --format='{{.State.Health.Status}}' fraud-risk-api-test
healthy   # after 6 seconds — unchanged from Phase 8

$ curl http://127.0.0.1:8123/health
{"status":"ok","model_loaded":true,"policy_loaded":true}

$ curl -X POST http://127.0.0.1:8123/predict -H "Content-Type: application/json" \
  -d '{"TransactionID": 6000001, "TransactionDT": 100000, "TransactionAmt": 60.0, "card1": 33333, "ProductCD": "W"}'
{"transaction_id":6000001,"risk_score":0.0616832113578294,"decision":"APPROVE", ...}

$ curl http://127.0.0.1:8123/metrics | grep -v '^#' | head -5
fraud_api_requests_total 4
fraud_api_requests_by_endpoint_total{endpoint="/health"} 2
fraud_api_requests_by_endpoint_total{endpoint="/metadata"} 1
fraud_api_requests_by_endpoint_total{endpoint="/predict"} 1
fraud_api_requests_by_status_class_total{status_class="2xx"} 4

$ curl http://127.0.0.1:8123/monitoring/summary
{"uptime_seconds":4.69,...,"predictions":{"attempts":1,"successes":1,"failures":0,...},
 "decisions":{"counts":{"APPROVE":1,"REVIEW":0,"BLOCK":0},"total":1,...}, ...}
```

`/health`, `/metadata`, `/predict` unchanged and confirmed working; `/metrics`
and `/monitoring/summary` confirmed working with accurate, real-time-updated
counts; the Phase 8 healthcheck (`/health`-based) unaffected. Container
stopped and removed cleanly afterward. No new runtime dependency was
needed (the monitoring layer uses only the standard library plus FastAPI
components already present), so `requirements.txt` did not change for
this phase.

## 15. Demonstration results (real model, real data)

`python scripts/demo_monitoring.py` — real run, real predictions:

```
Step 1: predictions.successes = 0, decisions.total = 0  (clean start)
Step 3: 8 valid transactions -> 8x "200 ... decision=APPROVE" (all real predictions)
Step 4: 1 invalid transaction (negative amount) -> 422
Step 6: predictions.successes = 8, predictions.failures = 0
        decisions.counts = {'APPROVE': 8, 'REVIEW': 0, 'BLOCK': 0}
        risk_scores.count = 8, risk_scores.mean = 0.0121
        errors.request_validation_failures = 1
```

All assertions in the script passed: successes matched the number of
valid requests sent, the invalid request was counted as a failure (not a
success), and decision totals matched successful predictions exactly.

## 16. Test suite results

**206/206 Python tests passed** (179 from Phases 0-8 + 27 new: 16 unit
tests for `MetricsRegistry` in isolation, 11 FastAPI integration tests
including the two non-interference tests above). All existing Phase 7
API tests re-run unchanged and still passing, confirming the monitoring
additions didn't alter existing behavior.

## 17. Errors encountered and fixes

1. **Middleware registered before the registry existed.** The first
   implementation tried to pass a concrete `MetricsRegistry` instance to
   `app.add_middleware(...)` at import time — but `add_middleware` runs
   when `main.py` is imported, before `lifespan` has run and actually
   created the registry. Fixed by having the middleware accept a
   zero-argument `registry_provider` callable
   (`deps.get_metrics_registry_or_none`) and look up the current registry
   fresh on every request, rather than binding to a fixed instance at
   construction time.
2. **Demo script OOM from double-loading the raw dataset.** Running
   `scripts/demo_monitoring.py` with the host-default `WARM_START_STATE=true`
   meant the API's own startup loaded the ~683MB `train_transaction.csv`
   for warm-start, and the script's own Step 2 (fetching sample
   transactions) loaded it a second time — real memory pressure on this
   environment's 3.9GB RAM, and the process was killed. Fixed by having
   the demo script set `WARM_START_STATE=false` before importing the app
   (documented in the script itself as specific to this demo's own
   double-load risk, not a change to the documented default for any other
   entry point).

## 18. Known limitations

- **In-memory metrics only** — no persistence; a process restart (or
  container restart) resets every counter to zero. This is documented,
  not hidden, per the brief's explicit instruction.
- **Simple running-mean latency, not real percentiles** — `avg_latency_ms`
  is exactly what it says; no p50/p95/p99 is computed or claimed.
- **No cross-request correlation IDs** — structured logs record
  per-event metadata but don't yet thread a single ID through an entire
  request's lifecycle for easy log correlation.
- **No dashboard** — `/metrics` and `/monitoring/summary` are
  machine/human-readable endpoints; no Grafana or other visualization
  layer exists (explicitly out of scope).
- **Single-process metrics** — like the engine's behavioral state, this
  design does not support multiple replicas sharing one metrics view;
  each process has its own independent counters.
- **`internal_errors` counter is new and unexercised by real traffic in
  this session** — verified structurally (handler registration doesn't
  intercept more specific handlers) but not triggered by an actual
  unexpected production-style failure during validation, since none
  occurred.

## 19. Future path toward drift detection

Phase 9 gives the raw material a future drift-detection phase would need
(risk-score distributions, decision-rate distributions, data-quality
counters over time) but does not implement any comparison against a
baseline, statistical drift test (e.g. PSI, KS-test), alerting, or
automated retraining trigger — all of that is future work, explicitly out
of scope here per the brief.

## What was intentionally NOT implemented

Automated drift detection, model retraining, retraining pipelines, a real
Prometheus server deployment, Grafana deployment, databases, Kafka, Redis,
cloud monitoring, Kubernetes, authentication, dashboard redesign, new
fraud models, threshold redesign — all reserved for later phases, per the
Phase 9 scope restrictions.
