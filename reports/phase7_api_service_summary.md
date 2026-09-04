# Phase 7 Summary — FastAPI Risk Scoring Service

This report documents a local FastAPI service exposing the approved Phase
6 `RiskDecisionEngine` over HTTP. All numbers are from real runs — both an
automated synthetic-data test suite and a genuine live verification
against the real Phase 4/5 model, real data, and (separately) a real
`uvicorn` server process.

**This is a local engineering/demonstration service, not a deployed
production financial system.** No authentication, no database
persistence, no distributed infrastructure — see the Limitations section.

## 1. API architecture

```
src/api/
    __init__.py
    schemas.py        - Pydantic request/response models only
    dependencies.py    - engine lifecycle (build/get/set), no ML logic
    main.py             - routes; each one is a thin call into the engine
```

No route file contains feature generation, model prediction, or decision
logic — every one of those already lives in `src/engine/risk_engine.py`
(Phase 6), `src/models/`, and `src/decision/policy.py` (Phase 5), and is
called, not reimplemented.

## 2. Why the API layer is thin

The brief is explicit that the API must not duplicate approved logic. In
practice this means `POST /predict`'s entire body is: build a full
transaction dict from the validated request, call
`engine.process_transaction(transaction)`, and repackage the fields of
interest into the response schema. If Phase 6's engine changes, the API
changes with it automatically — there is no second implementation to keep
in sync.

## 3. Application lifecycle

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    engine, policy_config = deps.build_engine(warm_start=True)
    deps.set_engine(engine, policy_config)
    yield
```

`build_engine()` loads the (cached, pickled) Phase 4/5 model bundle,
loads the frozen Phase 5 policy from `config/decision_policy.yaml`, and
warm-starts behavioral state from TRAIN+VALIDATION history via
`bulk_initialize` (the same approach Phase 6 used) — all ONCE, at process
startup, not per request. The resulting `RiskDecisionEngine` (and its
`BehavioralStateManager`) is held in a module-level singleton
(`src/api/dependencies.py`) for the process's lifetime, which is what
makes the stateful behavior possible. Real measured startup time:
**~31-41 seconds** (dominated by loading the raw 683MB transaction CSV and
computing the temporal split to build warm-start history — the model
bundle itself loads from a 3.6MB pickle in well under a second).

## 4. Model-loading strategy

The model, preprocessing pipelines, and schema are loaded from
`data/interim/phase6_model_bundle.pkl` (built in Phase 6) rather than
retrained — `build_or_load_bundle()` is reused unchanged from
`src/run_phase6_simulation.py`. If that file is ever missing, the same
function transparently retrains and re-caches it (reproducing Phase 4/5's
exact hyperparameters/split/seed), so the API never silently runs with a
different model than the rest of this project.

## 5. Stateful engine behavior

Because the engine is a process-wide singleton, sequential requests
correctly see each other's history:

```
Request 1 (card1=7919, TransactionID=3488959) -> predict -> update state
Request 2 (card1=7919, TransactionID=3488962) -> sees request 1's history
```

Verified directly (not just assumed) with real data via
`scripts/demo_api.py`. A `POST /dev/reset-state` endpoint exists purely
for local testing/demonstration and is explicitly tagged
`development-only` in the OpenAPI docs — the brief's instruction not to
expose unsafe reset functionality as ordinary API behavior is honored by
naming, tagging, and documentation, not by hiding it outright (a real
deployment would need to remove or access-control this route entirely,
noted in Limitations).

## 6. Request schema

`TransactionRequest` (`src/api/schemas.py`) explicitly types the four
structurally-required fields — `TransactionID`, `TransactionDT`,
`TransactionAmt` (must be `>= 0`), `card1` — plus `ProductCD`. The
remaining ~386 Phase 1 raw columns (mostly opaque `V*`/`C*`/`D*`/`M*`
Vesta features) are accepted via Pydantic's `extra="allow"` rather than
hand-declared one by one; any not supplied are filled with `null` before
reaching the engine (`build_full_transaction_dict`), consistent with Phase
1's existing missing-value tolerance. `isFraud` is explicitly forbidden by
a model validator (see the real bug this required fixing, below).

## 7. Response schema

`PredictResponse` returns `transaction_id`, `risk_score` (bounded `[0,1]`
by the schema itself), `decision`, plus `model_version`, `policy_version`,
`processing_status`, `processing_time_ms`. **Behavioral features are
deliberately NOT included** — they remain internal engine state, per the
brief. The docstring states plainly: "Higher `risk_score` means higher
predicted fraud risk."

## 8. Endpoint definitions

| Method | Path | Purpose |
|---|---|---|
| POST | `/predict` | Score one transaction, return risk score + decision |
| GET | `/health` | Service/model/policy operational status |
| GET | `/metadata` | Model type/version, policy name/version/thresholds, supported decisions |
| POST | `/dev/reset-state` | **Dev/testing only** - clears all behavioral history |

Plus FastAPI's automatic `/docs` (Swagger UI) and `/openapi.json` —
confirmed reachable on both the in-process `TestClient` and a real live
`uvicorn` server.

## 9. Error-handling strategy

| Condition | Response |
|---|---|
| Pydantic schema violation (missing required field, negative amount, wrong type, `isFraud` present) | `422`, FastAPI's standard validation error body |
| Engine-level `TransactionValidationError` (e.g. a raw column genuinely missing from the built dict) | `422`, custom handler, `{"error": "invalid_transaction", "detail": ...}` |
| Engine/model not initialized (`RuntimeError`) | `503`, `{"error": "engine_not_initialized", "detail": ...}` |
| Any other unexpected exception | Starlette's default `500` (no traceback leaked — confirmed by a dedicated test checking the response body contains no `Traceback` or `.py` file-path text) |

No Python traceback or internal file path is ever returned to the client.

## 10. Example `/predict` results (real model, real data)

```
POST /predict  (TransactionID=3488959, card1=7919, first transaction)
200 {
  "transaction_id": 3488959, "risk_score": 0.0009101570052104251,
  "decision": "APPROVE", "model_version": "phase4_lightgbm_transaction_plus_behavioral_v1",
  "policy_version": "balanced", "processing_status": "success",
  "processing_time_ms": 408.99
}

POST /predict  (TransactionID=3488962, SAME card1=7919 - sees txn 1's history)
200 {
  "transaction_id": 3488962, "risk_score": 0.0008958150995441318,
  "decision": "APPROVE", "processing_status": "success",
  "processing_time_ms": 54.22
}

POST /predict  (TransactionAmt=-1.0, invalid)
422 {"detail": [{"type": "greater_than_equal", "loc": ["body", "TransactionAmt"],
                  "msg": "Input should be greater than or equal to 0", ...}]}
```

(Full transcript: `python scripts/demo_api.py`.)

## 11. `/health` and `/metadata` results (real, live)

```
GET /health   -> 200 {"status": "ok", "model_loaded": true, "policy_loaded": true}
GET /metadata -> 200 {
  "api_version": "0.1.0",
  "model_type": "LightGBM (transaction-level + card1 behavioral features)",
  "model_version": "phase4_lightgbm_transaction_plus_behavioral_v1",
  "policy_name": "balanced", "policy_version": "balanced",
  "supported_decisions": ["APPROVE", "REVIEW", "BLOCK"],
  "approve_threshold": 0.32061562877852007, "block_threshold": 0.631136794838037
}
```

Also confirmed against a **real, separately-launched `uvicorn` process**
(`uvicorn src.api.main:app`) reached over real HTTP via `curl` on
`127.0.0.1` — not just the in-process `TestClient` — with identical
results, and `/docs` returning `200`.

## 12. Direct-engine vs. API parity

Two checks, both real:

1. **Synthetic** (automated test, `test_direct_engine_vs_api_parity`): two
   independently-built synthetic engines from the same seed, one called
   directly, one through the API — scores match to `<1e-6`, decisions
   identical.
2. **Real data** (manual verification, this session): two independently
   warm-started engines built from the real Phase 4/5 bundle and real
   TRAIN+VALIDATION history, compared on the first 10 real TEST
   transactions — **0/10 mismatches**, exact score and decision agreement.

## 13. API documentation status

`/docs` (Swagger UI) and `/openapi.json` both confirmed reachable and
correctly populated (all four routes listed, response models attached) —
verified against both the in-process TestClient and a real live server.
Every route has a `summary`/`description`; `TransactionRequest` and
`PredictResponse` fields carry `Field(..., description=...)` explaining
units, direction (`higher = higher fraud risk`), and the `card1`
pseudo-entity caveat inline.

## Local usage instructions

```bash
# From the repository root:
pip install -r requirements.txt

# Run the in-process demonstration (no server process needed):
python scripts/demo_api.py

# Or run a real live server:
uvicorn src.api.main:app --reload
# then, in another terminal:
curl http://127.0.0.1:8000/health
# or open http://127.0.0.1:8000/docs in a browser

# Run the automated test suite:
pytest tests/test_phase7_api.py -v
```

## Test suite results

**163/163 passed** across the full repository (12 new Phase 7 API tests +
151 from Phases 0-6). Phase 7 tests cover: health (initialized and
not-initialized states), metadata content, valid prediction shape/bounds,
missing-field/negative-amount/`isFraud`-present rejection (422),
no-traceback-leakage, sequential stateful requests, dev-only state reset,
invalid-request-does-not-update-state, and direct-engine-vs-API parity.
Phase 0 and Phase 1 scripts re-verified working on real data.

## Errors encountered and fixes

Two genuine bugs, both found by testing before being accepted as done —
not by inspection alone:

1. **`isFraud` was silently dropped, not rejected.** The initial
   implementation filtered the incoming request down to only the columns
   the engine's schema expects (which deliberately excludes `isFraud`)
   before ever calling `process_transaction` — so a client-supplied
   `isFraud` field was quietly discarded instead of triggering the
   engine's own rejection, which never got the chance to see it. Caught
   by a real test failure (`200` instead of the expected `422`). Fixed
   with a Pydantic `model_validator(mode="before")` on `TransactionRequest`
   that rejects `isFraud` at the schema layer, before the column-filtering
   step can hide it.
2. **A single-row transaction with a `None` (JSON `null`) numeric field
   crashed prediction.** `pd.DataFrame([{"D1": None, ...}])` cannot infer
   a numeric dtype from a lone `None` the way a batch DataFrame can from a
   column of real floats — pandas leaves the column as `object`, which
   LightGBM's native `predict()` rejects outright
   ("pandas dtypes must be int, float or bool"). This was NOT caught by
   Phase 6's simulation (which always built rows from real pandas `NaN`
   values, not JSON `null`/Python `None`) — it surfaced only during real
   end-to-end API testing with a genuinely high-missingness transaction.
   Fixed in `RiskDecisionEngine.process_transaction` itself (not just the
   API layer), by explicitly casting numeric schema columns to `float64`
   on the single-row DataFrame before preprocessing — so every caller
   (API, direct use, future tooling) gets the fix, not just this phase's
   route. A regression test was added to Phase 6's engine test file.
   Verified fixed against 30 real, diverse-missingness test transactions
   after the fix (30/30 succeeded).

## Known limitations

- **No authentication or authorization** — anyone who can reach the
  process can call every route, including `/dev/reset-state`.
- **No persistence across restarts** — behavioral state lives in the
  single engine process's memory only; restarting the service loses all
  state accumulated since the last warm-start (though warm-start itself
  correctly rebuilds from historical TRAIN+VALIDATION data every time).
- **Single-process, no concurrency testing** — no load testing, no
  verification of behavior under concurrent overlapping requests to the
  same `card1` (a real race condition could exist there; not evaluated).
- **`/dev/reset-state` is a real, callable, unauthenticated endpoint** —
  clearly named and tagged as dev-only, but a genuine production system
  would need to remove it or gate it behind real access control rather
  than rely on naming convention alone.
- **Slow first request per process** (~30-40s) due to the warm-start
  data load — acceptable for a local demo, not for a service expected to
  scale instances up/down frequently.
- **No database or persistent storage** — explicitly out of scope per the
  brief.

## Future deployment path

A genuine production deployment would need, at minimum: authentication
and authorization, a persistent/distributed store for behavioral state
(not an in-process dict), horizontal scaling with shared state (the
current single-process-singleton design does not support multiple
replicas correctly, since each would have independent, diverging state),
containerization (Docker) and orchestration, structured logging and
monitoring/alerting, load and concurrency testing, a real request queue
or streaming ingestion layer (Kafka, etc.), rate limiting, and a plan for
periodic model retraining and redeployment. None of this exists yet — all
explicitly out of scope for Phase 7, per the brief.

## What was intentionally NOT implemented

Cloud deployment, Docker, Kubernetes, Kafka, distributed streaming,
authentication, payment processing, real banking integration, database
persistence, dashboard redesign, model retraining, new fraud models,
threshold redesign — all reserved for later phases, per the Phase 7 scope
restrictions.
