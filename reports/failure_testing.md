# Failure Testing

All scenarios below were executed as real, passing tests in
`tests/test_production_failure_scenarios.py` and
`tests/test_production_streaming.py` in this session — including a REAL
induced Redis process failure (the local `redis-server` was genuinely
killed via `redis-cli shutdown nosave` and restarted mid-test-run), not
simulated. Where a real induction wasn't practical (e.g. killing a live
API process), the equivalent condition was exercised through the same
code path a real failure would use.

## Redis failure

**Test**: `test_redis_becomes_unavailable_mid_operation_raises_clean_error`
— genuinely stopped the local Redis server, confirmed
`RedisStateBackend.get()` raises `StateBackendUnavailableError` (not a
hang, not a raw `redis-py` traceback), then restarted Redis.

**Expected/actual behavior**: `RedisStateBackend` fails cleanly, AND (a
real gap found during this same test, then fixed) a live API request
hitting this failure now gets a clean, documented `503
state_backend_unavailable` response — via a new dedicated
`handle_state_backend_unavailable` exception handler in `src/api/main.py`
— rather than a generic, unhandled 500. Verified end-to-end in the same
test: a real `TestClient` request against a real (broken) engine returns
exactly this response. This mirrors the pattern already established for
"engine not initialized" (`RuntimeError` -> 503), and required registering
a handler for the more-specific `StateBackendUnavailableError` subclass
so Starlette's exception dispatch prefers it over the generic
`RuntimeError` handler for this exact case, without changing that
handler's behavior for genuine engine-not-initialized errors.

## Kafka unavailable

**Test**: `test_kafka_unavailable_raises_clean_error_not_hang`.
**Behavior**: `TransactionProducer.send()` raises `StreamingUnavailableError`
within the configured timeout (verified under 5 seconds, using an
explicit `api_version` to avoid `kafka-python`'s observed indefinite-hang
behavior during auto-negotiation against an unreachable broker — a real
finding from this session, documented in
`src/streaming/kafka_client.py`'s module docstring).

## Model unavailable

**Test**: `test_model_unavailable_returns_clean_error_not_crash`
(re-confirming Phase 7's own `handle_runtime_error` handler, not
reimplementing it). **Behavior**: a clean 503, never a crash or a leaked
traceback — this was already correct before this upgrade.

## Invalid transaction

**Test**: `test_invalid_transaction_rejected_cleanly`. **Behavior**:
`TransactionValidationError` raised for a disallowed field (`isFraud`);
already extensively covered by the pre-existing Phase 7/9 test suite —
re-confirmed here as part of this consolidated failure-testing pass.

## Duplicate Kafka message — is processing idempotent?

**Real, measured answer, in two parts:**

1. **At the raw engine layer**: NO. `test_duplicate_message_processing_is_not_idempotent_by_default`
   proves that calling `RiskDecisionEngine.process_transaction()` twice
   with the identical transaction double-counts it in behavioral state
   (`bhv_prev_txn_count` increments twice). This is real, and by design —
   the engine has no concept of message delivery semantics, and should
   not.
2. **At the streaming consumer layer**: YES, within a bounded window. A
   real gap was found and FIXED: `FraudProcessingHandler` now maintains a
   bounded in-memory cache of recently-seen transaction IDs
   (`idempotency_cache_size`, default 10,000) and skips re-processing a
   duplicate, returning the cached original result instead. Verified by
   `test_fraud_processing_handler_is_idempotent_for_duplicate_transaction_ids`:
   the engine is called exactly once for two identical deliveries, and
   behavioral state is confirmed NOT double-updated.

**Honest limit of the fix**: the cache is bounded and in-memory — a
duplicate arriving after the cache has evicted that transaction ID (more
than 10,000 distinct transactions later, by default), or after a consumer
process restart, would NOT be caught. A production-grade fix would need a
persistent, shared idempotency store (e.g. a Redis SET with TTL) — not
implemented here; see "Not implemented" below.

## Consumer restart

**Test**: `test_consumer_restart_with_redis_backend_preserves_state` and
`test_consumer_restart_with_inmemory_backend_loses_state` — both real,
both passing, deliberately showing the two different real behaviors:

- **With `STATE_BACKEND=redis`**: a "restarted" consumer (a fresh engine
  object, fresh Python-side Redis client) sees the exact same state a
  prior process left behind — verified directly.
- **With the default in-memory backend**: a "restarted" consumer has NO
  memory of prior state — the real, documented tradeoff of the default
  configuration, not glossed over.

## API restart

**Test**: `test_api_restart_reloads_model_and_policy_correctly` — two
independently-built engines (simulating "before" and "after" a restart)
produce identical risk scores and decisions for the same transaction,
confirming the model/policy reload path is deterministic and correct.

## What was intentionally NOT implemented (honest limitations)

- **No persistent, cross-restart idempotency store** — the current fix
  (bounded in-memory cache) closes the common case (a duplicate arriving
  shortly after the original, before a restart) but not the full
  production requirement.
- **No automatic Redis reconnection/circuit-breaker logic** — a genuinely
  unavailable Redis stays unavailable until the next request tries again;
  no backoff/retry loop was added around Redis operations specifically
  (Kafka's producer/consumer DO have retry logic; Redis operations
  currently do not). The fix above makes the FAILURE clean (503, not a
  crash) — it does not make Redis automatically recover or retry.
- **No chaos/fault-injection testing beyond what's listed above** — e.g.
  network partition simulation, slow-Redis-under-load, or Kafka
  broker-leader-election scenarios were not tested (the last one couldn't
  be, given no broker was available at all).
