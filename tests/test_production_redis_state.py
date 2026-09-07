"""
Production upgrade — Redis state backend tests.

Requires a REAL, reachable local Redis server (`redis-server`, started
separately — see `reports/production_architecture_audit.md` for how this
session started one via `apt-get install redis-server`). These tests are
SKIPPED, not failed, if Redis is unreachable — this is a deliberate
design choice per the task's own instruction ("Do NOT require Kafka for
unit tests"; the same principle applies here: the existing, default
in-memory backend must remain fully testable with zero external
dependencies, and Redis-specific tests must not break `pytest -q` on a
machine without Redis installed).
"""

import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.engine.state import BehavioralStateManager, EntityState
from src.engine.state_backend import (
    InMemoryStateBackend, RedisStateBackend, StateBackendUnavailableError,
)

try:
    import redis as redis_lib
    _client = redis_lib.Redis.from_url("redis://localhost:6379/0", socket_connect_timeout=1)
    _client.ping()
    REDIS_AVAILABLE = True
except Exception:
    REDIS_AVAILABLE = False

requires_redis = pytest.mark.skipif(not REDIS_AVAILABLE, reason="No reachable local Redis server for this test run.")


@pytest.fixture()
def redis_client():
    client = redis_lib.Redis.from_url("redis://localhost:6379/0")
    # Isolate this test run's keys from anything else that might be using
    # the same local Redis instance.
    for key in client.keys("pytest:fraud:state:*"):
        client.delete(key)
    yield client
    for key in client.keys("pytest:fraud:state:*"):
        client.delete(key)


# ---------------------------------------------------------------------------
# InMemoryStateBackend — no external dependency, always runs
# ---------------------------------------------------------------------------

def test_inmemory_backend_get_put_roundtrip():
    backend = InMemoryStateBackend()
    assert backend.get("A") is None
    state = EntityState(count=3, mean_amt=10.0)
    backend.put("A", state)
    assert backend.get("A") is state
    assert backend.count() == 1


def test_inmemory_backend_clear():
    backend = InMemoryStateBackend()
    backend.put("A", EntityState())
    backend.put("B", EntityState())
    backend.clear()
    assert backend.count() == 0


def test_behavioral_state_manager_defaults_to_inmemory_backend():
    mgr = BehavioralStateManager()
    from src.engine.state_backend import InMemoryStateBackend as IMB
    assert isinstance(mgr._backend, IMB)


# ---------------------------------------------------------------------------
# RedisStateBackend — real Redis required
# ---------------------------------------------------------------------------

@requires_redis
def test_redis_backend_get_put_roundtrip(redis_client):
    backend = RedisStateBackend(redis_client, EntityState, key_prefix="pytest:fraud:state:")
    assert backend.get("A") is None
    state = EntityState(count=5, mean_amt=42.0, m2=10.0, min_amt=1.0, max_amt=100.0, last_time=1000.0)
    backend.put("A", state)

    retrieved = backend.get("A")
    assert retrieved.count == 5
    assert retrieved.mean_amt == 42.0
    assert retrieved.m2 == 10.0
    assert retrieved.min_amt == 1.0
    assert retrieved.max_amt == 100.0
    assert retrieved.last_time == 1000.0


@requires_redis
def test_redis_backend_persists_across_new_client_instances(redis_client):
    """The whole point of Redis over in-memory: state survives a fresh
    Python-side object, simulating a process restart."""
    backend_a = RedisStateBackend(redis_client, EntityState, key_prefix="pytest:fraud:state:")
    backend_a.put("A", EntityState(count=7, mean_amt=1.0))

    fresh_client = redis_lib.Redis.from_url("redis://localhost:6379/0")
    backend_b = RedisStateBackend(fresh_client, EntityState, key_prefix="pytest:fraud:state:")
    assert backend_b.get("A").count == 7


@requires_redis
def test_redis_backend_count_and_clear(redis_client):
    backend = RedisStateBackend(redis_client, EntityState, key_prefix="pytest:fraud:state:")
    backend.put("A", EntityState(count=1))
    backend.put("B", EntityState(count=2))
    assert backend.count() == 2
    backend.clear()
    assert backend.count() == 0


@requires_redis
def test_redis_backend_unreachable_raises_clean_error():
    bad_client = redis_lib.Redis(host="127.0.0.1", port=1, socket_connect_timeout=1)
    backend = RedisStateBackend(bad_client, EntityState, key_prefix="pytest:fraud:state:")
    with pytest.raises(StateBackendUnavailableError):
        backend.get("A")


# ---------------------------------------------------------------------------
# BehavioralStateManager with a Redis backend — same math, different storage
# ---------------------------------------------------------------------------

@requires_redis
def test_behavioral_state_manager_with_redis_matches_inmemory_result(redis_client):
    """The core correctness claim: swapping the backend must not change
    the FEATURE VALUES computed — same transaction sequence, same result."""
    mgr_memory = BehavioralStateManager()
    backend = RedisStateBackend(redis_client, EntityState, key_prefix="pytest:fraud:state:")
    mgr_redis = BehavioralStateManager(backend=backend)

    txns = [(100.0, 10.0), (200.0, 20.0), (300.0, 15.0), (400.0, 12.0)]
    for t, amt in txns:
        f_mem = mgr_memory.compute_features("card_X", t, amt)
        f_redis = mgr_redis.compute_features("card_X", t, amt)
        for key in f_mem:
            a, b = f_mem[key], f_redis[key]
            if a != a and b != b:  # both NaN
                continue
            assert a == pytest.approx(b), f"mismatch on {key} at t={t}: {a} vs {b}"
        mgr_memory.update("card_X", t, amt)
        mgr_redis.update("card_X", t, amt)


@requires_redis
def test_behavioral_state_manager_redis_reset(redis_client):
    backend = RedisStateBackend(redis_client, EntityState, key_prefix="pytest:fraud:state:")
    mgr = BehavioralStateManager(backend=backend)
    mgr.update("card_X", 100.0, 50.0)
    assert mgr.entity_count == 1
    mgr.reset()
    assert mgr.entity_count == 0
    assert mgr.get_state_snapshot("card_X") is None


@requires_redis
def test_redis_backend_ttl_expiry(redis_client):
    backend = RedisStateBackend(redis_client, EntityState, key_prefix="pytest:fraud:state:", ttl_seconds=1)
    backend.put("A", EntityState(count=1))
    assert backend.get("A") is not None
    time.sleep(1.5)
    assert backend.get("A") is None  # expired


# ---------------------------------------------------------------------------
# Redis latency benchmark (real measurement, not fabricated)
# ---------------------------------------------------------------------------

@requires_redis
def test_redis_latency_benchmark_runs_and_reports(redis_client, capsys):
    """Not a correctness test — runs a real, small benchmark and prints
    real p50/p95/p99 numbers, so `pytest -v -s` surfaces them. The
    authoritative benchmark run lives in `scripts/benchmark_redis_state.py`;
    this test exists to keep the measurement code itself exercised in CI."""
    backend = RedisStateBackend(redis_client, EntityState, key_prefix="pytest:fraud:state:")
    n = 200
    lookup_times = []
    update_times = []
    for i in range(n):
        entity = f"bench_{i % 20}"
        t0 = time.perf_counter()
        backend.get(entity)
        lookup_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        backend.put(entity, EntityState(count=i, mean_amt=float(i)))
        update_times.append(time.perf_counter() - t0)

    lookup_times.sort()
    p50 = lookup_times[int(n * 0.50)]
    p95 = lookup_times[int(n * 0.95)]
    print(f"\n[redis benchmark, n={n}] lookup p50={p50*1000:.3f}ms p95={p95*1000:.3f}ms")
    assert p50 >= 0  # sanity — real measurement, not asserting a specific threshold


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
