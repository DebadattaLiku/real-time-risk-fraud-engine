#!/usr/bin/env python3
"""
Production upgrade — Redis state store benchmark.

Real measurement, not fabricated. Requires a reachable local Redis server.
Reports methodology explicitly so the numbers are interpretable, not just
asserted.

Usage:
    python scripts/benchmark_redis_state.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import redis

from src.engine.state import EntityState, BehavioralStateManager
from src.engine.state_backend import RedisStateBackend, InMemoryStateBackend


def percentile(sorted_values: list, p: float) -> float:
    idx = min(int(len(sorted_values) * p), len(sorted_values) - 1)
    return sorted_values[idx]


def benchmark_backend(name: str, backend, n_ops: int = 2000, n_entities: int = 200) -> dict:
    lookup_times, update_times = [], []
    # Warm-up (not measured) — matches standard benchmarking practice of
    # excluding first-connection/JIT-adjacent effects from the measured window.
    for i in range(50):
        backend.put(f"warmup_{i}", EntityState(count=1))

    for i in range(n_ops):
        entity = f"entity_{i % n_entities}"

        t0 = time.perf_counter()
        backend.get(entity)
        lookup_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        backend.put(entity, EntityState(count=i, mean_amt=float(i), m2=1.0, min_amt=0.0, max_amt=100.0, last_time=float(i)))
        update_times.append(time.perf_counter() - t0)

    lookup_times.sort()
    update_times.sort()
    return {
        "backend": name,
        "n_ops": n_ops,
        "lookup_p50_ms": percentile(lookup_times, 0.50) * 1000,
        "lookup_p95_ms": percentile(lookup_times, 0.95) * 1000,
        "lookup_p99_ms": percentile(lookup_times, 0.99) * 1000,
        "lookup_mean_ms": statistics.mean(lookup_times) * 1000,
        "update_p50_ms": percentile(update_times, 0.50) * 1000,
        "update_p95_ms": percentile(update_times, 0.95) * 1000,
        "update_p99_ms": percentile(update_times, 0.99) * 1000,
        "update_mean_ms": statistics.mean(update_times) * 1000,
    }


def main() -> int:
    print("=" * 70)
    print("Redis state store benchmark (real measurement)")
    print("=" * 70)
    print(f"Python: {sys.version.split()[0]}")
    print("Redis: local, unix socket via localhost:6379, single client connection")
    print("Concurrency: single-threaded, sequential operations (no concurrent load)")
    print()

    try:
        client = redis.Redis.from_url("redis://localhost:6379/0", socket_connect_timeout=2)
        client.ping()
    except Exception as e:
        print(f"Redis is NOT reachable at localhost:6379 — benchmark cannot run: {e}")
        print("Result: NOT MEASURED")
        return 1

    client.flushdb()
    redis_backend = RedisStateBackend(client, EntityState, key_prefix="bench:fraud:state:")
    memory_backend = InMemoryStateBackend()

    n_ops = 2000
    results = [
        benchmark_backend("InMemoryStateBackend", memory_backend, n_ops=n_ops),
        benchmark_backend("RedisStateBackend", redis_backend, n_ops=n_ops),
    ]

    print(f"{'Backend':<22} {'lookup p50':>12} {'lookup p95':>12} {'lookup p99':>12} {'update p50':>12} {'update p95':>12} {'update p99':>12}")
    for r in results:
        print(f"{r['backend']:<22} {r['lookup_p50_ms']:>10.4f}ms {r['lookup_p95_ms']:>10.4f}ms {r['lookup_p99_ms']:>10.4f}ms "
              f"{r['update_p50_ms']:>10.4f}ms {r['update_p95_ms']:>10.4f}ms {r['update_p99_ms']:>10.4f}ms")

    client.flushdb()
    print()
    print("Note: this is a SINGLE local Redis instance over a loopback TCP")
    print("connection with no concurrent load and no network latency beyond")
    print("localhost — it is NOT representative of production latency under")
    print("real network conditions, connection pooling contention, or")
    print("concurrent request volume. It demonstrates the real, measured")
    print("relative overhead of an external state store vs. an in-process")
    print("dict, in this specific environment, nothing more.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
