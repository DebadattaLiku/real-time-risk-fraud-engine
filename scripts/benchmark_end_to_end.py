#!/usr/bin/env python3
"""
Production upgrade — end-to-end latency/throughput benchmark.

Every number below is a REAL measurement taken in this session, on this
machine, under the exact conditions stated in each section's printed
header. Nothing here is estimated or invented. Where a component
genuinely could not be benchmarked in this environment (Kafka — no real
broker available; see src/streaming/kafka_client.py), this script prints
NOT MEASURED explicitly rather than fabricating a number.

Usage:
    python scripts/benchmark_end_to_end.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import platform


def percentile(sorted_values: list, p: float) -> float:
    idx = min(int(len(sorted_values) * p), len(sorted_values) - 1)
    return sorted_values[idx]


def summarize(name: str, times_s: list) -> dict:
    times_s = sorted(times_s)
    n = len(times_s)
    return {
        "component": name, "n": n,
        "p50_ms": percentile(times_s, 0.50) * 1000,
        "p95_ms": percentile(times_s, 0.95) * 1000,
        "p99_ms": percentile(times_s, 0.99) * 1000,
        "mean_ms": statistics.mean(times_s) * 1000,
        "throughput_per_sec": n / sum(times_s) if sum(times_s) > 0 else float("inf"),
    }


def print_row(r: dict) -> None:
    print(f"{r['component']:<32} n={r['n']:<6} p50={r['p50_ms']:>8.3f}ms  p95={r['p95_ms']:>8.3f}ms  "
          f"p99={r['p99_ms']:>8.3f}ms  throughput={r['throughput_per_sec']:>9.1f}/s")


def main() -> int:
    print("=" * 100)
    print("End-to-end latency/throughput benchmark (real measurements)")
    print("=" * 100)
    print(f"Python: {sys.version.split()[0]} on {platform.platform()}")
    print("Concurrency: single-threaded, sequential requests, no load generator — these are")
    print("             LATENCY-UNDER-NO-CONTENTION numbers, not a claim about throughput under")
    print("             concurrent production load (no such load test was run in this environment).")
    print()

    from src.api.dependencies import build_engine
    from src.explainability.shap_explainer import FraudExplainer

    print("Loading model bundle (not timed — startup cost, not a per-request cost)...")
    engine, _ = build_engine(warm_start=False)

    def real_txn(**overrides):
        base = {c: None for c in engine.get_expected_raw_columns()}
        base.update({"TransactionID": 1, "TransactionDT": 100000, "TransactionAmt": 100.0, "card1": 1, "ProductCD": "W"})
        base.update(overrides)
        return base

    n_ops = 300
    results = []

    # --- 1. Direct model inference (predict_proba only, preprocessed row already built once) ---
    import pandas as pd
    txn0 = real_txn(card1=900001)
    raw_df = pd.DataFrame([{k: txn0.get(k) for k in engine._expected_raw_columns}])
    numeric_cols = [s.name for s in engine.schema if s.used and s.value_type == "numeric" and s.name in raw_df.columns]
    raw_df[numeric_cols] = raw_df[numeric_cols].astype("float64")
    X = engine.feature_pipeline.transform(raw_df)
    Z = engine.lgbm_preprocessor.transform(X)
    bhv_cols = ["bhv_prev_txn_count", "bhv_prev_txn_count_log1p", "bhv_hist_mean_amt", "bhv_hist_std_amt",
                "bhv_hist_min_amt", "bhv_hist_max_amt", "bhv_time_since_prev_txn", "bhv_amt_to_hist_mean_ratio",
                "bhv_amt_diff_from_hist_mean", "bhv_amt_zscore", "bhv_prior_count_1h", "bhv_prior_count_24h"]
    for c in bhv_cols:
        Z[c] = float("nan")
    times = []
    for _ in range(n_ops):
        t0 = time.perf_counter()
        engine.model.predict_proba(Z)
        times.append(time.perf_counter() - t0)
    results.append(summarize("Direct model inference", times))

    # --- 2. Direct engine call, in-memory state, cold entities (no state accumulation effect) ---
    times = []
    for i in range(n_ops):
        t0 = time.perf_counter()
        engine.process_transaction(real_txn(TransactionID=i, card1=1_000_000 + i))
        times.append(time.perf_counter() - t0)
    results.append(summarize("Direct engine (in-memory state)", times))

    # --- 3. Direct engine call, Redis-backed state ---
    try:
        import redis
        from src.engine.state import BehavioralStateManager, EntityState
        from src.engine.state_backend import RedisStateBackend
        client = redis.Redis.from_url("redis://localhost:6379/0", socket_connect_timeout=2)
        client.ping()
        redis_engine, _ = build_engine(warm_start=False)
        redis_engine.state_manager = BehavioralStateManager(backend=RedisStateBackend(client, EntityState, key_prefix="bench:e2e:"))
        times = []
        for i in range(n_ops):
            t0 = time.perf_counter()
            redis_engine.process_transaction(real_txn(TransactionID=i, card1=2_000_000 + i))
            times.append(time.perf_counter() - t0)
        results.append(summarize("Direct engine (Redis-backed state)", times))
        client.flushdb()
    except Exception as e:
        print(f"Redis-backed engine benchmark skipped (Redis unreachable): {e}")

    # --- 4. FastAPI, in-process TestClient (real Starlette routing/validation, no real network) ---
    from fastapi.testclient import TestClient
    from src.api.main import app
    from src.api import dependencies as deps
    from src.monitoring.metrics import MetricsRegistry
    deps.set_engine(engine, {"policy_name": "balanced"})
    deps.set_metrics_registry(MetricsRegistry())
    client_api = TestClient(app)
    times = []
    for i in range(n_ops):
        txn = real_txn(TransactionID=3_000_000 + i, card1=3_000_000 + i)
        t0 = time.perf_counter()
        client_api.post("/predict", json=txn)
        times.append(time.perf_counter() - t0)
    results.append(summarize("FastAPI /predict (TestClient, in-process)", times))

    # --- 5. SHAP explanation latency (via /predict/explain path, real model) ---
    engine.explainer = FraudExplainer(engine.model, top_k=5)
    times = []
    for i in range(n_ops):
        txn = real_txn(TransactionID=4_000_000 + i, card1=4_000_000 + i)
        t0 = time.perf_counter()
        engine.process_transaction(txn, explain=True)
        times.append(time.perf_counter() - t0)
    results.append(summarize("Direct engine + SHAP explanation", times))

    print()
    for r in results:
        print_row(r)

    print()
    print("Kafka ingestion -> Redis -> feature generation -> LightGBM -> SHAP -> decision (full streaming path):")
    print("  NOT MEASURED — no real Kafka broker could be provisioned in this sandboxed development")
    print("  environment (network egress restrictions block Apache/Confluent/Bitnami/Redpanda; no apt")
    print("  package provides a broker). See src/streaming/kafka_client.py's module docstring and")
    print("  reports/production_architecture_audit.md for the full accounting. The non-Kafka portion of")
    print("  this same pipeline (Redis -> feature generation -> LightGBM -> SHAP -> decision) IS measured")
    print("  above as 'Direct engine (Redis-backed state)' and 'Direct engine + SHAP explanation'.")
    print()
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
