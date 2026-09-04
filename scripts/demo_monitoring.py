#!/usr/bin/env python3
"""
Phase 9 demonstration script — observability walkthrough.

    Start with a clean monitoring state
        -> send several valid prediction requests
        -> send at least one invalid request
        -> retrieve /metrics
        -> retrieve /monitoring/summary
        -> show decision/request metrics updated

Uses FastAPI's in-process TestClient (same approach as Phase 7's
`scripts/demo_api.py`) — no separate server process required, works
reliably in any environment. Does not retrain the model or modify the
decision policy.

Usage (from the repository root):

    python scripts/demo_monitoring.py
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# This demo script itself loads the raw dataset once (Step 2, below) to
# pull a few real sample transactions. If the API's own startup ALSO
# warm-started behavioral state (the host-default behavior — see
# src/api/main.py), that would mean loading the ~683MB
# train_transaction.csv TWICE concurrently, which is unnecessary memory
# pressure purely for this demonstration (a real deployment would not do
# this — it either warm-starts once, as tested in Phase 7, or runs cold,
# as Phase 8's container does). Set explicitly before importing the app so
# this demo's own startup is fast and lean; this does not change the
# documented WARM_START_STATE default for any other entry point.
os.environ.setdefault("WARM_START_STATE", "false")

import pandas as pd
from fastapi.testclient import TestClient

from src.data.load import load_train_transaction, load_config
from src.data.split import compute_temporal_split


def _to_json_payload(row: pd.Series) -> dict:
    d = row.to_dict()
    d.pop("isFraud", None)
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in d.items()}


def main() -> int:
    print("=" * 70)
    print("Phase 9 observability demonstration")
    print("=" * 70)
    print("\nStarting API (in-process TestClient — triggers real application "
          "startup, which also initializes a CLEAN MetricsRegistry)...\n")

    from src.api.main import app

    with TestClient(app) as client:
        print("Step 1: confirm a clean monitoring state at startup")
        summary = client.get("/monitoring/summary").json()
        print(f"  predictions.successes = {summary['predictions']['successes']} (expect 0)")
        print(f"  decisions.total       = {summary['decisions']['total']} (expect 0)\n")

        print("Step 2: load a handful of real test transactions...")
        config = load_config()
        df, _ = load_train_transaction(config=config)
        train_df, val_df, test_df, _ = compute_temporal_split(
            df, "TransactionDT", "TransactionID",
            config["split"]["train_ratio"], config["split"]["val_ratio"], config["split"]["test_ratio"],
        )
        del df
        test_sorted = test_df.sort_values(["TransactionDT", "TransactionID"], kind="mergesort").reset_index(drop=True)
        sample = test_sorted.iloc[:8]
        print(f"  loaded {len(sample)} transactions\n")

        print("Step 3: send valid prediction requests...")
        for _, row in sample.iterrows():
            payload = _to_json_payload(row)
            resp = client.post("/predict", json=payload)
            print(f"  txn {payload['TransactionID']}: {resp.status_code} "
                  f"risk_score={resp.json().get('risk_score'):.4f} decision={resp.json().get('decision')}")

        print("\nStep 4: send one invalid request (negative amount)...")
        bad_payload = _to_json_payload(sample.iloc[0])
        bad_payload["TransactionID"] = 999999001
        bad_payload["TransactionAmt"] = -10.0
        resp = client.post("/predict", json=bad_payload)
        print(f"  invalid txn: {resp.status_code} {resp.json()}\n")

        print("Step 5: retrieve GET /metrics (Prometheus text format, excerpt)...")
        metrics_resp = client.get("/metrics")
        lines = metrics_resp.text.splitlines()
        interesting = [l for l in lines if l.startswith("fraud_api_") and not l.startswith("#")]
        for line in interesting[:12]:
            print(f"  {line}")
        print(f"  ... ({len(interesting)} metric lines total)\n")

        print("Step 6: retrieve GET /monitoring/summary...")
        summary = client.get("/monitoring/summary").json()
        print(f"  requests.total              = {summary['requests']['total']}")
        print(f"  requests.by_endpoint         = {summary['requests']['by_endpoint']}")
        print(f"  predictions.attempts          = {summary['predictions']['attempts']}")
        print(f"  predictions.successes          = {summary['predictions']['successes']}")
        print(f"  predictions.failures            = {summary['predictions']['failures']}")
        print(f"  decisions.counts                 = {summary['decisions']['counts']}")
        print(f"  risk_scores.count                 = {summary['risk_scores']['count']}")
        print(f"  risk_scores.mean                   = {summary['risk_scores']['mean']}")
        print(f"  errors.request_validation_failures  = {summary['errors']['request_validation_failures']}")

        print("\n" + "=" * 70)
        assert summary["predictions"]["successes"] == len(sample), "expected all valid requests to succeed"
        assert summary["errors"]["request_validation_failures"] >= 1, "expected the invalid request to be counted"
        assert summary["decisions"]["total"] == len(sample), "decision total should match successful predictions"
        print("Demonstration complete: decision and request metrics updated "
              "as expected, and the invalid request was correctly counted "
              "as a failure, not a success.")
        print("Model was NOT retrained; decision policy was NOT modified.")
        print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
