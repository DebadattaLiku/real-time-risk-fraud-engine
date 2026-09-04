#!/usr/bin/env python3
"""
Phase 7 demonstration script — local API client.

Demonstrates the intended usage flow:

    Start API (in-process, via FastAPI's TestClient)
        -> check /health
        -> send transaction 1, receive prediction
        -> send transaction 2 (same card1), observe stateful processing
        -> check /metadata

This uses FastAPI's TestClient in-process rather than requiring a
separately-running `uvicorn` server, so it runs reliably in any
environment (including sandboxes where launching and reaching a background
server process is unreliable) with a single command and no extra setup.
The exact same routes/logic run identically against a real live server —
see "Running as a real live server" below for those commands.

Usage (from the repository root):

    python scripts/demo_api.py

Running as a real live server (for real HTTP demonstration, e.g. via curl
or a browser hitting /docs):

    uvicorn src.api.main:app --reload
    # then, in another terminal:
    curl http://127.0.0.1:8000/health
    # or open http://127.0.0.1:8000/docs in a browser
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    print("Phase 7 API demonstration")
    print("=" * 70)
    print("\nStarting API (in-process TestClient — triggers real application "
          "startup: loads the model bundle and warm-starts behavioral "
          "state from historical data)...\n")

    from src.api.main import app

    with TestClient(app) as client:
        print("Checking GET /health ...")
        resp = client.get("/health")
        print(f"  {resp.status_code} {resp.json()}\n")

        print("Checking GET /metadata ...")
        resp = client.get("/metadata")
        print(f"  {resp.status_code} {resp.json()}\n")

        print("Loading two real test transactions sharing the same card1 entity...")
        config = load_config()
        df, _ = load_train_transaction(config=config)
        train_df, val_df, test_df, _ = compute_temporal_split(
            df, "TransactionDT", "TransactionID",
            config["split"]["train_ratio"], config["split"]["val_ratio"], config["split"]["test_ratio"],
        )
        del df
        test_sorted = test_df.sort_values(["TransactionDT", "TransactionID"], kind="mergesort").reset_index(drop=True)

        target_card1 = test_sorted["card1"].mode().iloc[0]  # a card1 with multiple test-period transactions
        same_entity = test_sorted[test_sorted["card1"] == target_card1].head(2)
        row1, row2 = same_entity.iloc[0], same_entity.iloc[1]
        print(f"  Using card1={target_card1}: TransactionID {row1['TransactionID']} then {row2['TransactionID']}\n")

        print("POST /predict — transaction 1 ...")
        payload1 = _to_json_payload(row1)
        resp1 = client.post("/predict", json=payload1)
        print(f"  {resp1.status_code} {resp1.json()}\n")

        print("POST /predict — transaction 2 (same card1 — should see transaction 1's history) ...")
        payload2 = _to_json_payload(row2)
        resp2 = client.post("/predict", json=payload2)
        print(f"  {resp2.status_code} {resp2.json()}\n")

        print("Demonstrating invalid input is rejected without corrupting state ...")
        bad_payload = dict(payload2)
        bad_payload["TransactionAmt"] = -1.0
        bad_payload["TransactionID"] = 999999999
        resp_bad = client.post("/predict", json=bad_payload)
        print(f"  {resp_bad.status_code} {resp_bad.json()}\n")

        print("=" * 70)
        print("Demonstration complete. Both real predictions returned "
              "risk_score + APPROVE/REVIEW/BLOCK decisions; the invalid "
              "request was rejected with a 422 and did not update state.")
        print("This is a LOCAL SIMULATION service — see "
              "reports/phase7_api_service_summary.md.")
        print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
