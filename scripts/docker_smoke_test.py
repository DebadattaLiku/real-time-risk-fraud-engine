#!/usr/bin/env python3
"""
Phase 8: Docker smoke test.

Validates a RUNNING container's `/health`, `/metadata`, and `/predict`
endpoints over real HTTP. Does not build or start anything itself — run it
against an already-running container (see usage below). Fails loudly
(non-zero exit, clear message) on any unexpected response; never retrains
or touches the model.

Usage:
    docker run -d --name fraud-risk-api -p 8000:8000 fraud-risk-api:local
    # wait for the container to report healthy, then:
    python scripts/docker_smoke_test.py --base-url http://localhost:8000
    docker stop fraud-risk-api && docker rm fraud-risk-api
"""

from __future__ import annotations

import argparse
import sys

import requests


def check(condition: bool, message: str, failures: list) -> None:
    if not condition:
        failures.append(message)
        print(f"  FAIL: {message}")
    else:
        print(f"  ok: {message}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test a running fraud-risk-api container.")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL of the running service.")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    failures: list = []

    print(f"Smoke-testing {args.base_url} ...\n")

    print("GET /health")
    try:
        resp = requests.get(f"{args.base_url}/health", timeout=args.timeout)
    except requests.RequestException as e:
        print(f"  FAIL: could not reach /health: {e}")
        return 1
    check(resp.status_code == 200, f"/health status 200 (got {resp.status_code})", failures)
    body = resp.json()
    check(body.get("status") == "ok", f"/health status field is 'ok' (got {body.get('status')!r})", failures)
    check(body.get("model_loaded") is True, "/health model_loaded is true", failures)
    check(body.get("policy_loaded") is True, "/health policy_loaded is true", failures)

    print("\nGET /metadata")
    resp = requests.get(f"{args.base_url}/metadata", timeout=args.timeout)
    check(resp.status_code == 200, f"/metadata status 200 (got {resp.status_code})", failures)
    meta = resp.json()
    for field in ("model_type", "model_version", "policy_name", "approve_threshold", "block_threshold", "supported_decisions"):
        check(field in meta, f"/metadata includes '{field}'", failures)
    check(
        set(meta.get("supported_decisions", [])) == {"APPROVE", "REVIEW", "BLOCK"},
        f"/metadata supported_decisions is exactly APPROVE/REVIEW/BLOCK (got {meta.get('supported_decisions')})",
        failures,
    )

    print("\nPOST /predict (valid transaction)")
    payload = {
        "TransactionID": 9000001, "TransactionDT": 100000, "TransactionAmt": 50.0,
        "card1": 88888, "ProductCD": "W",
    }
    resp = requests.post(f"{args.base_url}/predict", json=payload, timeout=args.timeout)
    check(resp.status_code == 200, f"/predict status 200 (got {resp.status_code}: {resp.text[:200]})", failures)
    if resp.status_code == 200:
        body = resp.json()
        check(body.get("transaction_id") == payload["TransactionID"], "/predict echoes transaction_id", failures)
        score = body.get("risk_score")
        check(isinstance(score, (int, float)) and 0.0 <= score <= 1.0, f"/predict risk_score in [0,1] (got {score!r})", failures)
        check(body.get("decision") in ("APPROVE", "REVIEW", "BLOCK"), f"/predict decision is valid (got {body.get('decision')!r})", failures)

    print("\nPOST /predict (invalid transaction — negative amount, must be rejected)")
    bad_payload = dict(payload, TransactionID=9000002, TransactionAmt=-1.0)
    resp = requests.post(f"{args.base_url}/predict", json=bad_payload, timeout=args.timeout)
    check(resp.status_code == 422, f"/predict rejects negative amount with 422 (got {resp.status_code})", failures)

    print("\nPOST /predict (sequential requests, same card1 — stateful behavior)")
    entity_payload_1 = dict(payload, TransactionID=9000003, TransactionDT=200000, card1=77777)
    entity_payload_2 = dict(payload, TransactionID=9000004, TransactionDT=201000, card1=77777)
    resp1 = requests.post(f"{args.base_url}/predict", json=entity_payload_1, timeout=args.timeout)
    resp2 = requests.post(f"{args.base_url}/predict", json=entity_payload_2, timeout=args.timeout)
    check(resp1.status_code == 200 and resp2.status_code == 200, "both sequential requests succeed", failures)

    print("\n" + "=" * 60)
    if failures:
        print(f"SMOKE TEST FAILED: {len(failures)} check(s) failed.")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("SMOKE TEST PASSED: all checks succeeded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
