"""
Phase 12: Thin HTTP client for the existing Phase 7-11 FastAPI service.

No inference, monitoring, drift, or governance LOGIC lives here — every
function is a plain HTTP call to an already-approved endpoint
(`/predict`, `/health`, `/metadata`, `/monitoring/summary`,
`/drift/summary`, `/drift/analyze`, `/model-governance/summary`). The
dashboard reuses the real system through its real API contract rather
than reimplementing anything it does.

All functions fail soft: on any connection error, timeout, or non-2xx
response, they return `(False, error_dict)` rather than raising, so a
dashboard page can render a clean "API not available" state instead of
crashing.
"""

from __future__ import annotations

from typing import Any

import requests

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 5.0


def _get(base_url: str, path: str, timeout: float = DEFAULT_TIMEOUT) -> tuple[bool, Any]:
    try:
        resp = requests.get(f"{base_url}{path}", timeout=timeout)
    except requests.RequestException as e:
        return False, {"error": "connection_error", "detail": str(e)}
    if resp.status_code != 200:
        return False, {"error": "http_error", "status_code": resp.status_code, "detail": resp.text[:500]}
    try:
        return True, resp.json()
    except ValueError:
        return True, resp.text


def _post(base_url: str, path: str, json_body: dict, timeout: float = DEFAULT_TIMEOUT) -> tuple[bool, Any]:
    try:
        resp = requests.post(f"{base_url}{path}", json=json_body, timeout=timeout)
    except requests.RequestException as e:
        return False, {"error": "connection_error", "detail": str(e)}
    try:
        body = resp.json()
    except ValueError:
        body = {"raw_text": resp.text[:500]}
    if resp.status_code >= 400:
        return False, {"error": "http_error", "status_code": resp.status_code, "detail": body}
    return True, body


def is_api_available(base_url: str = DEFAULT_BASE_URL, timeout: float = 2.0) -> bool:
    ok, _ = _get(base_url, "/health", timeout=timeout)
    return ok


def get_health(base_url: str = DEFAULT_BASE_URL) -> tuple[bool, Any]:
    return _get(base_url, "/health")


def get_metadata(base_url: str = DEFAULT_BASE_URL) -> tuple[bool, Any]:
    return _get(base_url, "/metadata")


def predict(transaction: dict, base_url: str = DEFAULT_BASE_URL) -> tuple[bool, Any]:
    return _post(base_url, "/predict", transaction)


def get_monitoring_summary(base_url: str = DEFAULT_BASE_URL) -> tuple[bool, Any]:
    return _get(base_url, "/monitoring/summary")


def get_drift_summary(base_url: str = DEFAULT_BASE_URL) -> tuple[bool, Any]:
    return _get(base_url, "/drift/summary")


def analyze_drift_batch(records: list, base_url: str = DEFAULT_BASE_URL) -> tuple[bool, Any]:
    return _post(base_url, "/drift/analyze", {"records": records})


def get_governance_summary(base_url: str = DEFAULT_BASE_URL) -> tuple[bool, Any]:
    return _get(base_url, "/model-governance/summary")


def get_metrics_text(base_url: str = DEFAULT_BASE_URL) -> tuple[bool, str]:
    """Raw Prometheus text exposition — returned as plain text, not parsed."""
    try:
        resp = requests.get(f"{base_url}/metrics", timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as e:
        return False, str(e)
    if resp.status_code != 200:
        return False, resp.text[:500]
    return True, resp.text
