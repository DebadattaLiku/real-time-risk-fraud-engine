"""
Phase 9: `/metrics` text rendering.

Produces output in the Prometheus text exposition format (plain counters
and gauges, `# HELP`/`# TYPE` comments) using only the standard library —
no `prometheus_client` dependency, no Prometheus server, no Grafana. This
means the service exposes metrics in a widely-understood, scrapeable
format; it does NOT mean a Prometheus server is deployed or configured
anywhere in this project. See the Phase 9 report for the explicit
distinction.
"""

from __future__ import annotations


def _sanitize_label_value(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def render_prometheus_text(snapshot: dict) -> str:
    lines: list = []

    def counter(name: str, help_text: str, value) -> None:
        if value is None:
            return
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} counter")
        lines.append(f"{name} {value}")

    def gauge(name: str, help_text: str, value) -> None:
        if value is None:
            return
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name} {value}")

    req = snapshot["requests"]
    counter("fraud_api_requests_total", "Total number of API requests received.", req["total"])
    lines.append("# HELP fraud_api_requests_by_endpoint_total Requests received, by endpoint.")
    lines.append("# TYPE fraud_api_requests_by_endpoint_total counter")
    for endpoint, count in req["by_endpoint"].items():
        lines.append(f'fraud_api_requests_by_endpoint_total{{endpoint="{_sanitize_label_value(endpoint)}"}} {count}')
    lines.append("# HELP fraud_api_requests_by_status_class_total Requests received, by HTTP status class.")
    lines.append("# TYPE fraud_api_requests_by_status_class_total counter")
    for status_class, count in req["by_status_class"].items():
        lines.append(f'fraud_api_requests_by_status_class_total{{status_class="{status_class}"}} {count}')
    gauge("fraud_api_request_latency_avg_ms", "Average end-to-end request latency in milliseconds.", req["avg_latency_ms"])

    pred = snapshot["predictions"]
    counter("fraud_api_prediction_attempts_total", "Prediction requests that reached the RiskDecisionEngine.", pred["attempts"])
    counter("fraud_api_prediction_successes_total", "Predictions that completed successfully.", pred["successes"])
    counter("fraud_api_prediction_failures_total", "Predictions that failed engine-level validation/processing.", pred["failures"])
    gauge("fraud_api_prediction_latency_avg_ms", "Average engine-reported prediction processing time in milliseconds.", pred["avg_engine_latency_ms"])

    dec = snapshot["decisions"]
    lines.append("# HELP fraud_api_decisions_total Decisions returned, by decision type.")
    lines.append("# TYPE fraud_api_decisions_total counter")
    for decision, count in dec["counts"].items():
        lines.append(f'fraud_api_decisions_total{{decision="{decision}"}} {count}')

    rs = snapshot["risk_scores"]
    counter("fraud_api_risk_scores_observed_total", "Total number of risk scores observed.", rs["count"])
    gauge("fraud_api_risk_score_mean", "Mean of all observed risk scores.", rs["mean"])
    gauge("fraud_api_risk_score_min", "Minimum observed risk score.", rs["min"])
    gauge("fraud_api_risk_score_max", "Maximum observed risk score.", rs["max"])
    lines.append("# HELP fraud_api_risk_score_bucket_total Risk score distribution, decile buckets.")
    lines.append("# TYPE fraud_api_risk_score_bucket_total counter")
    for bucket, count in rs["bucket_counts"].items():
        lines.append(f'fraud_api_risk_score_bucket_total{{bucket="{bucket}"}} {count}')

    err = snapshot["errors"]
    counter("fraud_api_request_validation_failures_total", "Requests rejected by request-schema validation (never reached the engine).", err["request_validation_failures"])
    counter("fraud_api_engine_validation_failures_total", "Requests rejected by the engine's own validation.", err["engine_validation_failures"])
    counter("fraud_api_engine_not_initialized_errors_total", "Requests received while the engine was not initialized.", err["engine_not_initialized_errors"])
    counter("fraud_api_internal_errors_total", "Unexpected internal errors.", err["internal_errors"])

    gauge("fraud_api_uptime_seconds", "Seconds since the application process started.", snapshot["uptime_seconds"])

    return "\n".join(lines) + "\n"
