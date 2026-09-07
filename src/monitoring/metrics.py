"""
Phase 9: Core metrics registry.

Pure observation — this module holds counters/gauges and provides methods
to record events, but computes nothing that feeds back into a request's
response. Nothing here can change a risk score, a decision, or behavioral
state; it only reads already-computed results after the fact (see
`src/api/main.py`'s `/predict` route for exactly where recording happens
— always AFTER `engine.process_transaction()` has already returned).

Thread-safety note: FastAPI runs synchronous (`def`, not `async def`)
route handlers — which is what every route in this project uses — in a
worker thread pool, so concurrent requests can genuinely execute this
code from different threads simultaneously. A single `threading.Lock`
protects every mutation here; snapshots are taken under the same lock so
a caller never sees a torn/partially-updated state.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict

# Decile buckets for the risk-score histogram: [0.0-0.1), [0.1-0.2), ...,
# [0.9-1.0]. Chosen for simplicity and interpretability (10 equal-width
# bins spanning the full [0,1] probability range) — NOT tied to the
# decision policy's approve/block thresholds, so this histogram stays a
# neutral distribution summary independent of whatever policy happens to
# be loaded, and doesn't need to change if the policy does.
RISK_SCORE_BUCKET_EDGES = [round(i * 0.1, 1) for i in range(11)]  # 0.0, 0.1, ..., 1.0


def _bucket_label(score: float) -> str:
    idx = min(int(score * 10), 9)  # score==1.0 falls into the last bucket, not a new one
    lo, hi = RISK_SCORE_BUCKET_EDGES[idx], RISK_SCORE_BUCKET_EDGES[idx + 1]
    return f"{lo:.1f}-{hi:.1f}"


class MetricsRegistry:
    """
    One instance lives for the lifetime of the application process
    (installed the same way as the engine singleton — see
    `src/api/dependencies.py`). Metrics are IN-MEMORY ONLY: they reset to
    zero whenever the process restarts. No persistence, no database, per
    Phase 9 scope.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.start_time = time.time()

        # --- Request metrics (recorded by middleware, every endpoint) ---
        self.request_count_total = 0
        self.request_count_by_endpoint: dict = defaultdict(int)
        self.request_count_by_status_class: dict = defaultdict(int)  # "2xx"/"4xx"/"5xx"
        self._request_latency_sum_ms = 0.0
        self._request_latency_count = 0
        self._request_latency_by_endpoint_sum_ms: dict = defaultdict(float)
        self._request_latency_by_endpoint_count: dict = defaultdict(int)

        # --- Prediction metrics (recorded explicitly in POST /predict) ---
        self.prediction_attempts = 0     # reached the engine (passed request-schema validation)
        self.prediction_successes = 0
        self.prediction_failures = 0     # engine-level validation/processing failures
        self._prediction_latency_sum_ms = 0.0  # engine-reported processing time, not HTTP round-trip
        self._prediction_latency_count = 0

        # --- Decision metrics ---
        self.decision_counts: dict = {"APPROVE": 0, "REVIEW": 0, "BLOCK": 0}

        # --- Risk-score metrics ---
        self.risk_score_count = 0
        self._risk_score_sum = 0.0
        self.risk_score_min = None
        self.risk_score_max = None
        self.risk_score_bucket_counts: dict = defaultdict(int)

        # --- Data-quality / error metrics ---
        self.request_validation_failures = 0   # pydantic schema-level (never reached the engine)
        self.engine_validation_failures = 0    # engine-level (e.g. isFraud present, bad numeric value)
        self.engine_not_initialized_errors = 0
        self.internal_errors = 0
        self.transaction_amount_count = 0
        self._transaction_amount_sum = 0.0
        self.transaction_amount_min = None
        self.transaction_amount_max = None

        # --- Production upgrade: SHAP explanation / Redis / streaming ---
        # Additive only — none of the metrics above were touched or
        # renamed, so every existing snapshot consumer keeps working
        # unchanged; these appear under a new top-level "extensions" key
        # in snapshot() (see below).
        self._explanation_latency_sum_ms = 0.0
        self._explanation_latency_count = 0
        self.redis_lookup_count = 0
        self.redis_lookup_failures = 0
        self._redis_lookup_latency_sum_ms = 0.0
        self.streaming_messages_processed = 0
        self.streaming_messages_failed = 0
        self.streaming_messages_dead_lettered = 0
        self.model_load_time_ms: float | None = None

    # ------------------------------------------------------------------
    # Recording methods
    # ------------------------------------------------------------------

    def record_request(self, endpoint: str, status_code: int, duration_ms: float) -> None:
        status_class = f"{status_code // 100}xx"
        with self._lock:
            self.request_count_total += 1
            self.request_count_by_endpoint[endpoint] += 1
            self.request_count_by_status_class[status_class] += 1
            self._request_latency_sum_ms += duration_ms
            self._request_latency_count += 1
            self._request_latency_by_endpoint_sum_ms[endpoint] += duration_ms
            self._request_latency_by_endpoint_count[endpoint] += 1

    def record_prediction_attempt(self) -> None:
        with self._lock:
            self.prediction_attempts += 1

    def record_prediction_success(self, decision: str, risk_score: float, duration_ms: float, transaction_amount: float | None = None) -> None:
        with self._lock:
            self.prediction_successes += 1
            if decision in self.decision_counts:
                self.decision_counts[decision] += 1

            self.risk_score_count += 1
            self._risk_score_sum += risk_score
            self.risk_score_min = risk_score if self.risk_score_min is None else min(self.risk_score_min, risk_score)
            self.risk_score_max = risk_score if self.risk_score_max is None else max(self.risk_score_max, risk_score)
            self.risk_score_bucket_counts[_bucket_label(risk_score)] += 1

            self._prediction_latency_sum_ms += duration_ms
            self._prediction_latency_count += 1

            if transaction_amount is not None:
                self.transaction_amount_count += 1
                self._transaction_amount_sum += transaction_amount
                self.transaction_amount_min = (
                    transaction_amount if self.transaction_amount_min is None else min(self.transaction_amount_min, transaction_amount)
                )
                self.transaction_amount_max = (
                    transaction_amount if self.transaction_amount_max is None else max(self.transaction_amount_max, transaction_amount)
                )

    def record_engine_validation_failure(self) -> None:
        with self._lock:
            self.prediction_failures += 1
            self.engine_validation_failures += 1

    def record_request_validation_failure(self) -> None:
        with self._lock:
            self.request_validation_failures += 1

    def record_engine_not_initialized(self) -> None:
        with self._lock:
            self.engine_not_initialized_errors += 1

    def record_internal_error(self) -> None:
        with self._lock:
            self.internal_errors += 1

    # --- Production upgrade recording methods -------------------------

    def record_explanation(self, duration_ms: float) -> None:
        with self._lock:
            self._explanation_latency_sum_ms += duration_ms
            self._explanation_latency_count += 1

    def record_redis_lookup(self, duration_ms: float, failed: bool = False) -> None:
        with self._lock:
            self.redis_lookup_count += 1
            self._redis_lookup_latency_sum_ms += duration_ms
            if failed:
                self.redis_lookup_failures += 1

    def record_streaming_message(self, status: str) -> None:
        """`status` is one of 'processed', 'failed', 'dead_lettered'."""
        with self._lock:
            if status == "processed":
                self.streaming_messages_processed += 1
            elif status == "failed":
                self.streaming_messages_failed += 1
            elif status == "dead_lettered":
                self.streaming_messages_dead_lettered += 1

    def record_model_load_time(self, duration_ms: float) -> None:
        with self._lock:
            self.model_load_time_ms = duration_ms

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """Returns a fully-consistent point-in-time view of all metrics."""
        with self._lock:
            avg_request_latency_ms = (
                self._request_latency_sum_ms / self._request_latency_count if self._request_latency_count else None
            )
            avg_prediction_latency_ms = (
                self._prediction_latency_sum_ms / self._prediction_latency_count if self._prediction_latency_count else None
            )
            avg_latency_by_endpoint = {
                ep: self._request_latency_by_endpoint_sum_ms[ep] / self._request_latency_by_endpoint_count[ep]
                for ep in self._request_latency_by_endpoint_count
            }
            mean_risk_score = self._risk_score_sum / self.risk_score_count if self.risk_score_count else None
            mean_transaction_amount = (
                self._transaction_amount_sum / self.transaction_amount_count if self.transaction_amount_count else None
            )

            return {
                "uptime_seconds": time.time() - self.start_time,
                "requests": {
                    "total": self.request_count_total,
                    "by_endpoint": dict(self.request_count_by_endpoint),
                    "by_status_class": dict(self.request_count_by_status_class),
                    "avg_latency_ms": avg_request_latency_ms,
                    "avg_latency_ms_by_endpoint": avg_latency_by_endpoint,
                },
                "predictions": {
                    "attempts": self.prediction_attempts,
                    "successes": self.prediction_successes,
                    "failures": self.prediction_failures,
                    "avg_engine_latency_ms": avg_prediction_latency_ms,
                },
                "decisions": {
                    "counts": dict(self.decision_counts),
                    "total": sum(self.decision_counts.values()),
                    "distribution": (
                        {k: v / sum(self.decision_counts.values()) for k, v in self.decision_counts.items()}
                        if sum(self.decision_counts.values()) > 0 else {}
                    ),
                },
                "risk_scores": {
                    "count": self.risk_score_count,
                    "mean": mean_risk_score,
                    "min": self.risk_score_min,
                    "max": self.risk_score_max,
                    "bucket_counts": dict(self.risk_score_bucket_counts),
                    "bucket_edges": RISK_SCORE_BUCKET_EDGES,
                },
                "data_quality": {
                    "request_validation_failures": self.request_validation_failures,
                    "engine_validation_failures": self.engine_validation_failures,
                    "transaction_amount": {
                        "count": self.transaction_amount_count,
                        "mean": mean_transaction_amount,
                        "min": self.transaction_amount_min,
                        "max": self.transaction_amount_max,
                    },
                },
                "errors": {
                    "request_validation_failures": self.request_validation_failures,
                    "engine_validation_failures": self.engine_validation_failures,
                    "engine_not_initialized_errors": self.engine_not_initialized_errors,
                    "internal_errors": self.internal_errors,
                },
                "extensions": {
                    "shap": {
                        "explanations_computed": self._explanation_latency_count,
                        "avg_latency_ms": (
                            self._explanation_latency_sum_ms / self._explanation_latency_count
                            if self._explanation_latency_count else None
                        ),
                    },
                    "redis": {
                        "lookup_count": self.redis_lookup_count,
                        "lookup_failures": self.redis_lookup_failures,
                        "avg_lookup_latency_ms": (
                            self._redis_lookup_latency_sum_ms / self.redis_lookup_count
                            if self.redis_lookup_count else None
                        ),
                    },
                    "streaming": {
                        "messages_processed": self.streaming_messages_processed,
                        "messages_failed": self.streaming_messages_failed,
                        "messages_dead_lettered": self.streaming_messages_dead_lettered,
                    },
                    "model_load_time_ms": self.model_load_time_ms,
                },
            }
