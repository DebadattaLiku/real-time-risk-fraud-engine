"""
Phase 9: Structured logging configuration.

Produces `key=value`-style log lines (easy to grep, easy to later parse
into a log aggregator) covering the request lifecycle: received ->
processed -> decision returned. Deliberately logs only safe, aggregate
operational metadata:

    transaction_id, decision, risk_score, duration_ms, endpoint, status_code

NEVER logged: the full transaction payload, individual feature values,
`isFraud` (never present to begin with — the engine rejects it), or raw
Python tracebacks/internal paths.
"""

from __future__ import annotations

import logging
import sys

_LOGGER_NAME = "fraud_api"


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:  # avoid duplicate handlers if called more than once (e.g. in tests)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s level=%(levelname)s logger=%(name)s %(message)s",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(_LOGGER_NAME)


def log_request_received(endpoint: str, method: str) -> None:
    get_logger().info(f"event=request_received endpoint={endpoint} method={method}")


def log_prediction_processed(transaction_id, decision: str, risk_score: float, duration_ms: float) -> None:
    get_logger().info(
        f"event=prediction_processed transaction_id={transaction_id} "
        f"decision={decision} risk_score={risk_score:.6f} duration_ms={duration_ms:.2f}"
    )


def log_validation_failure(source: str, detail: str) -> None:
    """`source` is 'request_schema' or 'engine' — `detail` is a short,
    already-safe message (e.g. the engine's own validation error text,
    which never includes raw feature values), not a traceback."""
    get_logger().warning(f"event=validation_failure source={source} detail={detail!r}")


def log_internal_error(endpoint: str) -> None:
    # Deliberately does NOT log the exception object/traceback — that
    # stays in the server process's own crash/exception visibility (e.g.
    # uvicorn's own error log), not in this application-level structured
    # log, and never in the HTTP response (see main.py's exception
    # handlers).
    get_logger().error(f"event=internal_error endpoint={endpoint}")
