"""
Phase 7: FastAPI risk scoring service.

Thin HTTP layer around the approved Phase 6 `RiskDecisionEngine` — no ML,
behavioral-feature, or decision logic is implemented in this file. Every
route either calls into the engine directly or reads simple metadata off
it. This is a LOCAL ENGINEERING SERVICE for this project, not a
production-deployed financial system — see
`reports/phase7_api_service_summary.md`.
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.responses import JSONResponse, PlainTextResponse
import pandas as pd

from src.api import dependencies as deps
from src.api.schemas import (
    HealthResponse, MetadataResponse, PredictResponse, TransactionRequest, DriftAnalysisRequest,
)
from src.decision.policy import DECISIONS
from src.engine.risk_engine import RiskDecisionEngine, TransactionValidationError
from src.engine.state_backend import StateBackendUnavailableError
from src.monitoring.metrics import MetricsRegistry
from src.monitoring.middleware import MetricsMiddleware
from src.monitoring.prometheus_export import render_prometheus_text
from src.monitoring.data_quality import extract_transaction_amount
from src.monitoring.logging_config import (
    configure_logging, log_request_received, log_prediction_processed,
    log_validation_failure, log_internal_error,
)

APP_TITLE = "Fraud Risk Decision API (Local Simulation)"
APP_DESCRIPTION = (
    "A local FastAPI service exposing the approved fraud risk decision "
    "engine (transaction + behavioral features -> LightGBM -> frozen "
    "APPROVE/REVIEW/BLOCK policy).\n\n"
    "**This is a local engineering/demonstration service for this "
    "project. It is NOT a production-deployed financial system** — no "
    "authentication, no persistence beyond the process lifetime, no "
    "distributed infrastructure. See `reports/phase7_api_service_summary.md`.\n\n"
    "`risk_score` is a fraud probability in [0, 1]. **Higher risk_score "
    "means higher predicted fraud risk.**"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loads the (cached) model bundle + frozen policy and warm-starts
    # behavioral state ONCE at process startup — never per-request.
    #
    # WARM_START_STATE (env var, default "true"): whether to replay
    # TRAIN+VALIDATION history into behavioral state at startup, which
    # requires loading the raw ~683MB train_transaction.csv (see
    # `src/api/dependencies.py:build_engine`). This defaults to "true" so
    # running the API directly on a host with the full dataset present
    # (Phase 7's tested, approved behavior) is unchanged. Phase 8's Docker
    # image explicitly sets this to "false" (see Dockerfile) — the raw
    # dataset is deliberately NOT copied into the container (kept lean;
    # the pretrained model bundle already encodes what training needed
    # from it), so the container starts cold instead: every entity begins
    # as a genuine, leakage-safe cold-start rather than failing to start
    # at all. This is a deployment/configuration decision, not a change to
    # any approved modeling or behavioral-feature logic.
    configure_logging()
    warm_start = os.environ.get("WARM_START_STATE", "true").strip().lower() not in ("0", "false", "no")
    engine, policy_config = deps.build_engine(warm_start=warm_start)
    deps.set_engine(engine, policy_config)
    # Phase 9: a fresh MetricsRegistry every process start — in-memory
    # only, explicitly reset on restart (see reports/phase9_monitoring_observability_summary.md).
    deps.set_metrics_registry(MetricsRegistry())
    # Phase 10: load the pre-built drift reference profile (a small JSON
    # artifact, not the raw dataset) — None if it hasn't been built yet
    # (see src/run_phase10_build_reference.py), in which case drift
    # endpoints report a clear "not available" state rather than failing
    # startup.
    deps.set_drift_monitor(deps.build_drift_monitor())
    # Phase 11: read-only load of the governance registry — this route
    # layer NEVER calls promote()/rollback()/register_candidate(); those
    # are explicit, human-invoked actions via scripts only (see
    # src/run_phase11_register_champion.py, scripts/demo_model_governance.py).
    deps.set_model_registry(deps.build_model_registry_or_none())
    yield
    # No teardown needed: this is a local, in-memory-only service with no
    # external connections to close.


app = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION, version=deps.API_VERSION, lifespan=lifespan)

# Phase 9: passive request-metrics middleware — observes every request's
# status/latency without touching request or response content. Installed
# once, applies to every route automatically (including /health,
# /metadata, /metrics itself, and any future route) with no per-route code.
app.add_middleware(MetricsMiddleware, registry_provider=deps.get_metrics_registry_or_none)


@app.exception_handler(TransactionValidationError)
async def handle_validation_error(request: Request, exc: TransactionValidationError) -> JSONResponse:
    """Invalid transaction data (per the engine's own validation) -> 422.
    Never leaks internal details beyond the engine's own clear message."""
    log_validation_failure(source="engine", detail=str(exc))
    try:
        deps.get_metrics_registry().record_engine_validation_failure()
    except RuntimeError:
        pass  # metrics not initialized yet — never block the actual error response on this
    return JSONResponse(status_code=422, content={"error": "invalid_transaction", "detail": str(exc)})


@app.exception_handler(RequestValidationError)
async def handle_request_validation_error(request: Request, exc: RequestValidationError):
    """
    Pydantic-level schema failures (missing required field, wrong type,
    `isFraud` present, etc.) — these never reach `predict()`'s body at
    all, so this is the only place to observe them. The actual HTTP
    response is FastAPI's own standard validation-error body, unchanged;
    this handler only ADDS metrics/logging around it.
    """
    log_validation_failure(source="request_schema", detail=str(exc)[:200])
    try:
        deps.get_metrics_registry().record_request_validation_failure()
    except RuntimeError:
        pass
    return await request_validation_exception_handler(request, exc)


@app.exception_handler(RuntimeError)
async def handle_runtime_error(request: Request, exc: RuntimeError) -> JSONResponse:
    """Engine/model not initialized -> 503, distinguished from a client
    input error (422) or an unexpected internal failure (500)."""
    try:
        deps.get_metrics_registry().record_engine_not_initialized()
    except RuntimeError:
        pass
    return JSONResponse(status_code=503, content={"error": "engine_not_initialized", "detail": str(exc)})


@app.exception_handler(StateBackendUnavailableError)
async def handle_state_backend_unavailable(request: Request, exc: StateBackendUnavailableError) -> JSONResponse:
    """
    Production upgrade — a real gap found and fixed during Phase 12
    failure testing (see reports/failure_testing.md's "Redis failure"
    section). `StateBackendUnavailableError` IS a `RuntimeError`
    subclass, so without this dedicated handler it would still resolve
    to SOME response (Starlette's exception dispatch walks the MRO), but
    it would be indistinguishable from "engine not initialized" in the
    response body/metrics — this handler gives it its own clear error
    code and its own metrics counter, matching the pattern already
    established for every other error category in this file. Registering
    a handler for a subclass makes it MORE specific than the existing
    bare `RuntimeError` handler above, so Starlette dispatches here first
    for this exact exception type — the existing handler's behavior for
    genuine engine-not-initialized errors is completely unaffected.
    """
    try:
        deps.get_metrics_registry().record_redis_lookup(duration_ms=0.0, failed=True)
    except RuntimeError:
        pass
    return JSONResponse(status_code=503, content={"error": "state_backend_unavailable", "detail": str(exc)})


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """
    Genuinely unexpected errors -> 500, generic message only (never a
    traceback or internal path — same guarantee Phase 7 already tested
    for Starlette's own default handling; this handler ADDS metrics
    visibility without weakening that guarantee). Registering a handler
    for the base `Exception` class does not intercept more specific
    exceptions that already have their own handler (`HTTPException`,
    `RequestValidationError`, `TransactionValidationError`, `RuntimeError`
    above) — Starlette dispatches to the most specific registered handler
    for a given exception type.
    """
    log_internal_error(endpoint=str(request.url.path))
    try:
        deps.get_metrics_registry().record_internal_error()
    except RuntimeError:
        pass
    return JSONResponse(status_code=500, content={"error": "internal_error", "detail": "An internal error occurred while processing the request."})


@app.post(
    "/predict", response_model=PredictResponse, tags=["prediction"],
    summary="Score one transaction and receive an APPROVE/REVIEW/BLOCK decision.",
    description=(
        "Submits one transaction to the approved RiskDecisionEngine. "
        "Behavioral features are generated internally from existing "
        "historical state (never from the current transaction's own "
        "data), the frozen Phase 5 policy is applied, and behavioral "
        "state is updated ONLY AFTER the prediction and decision are "
        "finalized. isFraud must never be included in the request."
    ),
)
def predict(request: TransactionRequest, engine: RiskDecisionEngine = Depends(deps.get_engine)) -> PredictResponse:
    log_request_received(endpoint="/predict", method="POST")
    metrics = deps.get_metrics_registry_or_none()
    if metrics is not None:
        metrics.record_prediction_attempt()

    transaction = deps.build_full_transaction_dict(engine, request)
    result = engine.process_transaction(transaction)  # validation errors propagate to the handlers above

    # Phase 9: record AFTER the engine call already returned — purely
    # observing an already-final result, never influencing it. If this
    # were removed entirely, `predict()`'s return value would be
    # byte-for-byte identical (see the non-interference tests).
    if metrics is not None:
        metrics.record_prediction_success(
            decision=result["decision"],
            risk_score=result["risk_score"],
            duration_ms=result["processing_metadata"]["processing_time_ms"],
            transaction_amount=extract_transaction_amount(transaction),
        )
    log_prediction_processed(
        transaction_id=result["transaction_id"], decision=result["decision"],
        risk_score=result["risk_score"], duration_ms=result["processing_metadata"]["processing_time_ms"],
    )

    return PredictResponse(
        transaction_id=result["transaction_id"],
        risk_score=result["risk_score"],
        decision=result["decision"],
        model_version=deps.MODEL_VERSION,
        policy_version=result["processing_metadata"]["policy_name"],
        processing_status="success",
        processing_time_ms=result["processing_metadata"]["processing_time_ms"],
    )


@app.post(
    "/predict/explain", tags=["prediction"],
    summary="Score one transaction AND return a SHAP-based explanation (presentation-only).",
    description=(
        "Production upgrade. Identical inference/decision path to "
        "/predict — the SAME RiskDecisionEngine.process_transaction() "
        "call, the SAME model, the SAME frozen policy — with one addition: "
        "a SHAP TreeExplainer breakdown of the top contributing features, "
        "computed strictly AFTER risk_score/decision are already final. "
        "The explanation can never change the score or decision (verified "
        "by a dedicated test — see tests/test_production_shap.py). This is "
        "a SEPARATE endpoint from /predict specifically so the default, "
        "high-frequency prediction path never pays the SHAP computation "
        "cost — see reports/streaming_benchmark.md for real measured "
        "explanation latency. Returns 503 if SHAP was not configured for "
        "this running instance (ENABLE_SHAP_EXPLAINABILITY=false)."
    ),
)
def predict_explain(request: TransactionRequest, engine: RiskDecisionEngine = Depends(deps.get_engine)) -> dict:
    log_request_received(endpoint="/predict/explain", method="POST")
    metrics = deps.get_metrics_registry_or_none()
    if metrics is not None:
        metrics.record_prediction_attempt()

    transaction = deps.build_full_transaction_dict(engine, request)
    result = engine.process_transaction(transaction, explain=True)

    if metrics is not None:
        metrics.record_prediction_success(
            decision=result["decision"],
            risk_score=result["risk_score"],
            duration_ms=result["processing_metadata"]["processing_time_ms"],
            transaction_amount=extract_transaction_amount(transaction),
        )
        explanation_ms = (result.get("explanation") or {}).get("explanation_time_ms")
        if explanation_ms is not None:
            metrics.record_explanation(duration_ms=explanation_ms)
    log_prediction_processed(
        transaction_id=result["transaction_id"], decision=result["decision"],
        risk_score=result["risk_score"], duration_ms=result["processing_metadata"]["processing_time_ms"],
    )

    if result["explanation"] is not None and "error" in result["explanation"]:
        raise HTTPException(status_code=503, detail=result["explanation"]["error"])

    return {
        "transaction_id": result["transaction_id"],
        "risk_score": result["risk_score"],
        "decision": result["decision"],
        "model_version": deps.MODEL_VERSION,
        "policy_version": result["processing_metadata"]["policy_name"],
        "explanation": result["explanation"],
    }


@app.get(
    "/health", response_model=HealthResponse, tags=["operations"],
    summary="Report whether the service is operational.",
)
def health() -> HealthResponse:
    engine = deps.get_engine_or_none()
    model_loaded = engine is not None and engine.model is not None
    policy_loaded = engine is not None and engine.policy is not None
    status = "ok" if (model_loaded and policy_loaded) else "degraded"
    return HealthResponse(status=status, model_loaded=model_loaded, policy_loaded=policy_loaded)


@app.get(
    "/metadata", response_model=MetadataResponse, tags=["operations"],
    summary="Return safe, non-sensitive model and policy metadata.",
)
def metadata(engine: RiskDecisionEngine = Depends(deps.get_engine)) -> MetadataResponse:
    return MetadataResponse(
        api_version=deps.API_VERSION,
        model_type="LightGBM (transaction-level + card1 behavioral features)",
        model_version=deps.MODEL_VERSION,
        policy_name=engine.policy.name,
        policy_version=engine.policy.name,
        supported_decisions=list(DECISIONS),
        approve_threshold=engine.policy.approve_threshold,
        block_threshold=engine.policy.block_threshold,
    )


@app.post(
    "/dev/reset-state", tags=["development-only"],
    summary="[DEV/TESTING ONLY] Reset all behavioral state.",
    description=(
        "Clears ALL accumulated behavioral history for every entity. This "
        "is a development/testing utility, explicitly NOT part of the "
        "production-style prediction API — a real deployment would not "
        "expose this without strong access control, if at all."
    ),
)
def dev_reset_state() -> dict:
    deps.reset_engine_state()
    return {"status": "state_reset"}


@app.get(
    "/metrics", tags=["observability"],
    summary="Prometheus-style text metrics.",
    description=(
        "Exposes service metrics in the Prometheus text exposition "
        "format (plain counters/gauges with # HELP / # TYPE comments). "
        "This means the format is scrape-compatible — it does NOT mean a "
        "Prometheus server or Grafana is deployed anywhere in this "
        "project (neither is; see reports/phase9_monitoring_observability_summary.md). "
        "Metrics are in-memory only and reset when the process restarts."
    ),
)
def metrics_endpoint(registry: MetricsRegistry = Depends(deps.get_metrics_registry)) -> PlainTextResponse:
    return PlainTextResponse(render_prometheus_text(registry.snapshot()), media_type="text/plain; version=0.0.4")


@app.get(
    "/monitoring/summary", tags=["observability"],
    summary="Human/machine-readable monitoring snapshot.",
    description=(
        "A structured JSON snapshot of current service activity, latency, "
        "decisions, risk-score distribution, and data-quality/error "
        "counters — the same underlying data as /metrics, in a friendlier "
        "shape. In-memory only; resets to zero on process restart."
    ),
)
def monitoring_summary(registry: MetricsRegistry = Depends(deps.get_metrics_registry)) -> dict:
    return registry.snapshot()


@app.get(
    "/drift/summary", tags=["drift"],
    summary="Latest known drift-monitoring status (read-only, cheap).",
    description=(
        "Reports the CURRENTLY KNOWN drift-monitoring state — batches "
        "analyzed so far, a severity-level histogram, and the most recent "
        "batch's overall status/explanation. Does NOT run a new analysis "
        "(that only happens via POST /drift/analyze, a deliberately "
        "separate, batch-only operation — never triggered by /predict). "
        "If no reference profile has been built yet "
        "(`python -m src.run_phase10_build_reference`), reports "
        "`available: false` rather than erroring."
    ),
)
def drift_summary() -> dict:
    monitor = deps.get_drift_monitor_or_none()
    if monitor is None:
        return {"available": False, "detail": "No drift reference profile is loaded."}
    return {"available": True, **monitor.summary()}


@app.post(
    "/drift/analyze", tags=["drift"],
    summary="Run drift analysis on a batch of ALREADY-SCORED records.",
    description=(
        "A deliberately separate, batch-level, investigation-only "
        "operation — NOT part of the real-time /predict path (per Phase "
        "10's passive-only design principle, drift analysis must never "
        "run automatically on every prediction). Each record in the "
        "request body should already contain the monitored raw fields "
        "(e.g. TransactionAmt, ProductCD), the already-computed Phase 4 "
        "behavioral (`bhv_*`) fields, and the model's own `risk_score`/"
        "`decision` for that transaction — this endpoint does not call "
        "the model or the behavioral-state engine itself, so it cannot "
        "duplicate or diverge from their approved logic. Requires at "
        "least 1 record; batches smaller than the documented minimum are "
        "still analyzed but flagged as low-confidence in the response."
    ),
)
def drift_analyze(request: DriftAnalysisRequest) -> dict:
    monitor = deps.get_drift_monitor_or_none()
    if monitor is None:
        raise HTTPException(status_code=503, detail="No drift reference profile is loaded — run `python -m src.run_phase10_build_reference` first.")
    if not request.records:
        raise HTTPException(status_code=422, detail="records must contain at least one item.")

    batch_df = pd.DataFrame(request.records)
    risk_scores = batch_df.pop("risk_score") if "risk_score" in batch_df.columns else None
    decisions = batch_df.pop("decision") if "decision" in batch_df.columns else None
    return monitor.analyze_batch(batch_df, risk_scores=risk_scores, decisions=decisions)


@app.get(
    "/model-governance/summary", tags=["governance"],
    summary="Read-only model governance/lifecycle status.",
    description=(
        "Reports the current champion model, its version/status, and "
        "counts of registered candidates/rejected/retired models — from "
        "the Phase 11 local model registry. This route is STRICTLY "
        "READ-ONLY: it never promotes, rolls back, or registers anything. "
        "Promotion/rollback are explicit, human-invoked actions performed "
        "via `src/governance/registry.py`'s `ModelRegistry.promote()`/"
        "`rollback()`, never triggered by this or any other HTTP endpoint. "
        "If no registry has been built yet "
        "(`python -m src.run_phase11_register_champion`), reports "
        "`available: false` rather than erroring."
    ),
)
def model_governance_summary() -> dict:
    registry = deps.get_model_registry_or_none()
    if registry is None:
        return {"available": False, "detail": "No model governance registry is loaded."}
    summary = {"available": True, **registry.summary()}

    # Production upgrade — best-effort MLflow cross-reference (additive
    # only; never raises, never blocks this endpoint if MLflow hasn't
    # been used yet). The existing governance registry above remains the
    # sole AUTHORITATIVE source for champion/promotion status — see
    # src/mlops/mlflow_tracking.py's module docstring.
    try:
        from src.mlops.mlflow_tracking import get_champion_run_summary
        mlflow_summary = get_champion_run_summary()
        summary["mlflow"] = mlflow_summary if mlflow_summary is not None else {"available": False}
    except Exception:
        summary["mlflow"] = {"available": False}

    return summary
