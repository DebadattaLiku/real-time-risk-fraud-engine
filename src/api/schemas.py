"""
Phase 7: Pydantic request/response schemas for the FastAPI risk scoring
service.

No ML, feature, or decision logic lives here — these are pure data-shape
and basic-type/range validation definitions. The actual leakage-safe
behavioral feature generation, model prediction, and decision policy are
entirely inside the already-approved `RiskDecisionEngine`
(`src/engine/risk_engine.py`), called from `src/api/main.py`.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TransactionRequest(BaseModel):
    """
    One transaction to be scored.

    The full Phase 1 feature schema has ~390 raw columns (many opaque
    Vesta-engineered `V*`/`C*`/`D*`/`M*` columns). Rather than hand-declare
    every one, the handful that are structurally required or most relevant
    are typed explicitly below; any additional Phase 1 columns the client
    supplies are accepted via `extra="allow"` and passed through to the
    engine exactly as Phase 1's `FeaturePipeline` already tolerates missing
    values for them (categorical -> "__missing__", numeric -> NaN,
    preserved for LightGBM's native missing-value handling — unchanged
    from Phase 1/2B). Any expected column NOT supplied at all (missing key)
    is filled with `null` before reaching the engine
    (`src/api/dependencies.py:build_full_transaction_dict`), which is safe
    for every column except the four structurally required ones below,
    which Pydantic itself rejects with a 422 if omitted.

    `isFraud` is deliberately NOT a field here — the engine raises if it
    is present at all (a client should never send it), and this schema
    does not declare or reserve it, so it can only arrive via `extra`,
    where the engine's own guard still catches and rejects it.
    """

    model_config = ConfigDict(extra="allow", json_schema_extra={
        "example": {
            "TransactionID": 3488959,
            "TransactionDT": 13268363,
            "TransactionAmt": 106.0,
            "ProductCD": "W",
            "card1": 15775,
            "card4": "visa",
            "card6": "debit",
        }
    })

    TransactionID: int = Field(..., description="Unique transaction identifier.")
    TransactionDT: float = Field(
        ..., description=(
            "Transaction time delta in seconds (dataset-relative, not a "
            "calendar timestamp). Used only for temporal ordering and "
            "behavioral-state lookups — never as a raw model feature "
            "(Phase 1/4 decision, unchanged here)."
        ),
    )
    TransactionAmt: float = Field(..., ge=0, description="Transaction amount; must be non-negative.")
    card1: int = Field(
        ..., description=(
            "card1 pseudo-entity identifier, used for behavioral-history "
            "lookups. This is Vesta's anonymized card identifier — NOT a "
            "verified customer ID (Phase 0/4 caveat, unchanged)."
        ),
    )
    ProductCD: Optional[str] = Field(None, description="Product code category.")

    @field_validator("TransactionDT")
    @classmethod
    def _dt_must_be_finite(cls, v):
        if v != v or v in (float("inf"), float("-inf")):  # NaN/inf check without importing math
            raise ValueError("TransactionDT must be a finite number")
        return v

    @model_validator(mode="before")
    @classmethod
    def _forbid_isfraud(cls, data):
        """
        Rejects `isFraud` at the schema layer, BEFORE it can reach
        `build_full_transaction_dict` — which, since it only forwards
        columns the engine's schema actually expects (deliberately
        excluding `isFraud`), would otherwise silently DROP a
        client-supplied `isFraud` rather than reject it, defeating the
        intended protection. Caught by a real test failure
        (`test_predict_with_isfraud_field_rejected` returned 200 instead
        of 422 until this validator was added) — the engine-level guard in
        `RiskDecisionEngine._validate_transaction` alone was not sufficient
        because the API layer's own column-filtering step ran first and
        stripped the evidence before the engine ever saw it.
        """
        if isinstance(data, dict) and "isFraud" in data:
            raise ValueError("isFraud must not be included in a prediction request — labels are not available at prediction time.")
        return data


class PredictResponse(BaseModel):
    """
    Response for `POST /predict`. Higher `risk_score` means HIGHER
    predicted fraud risk (consistent with every score in this project
    since Phase 2A). Behavioral features are intentionally NOT included —
    they are internal engine state, not part of the public API contract.
    """

    transaction_id: int
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Predicted fraud probability in [0, 1]. Higher = higher fraud risk.")
    decision: str = Field(..., description="One of APPROVE, REVIEW, BLOCK.")
    model_version: str
    policy_version: str
    processing_status: str = "success"
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' if fully operational, 'degraded' otherwise.")
    model_loaded: bool
    policy_loaded: bool


class MetadataResponse(BaseModel):
    api_version: str
    model_type: str
    model_version: str
    policy_name: str
    policy_version: str
    supported_decisions: list
    approve_threshold: float
    block_threshold: float


class ErrorResponse(BaseModel):
    error: str
    detail: str


class DriftAnalysisRequest(BaseModel):
    """
    Request body for `POST /drift/analyze`. Each item in `records` should
    be an already-scored transaction: the monitored raw fields (e.g.
    `TransactionAmt`, `ProductCD`), the already-computed Phase 4
    behavioral (`bhv_*`) fields, and this transaction's `risk_score`/
    `decision` — NOT raw, unscored transactions. This endpoint never calls
    the model or the behavioral-state engine itself (see
    `src/drift/monitor.py`'s module docstring for why that separation
    matters).
    """

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "records": [
                {"TransactionAmt": 75.5, "ProductCD": "W", "bhv_prev_txn_count": 12.0,
                 "risk_score": 0.02, "decision": "APPROVE"},
            ]
        }
    })

    records: list[dict] = Field(..., min_length=1, description="Already-scored transaction records to analyze for drift.")
