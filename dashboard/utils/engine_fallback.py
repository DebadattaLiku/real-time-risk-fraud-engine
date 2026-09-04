"""
Phase 12: Local RiskDecisionEngine fallback for when the FastAPI service
isn't running.

Zero new inference logic — `build_engine_for_dashboard()` calls
`src.api.dependencies.build_engine()`, the EXACT function Phase 7's real
API startup uses, just with `warm_start=False` for a fast dashboard
session (consistent with Phase 8's documented container default, not a
new behavior). `score_transaction_locally()` calls
`RiskDecisionEngine.process_transaction()` directly — the same approved
Phase 6 engine method the API's `/predict` route calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.api import dependencies as deps
from src.engine.risk_engine import RiskDecisionEngine, TransactionValidationError


def build_engine_for_dashboard(warm_start: bool = False) -> RiskDecisionEngine:
    """
    Builds a real engine backed by the real Phase 4/5 model bundle and the
    real frozen Phase 5 policy — identical to what the FastAPI service
    itself constructs at startup (`src/api/dependencies.py:build_engine`).
    `warm_start=False` by default so a dashboard session starts in
    seconds rather than replaying the full historical dataset; this
    mirrors Phase 8's own container default, not a new design decision.
    """
    engine, _policy_config = deps.build_engine(warm_start=warm_start)
    return engine


def score_transaction_locally(engine: RiskDecisionEngine, transaction: dict) -> tuple[bool, dict]:
    """
    Scores one transaction directly through the real engine — the exact
    same `process_transaction()` method `POST /predict` calls. Returns
    `(True, result_dict)` on success (the engine's own raw result, which
    — unlike the API's `PredictResponse` — DOES include
    `behavioral_features`, useful for this dashboard's "Decision Signals"
    view), or `(False, {"error": ...})` on a real
    `TransactionValidationError` from the engine's own validation.

    `isFraud` is checked BEFORE filtering to the engine's expected raw
    columns, not after — the same Phase 7 lesson applies here: filtering
    down to `engine.get_expected_raw_columns()` (which deliberately
    excludes `isFraud`) would otherwise silently DROP a caller-supplied
    `isFraud` instead of rejecting it, since the engine's own guard would
    never get the chance to see it. Caught by a real test failure
    (`test_score_transaction_locally_isfraud_rejected` returned success
    instead of a rejection) before being fixed here.
    """
    if "isFraud" in transaction:
        return False, {
            "error": "invalid_transaction",
            "detail": "isFraud must not be included in a prediction request — labels are not available at prediction time.",
        }
    full_row = {col: transaction.get(col) for col in engine.get_expected_raw_columns()}
    try:
        result = engine.process_transaction(full_row)
        return True, result
    except TransactionValidationError as e:
        return False, {"error": "invalid_transaction", "detail": str(e)}
