"""
Phase 7: FastAPI dependency / lifecycle management.

Builds ONE `RiskDecisionEngine` instance (reusing Phase 6's
`build_or_load_bundle` for the model/pipelines and Phase 5's frozen
`DecisionPolicy` — never duplicated or reimplemented here), and holds it
in a module-level singleton for the lifetime of the application process.
This is what makes the API's stateful behavior work: the SAME engine
instance (and therefore the SAME `BehavioralStateManager`) is reused
across every request, so behavioral history accumulates exactly as Phase 6
intended — the model is not reloaded and state is not reset per request.

Tests replace the singleton via FastAPI's `app.dependency_overrides`
mechanism (see `tests/test_phase7_api.py`), so the automated test suite
never has to pay the cost of loading the real ~600MB dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import yaml

from src.data.load import load_train_transaction, load_config
from src.data.split import compute_temporal_split
from src.decision.policy import DecisionPolicy
from src.engine.state import BehavioralStateManager
from src.engine.risk_engine import RiskDecisionEngine
from src.run_phase6_simulation import build_or_load_bundle
from src.monitoring.metrics import MetricsRegistry
from src.drift.monitor import DriftMonitor
from src.drift.reference import load_reference_profile
from src.governance.registry import ModelRegistry

MODEL_VERSION = "phase4_lightgbm_transaction_plus_behavioral_v1"
API_VERSION = "0.1.0"
REFERENCE_PROFILE_PATH = REPO_ROOT / "artifacts" / "drift" / "reference_profile.json"
MODEL_REGISTRY_PATH = REPO_ROOT / "artifacts" / "models" / "registry.json"

_engine: RiskDecisionEngine | None = None
_policy_config: dict | None = None
_metrics_registry: MetricsRegistry | None = None
_drift_monitor: DriftMonitor | None = None
_model_registry: ModelRegistry | None = None


def build_engine(warm_start: bool = True) -> tuple[RiskDecisionEngine, dict]:
    """
    Builds a real engine backed by the cached (or freshly trained) Phase
    4/5 model bundle and the frozen Phase 5 policy. `warm_start=True`
    (the default, used at real application startup) replays TRAIN+VALIDATION
    history into behavioral state via `bulk_initialize` — the same
    approach Phase 6's simulation used — so the very first live request
    already has realistic history to draw on, rather than starting stone
    cold. `warm_start=False` is for fast test/demo setups that don't need
    that ~10-30s historical load.
    """
    config = load_config()
    bundle = build_or_load_bundle(config)
    schema = bundle["schema"]
    fp = bundle["feature_pipeline"]
    lgbm_pre = bundle["lgbm_preprocessor"]
    model = bundle["model"]

    with open(REPO_ROOT / "config" / "decision_policy.yaml") as f:
        policy_config = yaml.safe_load(f)
    policy = DecisionPolicy(
        approve_threshold=policy_config["approve_threshold"],
        block_threshold=policy_config["block_threshold"],
        name=policy_config["policy_name"],
    )

    state_manager = BehavioralStateManager(entity_col="card1")
    if warm_start:
        id_col, time_col = "TransactionID", "TransactionDT"
        df, _ = load_train_transaction(config=config)
        train_df, val_df, test_df, _ = compute_temporal_split(
            df, time_col, id_col,
            config["split"]["train_ratio"], config["split"]["val_ratio"], config["split"]["test_ratio"],
        )
        del df
        historical = pd.concat([train_df, val_df], ignore_index=True)
        state_manager.bulk_initialize(
            historical, time_col=time_col, amount_col="TransactionAmt",
            as_of_time=float(historical[time_col].max()),
        )

    engine = RiskDecisionEngine(
        schema=schema, feature_pipeline=fp, lgbm_preprocessor=lgbm_pre, model=model,
        policy=policy, state_manager=state_manager,
    )
    return engine, policy_config


def set_engine(engine: RiskDecisionEngine, policy_config: dict) -> None:
    """
    Installs the given engine as the process-wide singleton. Called once
    from the application `lifespan` at real startup, and by tests via
    `app.dependency_overrides` to inject a fast synthetic engine instead.
    """
    global _engine, _policy_config
    _engine = engine
    _policy_config = policy_config


def get_engine() -> RiskDecisionEngine:
    """FastAPI dependency: returns the current engine singleton, or raises
    if startup never ran (mapped to a 503 by main.py's exception handler)."""
    if _engine is None:
        raise RuntimeError("RiskDecisionEngine is not initialized — application startup did not complete.")
    return _engine


def get_engine_or_none() -> RiskDecisionEngine | None:
    """
    Non-raising variant for `/health`, which must be able to report a
    graceful 'degraded' status rather than itself failing when the engine
    isn't initialized — a plain `Depends(get_engine)` would raise before
    the route body ever runs, which is the wrong behavior specifically for
    a health check.
    """
    return _engine


def get_policy_config() -> dict:
    if _policy_config is None:
        raise RuntimeError("Decision policy configuration is not initialized — application startup did not complete.")
    return _policy_config


def get_metrics_registry() -> MetricsRegistry:
    """
    FastAPI dependency for the process-wide `MetricsRegistry` singleton.
    Created once (`set_metrics_registry`, called from `main.py`'s
    `lifespan`, mirroring the engine's own lifecycle) and reused for every
    request for the life of the process — metrics accumulate across
    requests exactly like behavioral state does, and are equally reset on
    restart (in-memory only, no persistence, per Phase 9 scope).
    """
    if _metrics_registry is None:
        raise RuntimeError("MetricsRegistry is not initialized — application startup did not complete.")
    return _metrics_registry


def set_metrics_registry(registry: MetricsRegistry) -> None:
    global _metrics_registry
    _metrics_registry = registry


def get_metrics_registry_or_none() -> MetricsRegistry | None:
    """Non-raising variant for the middleware, which must never fail a
    request just because metrics aren't initialized yet."""
    return _metrics_registry


def build_drift_monitor() -> DriftMonitor | None:
    """
    Loads the Phase 10 reference profile (a small, pre-built JSON artifact
    — see `src/run_phase10_build_reference.py`) and wraps it in a
    `DriftMonitor`. Returns `None` (not an exception) if the artifact
    doesn't exist yet, so a fresh checkout that hasn't run the
    reference-building script still starts the API successfully — drift
    endpoints report a clear "not available" state instead (see
    `src/api/main.py`).
    """
    if not REFERENCE_PROFILE_PATH.is_file():
        return None
    profile = load_reference_profile(REFERENCE_PROFILE_PATH)
    return DriftMonitor(profile)


def set_drift_monitor(monitor: DriftMonitor | None) -> None:
    global _drift_monitor
    _drift_monitor = monitor


def get_drift_monitor_or_none() -> DriftMonitor | None:
    return _drift_monitor


def build_model_registry_or_none() -> ModelRegistry | None:
    """
    Loads the Phase 11 governance registry (a small local JSON file — see
    `src/run_phase11_register_champion.py`). Returns `None`, not an
    exception, if it doesn't exist yet, so a fresh checkout still starts
    the API — `/model-governance/summary` reports "not available" instead.
    This is READ-ONLY from the API's perspective: no route in this file
    ever calls `promote()`/`rollback()`/`register_candidate()` — those are
    explicit, human-invoked actions via scripts, never HTTP-triggered.
    """
    if not MODEL_REGISTRY_PATH.is_file():
        return None
    return ModelRegistry(MODEL_REGISTRY_PATH)


def set_model_registry(registry: ModelRegistry | None) -> None:
    global _model_registry
    _model_registry = registry


def get_model_registry_or_none() -> ModelRegistry | None:
    return _model_registry


def reset_engine_state() -> None:
    """
    DEV/TESTING ONLY. Clears all behavioral state on the current engine.
    Exposed via the explicitly-marked `POST /dev/reset-state` route in
    `main.py`, never as ordinary prediction-API behavior.
    """
    if _engine is not None:
        _engine.reset_state()


def build_full_transaction_dict(engine: RiskDecisionEngine, request) -> dict:
    """
    Expands a (possibly partial) `TransactionRequest` into a dict covering
    EVERY raw column the engine expects as a key — filling any column the
    client didn't supply with `None`. This is required because
    `RiskDecisionEngine._validate_transaction` checks for key PRESENCE
    (not just non-null values) for every Phase 1 schema column; the four
    structurally required fields (TransactionID/DT/Amt/card1) are already
    guaranteed present and valid by Pydantic before this function is ever
    called.
    """
    data = request.model_dump()
    expected_columns = engine.get_expected_raw_columns()
    return {col: data.get(col, None) for col in expected_columns}
