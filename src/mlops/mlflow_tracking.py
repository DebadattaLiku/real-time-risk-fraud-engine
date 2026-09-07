"""
Production upgrade — MLflow experiment tracking + model registry.

Adds MLflow for experiment/parameter/metric/artifact tracking and model
registration. Does NOT replace Phase 11's existing local governance
system (`src/governance/`), which remains the authoritative source for
promotion/rollback decisions:

    MLflow                                Existing Governance
        |                                        |
    Experiment / Artifact / Registry        Promotion / Compatibility /
    metadata (tracking, discoverability)     Operational gates (decision authority)

This module logs the REAL, ALREADY-COMPUTED champion metrics from Phase
4/5's saved evaluation artifacts (`data/interim/phase4_metrics.json`,
`phase5_metrics.json`) and the real trained model bundle — it does NOT
retrain anything or fabricate a new experiment run. The logged run
represents "registering the existing, already-approved champion into
MLflow for tracking," explicitly tagged as such, not a live new training
execution.

Local, SQLite-backed tracking (`sqlite:///mlflow.db`) by default — this
project's installed MLflow version (3.16) deprecated the plain filesystem
store in favor of a database backend, so SQLite (a real, working, local,
zero-extra-setup database — not a fabricated workaround) is used instead.
No server is required to use `mlflow.log_*`/`mlflow.register_model`
against this URI. A real MLflow Tracking Server (`mlflow server --host
0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db`) can be
pointed at with `MLFLOW_TRACKING_URI=http://localhost:5000` if one is
running; this module works identically either way since it only calls
the standard `mlflow` client API.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MLFLOW_EXPERIMENT_NAME = "fraud-risk-engine"
MLFLOW_REGISTERED_MODEL_NAME = "fraud-risk-lightgbm"


def log_champion_to_mlflow(tracking_uri: str = "sqlite:///mlflow.db") -> str:
    """
    Logs the REAL, already-approved champion (`fraud-risk-lightgbm-v1`)
    into MLflow: real hyperparameters, real Phase 4 validation/test
    ranking metrics, real Phase 5 decision-policy metrics, the real
    trained model bundle as an artifact, and the real Phase 11 artifact
    SHA-256 hash as a tag (so the MLflow record and the governance
    registry record can be cross-referenced by hash, not just by name).

    Returns the MLflow run ID.
    """
    import mlflow
    import mlflow.lightgbm
    import yaml

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with open(REPO_ROOT / "data" / "interim" / "phase4_metrics.json") as f:
        phase4 = json.load(f)
    with open(REPO_ROOT / "data" / "interim" / "phase5_metrics.json") as f:
        phase5 = json.load(f)
    with open(REPO_ROOT / "config" / "decision_policy.yaml") as f:
        policy_config = yaml.safe_load(f)
    with open(REPO_ROOT / "artifacts" / "models" / "registry.json") as f:
        governance_registry = json.load(f)

    champion = governance_registry["models"][governance_registry["champion_model_id"]]
    model_b = phase4["model_b_behavioral"]

    with mlflow.start_run(run_name="fraud-risk-lightgbm-v1-registration") as run:
        mlflow.set_tags({
            "model_id": champion["model_id"],
            "model_version": champion["model_version"],
            "run_type": "existing_champion_registration",  # NOT a new training run — see module docstring
            "artifact_sha256": champion["artifact_sha256"],
            "feature_schema_version": champion["feature_schema_version"],
            "decision_policy_version": champion["decision_policy_version"],
        })

        mlflow.log_params({
            **{f"hyperparam_{k}": v for k, v in champion.get("hyperparameters", {}).items()},
            "training_data_reference": champion["training_data_reference"],
            "validation_data_reference": champion["validation_data_reference"],
        })

        mlflow.log_metrics({
            "val_pr_auc": model_b["val_ranking"]["pr_auc"],
            "val_roc_auc": model_b["val_ranking"]["roc_auc"],
            "test_pr_auc": model_b["test_ranking"]["pr_auc"],
            "test_roc_auc": model_b["test_ranking"]["roc_auc"],
            "test_recall_at_1pct": model_b["test_budget_table"][0]["recall_at_k"],
            "test_recall_at_2pct": model_b["test_budget_table"][1]["recall_at_k"],
            "test_recall_at_5pct": model_b["test_budget_table"][2]["recall_at_k"],
            "test_precision_among_blocked": phase5["test_eval"]["precision_among_blocked"],
            "test_pct_legitimate_blocked": phase5["test_eval"]["pct_legitimate_blocked"],
            "test_fraud_captured_review_plus_block": phase5["test_eval"]["fraud_captured_review_plus_block"],
        })

        model_bundle_path = REPO_ROOT / "data" / "interim" / "phase6_model_bundle.pkl"
        mlflow.log_artifact(str(model_bundle_path), artifact_path="model_bundle")

        policy_path = REPO_ROOT / "config" / "decision_policy.yaml"
        mlflow.log_artifact(str(policy_path), artifact_path="policy")

        # The full serving bundle (schema + FeaturePipeline +
        # LightGBMPreprocessor + model) is logged above via log_artifact
        # for completeness/backward reference, but mlflow.register_model
        # requires a properly-FLAVORED MLflow Model (MLmodel metadata), not
        # a generic file — so the actual LightGBM classifier is logged a
        # second time via mlflow.lightgbm.log_model specifically so
        # registration succeeds genuinely. This does NOT change what the
        # live API serves (still the full bundle via
        # data/interim/phase6_model_bundle.pkl, unchanged) — this is a
        # tracking/registry concern only.
        with open(REPO_ROOT / "data" / "interim" / "phase6_model_bundle.pkl", "rb") as f:
            import pickle
            bundle_obj = pickle.load(f)
        try:
            mlflow.lightgbm.log_model(
                bundle_obj["model"], artifact_path="model", registered_model_name=MLFLOW_REGISTERED_MODEL_NAME,
            )
        except Exception as e:  # noqa: BLE001 — registry registration is best-effort; the run/metrics/artifacts above are still logged regardless
            print(f"MLflow model registration skipped (non-fatal): {e}")

        return run.info.run_id


def get_champion_run_summary(tracking_uri: str = "sqlite:///mlflow.db") -> dict | None:
    """Read-only lookup for the dashboard/API — the most recent
    'existing_champion_registration' run, or None if MLflow has no runs
    logged yet."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    try:
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if experiment is None:
            return None
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1)
        if len(runs) == 0:
            return None
        row = runs.iloc[0]
        return {
            "run_id": row["run_id"],
            "metrics": {k.replace("metrics.", ""): v for k, v in row.items() if k.startswith("metrics.")},
            "tags": {k.replace("tags.", ""): v for k, v in row.items() if k.startswith("tags.")},
        }
    except Exception:
        return None
