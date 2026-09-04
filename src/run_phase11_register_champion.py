"""
Phase 11: Register the existing, already-approved LightGBM model
(transaction-level + Phase 4 behavioral features) as the initial champion
in the governance registry.

Uses REAL, already-computed information — Phase 4's saved evaluation
metrics (`data/interim/phase4_metrics.json`), Phase 5's saved policy
evaluation (`data/interim/phase5_metrics.json`), the frozen policy
(`config/decision_policy.yaml`), the real trained hyperparameters
(`src/models/lightgbm_model.py`'s `DEFAULT_PARAMS`), and a real SHA-256 of
the actual model bundle artifact. Does NOT retrain or re-evaluate
anything — this is metadata registration, not a training run.

Usage:
    python -m src.run_phase11_register_champion
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml

from src.models.lightgbm_model import DEFAULT_PARAMS
from src.governance.metadata import build_model_metadata, compute_artifact_sha256
from src.governance.registry import ModelRegistry

REGISTRY_PATH = REPO_ROOT / "artifacts" / "models" / "registry.json"
MODEL_BUNDLE_PATH = REPO_ROOT / "data" / "interim" / "phase6_model_bundle.pkl"
CHAMPION_MODEL_ID = "fraud-risk-lightgbm-v1"


def main() -> int:
    print("Loading real, already-computed evaluation metrics (Phase 4/5) — not re-evaluating.")
    with open(REPO_ROOT / "data" / "interim" / "phase4_metrics.json") as f:
        phase4 = json.load(f)
    with open(REPO_ROOT / "data" / "interim" / "phase5_metrics.json") as f:
        phase5 = json.load(f)
    with open(REPO_ROOT / "config" / "decision_policy.yaml") as f:
        policy_config = yaml.safe_load(f)

    model_b = phase4["model_b_behavioral"]
    evaluation_metrics = {
        "validation": {
            "ranking": model_b["val_ranking"],
            "budget_table": model_b["val_budget_table"],
        },
        "test_final_evaluation_only": {
            "ranking": model_b["test_ranking"],
            "budget_table": model_b["test_budget_table"],
            "note": (
                "This is the ONE-TIME, final Phase 4 test-set evaluation — "
                "reported here for the historical record, never used again "
                "as a tuning or promotion-gate signal (see src/governance/evaluation.py)."
            ),
        },
        "decision_policy_validation_eval": phase5["candidate_policies"]["balanced"]["val_eval"],
        "decision_policy_test_eval": phase5["test_eval"],
    }

    print(f"Computing real SHA-256 of the actual model bundle artifact ({MODEL_BUNDLE_PATH})...")
    artifact_sha256 = compute_artifact_sha256(MODEL_BUNDLE_PATH)
    print(f"  {artifact_sha256}")

    metadata = build_model_metadata(
        model_id=CHAMPION_MODEL_ID,
        model_version="v1",
        model_type="LightGBM (transaction-level + card1 behavioral features, Phase 4)",
        status="CHAMPION",
        feature_schema_version="phase1-v1",
        training_data_reference=(
            "TRAIN partition, Phase 1 temporal split (0-413,377, 70%) — see "
            "data/interim/phase1_split_metadata.json for exact boundaries."
        ),
        validation_data_reference=(
            "VALIDATION partition, Phase 1 temporal split (413,378-501,958, 15%) — "
            "same source as the Phase 10 drift reference profile."
        ),
        test_data_reference=(
            "TEST partition, Phase 1 temporal split (501,959-590,539, 15%) — "
            "evaluated ONCE (Phase 4), final record only, never re-used for tuning."
        ),
        evaluation_metrics=evaluation_metrics,
        artifact_reference=str(MODEL_BUNDLE_PATH.relative_to(REPO_ROOT)),
        artifact_sha256=artifact_sha256,
        decision_policy_version=policy_config["policy_name"],
        hyperparameters=DEFAULT_PARAMS,
        notes=(
            "Initial champion, registered in Phase 11 from the model approved in Phases "
            "4-5. Not retrained or re-evaluated for this registration — all metrics are "
            "the real, already-saved values from those phases."
        ),
    )

    registry = ModelRegistry(REGISTRY_PATH)
    try:
        registry.register_champion(metadata)
        print(f"Registered '{CHAMPION_MODEL_ID}' as CHAMPION.")
    except Exception as e:
        print(f"Registration skipped/failed (likely already registered): {e}")

    registry.save()
    print(f"Saved registry to: {REGISTRY_PATH}")
    print(json.dumps(registry.summary(), indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
