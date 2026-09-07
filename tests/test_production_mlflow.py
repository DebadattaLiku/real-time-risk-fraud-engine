"""
Production upgrade — MLflow tracking tests. Uses a temporary SQLite
tracking URI per test (via tmp_path) so these tests never depend on or
pollute the real repo-root `mlflow.db` created by a real run of
`src.mlops.mlflow_tracking.log_champion_to_mlflow()` (documented in
reports/production_readiness.md as a real, already-executed run).
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mlops.mlflow_tracking import log_champion_to_mlflow, get_champion_run_summary, MLFLOW_EXPERIMENT_NAME


def _tmp_tracking_uri(tmp_path) -> str:
    return f"sqlite:///{tmp_path}/mlflow_test.db"


def test_log_champion_to_mlflow_real_run(tmp_path):
    """Real MLflow API calls, real repository artifacts (Phase 4/5 saved
    metrics, the real model bundle) — not mocked."""
    uri = _tmp_tracking_uri(tmp_path)
    run_id = log_champion_to_mlflow(tracking_uri=uri)
    assert run_id is not None
    assert len(run_id) > 0


def test_logged_metrics_match_real_phase4_phase5_artifacts(tmp_path):
    import json
    uri = _tmp_tracking_uri(tmp_path)
    log_champion_to_mlflow(tracking_uri=uri)
    summary = get_champion_run_summary(tracking_uri=uri)

    with open(REPO_ROOT / "data" / "interim" / "phase4_metrics.json") as f:
        phase4 = json.load(f)
    expected_pr_auc = phase4["model_b_behavioral"]["test_ranking"]["pr_auc"]

    assert summary is not None
    assert summary["metrics"]["test_pr_auc"] == pytest.approx(expected_pr_auc)


def test_logged_tags_identify_this_as_registration_not_training(tmp_path):
    """Critical honesty property: the run must be tagged as registering an
    EXISTING champion, never implying a new training experiment occurred."""
    uri = _tmp_tracking_uri(tmp_path)
    log_champion_to_mlflow(tracking_uri=uri)
    summary = get_champion_run_summary(tracking_uri=uri)
    assert summary["tags"]["run_type"] == "existing_champion_registration"
    assert summary["tags"]["model_id"] == "fraud-risk-lightgbm-v1"


def test_logged_artifact_hash_matches_governance_registry(tmp_path):
    import json
    uri = _tmp_tracking_uri(tmp_path)
    log_champion_to_mlflow(tracking_uri=uri)
    summary = get_champion_run_summary(tracking_uri=uri)

    with open(REPO_ROOT / "artifacts" / "models" / "registry.json") as f:
        governance_registry = json.load(f)
    champion = governance_registry["models"][governance_registry["champion_model_id"]]

    assert summary["tags"]["artifact_sha256"] == champion["artifact_sha256"]


def test_get_champion_run_summary_returns_none_for_empty_experiment(tmp_path):
    uri = _tmp_tracking_uri(tmp_path)
    # Never logged anything to this fresh URI.
    summary = get_champion_run_summary(tracking_uri=uri)
    assert summary is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
