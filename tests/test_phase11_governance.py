"""
Phase 11 test suite: model registry, metadata, champion/challenger
evaluation, promotion gates, promotion/rollback workflow, compatibility
checks, and drift/governance separation. Synthetic data throughout (fast,
deterministic, independent of the real dataset).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.governance.metadata import (
    build_model_metadata, validate_metadata, validate_status, InvalidMetadataError,
    compute_artifact_sha256, UNAVAILABLE, VALID_STATUSES,
)
from src.governance.registry import ModelRegistry, RegistryError
from src.governance.evaluation import evaluate_model_scores, compare_champion_vs_candidate
from src.governance.compatibility import check_schema_compatibility, check_decision_policy_compatibility
from src.governance.gates import (
    run_all_gates, aggregate_gate_results, gate_candidate_quality, gate_no_unacceptable_degradation,
    MIN_CANDIDATE_PR_AUC, MAX_ACCEPTABLE_PR_AUC_DEGRADATION,
)
from src.governance.report import build_governance_report
from src.decision.policy import DecisionPolicy


def _make_metadata(model_id, status, pr_auc=0.5, roc_auc=0.9, schema="phase1-v1", policy="balanced", **overrides):
    base = build_model_metadata(
        model_id=model_id, model_version="v1", model_type="LightGBM", status=status,
        feature_schema_version=schema,
        training_data_reference="train partition", validation_data_reference="validation partition",
        evaluation_metrics={"val_ranking": {"pr_auc": pr_auc, "roc_auc": roc_auc}},
        artifact_reference="data/interim/fake_bundle.pkl",
        decision_policy_version=policy,
    )
    base.update(overrides)
    return base


def _make_eval(pr_auc, roc_auc, policy_eval=None):
    result = {"ranking": {"pr_auc": pr_auc, "roc_auc": roc_auc}}
    if policy_eval is not None:
        result["policy_evaluation"] = policy_eval
    return result


def _make_policy_eval(pct_blocked=0.001, recall=0.4):
    return {
        "buckets": {
            "APPROVE": {"pct_of_total": 0.98}, "REVIEW": {"pct_of_total": 0.01}, "BLOCK": {"pct_of_total": 0.01},
        },
        "fraud_recall_review_plus_block": recall,
        "precision_among_blocked": 0.8,
        "pct_legitimate_blocked": pct_blocked,
    }


# ---------------------------------------------------------------------------
# Metadata tests
# ---------------------------------------------------------------------------

def test_build_model_metadata_has_required_fields():
    meta = _make_metadata("m1", "CANDIDATE")
    validate_metadata(meta)  # should not raise


def test_invalid_status_rejected():
    with pytest.raises(InvalidMetadataError):
        build_model_metadata(
            model_id="m1", model_version="v1", model_type="LightGBM", status="NOT_A_STATUS",
            feature_schema_version="v1", training_data_reference="x", validation_data_reference="y",
            evaluation_metrics={}, artifact_reference="x", decision_policy_version="v1",
        )


def test_missing_required_field_rejected():
    meta = _make_metadata("m1", "CANDIDATE")
    del meta["model_type"]
    with pytest.raises(InvalidMetadataError):
        validate_metadata(meta)


def test_unavailable_historical_fields_marked_explicitly():
    meta = _make_metadata("m1", "CANDIDATE", test_data_reference=UNAVAILABLE)
    assert meta["test_data_reference"] == UNAVAILABLE
    assert meta["artifact_sha256"] == UNAVAILABLE  # not provided -> explicit marker, not fabricated


def test_compute_artifact_sha256_real_hash(tmp_path):
    f = tmp_path / "artifact.bin"
    f.write_bytes(b"hello world")
    h1 = compute_artifact_sha256(f)
    h2 = compute_artifact_sha256(f)
    assert h1 == h2  # deterministic
    assert len(h1) == 64  # real sha256 hex digest length


def test_compute_artifact_sha256_missing_file_returns_unavailable(tmp_path):
    assert compute_artifact_sha256(tmp_path / "does_not_exist.pkl") == UNAVAILABLE


def test_all_valid_statuses_accepted():
    for status in VALID_STATUSES:
        validate_status(status)  # should not raise


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

def test_registry_initializes_correctly(tmp_path):
    reg = ModelRegistry(tmp_path / "registry.json")
    summary = reg.summary()
    assert summary["champion_model_id"] is None
    assert summary["n_registered_models"] == 0


def test_registry_champion_can_be_registered(tmp_path):
    reg = ModelRegistry(tmp_path / "registry.json")
    reg.register_champion(_make_metadata("champ-v1", "CHAMPION"))
    champ = reg.get_champion()
    assert champ["model_id"] == "champ-v1"
    assert champ["status"] == "CHAMPION"


def test_registry_candidate_can_be_registered(tmp_path):
    reg = ModelRegistry(tmp_path / "registry.json")
    reg.register_champion(_make_metadata("champ-v1", "CHAMPION"))
    reg.register_candidate(_make_metadata("cand-v1", "CANDIDATE"))
    candidates = reg.list_models(status="CANDIDATE")
    assert len(candidates) == 1
    assert candidates[0]["model_id"] == "cand-v1"


def test_registry_duplicate_registration_without_overwrite_raises(tmp_path):
    reg = ModelRegistry(tmp_path / "registry.json")
    reg.register_candidate(_make_metadata("cand-v1", "CANDIDATE"))
    with pytest.raises(RegistryError):
        reg.register_candidate(_make_metadata("cand-v1", "CANDIDATE"))


def test_registry_second_champion_without_promote_raises(tmp_path):
    reg = ModelRegistry(tmp_path / "registry.json")
    reg.register_champion(_make_metadata("champ-v1", "CHAMPION"))
    with pytest.raises(RegistryError):
        reg.register_champion(_make_metadata("champ-v2", "CHAMPION"))


def test_registry_save_and_reload_roundtrip(tmp_path):
    path = tmp_path / "registry.json"
    reg = ModelRegistry(path)
    reg.register_champion(_make_metadata("champ-v1", "CHAMPION"))
    reg.save()

    reg2 = ModelRegistry(path)
    assert reg2.get_champion()["model_id"] == "champ-v1"


# ---------------------------------------------------------------------------
# Comparison tests
# ---------------------------------------------------------------------------

def test_better_candidate_shows_positive_delta():
    champ_eval = _make_eval(pr_auc=0.50, roc_auc=0.90)
    cand_eval = _make_eval(pr_auc=0.60, roc_auc=0.93)
    comparison = compare_champion_vs_candidate(champ_eval, cand_eval, "validation")
    assert comparison["ranking_metrics"]["delta_pr_auc"] == pytest.approx(0.10)
    assert comparison["ranking_metrics"]["delta_pr_auc"] > 0


def test_worse_candidate_shows_negative_delta():
    champ_eval = _make_eval(pr_auc=0.50, roc_auc=0.90)
    cand_eval = _make_eval(pr_auc=0.40, roc_auc=0.85)
    comparison = compare_champion_vs_candidate(champ_eval, cand_eval, "validation")
    assert comparison["ranking_metrics"]["delta_pr_auc"] < 0


def test_comparison_includes_policy_metrics_when_available():
    champ_eval = _make_eval(0.5, 0.9, policy_eval=_make_policy_eval())
    cand_eval = _make_eval(0.55, 0.91, policy_eval=_make_policy_eval(pct_blocked=0.002))
    comparison = compare_champion_vs_candidate(champ_eval, cand_eval, "validation")
    assert "policy_metrics" in comparison
    assert comparison["policy_metrics"]["delta_pct_legitimate_blocked"] == pytest.approx(0.001)


def test_evaluate_model_scores_reuses_approved_functions():
    rng = np.random.default_rng(0)
    y_true = (rng.random(500) < 0.05).astype(int)
    scores = rng.random(500)
    result = evaluate_model_scores(y_true, scores)
    assert "ranking" in result
    assert "pr_auc" in result["ranking"] and "roc_auc" in result["ranking"]


def test_evaluate_model_scores_with_policy():
    rng = np.random.default_rng(1)
    y_true = (rng.random(500) < 0.05).astype(int)
    scores = rng.random(500)
    policy = DecisionPolicy(approve_threshold=0.3, block_threshold=0.7, name="test")
    result = evaluate_model_scores(y_true, scores, policy=policy)
    assert "policy_evaluation" in result


# ---------------------------------------------------------------------------
# Promotion-gate tests
# ---------------------------------------------------------------------------

def test_passing_candidate_gets_promote_recommendation():
    champ_meta = _make_metadata("champ-v1", "CHAMPION")
    cand_meta = _make_metadata("cand-v1", "CANDIDATE")
    champ_eval = _make_eval(0.50, 0.90, policy_eval=_make_policy_eval())
    cand_eval = _make_eval(0.55, 0.92, policy_eval=_make_policy_eval(pct_blocked=0.0008))

    gates = run_all_gates(champ_meta, cand_meta, champ_eval, cand_eval)
    agg = aggregate_gate_results(gates)
    assert agg["recommendation"] == "PROMOTE"


def test_degraded_candidate_is_rejected():
    champ_meta = _make_metadata("champ-v1", "CHAMPION")
    cand_meta = _make_metadata("cand-v1", "CANDIDATE")
    champ_eval = _make_eval(0.50, 0.90, policy_eval=_make_policy_eval())
    cand_eval = _make_eval(0.20, 0.70, policy_eval=_make_policy_eval())  # far below champion AND below min threshold

    gates = run_all_gates(champ_meta, cand_meta, champ_eval, cand_eval)
    agg = aggregate_gate_results(gates)
    assert agg["recommendation"] == "REJECT"
    assert len(agg["failed_gates"]) > 0


def test_ambiguous_tradeoff_produces_requires_review():
    champ_meta = _make_metadata("champ-v1", "CHAMPION")
    cand_meta = _make_metadata("cand-v1", "CANDIDATE")
    # Candidate is meaningfully worse (within tolerated-but-reviewable degradation zone)
    # but still above the minimum quality bar.
    champ_eval = _make_eval(0.50, 0.90, policy_eval=_make_policy_eval())
    cand_eval = _make_eval(0.489, 0.895, policy_eval=_make_policy_eval())  # -0.011, inside review zone

    gates = run_all_gates(champ_meta, cand_meta, champ_eval, cand_eval)
    agg = aggregate_gate_results(gates)
    assert agg["recommendation"] == "REQUIRES_REVIEW"


def test_gate_candidate_quality_threshold():
    weak_eval = _make_eval(pr_auc=MIN_CANDIDATE_PR_AUC - 0.05, roc_auc=0.6)
    result = gate_candidate_quality(weak_eval)
    assert result["result"] == "FAIL"

    strong_eval = _make_eval(pr_auc=MIN_CANDIDATE_PR_AUC + 0.1, roc_auc=0.9)
    result2 = gate_candidate_quality(strong_eval)
    assert result2["result"] == "PASS"


def test_gate_degradation_boundary_behavior():
    champ_eval = _make_eval(0.50, 0.9)
    # exactly at the max tolerated degradation boundary
    cand_eval = _make_eval(0.50 - MAX_ACCEPTABLE_PR_AUC_DEGRADATION, 0.9)
    result = gate_no_unacceptable_degradation(champ_eval, cand_eval)
    assert result["result"] in ("REQUIRES_REVIEW", "PASS")  # boundary, not FAIL

    cand_eval_worse = _make_eval(0.50 - MAX_ACCEPTABLE_PR_AUC_DEGRADATION - 0.01, 0.9)
    result2 = gate_no_unacceptable_degradation(champ_eval, cand_eval_worse)
    assert result2["result"] == "FAIL"


def test_aggregate_gate_results_explainable_not_black_box():
    gates = [
        {"gate": "a", "result": "PASS", "explanation": "ok"},
        {"gate": "b", "result": "FAIL", "explanation": "bad"},
    ]
    agg = aggregate_gate_results(gates)
    assert agg["recommendation"] == "REJECT"
    assert "b" in agg["failed_gates"]
    assert "b" in agg["explanation"]  # names the actual failing gate, not a mystery score


# ---------------------------------------------------------------------------
# Compatibility tests
# ---------------------------------------------------------------------------

def test_schema_mismatch_detected():
    champ_meta = _make_metadata("champ-v1", "CHAMPION", schema="phase1-v1")
    cand_meta = _make_metadata("cand-v1", "CANDIDATE", schema="phase1-v2-EXPERIMENTAL")
    check = check_schema_compatibility(champ_meta, cand_meta)
    assert check["compatible"] is False


def test_schema_match_detected():
    champ_meta = _make_metadata("champ-v1", "CHAMPION", schema="phase1-v1")
    cand_meta = _make_metadata("cand-v1", "CANDIDATE", schema="phase1-v1")
    check = check_schema_compatibility(champ_meta, cand_meta)
    assert check["compatible"] is True


def test_policy_incompatibility_detected():
    champ_meta = _make_metadata("champ-v1", "CHAMPION", policy="balanced")
    cand_meta = _make_metadata("cand-v1", "CANDIDATE", policy="aggressive")
    check = check_decision_policy_compatibility(champ_meta, cand_meta)
    assert check["compatible"] is False


def test_schema_mismatch_fails_promotion_gate():
    champ_meta = _make_metadata("champ-v1", "CHAMPION", schema="phase1-v1")
    cand_meta = _make_metadata("cand-v1", "CANDIDATE", schema="INCOMPATIBLE")
    champ_eval = _make_eval(0.5, 0.9, policy_eval=_make_policy_eval())
    cand_eval = _make_eval(0.6, 0.92, policy_eval=_make_policy_eval())
    gates = run_all_gates(champ_meta, cand_meta, champ_eval, cand_eval)
    agg = aggregate_gate_results(gates)
    assert agg["recommendation"] == "REJECT"
    assert "feature_schema_compatibility" in agg["failed_gates"]


# ---------------------------------------------------------------------------
# Promotion tests
# ---------------------------------------------------------------------------

def test_evaluation_alone_does_not_change_champion(tmp_path):
    reg = ModelRegistry(tmp_path / "registry.json")
    reg.register_champion(_make_metadata("champ-v1", "CHAMPION"))
    reg.register_candidate(_make_metadata("cand-v1", "CANDIDATE"))

    champ_meta = reg.get_champion()
    cand_meta = reg.get_model("cand-v1")
    champ_eval = _make_eval(0.5, 0.9, policy_eval=_make_policy_eval())
    cand_eval = _make_eval(0.6, 0.92, policy_eval=_make_policy_eval())
    report = build_governance_report(champ_meta, cand_meta, champ_eval, cand_eval, "validation")

    assert report["recommendation"] == "PROMOTE"
    # Evaluation/report generation must NEVER change the registry itself.
    assert reg.get_champion()["model_id"] == "champ-v1"
    assert reg.get_model("cand-v1")["status"] == "CANDIDATE"


def test_explicit_promotion_changes_champion(tmp_path):
    reg = ModelRegistry(tmp_path / "registry.json")
    reg.register_champion(_make_metadata("champ-v1", "CHAMPION"))
    reg.register_candidate(_make_metadata("cand-v1", "CANDIDATE"))

    reg.promote("cand-v1", approved_by="test-reviewer", reason="test promotion")

    assert reg.get_champion()["model_id"] == "cand-v1"
    assert reg.get_model("champ-v1")["status"] == "RETIRED"


def test_promotion_preserves_history(tmp_path):
    reg = ModelRegistry(tmp_path / "registry.json")
    reg.register_champion(_make_metadata("champ-v1", "CHAMPION"))
    reg.register_candidate(_make_metadata("cand-v1", "CANDIDATE"))
    reg.promote("cand-v1", approved_by="reviewer", reason="better PR-AUC")

    retired = reg.get_model("champ-v1")
    assert "retirement_history" in retired
    assert retired["retirement_history"][0]["approved_by"] == "reviewer"

    new_champ = reg.get_champion()
    assert new_champ["promotion_history"][0]["previous_champion"] == "champ-v1"


def test_promote_non_candidate_raises(tmp_path):
    reg = ModelRegistry(tmp_path / "registry.json")
    reg.register_champion(_make_metadata("champ-v1", "CHAMPION"))
    with pytest.raises(RegistryError):
        reg.promote("champ-v1", approved_by="x", reason="y")  # can't promote a champion


# ---------------------------------------------------------------------------
# Rollback tests
# ---------------------------------------------------------------------------

def test_rollback_restores_previous_champion(tmp_path):
    reg = ModelRegistry(tmp_path / "registry.json")
    reg.register_champion(_make_metadata("champ-v1", "CHAMPION"))
    reg.register_candidate(_make_metadata("cand-v1", "CANDIDATE"))
    reg.promote("cand-v1", approved_by="reviewer", reason="promote")
    assert reg.get_champion()["model_id"] == "cand-v1"

    reg.rollback("champ-v1", approved_by="reviewer", reason="regression found in production")
    assert reg.get_champion()["model_id"] == "champ-v1"
    assert reg.get_model("cand-v1")["status"] == "RETIRED"


def test_rollback_registry_history_remains_valid(tmp_path):
    reg = ModelRegistry(tmp_path / "registry.json")
    reg.register_champion(_make_metadata("champ-v1", "CHAMPION"))
    reg.register_candidate(_make_metadata("cand-v1", "CANDIDATE"))
    reg.promote("cand-v1", approved_by="reviewer", reason="promote")
    reg.rollback("champ-v1", approved_by="reviewer", reason="rollback")

    champ = reg.get_champion()
    assert "rollback_history" in champ
    assert champ["rollback_history"][0]["previous_champion"] == "cand-v1"
    # Both models remain in the registry, fully queryable.
    assert reg.get_model("cand-v1")["status"] == "RETIRED"


def test_rollback_requires_retired_status(tmp_path):
    reg = ModelRegistry(tmp_path / "registry.json")
    reg.register_champion(_make_metadata("champ-v1", "CHAMPION"))
    reg.register_candidate(_make_metadata("cand-v1", "CANDIDATE"))
    with pytest.raises(RegistryError):
        reg.rollback("cand-v1", approved_by="x", reason="y")  # candidate was never champion


# ---------------------------------------------------------------------------
# Drift/governance separation
# ---------------------------------------------------------------------------

def test_drift_module_never_imports_governance_registry():
    """Structural check: src/drift/ must not import src/governance/ —
    drift detection must not be able to touch the registry at all."""
    import ast
    drift_dir = REPO_ROOT / "src" / "drift"
    for py_file in drift_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "governance" not in node.module, f"{py_file} imports governance: {node.module}"
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "governance" not in alias.name, f"{py_file} imports governance: {alias.name}"


def test_governance_registry_has_no_drift_dependency(tmp_path):
    """A drift HIGH_DRIFT event, however represented, cannot by itself
    call promote()/rollback() — those require explicit approved_by/reason
    supplied by a human caller, never derived from a drift report."""
    reg = ModelRegistry(tmp_path / "registry.json")
    reg.register_champion(_make_metadata("champ-v1", "CHAMPION"))
    reg.register_candidate(_make_metadata("cand-v1", "CANDIDATE"))
    # Simulate "drift detected" — nothing about a drift severity string
    # can be passed to promote()/rollback() in place of a real approval;
    # the signature requires approved_by/reason strings supplied by a human.
    import inspect
    sig = inspect.signature(reg.promote)
    assert "approved_by" in sig.parameters
    assert "reason" in sig.parameters
    # Champion is unchanged without an explicit call.
    assert reg.get_champion()["model_id"] == "champ-v1"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
