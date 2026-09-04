"""
Phase 12: Loaders for existing, already-validated project artifacts.

Every function here reads a JSON file this project already produced in an
earlier, approved phase — nothing is recomputed, re-evaluated, or
re-derived differently from the established protocol. If an artifact is
missing, the loader returns `None` and the calling page renders a clear
"not available" state instead of fabricating a number.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

PHASE4_METRICS_PATH = REPO_ROOT / "data" / "interim" / "phase4_metrics.json"
PHASE5_METRICS_PATH = REPO_ROOT / "data" / "interim" / "phase5_metrics.json"
PHASE6_METRICS_PATH = REPO_ROOT / "data" / "interim" / "phase6_metrics.json"
REFERENCE_PROFILE_PATH = REPO_ROOT / "artifacts" / "drift" / "reference_profile.json"
MODEL_REGISTRY_PATH = REPO_ROOT / "artifacts" / "models" / "registry.json"
DECISION_POLICY_PATH = REPO_ROOT / "config" / "decision_policy.yaml"
TEST_SUMMARY_CACHE_PATH = REPO_ROOT / "artifacts" / "dashboard_test_summary.json"


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


def load_phase4_metrics() -> dict | None:
    return _load_json(PHASE4_METRICS_PATH)


def load_phase5_metrics() -> dict | None:
    return _load_json(PHASE5_METRICS_PATH)


def load_phase6_metrics() -> dict | None:
    return _load_json(PHASE6_METRICS_PATH)


def load_reference_profile() -> dict | None:
    return _load_json(REFERENCE_PROFILE_PATH)


def load_model_registry() -> dict | None:
    return _load_json(MODEL_REGISTRY_PATH)


def load_decision_policy() -> dict | None:
    if not DECISION_POLICY_PATH.is_file():
        return None
    import yaml
    with open(DECISION_POLICY_PATH) as f:
        return yaml.safe_load(f)


def get_champion_test_metrics() -> dict | None:
    """
    Real Phase 4 champion (`model_b_behavioral`) test-set ranking metrics
    and review-budget recall/precision — the SAME numbers already reported
    in `reports/phase4_behavioral_features_summary.md`, read from the same
    saved artifact, not recomputed.
    """
    phase4 = load_phase4_metrics()
    if phase4 is None:
        return None
    model_b = phase4.get("model_b_behavioral")
    if model_b is None:
        return None
    return {
        "test_ranking": model_b["test_ranking"],
        "test_budget_table": model_b["test_budget_table"],
        "val_ranking": model_b["val_ranking"],
    }


def get_champion_policy_test_eval() -> dict | None:
    """Real Phase 5 frozen-policy test-set evaluation (decision
    distribution, fraud capture, friction) — read from the saved artifact."""
    phase5 = load_phase5_metrics()
    if phase5 is None:
        return None
    return phase5.get("test_eval")


def read_cached_test_summary() -> dict | None:
    """
    Returns the LAST RECORDED pytest run summary (count, pass/fail), if a
    cache file exists — never runs pytest live from inside a dashboard
    page load (far too slow for an interactive UI). The cache is written
    by `scripts/refresh_dashboard_test_summary.py`, a separate, explicit
    step, not by the dashboard itself.
    """
    return _load_json(TEST_SUMMARY_CACHE_PATH)
