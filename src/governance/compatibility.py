"""
Phase 11: Version compatibility checks.

A candidate must not be silently promoted if it's incompatible with the
current approved inference architecture — these checks make that
explicit rather than assumed.
"""

from __future__ import annotations


def check_schema_compatibility(champion_metadata: dict, candidate_metadata: dict) -> dict:
    champion_schema = champion_metadata["feature_schema_version"]
    candidate_schema = candidate_metadata["feature_schema_version"]
    compatible = champion_schema == candidate_schema
    return {
        "check": "feature_schema_compatibility",
        "champion_feature_schema_version": champion_schema,
        "candidate_feature_schema_version": candidate_schema,
        "compatible": compatible,
        "explanation": (
            f"Feature schema versions match ({champion_schema})."
            if compatible else
            f"MISMATCH: champion uses schema '{champion_schema}', candidate uses "
            f"'{candidate_schema}' — promoting without explicit review could silently "
            f"break the approved FeaturePipeline/LightGBMPreprocessor contract."
        ),
    }


def check_decision_policy_compatibility(champion_metadata: dict, candidate_metadata: dict) -> dict:
    champion_policy = champion_metadata["decision_policy_version"]
    candidate_policy = candidate_metadata["decision_policy_version"]
    compatible = champion_policy == candidate_policy
    return {
        "check": "decision_policy_compatibility",
        "champion_decision_policy_version": champion_policy,
        "candidate_decision_policy_version": candidate_policy,
        "compatible": compatible,
        "explanation": (
            f"Both models are evaluated against the same frozen decision policy ('{champion_policy}')."
            if compatible else
            f"MISMATCH: champion metadata references policy '{champion_policy}', candidate "
            f"references '{candidate_policy}' — a candidate must be compatible with the "
            f"CURRENTLY frozen Phase 5 policy to be promoted without a separate policy review."
        ),
    }


def run_compatibility_checks(champion_metadata: dict, candidate_metadata: dict) -> list:
    return [
        check_schema_compatibility(champion_metadata, candidate_metadata),
        check_decision_policy_compatibility(champion_metadata, candidate_metadata),
    ]
