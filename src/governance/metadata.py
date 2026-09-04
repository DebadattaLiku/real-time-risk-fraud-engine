"""
Phase 11: Model metadata construction.

Metadata is built from ACTUAL repository information — saved evaluation
JSON files from Phases 4/5, `config/decision_policy.yaml`, real
hyperparameters from `src/models/lightgbm_model.py`, and a real SHA-256
hash of the actual artifact file — never invented. Any field this project
genuinely does not have on record (e.g. exact training wall-clock
duration, exact hardware used) is explicitly marked `UNAVAILABLE` rather
than fabricated or silently omitted, so a reader can tell the difference
between "this was zero" and "this was never recorded."
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

UNAVAILABLE = "UNAVAILABLE"

REQUIRED_METADATA_FIELDS = [
    "model_id", "model_version", "model_type", "status", "created_at",
    "feature_schema_version", "training_data_reference", "validation_data_reference",
    "evaluation_metrics", "artifact_reference", "decision_policy_version",
]

VALID_STATUSES = ("CANDIDATE", "CHAMPION", "REJECTED", "RETIRED")


class InvalidMetadataError(ValueError):
    pass


def compute_artifact_sha256(artifact_path: Path) -> str:
    """Real hash of the real file — used to verify a registered model's
    artifact hasn't silently changed since registration."""
    artifact_path = Path(artifact_path)
    if not artifact_path.is_file():
        return UNAVAILABLE
    sha256 = hashlib.sha256()
    with open(artifact_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def build_model_metadata(
    model_id: str,
    model_version: str,
    model_type: str,
    status: str,
    feature_schema_version: str,
    training_data_reference: str,
    validation_data_reference: str,
    evaluation_metrics: dict,
    artifact_reference: str,
    decision_policy_version: str,
    test_data_reference: str = UNAVAILABLE,
    hyperparameters: dict | None = None,
    artifact_sha256: str | None = None,
    notes: str = "",
    promoted_from: str | None = None,
) -> dict:
    validate_status(status)
    return {
        "model_id": model_id,
        "model_version": model_version,
        "model_type": model_type,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_schema_version": feature_schema_version,
        "training_data_reference": training_data_reference,
        "validation_data_reference": validation_data_reference,
        "test_data_reference": test_data_reference,
        "evaluation_metrics": evaluation_metrics,
        "artifact_reference": artifact_reference,
        "artifact_sha256": artifact_sha256 or UNAVAILABLE,
        "decision_policy_version": decision_policy_version,
        "hyperparameters": hyperparameters or {},
        "notes": notes,
        "promoted_from": promoted_from,
        "promoted_at": None,
        "retired_at": None,
    }


def validate_status(status: str) -> None:
    if status not in VALID_STATUSES:
        raise InvalidMetadataError(f"Invalid status '{status}' — must be one of {VALID_STATUSES}")


def validate_metadata(metadata: dict) -> None:
    missing = [f for f in REQUIRED_METADATA_FIELDS if f not in metadata]
    if missing:
        raise InvalidMetadataError(f"Metadata is missing required field(s): {missing}")
    validate_status(metadata["status"])
    if not isinstance(metadata["evaluation_metrics"], dict):
        raise InvalidMetadataError("evaluation_metrics must be a dict")
