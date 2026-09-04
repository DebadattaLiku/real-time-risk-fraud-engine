"""
Phase 11: Lightweight local model registry.

This is a JSON-file-backed registry — NOT a claim of a full enterprise
model registry (no database, no access control, no distributed storage,
no MLflow/cloud integration). It tracks model metadata and lifecycle
status (`CANDIDATE`/`CHAMPION`/`REJECTED`/`RETIRED`) and enforces the
core governance rule: nothing in this module ever changes `champion_model_id`
except `promote()` and `rollback()`, both of which must be called
EXPLICITLY — registering or evaluating a candidate never does this as a
side effect.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.governance.metadata import validate_metadata, validate_status, InvalidMetadataError

REGISTRY_VERSION = "v1"


class RegistryError(ValueError):
    pass


class ModelRegistry:
    def __init__(self, registry_path: Path):
        self.registry_path = Path(registry_path)
        if self.registry_path.is_file():
            with open(self.registry_path) as f:
                self._data = json.load(f)
        else:
            self._data = {
                "registry_version": REGISTRY_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_updated_at": None,
                "champion_model_id": None,
                "models": {},
            }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        self._data["last_updated_at"] = datetime.now(timezone.utc).isoformat()
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w") as f:
            json.dump(self._data, f, indent=2)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_model(self, metadata: dict, overwrite: bool = False) -> None:
        """Low-level registration — validates and stores metadata as-is
        (whatever `status` it already has). Prefer `register_champion` /
        `register_candidate` below for the common cases."""
        validate_metadata(metadata)
        model_id = metadata["model_id"]
        if model_id in self._data["models"] and not overwrite:
            raise RegistryError(
                f"Model '{model_id}' is already registered. Pass overwrite=True to "
                f"intentionally replace its metadata (this does NOT change champion status)."
            )
        self._data["models"][model_id] = metadata
        if metadata["status"] == "CHAMPION":
            self._data["champion_model_id"] = model_id

    def register_champion(self, metadata: dict) -> None:
        """Registers a model as the (typically initial) champion. Only
        appropriate when there is currently NO champion, or when
        explicitly re-registering the same already-approved champion
        (e.g. after a restart) — this does NOT implement promotion logic;
        use `promote()` to replace an existing champion with a candidate."""
        if metadata["status"] != "CHAMPION":
            raise RegistryError("register_champion requires metadata with status='CHAMPION'.")
        if self._data["champion_model_id"] is not None and self._data["champion_model_id"] != metadata["model_id"]:
            raise RegistryError(
                f"A champion ('{self._data['champion_model_id']}') is already registered. "
                f"Use promote() to replace it through the controlled workflow."
            )
        self.register_model(metadata, overwrite=True)

    def register_candidate(self, metadata: dict) -> None:
        if metadata["status"] != "CANDIDATE":
            raise RegistryError("register_candidate requires metadata with status='CANDIDATE'.")
        self.register_model(metadata)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_model(self, model_id: str) -> dict:
        if model_id not in self._data["models"]:
            raise RegistryError(f"No model registered with id '{model_id}'.")
        return self._data["models"][model_id]

    def get_champion(self) -> dict | None:
        champion_id = self._data["champion_model_id"]
        return self._data["models"][champion_id] if champion_id else None

    def list_models(self, status: str | None = None) -> list:
        models = list(self._data["models"].values())
        if status is not None:
            validate_status(status)
            models = [m for m in models if m["status"] == status]
        return models

    def summary(self) -> dict:
        champion = self.get_champion()
        candidates = self.list_models(status="CANDIDATE")
        return {
            "champion_model_id": champion["model_id"] if champion else None,
            "champion_version": champion["model_version"] if champion else None,
            "champion_status": champion["status"] if champion else None,
            "n_registered_models": len(self._data["models"]),
            "n_candidates": len(candidates),
            "n_rejected": len(self.list_models(status="REJECTED")),
            "n_retired": len(self.list_models(status="RETIRED")),
            "registry_version": self._data["registry_version"],
            "last_updated_at": self._data["last_updated_at"],
        }

    # ------------------------------------------------------------------
    # Explicit lifecycle actions — the ONLY places champion_model_id changes
    # ------------------------------------------------------------------

    def promote(self, candidate_model_id: str, approved_by: str, reason: str) -> dict:
        """
        Explicitly promotes a registered CANDIDATE to CHAMPION, retiring
        the previous champion (status -> RETIRED, history preserved, never
        deleted). Requires an explicit caller-supplied `approved_by` and
        `reason` — this method is never called automatically by evaluation
        or drift-monitoring code anywhere in this project (see
        `src/governance/gates.py` and `src/drift/` — neither imports this
        module).
        """
        candidate = self.get_model(candidate_model_id)
        if candidate["status"] != "CANDIDATE":
            raise RegistryError(
                f"Only a model with status='CANDIDATE' can be promoted (got '{candidate['status']}')."
            )

        previous_champion = self.get_champion()
        now = datetime.now(timezone.utc).isoformat()

        if previous_champion is not None:
            previous_champion["status"] = "RETIRED"
            previous_champion["retired_at"] = now
            previous_champion.setdefault("retirement_history", []).append({
                "retired_at": now, "reason": f"Superseded by promotion of '{candidate_model_id}'",
                "approved_by": approved_by,
            })

        candidate["status"] = "CHAMPION"
        candidate["promoted_at"] = now
        candidate["promoted_from"] = previous_champion["model_id"] if previous_champion else None
        candidate.setdefault("promotion_history", []).append({
            "promoted_at": now, "approved_by": approved_by, "reason": reason,
            "previous_champion": previous_champion["model_id"] if previous_champion else None,
        })

        self._data["champion_model_id"] = candidate_model_id
        return candidate

    def rollback(self, target_model_id: str, approved_by: str, reason: str) -> dict:
        """
        Explicitly restores a previously-RETIRED model to CHAMPION status,
        retiring the current champion in turn. `target_model_id` must
        currently have status RETIRED — this only works because
        `promote()` never deletes retired models, only marks them.
        Requires explicit `approved_by`/`reason`, same as `promote()`.
        """
        target = self.get_model(target_model_id)
        if target["status"] != "RETIRED":
            raise RegistryError(
                f"Only a RETIRED model can be rolled back to (got status='{target['status']}' for '{target_model_id}')."
            )

        current_champion = self.get_champion()
        now = datetime.now(timezone.utc).isoformat()

        if current_champion is not None:
            current_champion["status"] = "RETIRED"
            current_champion["retired_at"] = now
            current_champion.setdefault("retirement_history", []).append({
                "retired_at": now, "reason": f"Rolled back in favor of '{target_model_id}'",
                "approved_by": approved_by,
            })

        target["status"] = "CHAMPION"
        target["promoted_at"] = now
        target.setdefault("rollback_history", []).append({
            "rolled_back_at": now, "approved_by": approved_by, "reason": reason,
            "previous_champion": current_champion["model_id"] if current_champion else None,
        })

        self._data["champion_model_id"] = target_model_id
        return target

    def reject(self, candidate_model_id: str, reason: str) -> dict:
        candidate = self.get_model(candidate_model_id)
        if candidate["status"] != "CANDIDATE":
            raise RegistryError("Only a CANDIDATE can be rejected.")
        candidate["status"] = "REJECTED"
        candidate["notes"] = (candidate.get("notes") or "") + f"\nRejected: {reason}"
        return candidate
