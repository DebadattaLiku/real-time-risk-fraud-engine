"""
Phase 6: Stateful real-time risk decision engine.

Wraps the already-approved pipeline (Phase 1 `FeaturePipeline`, Phase 4
`LightGBMPreprocessor`, the selected transaction+behavioral LightGBM model,
Phase 4's `BehavioralStateManager`, and the frozen Phase 5 `DecisionPolicy`)
behind a single transaction-at-a-time interface, in the exact processing
order the Phase 6 brief specifies:

    1. Receive transaction
    2. Validate required fields
    3. Read current historical state         (implicit in step 4 — read-only)
    4. Generate behavioral features from EXISTING state only
    5. Combine transaction + behavioral features
    6. Apply preprocessing (Phase 1 + Phase 4, unchanged)
    7. Generate fraud risk prediction
    8. Apply frozen decision policy
    9. Return prediction and decision
    10. Update behavioral state with the CURRENT transaction

Step 10 happens LAST, after the prediction and decision are already
computed and packaged into the result — enforced by this method's plain
sequential control flow (the `state_manager.update(...)` call is the final
statement in `process_transaction`), so a transaction's own data can never
reach its own feature computation, and any validation failure or exception
earlier in the sequence prevents state from being touched at all (no
partial updates).
"""

from __future__ import annotations

import time as _time

import numpy as np
import pandas as pd

from src.features.schema import FeatureSpec
from src.features.pipeline import FeaturePipeline
from src.models.lightgbm_preprocessing import LightGBMPreprocessor
from src.decision.policy import DecisionPolicy
from src.engine.state import BehavioralStateManager


class TransactionValidationError(ValueError):
    pass


class RiskDecisionEngine:
    def __init__(
        self,
        schema: list[FeatureSpec],
        feature_pipeline: FeaturePipeline,
        lgbm_preprocessor: LightGBMPreprocessor,
        model,
        policy: DecisionPolicy,
        state_manager: BehavioralStateManager | None = None,
        entity_col: str = "card1",
        time_col: str = "TransactionDT",
        id_col: str = "TransactionID",
        amount_col: str = "TransactionAmt",
    ):
        self.schema = schema
        self.feature_pipeline = feature_pipeline
        self.lgbm_preprocessor = lgbm_preprocessor
        self.model = model
        self.policy = policy
        self.state_manager = state_manager if state_manager is not None else BehavioralStateManager(entity_col)
        self.entity_col = entity_col
        self.time_col = time_col
        self.id_col = id_col
        self.amount_col = amount_col

        # Every raw column FeaturePipeline expects to find, PLUS the
        # identifier/time columns it explicitly excludes but this engine
        # still needs to read (id_col, time_col are not "features" but are
        # required transaction fields).
        self._expected_raw_columns = sorted(
            {s.name for s in schema} - {"isFraud"}
        )

    def reset_state(self) -> None:
        self.state_manager.reset()

    def get_expected_raw_columns(self) -> list:
        """
        Public accessor for the full set of raw transaction-level columns
        this engine expects as dict keys (see `_validate_transaction`).
        Exposed so callers outside this module (e.g. the Phase 7 API
        layer) can build a complete transaction dict without reaching into
        a private attribute.
        """
        return list(self._expected_raw_columns)

    def _validate_transaction(self, transaction: dict) -> None:
        if "isFraud" in transaction:
            raise TransactionValidationError(
                "Real-time prediction input must never include 'isFraud' — "
                "labels are not available at prediction time and must not "
                "be used as a feature."
            )

        missing_keys = set(self._expected_raw_columns) - set(transaction.keys())
        if missing_keys:
            raise TransactionValidationError(
                f"Transaction is missing required field(s): {sorted(missing_keys)}"
            )

        for required in (self.id_col, self.time_col, self.amount_col, self.entity_col):
            value = transaction.get(required)
            if value is None:
                raise TransactionValidationError(f"Required field '{required}' is missing (None).")

        for numeric_field in (self.time_col, self.amount_col):
            value = transaction[numeric_field]
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise TransactionValidationError(
                    f"Field '{numeric_field}' must be numeric, got {transaction[numeric_field]!r}"
                )
            if not np.isfinite(value):
                raise TransactionValidationError(
                    f"Field '{numeric_field}' must be a finite number, got {transaction[numeric_field]!r}"
                )
            if numeric_field == self.amount_col and value < 0:
                raise TransactionValidationError(f"'{self.amount_col}' cannot be negative, got {value}")

    def process_transaction(self, transaction: dict) -> dict:
        t0 = _time.perf_counter()

        # STEP 2: validate.
        self._validate_transaction(transaction)

        entity_id = transaction[self.entity_col]
        current_time = float(transaction[self.time_col])
        current_amount = float(transaction[self.amount_col])
        transaction_id = transaction[self.id_col]

        # STEP 3 + 4: read existing state, compute behavioral features.
        # (compute_features() is read-only — see src/engine/state.py.)
        behavioral_features = self.state_manager.compute_features(entity_id, current_time, current_amount)

        # STEP 5: combine transaction features + behavioral features.
        raw_row = {k: transaction.get(k) for k in self._expected_raw_columns}
        raw_df = pd.DataFrame([raw_row])

        # A single-row DataFrame built from a dict cannot infer a numeric
        # dtype from a lone `None` (unlike a multi-row batch, where other
        # real values in the column pin its dtype to float64) — pandas
        # leaves such a column as `object`, which LightGBM's native
        # predict() rejects outright ("pandas dtypes must be int, float or
        # bool"). This is specifically a JSON-`null` -> Python-`None`
        # problem: a lone `float('nan')` (what Phase 6's batch-derived rows
        # always had) is still correctly inferred as float64, which is why
        # this was never caught until real end-to-end API testing sent a
        # genuine high-missingness transaction (several `V`/`D`/`dist`
        # columns are NaN for many real transactions). Fixed here, in the
        # engine itself, so every caller (API, direct use, future batch
        # tooling) gets the same correct behavior — not just the API layer.
        numeric_cols = [
            s.name for s in self.schema
            if s.used and s.value_type == "numeric" and s.name in raw_df.columns
        ]
        if numeric_cols:
            raw_df[numeric_cols] = raw_df[numeric_cols].astype("float64")

        # STEP 6: apply preprocessing (Phase 1 -> Phase 4, unchanged).
        X = self.feature_pipeline.transform(raw_df)
        Z = self.lgbm_preprocessor.transform(X)
        bhv_df = pd.DataFrame([behavioral_features])
        Z_full = pd.concat([Z.reset_index(drop=True), bhv_df.reset_index(drop=True)], axis=1)

        # STEP 7: fraud risk prediction.
        risk_score = float(self.model.predict_proba(Z_full)[:, 1][0])

        # STEP 8: apply frozen decision policy.
        decision = str(self.policy.decide([risk_score])[0])

        # STEP 9: package result (state not yet updated).
        elapsed_ms = (_time.perf_counter() - t0) * 1000
        result = {
            "transaction_id": transaction_id,
            "risk_score": risk_score,
            "decision": decision,
            "behavioral_features": behavioral_features,
            "state_updated": False,
            "processing_metadata": {
                "policy_name": self.policy.name,
                "approve_threshold": self.policy.approve_threshold,
                "block_threshold": self.policy.block_threshold,
                "processing_time_ms": elapsed_ms,
            },
        }

        # STEP 10: update state with the CURRENT transaction — LAST,
        # after prediction/decision are already finalized above.
        self.state_manager.update(entity_id, current_time, current_amount)
        result["state_updated"] = True

        return result
