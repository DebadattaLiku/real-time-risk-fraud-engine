"""
Production upgrade — fraud processing consumer.

This is the "Fraud Processing Consumer" from the target architecture:

    Kafka topic (fraud.transactions)
        -> FraudProcessingHandler
        -> the EXISTING, REAL RiskDecisionEngine.process_transaction()
        -> Kafka topic (fraud.decisions)

It does NOT reimplement inference, feature generation, or decisioning —
it is a thin adapter calling the exact same
`RiskDecisionEngine.process_transaction()` method the FastAPI `/predict`
route calls (`src/api/main.py`), so a transaction produces an IDENTICAL
result whether it arrives via HTTP or via this consumer. This is the same
"reuse, never duplicate" discipline already established for the
dashboard (Phase 12) and drift/governance layers (Phases 10-11).
"""

from __future__ import annotations

import logging

from src.engine.risk_engine import RiskDecisionEngine, TransactionValidationError

logger = logging.getLogger("fraud_streaming")


class FraudProcessingHandler:
    """
    Callable handler for `TransactionConsumer`/`FakeTransactionConsumer`.
    Wraps the real engine; optionally publishes each decision to a
    decisions producer (real or fake) — publishing failures are logged,
    never allowed to mask a real processing failure that should still
    count as "processed" (the risk decision was made correctly; failing
    to ALSO publish it is a separate, secondary concern).

    ## Idempotency (a real gap found during Phase 12 failure testing, fixed here)

    Kafka's standard at-least-once delivery semantics mean the SAME
    message can genuinely be delivered more than once (e.g. a consumer
    crashes after processing but before committing its offset). Without
    protection, replaying `RiskDecisionEngine.process_transaction()` for
    an already-processed transaction ID double-counts it in behavioral
    state (`tests/test_production_failure_scenarios.py::test_duplicate_message_processing_is_not_idempotent_by_default`
    demonstrates this directly against the raw engine). This handler adds
    a bounded, in-memory "already-processed transaction ID" cache
    (`maxlen=idempotency_cache_size`) checked BEFORE calling the engine —
    a duplicate is detected and skipped (the cached original result is
    returned again, the engine is never called a second time, so state is
    never double-updated).

    This is a real, working fix for the specific "duplicate message
    within the cache window" case — it is NOT a claim of perfect exactly-
    once semantics (the cache is bounded and in-memory, so a duplicate
    arriving after the cache has evicted that transaction ID, or after a
    process restart, would NOT be caught — see
    `reports/failure_testing.md`'s honest accounting of this limit).
    """

    def __init__(self, engine: RiskDecisionEngine, decision_producer=None, idempotency_cache_size: int = 10000):
        self.engine = engine
        self.decision_producer = decision_producer
        self.transactions_processed = 0
        self.transactions_rejected = 0
        self.duplicate_transactions_skipped = 0
        self._idempotency_cache_size = idempotency_cache_size
        self._seen_transaction_ids: dict = {}  # transaction_id -> cached result
        self._seen_order: list = []  # insertion order, for bounded eviction

    def _remember(self, transaction_id, result: dict) -> None:
        self._seen_transaction_ids[transaction_id] = result
        self._seen_order.append(transaction_id)
        if len(self._seen_order) > self._idempotency_cache_size:
            oldest = self._seen_order.pop(0)
            self._seen_transaction_ids.pop(oldest, None)

    def __call__(self, transaction: dict) -> dict:
        transaction_id = transaction.get("TransactionID")
        if transaction_id is not None and transaction_id in self._seen_transaction_ids:
            self.duplicate_transactions_skipped += 1
            logger.info(f"event=duplicate_transaction_skipped transaction_id={transaction_id}")
            return self._seen_transaction_ids[transaction_id]

        try:
            result = self.engine.process_transaction(transaction)
        except TransactionValidationError:
            self.transactions_rejected += 1
            raise  # let the consumer's retry/DLQ logic handle it — never swallowed here
        self.transactions_processed += 1
        if transaction_id is not None:
            self._remember(transaction_id, result)

        if self.decision_producer is not None:
            try:
                self.decision_producer.send({
                    "transaction_id": result["transaction_id"],
                    "risk_score": result["risk_score"],
                    "decision": result["decision"],
                    "policy_version": result["processing_metadata"].get("policy_name"),
                })
            except Exception as e:  # noqa: BLE001 — publishing the decision downstream must never fail the transaction's own processing
                logger.warning(f"event=decision_publish_failed error={e!r}")

        return result
