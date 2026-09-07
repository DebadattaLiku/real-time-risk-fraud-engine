"""
Production upgrade — streaming layer tests.

`InMemoryBroker`-based tests require no external service and always run
(per the task's explicit instruction that Kafka must not be required for
unit tests). Real-`kafka-python`-client tests are marked and skipped if
no broker is reachable (mirroring `tests/test_production_redis_state.py`'s
pattern for Redis) — in THIS environment they are expected to be skipped,
since no real Kafka broker exists here (see `src/streaming/kafka_client.py`'s
module docstring).
"""

import socket
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.split import compute_temporal_split
from src.features.schema import build_feature_schema
from src.features.pipeline import FeaturePipeline
from src.features.behavioral import compute_behavioral_features, get_behavioral_feature_names
from src.models.lightgbm_preprocessing import LightGBMPreprocessor
from src.models.lightgbm_model import train_lightgbm
from src.decision.policy import DecisionPolicy
from src.engine.state import BehavioralStateManager
from src.engine.risk_engine import RiskDecisionEngine

from src.streaming.kafka_client import TransactionProducer, TransactionConsumer, StreamingUnavailableError
from src.streaming.fake_broker import InMemoryBroker, FakeTransactionProducer, FakeTransactionConsumer
from src.streaming.fraud_processing_consumer import FraudProcessingHandler


def _kafka_broker_reachable(host="localhost", port=9092, timeout=0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


requires_real_kafka = pytest.mark.skipif(
    not _kafka_broker_reachable(), reason="No reachable Kafka broker on localhost:9092 for this test run.",
)


def _build_synthetic_engine(n=300, seed=0):
    rng = np.random.default_rng(seed)
    amt = rng.uniform(1, 500, size=n).astype("float32")
    c1 = rng.integers(0, 10, size=n).astype("float32")
    fraud_logit = -3.0 + 0.01 * amt + 0.3 * c1
    fraud_prob = 1 / (1 + np.exp(-fraud_logit))
    is_fraud = (rng.random(n) < fraud_prob).astype(int)
    df = pd.DataFrame({
        "TransactionID": np.arange(n), "TransactionDT": np.sort(rng.integers(0, 500000, size=n)),
        "isFraud": is_fraud, "TransactionAmt": amt, "ProductCD": rng.choice(["W", "C", "R"], size=n),
        "card1": rng.integers(1000, 1015, size=n), "card4": rng.choice(["visa", "mastercard"], size=n),
        "addr1": rng.integers(100, 130, size=n).astype("float32"), "C1": c1,
        "D1": rng.uniform(0, 100, size=n).astype("float32"), "V1": rng.uniform(0, 1, size=n).astype("float32"),
        "M1": rng.choice(["T", "F"], size=n),
    })
    train_df, val_df, test_df, _ = compute_temporal_split(df, "TransactionDT", "TransactionID", 0.7, 0.15, 0.15)
    schema = build_feature_schema(train_df)
    fp = FeaturePipeline(schema)
    fp.fit(train_df)
    X_train, X_val = fp.transform(train_df), fp.transform(val_df)
    y_train, y_val = fp.get_target(train_df, "isFraud"), fp.get_target(val_df, "isFraud")
    lgbm_pre = LightGBMPreprocessor(schema)
    Z_train, Z_val = lgbm_pre.fit_transform(X_train), lgbm_pre.transform(X_val)
    combined = pd.concat([train_df[["TransactionID", "TransactionDT", "card1", "TransactionAmt"]],
                           val_df[["TransactionID", "TransactionDT", "card1", "TransactionAmt"]],
                           test_df[["TransactionID", "TransactionDT", "card1", "TransactionAmt"]]], ignore_index=True)
    bhv = compute_behavioral_features(combined)
    bhv_names = get_behavioral_feature_names(bhv)
    bhv_idx = bhv.set_index("TransactionID")

    def attach(Z, ids):
        return pd.concat([Z.reset_index(drop=True), bhv_idx.loc[ids.values, bhv_names].reset_index(drop=True)], axis=1)

    model, _ = train_lightgbm(
        attach(Z_train, train_df["TransactionID"].reset_index(drop=True)), y_train,
        attach(Z_val, val_df["TransactionID"].reset_index(drop=True)), y_val,
        categorical_features=lgbm_pre.get_categorical_feature_names(), params={"n_estimators": 30}, early_stopping_rounds=10,
    )
    policy = DecisionPolicy(approve_threshold=0.3, block_threshold=0.7, name="test_policy")
    engine = RiskDecisionEngine(schema=schema, feature_pipeline=fp, lgbm_preprocessor=lgbm_pre, model=model,
                                 policy=policy, state_manager=BehavioralStateManager())
    return engine, test_df


def _txn_payload(row: pd.Series) -> dict:
    d = row.to_dict()
    d.pop("isFraud", None)
    return {k: (None if isinstance(v, float) and v != v else v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# InMemoryBroker — no external dependency
# ---------------------------------------------------------------------------

def test_inmemory_broker_fifo_order():
    broker = InMemoryBroker()
    broker.send({"id": 1})
    broker.send({"id": 2})
    assert broker.poll() == {"id": 1}
    assert broker.poll() == {"id": 2}
    assert broker.poll() is None


def test_fake_producer_consumer_roundtrip():
    broker = InMemoryBroker()
    producer = FakeTransactionProducer(broker)
    seen = []
    consumer = FakeTransactionConsumer(handler=lambda txn: seen.append(txn), broker=broker)

    producer.send({"TransactionID": 1})
    producer.send({"TransactionID": 2})
    consumer.run_until_empty()

    assert seen == [{"TransactionID": 1}, {"TransactionID": 2}]
    assert consumer.messages_processed == 2
    assert producer.sent_count == 2


def test_fake_consumer_retries_then_dead_letters_on_persistent_failure():
    broker = InMemoryBroker()
    dlq_broker = InMemoryBroker()
    dlq_producer = FakeTransactionProducer(dlq_broker)

    def always_fails(txn):
        raise ValueError("simulated processing failure")

    broker.send({"TransactionID": 99})
    consumer = FakeTransactionConsumer(handler=always_fails, broker=broker, max_retries=2, dlq_producer=dlq_producer)
    consumer.run_until_empty()

    assert consumer.messages_processed == 0
    assert consumer.messages_failed == 1
    assert consumer.messages_dead_lettered == 1
    assert len(dlq_broker) == 1
    dead_letter = dlq_broker.poll()
    assert dead_letter["original_transaction"] == {"TransactionID": 99}
    assert "simulated processing failure" in dead_letter["error"]


def test_fake_consumer_succeeds_after_transient_failure():
    """Retry logic must actually retry, not just count attempts."""
    broker = InMemoryBroker()
    attempts = {"n": 0}

    def fails_once_then_succeeds(txn):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ValueError("transient")
        return "ok"

    broker.send({"TransactionID": 1})
    consumer = FakeTransactionConsumer(handler=fails_once_then_succeeds, broker=broker, max_retries=2)
    consumer.run_until_empty()

    assert consumer.messages_processed == 1
    assert consumer.messages_failed == 0
    assert attempts["n"] == 2


# ---------------------------------------------------------------------------
# FraudProcessingHandler — wired to the REAL engine, no duplicated logic
# ---------------------------------------------------------------------------

def test_fraud_processing_handler_uses_real_engine_and_matches_direct_call():
    engine, test_df = _build_synthetic_engine(seed=11)
    handler = FraudProcessingHandler(engine)
    payload = _txn_payload(test_df.iloc[0])

    # A second, independent engine built from the SAME seed, called directly.
    engine_direct, _ = _build_synthetic_engine(seed=11)
    direct_result = engine_direct.process_transaction(dict(payload))

    handler_result = handler(dict(payload))

    assert handler_result["risk_score"] == pytest.approx(direct_result["risk_score"])
    assert handler_result["decision"] == direct_result["decision"]
    assert handler.transactions_processed == 1


def test_fraud_processing_handler_publishes_decision_to_producer():
    engine, test_df = _build_synthetic_engine(seed=12)
    broker = InMemoryBroker()
    decision_producer = FakeTransactionProducer(broker)
    handler = FraudProcessingHandler(engine, decision_producer=decision_producer)

    handler(_txn_payload(test_df.iloc[0]))

    assert len(broker) == 1
    published = broker.poll()
    assert "risk_score" in published and "decision" in published


def test_fraud_processing_handler_via_fake_consumer_end_to_end():
    """The full local pipeline: producer -> InMemoryBroker -> consumer ->
    REAL engine -> decision producer -> decisions broker."""
    engine, test_df = _build_synthetic_engine(seed=13)
    txn_broker = InMemoryBroker()
    decision_broker = InMemoryBroker()
    txn_producer = FakeTransactionProducer(txn_broker)
    decision_producer = FakeTransactionProducer(decision_broker)
    handler = FraudProcessingHandler(engine, decision_producer=decision_producer)
    consumer = FakeTransactionConsumer(handler=handler, broker=txn_broker)

    for i in range(5):
        txn_producer.send(_txn_payload(test_df.iloc[i]))
    consumer.run_until_empty()

    assert consumer.messages_processed == 5
    assert len(decision_broker) == 5


def test_fraud_processing_handler_validation_error_triggers_retry_and_dlq():
    engine, test_df = _build_synthetic_engine(seed=14)
    dlq_broker = InMemoryBroker()
    dlq_producer = FakeTransactionProducer(dlq_broker)
    handler = FraudProcessingHandler(engine)

    bad_payload = _txn_payload(test_df.iloc[0])
    bad_payload["isFraud"] = 0  # engine-level rejection (isFraud must never be accepted)

    broker = InMemoryBroker()
    broker.send(bad_payload)
    consumer = FakeTransactionConsumer(handler=handler, broker=broker, max_retries=1, dlq_producer=dlq_producer)
    consumer.run_until_empty()

    assert consumer.messages_failed == 1
    assert len(dlq_broker) == 1
    assert handler.transactions_rejected >= 1


# ---------------------------------------------------------------------------
# Real kafka-python client — construction/serialization, no broker required
# ---------------------------------------------------------------------------

def test_transaction_producer_serialization_helpers():
    from src.streaming.kafka_client import _serialize_transaction, _deserialize_transaction
    txn = {"TransactionID": 42, "TransactionAmt": 99.5}
    raw = _serialize_transaction(txn)
    assert isinstance(raw, bytes)
    assert _deserialize_transaction(raw) == txn


def test_transaction_producer_fails_cleanly_with_no_broker():
    """Real kafka-python client, real (short, bounded) connection attempt
    against a guaranteed-empty port — must raise StreamingUnavailableError,
    not hang and not leak a raw kafka-python exception type."""
    producer = TransactionProducer(bootstrap_servers="localhost:1", request_timeout_ms=1000, max_block_ms=1500)
    with pytest.raises(StreamingUnavailableError):
        producer.send({"TransactionID": 1})


@requires_real_kafka
def test_transaction_producer_consumer_against_real_broker():
    """Only runs if a real Kafka broker is genuinely reachable — expected
    to be SKIPPED in this sandboxed environment (see module docstring)."""
    producer = TransactionProducer(topic="fraud.transactions.pytest")
    producer.send({"TransactionID": 1})
    producer.close()


def test_fraud_processing_handler_is_idempotent_for_duplicate_transaction_ids():
    """The FIX for the real gap found in
    tests/test_production_failure_scenarios.py::test_duplicate_message_processing_is_not_idempotent_by_default —
    at the FraudProcessingHandler layer (where Kafka's at-least-once
    redelivery is a real concern), a duplicate transaction_id is detected
    and the engine is never called a second time."""
    engine, test_df = _build_synthetic_engine(seed=15)
    handler = FraudProcessingHandler(engine)
    payload = _txn_payload(test_df.iloc[0])

    result_1 = handler(dict(payload))
    snap_after_first = engine.state_manager.get_state_snapshot(payload["card1"])

    result_2 = handler(dict(payload))  # exact duplicate delivery
    snap_after_second = engine.state_manager.get_state_snapshot(payload["card1"])

    assert handler.duplicate_transactions_skipped == 1
    assert handler.transactions_processed == 1  # NOT 2 — the engine was only called once
    assert result_1 == result_2  # the cached original result is returned again
    assert snap_after_first == snap_after_second  # state was NOT double-updated


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
