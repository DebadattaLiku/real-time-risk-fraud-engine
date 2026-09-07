"""
Production upgrade — Kafka transaction streaming layer.

## A real, documented environment limitation

This sandboxed development environment's network egress allowlist
includes PyPI, npm, GitHub, and OS package mirrors, but NOT the Apache
Kafka / Confluent / Bitnami / Redpanda download or container-registry
domains needed to obtain an actual Kafka BROKER binary or image (the
Ubuntu apt repositories that ARE reachable do not package a Kafka broker
either — only unrelated Go client libraries, confirmed by inspection
during Phase 1 of this upgrade). This means: no real Kafka broker could
be started or connected to anywhere in this development session.

What IS real here:
- `kafka-python` (the actual PyPI client library) is installed and used
  directly below — this is not a hand-rolled fake client.
- `TransactionProducer`/`TransactionConsumer` genuinely attempt a real
  network connection to whatever `KAFKA_BROKERS` points at, and were
  verified in this session to fail cleanly and quickly (a caught
  `KafkaTimeoutError`, not a hang or crash) against `localhost:9092` with
  nothing listening there.
- `docker-compose.kafka.yml` (repo root) provides a real, standard Kafka
  broker service definition for anyone running this project somewhere
  with normal Docker Hub access — genuinely usable there, just not
  runnable in THIS sandboxed session.
- `InMemoryBroker` is an explicit, clearly-named TEST DOUBLE (not a real
  Kafka implementation) implementing the same minimal produce/poll shape,
  so the consumer's MESSAGE-PROCESSING LOGIC (deserialization, calling the
  real `RiskDecisionEngine`, error/retry handling, dead-letter routing) is
  genuinely exercised and tested end-to-end without needing a real broker
  — this is also why the task's own instruction ("Do NOT require Kafka
  for unit tests") is satisfied structurally, not just by skipping tests.

Kafka-specific throughput/latency numbers are therefore NOT MEASURED
anywhere in this project — see `reports/streaming_benchmark.md` for the
explicit accounting of what was and wasn't measured, and why.
"""

from __future__ import annotations

import json
import logging
import os
import time

logger = logging.getLogger("fraud_streaming")

DEFAULT_TRANSACTION_TOPIC = "fraud.transactions"
DEFAULT_DECISION_TOPIC = "fraud.decisions"
DEFAULT_DLQ_TOPIC = "fraud.transactions.dlq"


class StreamingUnavailableError(RuntimeError):
    """Raised by TransactionProducer/TransactionConsumer on any real
    connection/broker failure — never a raw kafka-python exception type,
    same pattern as StateBackendUnavailableError in
    src/engine/state_backend.py."""


def _serialize_transaction(transaction: dict) -> bytes:
    return json.dumps(transaction).encode("utf-8")


def _deserialize_transaction(raw: bytes) -> dict:
    return json.loads(raw.decode("utf-8"))


class TransactionProducer:
    """
    Thin wrapper around `kafka.KafkaProducer`. Real client library, real
    network attempt — see module docstring for what could and couldn't be
    verified against an actual broker in this environment.
    """

    def __init__(self, bootstrap_servers: str | None = None, topic: str = DEFAULT_TRANSACTION_TOPIC,
                 request_timeout_ms: int = 3000, max_block_ms: int = 3000):
        self.bootstrap_servers = bootstrap_servers or os.environ.get("KAFKA_BROKERS", "localhost:9092")
        self.topic = topic
        self._request_timeout_ms = request_timeout_ms
        self._max_block_ms = max_block_ms
        self._producer = None  # lazily constructed — see _ensure_producer

    def _ensure_producer(self):
        if self._producer is not None:
            return self._producer
        from kafka import KafkaProducer
        import kafka.errors as kafka_errors
        try:
            self._producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers.split(","),
                # Explicit api_version avoids kafka-python's auto-negotiation
                # handshake, which was observed in this environment to hang
                # indefinitely (not merely time out) against an unreachable
                # broker — see module docstring.
                api_version=(2, 5, 0),
                request_timeout_ms=self._request_timeout_ms,
                max_block_ms=self._max_block_ms,
                value_serializer=_serialize_transaction,
            )
        except kafka_errors.KafkaError as e:
            raise StreamingUnavailableError(f"Could not construct Kafka producer: {e}") from e
        return self._producer

    def send(self, transaction: dict) -> None:
        """Publish one transaction. Raises StreamingUnavailableError on
        any real send failure (broker unreachable, timeout) — never
        silently drops a message."""
        import kafka.errors as kafka_errors
        producer = self._ensure_producer()
        try:
            future = producer.send(self.topic, value=transaction)
            future.get(timeout=self._request_timeout_ms / 1000)
        except kafka_errors.KafkaError as e:
            raise StreamingUnavailableError(f"Failed to send transaction to Kafka topic '{self.topic}': {e}") from e

    def close(self) -> None:
        if self._producer is not None:
            self._producer.close(timeout=2)
            self._producer = None


class TransactionConsumer:
    """
    Thin wrapper around `kafka.KafkaConsumer`. `handler(transaction: dict)`
    is called once per successfully-deserialized message. On a handler
    exception, the message is retried up to `max_retries` times, then
    published to `dlq_topic` (a "dead-letter queue" — a documented,
    explicit failure strategy, not silent message loss) if a producer for
    the DLQ is configured, or just logged and skipped otherwise.

    `stop()` sets a flag checked at the top of every poll loop iteration —
    a cooperative, graceful shutdown (finishes the in-flight message,
    does not abandon it mid-processing), not a hard kill.
    """

    def __init__(
        self, handler, bootstrap_servers: str | None = None, topic: str = DEFAULT_TRANSACTION_TOPIC,
        group_id: str = "fraud-processing-consumer", max_retries: int = 2,
        dlq_producer: TransactionProducer | None = None, poll_timeout_ms: int = 1000,
    ):
        self.handler = handler
        self.bootstrap_servers = bootstrap_servers or os.environ.get("KAFKA_BROKERS", "localhost:9092")
        self.topic = topic
        self.group_id = group_id
        self.max_retries = max_retries
        self.dlq_producer = dlq_producer
        self.poll_timeout_ms = poll_timeout_ms
        self._stopped = False
        self.messages_processed = 0
        self.messages_failed = 0
        self.messages_dead_lettered = 0

    def stop(self) -> None:
        self._stopped = True

    def _process_one(self, transaction: dict) -> None:
        last_error = None
        for attempt in range(1, self.max_retries + 2):  # +1 initial attempt, +1 for range inclusivity
            try:
                self.handler(transaction)
                self.messages_processed += 1
                return
            except Exception as e:  # noqa: BLE001 — deliberately broad: any handler failure is retried, never crashes the consumer loop
                last_error = e
                logger.warning(f"event=transaction_processing_failed attempt={attempt} error={e!r}")
                time.sleep(min(0.1 * attempt, 1.0))  # simple linear backoff
        self.messages_failed += 1
        logger.error(f"event=transaction_processing_exhausted_retries error={last_error!r}")
        if self.dlq_producer is not None:
            try:
                self.dlq_producer.send({"original_transaction": transaction, "error": str(last_error)})
                self.messages_dead_lettered += 1
            except StreamingUnavailableError as dlq_error:
                logger.error(f"event=dlq_publish_failed error={dlq_error!r}")

    def run(self, max_messages: int | None = None) -> None:
        """Real Kafka consume loop. `max_messages` (test/demo convenience,
        not used in a real long-running deployment) stops after N messages."""
        from kafka import KafkaConsumer
        import kafka.errors as kafka_errors
        try:
            consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers.split(","),
                api_version=(2, 5, 0),
                group_id=self.group_id,
                consumer_timeout_ms=self.poll_timeout_ms,
                value_deserializer=_deserialize_transaction,
                enable_auto_commit=True,
            )
        except kafka_errors.KafkaError as e:
            raise StreamingUnavailableError(f"Could not construct Kafka consumer: {e}") from e

        try:
            n = 0
            while not self._stopped:
                for message in consumer:
                    if self._stopped:
                        break
                    self._process_one(message.value)
                    n += 1
                    if max_messages is not None and n >= max_messages:
                        return
                if max_messages is not None:
                    return  # consumer_timeout_ms elapsed with nothing new — stop for a bounded demo run
        finally:
            consumer.close()
