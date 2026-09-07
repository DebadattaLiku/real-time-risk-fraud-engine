"""
Production upgrade — in-memory Kafka test double.

An explicit, clearly-named TEST DOUBLE — not a real Kafka implementation,
not a claim of Kafka-protocol compatibility. It exists so
`FraudProcessingConsumer`'s message-handling logic (deserialization,
calling the real engine, retry/dead-letter behavior) can be tested
end-to-end without a real broker, per the task's own instruction that
Kafka must not be required for unit tests.

Interface deliberately mirrors just enough of `TransactionProducer`/
`TransactionConsumer` (`send()` / iterate-and-process) to be a drop-in
substitute in tests — it is NOT registered anywhere as a production
option, and `src/api/dependencies.py` never selects it.
"""

from __future__ import annotations

from collections import deque


class InMemoryBroker:
    """A shared, in-process FIFO queue standing in for a Kafka topic."""

    def __init__(self):
        self._queue = deque()

    def send(self, transaction: dict) -> None:
        self._queue.append(transaction)

    def poll(self) -> dict | None:
        return self._queue.popleft() if self._queue else None

    def __len__(self) -> int:
        return len(self._queue)


class FakeTransactionProducer:
    """Same `send()` shape as the real `TransactionProducer`, backed by an
    `InMemoryBroker` instead of a real Kafka connection."""

    def __init__(self, broker: InMemoryBroker):
        self.broker = broker
        self.sent_count = 0

    def send(self, transaction: dict) -> None:
        self.broker.send(transaction)
        self.sent_count += 1

    def close(self) -> None:
        pass


class FakeTransactionConsumer:
    """
    Same processing semantics as the real `TransactionConsumer`
    (`_process_one`'s retry/dead-letter logic is reused directly, not
    reimplemented — see `run_until_empty` below) but reads from an
    `InMemoryBroker` instead of a real Kafka topic.
    """

    def __init__(self, handler, broker: InMemoryBroker, max_retries: int = 2, dlq_producer=None):
        # Reuse the REAL TransactionConsumer's retry/DLQ logic rather than
        # reimplementing it a second time — this fake only replaces the
        # transport (`run`), never the processing semantics.
        from src.streaming.kafka_client import TransactionConsumer
        self._real = TransactionConsumer(handler, max_retries=max_retries, dlq_producer=dlq_producer)
        self.broker = broker

    @property
    def messages_processed(self) -> int:
        return self._real.messages_processed

    @property
    def messages_failed(self) -> int:
        return self._real.messages_failed

    @property
    def messages_dead_lettered(self) -> int:
        return self._real.messages_dead_lettered

    def run_until_empty(self) -> None:
        while True:
            transaction = self.broker.poll()
            if transaction is None:
                return
            self._real._process_one(transaction)
