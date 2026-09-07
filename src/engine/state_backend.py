"""
Production upgrade — pluggable state backend.

Extracts ONLY the storage/retrieval concern out of `BehavioralStateManager`
into a small interface (`StateBackend`: `get`/`put`/`items`/`clear`) so the
manager's feature-computation math (`compute_features`, `update`,
`bulk_initialize` in `src/engine/state.py`) is completely unchanged and
backend-agnostic. `InMemoryStateBackend` is the exact behavioral
equivalent of the original plain-dict implementation (default, used
everywhere existing tests already pass); `RedisStateBackend` is a new,
optional, real implementation backed by a real Redis server.

## Why a backend interface instead of two separate manager classes

`RiskDecisionEngine` and every existing test constructs
`BehavioralStateManager()` and calls the same five methods
(`compute_features`, `update`, `bulk_initialize`, `get_state_snapshot`,
`reset`) regardless of backend — none of that calling code needed to
change. Only `BehavioralStateManager.__init__` gained one new, optional,
backward-compatible parameter (`backend`).

## Redis serialization

`EntityState.to_dict()`/`from_dict()` (already existing, used for the
in-memory `serialize()`/`load_serialized()` methods) round-trip cleanly
to JSON, so `RedisStateBackend` stores one JSON string per entity under a
namespaced key (`{key_prefix}{entity_id}`) — no new serialization format
was invented for this.

## A real, documented behavioral nuance (not a correctness bug)

`BehavioralStateManager.compute_features()` prunes expired timestamps
from `state.recent_times` as a read-time optimization, mutating the
in-memory `EntityState` object it read. With `InMemoryStateBackend`, that
mutation is visible to the next call for free (same Python object,
same dict slot). With `RedisStateBackend`, `get()` returns a FRESH
deserialized copy on every call, so that particular read-time pruning
optimization does not persist back to Redis until the next real
`update()` call. This does NOT change the correctness of any returned
feature — `_velocity_counts()` always re-filters by the current cutoff
regardless of whether the deque was pre-pruned — it only means
`recent_times` may carry a handful of already-irrelevant timestamps for
slightly longer in Redis than in memory between `update()` calls. Documented
here explicitly rather than silently accepted.

## Failure handling

`RedisStateBackend` raises `StateBackendUnavailableError` (never a raw
`redis` exception) on connection failure, on the FIRST operation that
needs Redis — not eagerly at construction time, so a misconfigured Redis
URL doesn't crash application startup before the engine even processes a
request. Callers (see `src/api/dependencies.py`) decide what to do with
that error; this module does not silently fall back to in-memory storage,
because doing so silently would hide a real operational problem instead
of surfacing it.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod


class StateBackendUnavailableError(RuntimeError):
    """Raised by RedisStateBackend when Redis cannot be reached, instead
    of leaking a raw redis-py exception type into engine code."""


class StateBackend(ABC):
    @abstractmethod
    def get(self, entity_id):
        """Return the raw stored representation for entity_id, or None."""

    @abstractmethod
    def put(self, entity_id, state) -> None:
        """Store state for entity_id."""

    @abstractmethod
    def items(self):
        """Yield (entity_id, state) for every stored entity — used only by
        serialize()/entity_count, not on the hot prediction path."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all stored state."""

    @abstractmethod
    def count(self) -> int:
        """Number of entities currently stored."""


class InMemoryStateBackend(StateBackend):
    """The original behavior, extracted verbatim: a plain dict, process-local,
    non-persistent. Default backend — every existing test and the default
    `BehavioralStateManager()` constructor use this, unchanged."""

    def __init__(self):
        self._states: dict = {}

    def get(self, entity_id):
        return self._states.get(entity_id)

    def put(self, entity_id, state) -> None:
        self._states[entity_id] = state

    def items(self):
        return self._states.items()

    def clear(self) -> None:
        self._states = {}

    def count(self) -> int:
        return len(self._states)


class RedisStateBackend(StateBackend):
    """
    Real Redis-backed storage. Requires `redis-py` (`import redis`) and a
    reachable Redis server — this module does not start Redis itself.

    Each entity's `EntityState` is stored as one JSON string (via
    `state.to_dict()`/`EntityState.from_dict()`) under key
    `f"{key_prefix}{entity_id}"`. An optional `ttl_seconds` sets a
    per-entity expiry (useful in production to bound memory for entities
    that stop transacting) — `None` (the default) means no expiry, matching
    the in-memory backend's behavior of retaining state indefinitely.
    """

    def __init__(self, redis_client, entity_state_cls, key_prefix: str = "fraud:state:", ttl_seconds: int | None = None):
        self._redis = redis_client
        self._entity_state_cls = entity_state_cls
        self._key_prefix = key_prefix
        self._ttl_seconds = ttl_seconds

    def _key(self, entity_id) -> str:
        return f"{self._key_prefix}{entity_id}"

    def _wrap_errors(self, fn, *args, **kwargs):
        import redis as redis_lib
        try:
            return fn(*args, **kwargs)
        except redis_lib.exceptions.RedisError as e:
            raise StateBackendUnavailableError(f"Redis operation failed: {e}") from e

    def get(self, entity_id):
        raw = self._wrap_errors(self._redis.get, self._key(entity_id))
        if raw is None:
            return None
        return self._entity_state_cls.from_dict(json.loads(raw))

    def put(self, entity_id, state) -> None:
        payload = json.dumps(state.to_dict())
        if self._ttl_seconds is not None:
            self._wrap_errors(self._redis.set, self._key(entity_id), payload, ex=self._ttl_seconds)
        else:
            self._wrap_errors(self._redis.set, self._key(entity_id), payload)

    def items(self):
        keys = self._wrap_errors(self._redis.keys, f"{self._key_prefix}*")
        for key in keys:
            raw = self._wrap_errors(self._redis.get, key)
            if raw is None:
                continue
            key_str = key.decode() if isinstance(key, bytes) else key
            entity_id = key_str[len(self._key_prefix):]
            yield entity_id, self._entity_state_cls.from_dict(json.loads(raw))

    def clear(self) -> None:
        keys = self._wrap_errors(self._redis.keys, f"{self._key_prefix}*")
        if keys:
            self._wrap_errors(self._redis.delete, *keys)

    def count(self) -> int:
        keys = self._wrap_errors(self._redis.keys, f"{self._key_prefix}*")
        return len(keys)

    def ping(self) -> bool:
        """Explicit health check — used by /health, never called on the
        hot prediction path."""
        return bool(self._wrap_errors(self._redis.ping))
