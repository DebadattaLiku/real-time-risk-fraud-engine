"""
Phase 9: Passive request-metrics middleware.

Wraps every HTTP request/response pair to record count, status class, and
latency — WITHOUT reading, modifying, or re-serializing the response body.
`call_next` is awaited and its return value passed straight through
unchanged; only its status code and the elapsed wall-clock time are
observed. This is what makes the middleware genuinely passive: it cannot
alter what a client receives, only measure it.
"""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    `registry_provider` is a zero-argument callable returning the CURRENT
    `MetricsRegistry` (or `None` if not yet initialized) — not a fixed
    instance bound at construction time, because this middleware is
    registered on the FastAPI `app` object at IMPORT time (before
    `lifespan` has run and actually created the registry). Looking it up
    lazily, on every request, means the middleware works correctly
    regardless of exactly when the registry was installed.
    """

    def __init__(self, app, registry_provider):
        super().__init__(app)
        self.registry_provider = registry_provider

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # An unhandled exception still gets recorded as a request (as
            # a 500), then re-raised unchanged so FastAPI's normal error
            # handling proceeds exactly as it would without this
            # middleware installed.
            duration_ms = (time.perf_counter() - start) * 1000
            registry = self.registry_provider()
            if registry is not None:
                registry.record_request(endpoint=request.url.path, status_code=500, duration_ms=duration_ms)
            raise
        duration_ms = (time.perf_counter() - start) * 1000
        registry = self.registry_provider()
        if registry is not None:
            registry.record_request(endpoint=request.url.path, status_code=response.status_code, duration_ms=duration_ms)
        return response
