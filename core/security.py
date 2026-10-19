"""
Authentication, rate limiting and request logging for the HTTP layer.

Three problems this closes, all listed under "API and security" in the v2 plan:

  1. ``/ingest`` and ``DELETE /index/reset`` are destructive and were reachable
     by anyone who could open the port. When ``VANIVERSE_API_KEY`` is set they
     require an ``X-API-Key`` header. The key is compared in constant time.
  2. There was no rate limiting at all, so one client could pin a CPU-only box
     with an ingest loop. A small in-process sliding-window limiter covers the
     single-process deployments this project actually ships to.
  3. There was no telemetry, so a bad answer left no trace. Every request is
     logged with a correlation id, and the last N are kept in memory for
     ``GET /admin/requests``.

This is deliberately a dependency + middleware, not a framework. It is honest
about its limits: an in-process limiter does not coordinate across uvicorn
workers, and a shared demo key is not per-user auth.
"""
from __future__ import annotations

import hmac
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

from fastapi import Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

from core.config import (
    INGEST_LIMIT_PER_HOUR,
    RATE_LIMIT_ENABLED,
    RATE_LIMIT_PER_MINUTE,
    REQUIRE_AUTH_FOR_ALL,
    VANIVERSE_API_KEY,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Authentication
# ══════════════════════════════════════════════════════════════════════════════

def auth_configured() -> bool:
    """True when an API key is set, i.e. destructive routes are protected."""
    return bool(VANIVERSE_API_KEY)


def _key_matches(supplied: str | None) -> bool:
    if not VANIVERSE_API_KEY:
        return True
    if not supplied:
        return False
    return hmac.compare_digest(supplied, VANIVERSE_API_KEY)


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """
    Dependency for destructive routes (``/ingest``, ``/index/reset``).

    With no key configured the route stays open — that is the local-development
    default — and ``GET /health`` reports ``auth_configured: false`` so the
    exposure is visible rather than silent.
    """
    if not _key_matches(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A valid X-API-Key header is required for this endpoint.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


# ══════════════════════════════════════════════════════════════════════════════
# Rate limiting
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _Bucket:
    """Sliding-window request timestamps for a single key."""

    timestamps: deque[float] = field(default_factory=deque)

    def allow(self, now: float, window_s: float, limit: int) -> bool:
        cutoff = now - window_s
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.popleft()
        if len(self.timestamps) >= limit:
            return False
        self.timestamps.append(now)
        return True


class RateLimiter:
    """Thread-safe sliding-window limiter keyed by an arbitrary string."""

    def __init__(self) -> None:
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, *, limit: int, window_s: float) -> bool:
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.setdefault(key, _Bucket())
            # Opportunistic cleanup so a long-lived process does not leak keys.
            if len(self._buckets) > 4096:
                self._buckets = {k: v for k, v in self._buckets.items() if v.timestamps}
            return bucket.allow(now, window_s, limit)


_limiter = RateLimiter()

# The cheap interactive routes get the per-minute budget; ingestion gets an
# hourly budget because a single call is minutes of CPU, not milliseconds.
_MINUTE_ROUTES = ("/query", "/search")
_INGEST_ROUTES = ("/ingest",)


def _client_key(request: Request) -> str:
    # Trust X-Forwarded-For only as a hint; a public deployment should terminate
    # TLS and set it at a proxy. Falls back to the socket peer.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit_guard(request: Request) -> None:
    """Dependency applied to the interactive routes."""
    if not RATE_LIMIT_ENABLED:
        return
    if not _limiter.allow(
        f"minute:{_client_key(request)}", limit=RATE_LIMIT_PER_MINUTE, window_s=60.0
    ):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded ({RATE_LIMIT_PER_MINUTE}/min). Try again shortly.",
        )


def ingest_rate_limit_guard(request: Request) -> None:
    if not RATE_LIMIT_ENABLED:
        return
    if not _limiter.allow(
        f"ingest:{_client_key(request)}", limit=INGEST_LIMIT_PER_HOUR, window_s=3600.0
    ):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Ingest limit exceeded ({INGEST_LIMIT_PER_HOUR}/hour).",
        )


# ══════════════════════════════════════════════════════════════════════════════
# Request telemetry
# ══════════════════════════════════════════════════════════════════════════════

_RECENT_LIMIT = 200
_recent: deque[dict] = deque(maxlen=_RECENT_LIMIT)
_recent_lock = threading.Lock()


def recent_requests(limit: int = 50) -> list[dict]:
    """Most recent request records, newest last."""
    with _recent_lock:
        records = list(_recent)
    return records[-limit:]


async def telemetry_middleware(request: Request, call_next):
    """
    Attach a correlation id, time the request, and remember it.

    Logging every request is the minimum needed to debug "why was that answer
    wrong?" — without it, a bad answer on a hosted demo leaves no trace at all.
    """
    request_id = uuid.uuid4().hex[:12]
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.perf_counter() - started) * 1000
        _record(request, request_id, 500, elapsed_ms)
        raise

    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Request-ID"] = request_id
    _record(request, request_id, response.status_code, elapsed_ms)
    return response


def _record(request: Request, request_id: str, status_code: int, elapsed_ms: float) -> None:
    record = {
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "status": status_code,
        "latency_ms": round(elapsed_ms, 2),
        "client": _client_key(request),
    }
    with _recent_lock:
        _recent.append(record)
    logger.info(
        "%s %s %s → %s in %.1fms [%s]",
        request_id, request.method, request.url.path,
        status_code, elapsed_ms, record["client"],
    )


async def global_auth_middleware(request: Request, call_next):
    """
    Optional blanket auth for every route (``REQUIRE_AUTH_FOR_ALL=true``).

    Only inspects a header, and stays separate from the per-route dependency so
    read-only routes remain open by default.
    """
    if REQUIRE_AUTH_FOR_ALL and not _key_matches(request.headers.get("x-api-key")):
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "A valid X-API-Key header is required."},
        )
    return await call_next(request)
