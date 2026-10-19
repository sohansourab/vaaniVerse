"""Unit tests for API-key auth, rate limiting and request telemetry.

Fully offline: the auth check is a header comparison, the limiter is an
in-process sliding window, and the telemetry test drives a throwaway FastAPI app
rather than the real one (which would need a vector store and an LLM).
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from core import security
from core.security import (
    RateLimiter,
    _Bucket,
    auth_configured,
    recent_requests,
    require_api_key,
)


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    """Every test starts from the open local-development defaults."""
    monkeypatch.setattr(security, "VANIVERSE_API_KEY", "")
    monkeypatch.setattr(security, "REQUIRE_AUTH_FOR_ALL", False)
    security._recent.clear()
    yield
    security._recent.clear()


# ── authentication ─────────────────────────────────────────────────────────────

def test_routes_are_open_when_no_key_is_configured():
    require_api_key(None)          # no key configured → open
    require_api_key("anything")    # an irrelevant header is simply ignored


def test_auth_configured_tracks_the_key(monkeypatch):
    assert auth_configured() is False
    monkeypatch.setattr(security, "VANIVERSE_API_KEY", "sekret")
    assert auth_configured() is True


def test_missing_key_is_rejected_when_configured(monkeypatch):
    monkeypatch.setattr(security, "VANIVERSE_API_KEY", "sekret")
    with pytest.raises(HTTPException) as exc:
        require_api_key(None)
    assert exc.value.status_code == 401
    assert exc.value.headers["WWW-Authenticate"] == "ApiKey"


def test_wrong_key_is_rejected(monkeypatch):
    monkeypatch.setattr(security, "VANIVERSE_API_KEY", "sekret")
    with pytest.raises(HTTPException):
        require_api_key("guess")


def test_correct_key_passes(monkeypatch):
    monkeypatch.setattr(security, "VANIVERSE_API_KEY", "sekret")
    require_api_key("sekret")  # no raise


# ── rate limiting ──────────────────────────────────────────────────────────────

def test_bucket_allows_up_to_the_limit_then_denies():
    bucket = _Bucket()
    assert bucket.allow(100.0, 60.0, 2) is True
    assert bucket.allow(101.0, 60.0, 2) is True
    assert bucket.allow(102.0, 60.0, 2) is False


def test_bucket_slides_the_window():
    bucket = _Bucket()
    bucket.allow(100.0, 60.0, 2)
    bucket.allow(101.0, 60.0, 2)
    # By t=161 the request at t=100 has aged out of the 60 s window.
    assert bucket.allow(161.0, 60.0, 2) is True


def test_limiter_keys_are_independent():
    limiter = RateLimiter()
    assert limiter.allow("a", limit=1, window_s=60) is True
    assert limiter.allow("a", limit=1, window_s=60) is False
    assert limiter.allow("b", limit=1, window_s=60) is True


# ── telemetry middleware ───────────────────────────────────────────────────────

def _app() -> FastAPI:
    app = FastAPI()

    @app.get("/ping")
    def ping():
        return {"ok": True}

    # Same order as api.main: telemetry registered first so auth (added second)
    # is the outer layer, i.e. it runs first.
    app.middleware("http")(security.telemetry_middleware)
    app.middleware("http")(security.global_auth_middleware)
    return app


def test_telemetry_sets_a_request_id_and_records_the_call():
    resp = TestClient(_app()).get("/ping")
    assert resp.status_code == 200
    assert len(resp.headers["X-Request-ID"]) == 12

    record = recent_requests()[-1]
    assert record["path"] == "/ping"
    assert record["method"] == "GET"
    assert record["status"] == 200


def test_recent_requests_keeps_newest_last_and_honours_the_limit():
    for i in range(5):
        security._recent.append({"i": i})
    assert recent_requests(2) == [{"i": 3}, {"i": 4}]


def test_forwarded_for_header_selects_the_client_key():
    TestClient(_app()).get("/ping", headers={"X-Forwarded-For": "1.2.3.4, 5.6.7.8"})
    assert recent_requests()[-1]["client"] == "1.2.3.4"


# ── blanket auth middleware ────────────────────────────────────────────────────

def test_global_auth_blocks_and_admits_on_the_key(monkeypatch):
    monkeypatch.setattr(security, "REQUIRE_AUTH_FOR_ALL", True)
    monkeypatch.setattr(security, "VANIVERSE_API_KEY", "sekret")
    client = TestClient(_app())

    assert client.get("/ping").status_code == 401
    assert client.get("/ping", headers={"X-API-Key": "sekret"}).status_code == 200


def test_rejected_requests_are_not_recorded_as_usage(monkeypatch):
    """Auth runs before telemetry, so a 401 leaves no trace in the ring buffer."""
    monkeypatch.setattr(security, "REQUIRE_AUTH_FOR_ALL", True)
    monkeypatch.setattr(security, "VANIVERSE_API_KEY", "sekret")

    TestClient(_app()).get("/ping")
    assert recent_requests() == []


def test_routes_stay_open_when_blanket_auth_is_off():
    assert TestClient(_app()).get("/ping").status_code == 200
