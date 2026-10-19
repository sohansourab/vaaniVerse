"""Unit tests for scripts/load_sample.py.

No network access: `httpx.stream` is replaced with a stub that serves canned
bytes, so the download/validate/rename logic is exercised for real.
"""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from scripts import load_sample
from scripts.load_sample import SampleDownloadError, _is_pdf, download_sample

PDF_BYTES = b"%PDF-1.7\n" + b"0" * 4096 + b"\n%%EOF\n"


class _FakeResponse:
    def __init__(self, body: bytes, headers: dict[str, str] | None = None,
                 status: int = 200):
        self._body = body
        self.headers = headers or {}
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=None)  # type: ignore[arg-type]

    def iter_bytes(self, chunk_size: int = 65536):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]


class _FakeStream:
    def __init__(self, response: _FakeResponse | Exception):
        self._response = response

    def __enter__(self):
        if isinstance(self._response, Exception):
            raise self._response
        return self._response

    def __exit__(self, *exc):
        return False


@pytest.fixture
def fake_stream(monkeypatch):
    def install(response):
        monkeypatch.setattr(load_sample.httpx, "stream",
                            lambda *a, **k: _FakeStream(response))
    return install


# ── _is_pdf ────────────────────────────────────────────────────────────────────

def test_is_pdf_accepts_real_pdf(tmp_path):
    path = tmp_path / "a.pdf"
    path.write_bytes(PDF_BYTES)
    assert _is_pdf(path) is True


def test_is_pdf_rejects_html_and_missing(tmp_path):
    html = tmp_path / "a.pdf"
    html.write_bytes(b"<!DOCTYPE html><html>404</html>")
    assert _is_pdf(html) is False
    assert _is_pdf(tmp_path / "nope.pdf") is False


# ── download_sample ────────────────────────────────────────────────────────────

def test_download_writes_valid_pdf(tmp_path, fake_stream):
    fake_stream(_FakeResponse(PDF_BYTES, {"content-length": str(len(PDF_BYTES))}))
    dest = download_sample(dest_dir=tmp_path)
    assert dest.exists() and dest.read_bytes() == PDF_BYTES
    assert not (tmp_path / "Who_Am_I_Telugu.pdf.part").exists()


def test_existing_pdf_is_reused_without_network(tmp_path, monkeypatch):
    existing = tmp_path / "Who_Am_I_Telugu.pdf"
    existing.write_bytes(PDF_BYTES)

    def explode(*a, **k):
        raise AssertionError("download_sample must not hit the network")

    monkeypatch.setattr(load_sample.httpx, "stream", explode)
    assert download_sample(dest_dir=tmp_path) == existing


def test_force_redownloads(tmp_path, fake_stream):
    existing = tmp_path / "Who_Am_I_Telugu.pdf"
    existing.write_bytes(b"%PDF-1.4\nold\n")
    fake_stream(_FakeResponse(PDF_BYTES))
    assert download_sample(dest_dir=tmp_path, force=True).read_bytes() == PDF_BYTES


def test_error_page_is_rejected_and_part_removed(tmp_path, fake_stream):
    fake_stream(_FakeResponse(b"<html>not a pdf</html>"))
    with pytest.raises(SampleDownloadError):
        download_sample(dest_dir=tmp_path)
    assert list(tmp_path.glob("*.part")) == []
    assert not (tmp_path / "Who_Am_I_Telugu.pdf").exists()


def test_short_read_is_rejected(tmp_path, fake_stream):
    fake_stream(_FakeResponse(PDF_BYTES, {"content-length": str(len(PDF_BYTES) + 999)}))
    with pytest.raises(SampleDownloadError, match="Incomplete"):
        download_sample(dest_dir=tmp_path)


def test_network_error_becomes_sample_download_error(tmp_path, fake_stream):
    fake_stream(httpx.ConnectError("no route"))
    with pytest.raises(SampleDownloadError):
        download_sample(dest_dir=tmp_path)
