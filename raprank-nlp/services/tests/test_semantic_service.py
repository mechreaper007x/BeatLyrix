"""
raprank-semantic HTTP client — focus on graceful degradation.

The semantic axes are an additive enrichment layer, so analyze_semantics() must
return None (never raise) whenever the service is empty-input, down, slow, or
returns a malformed body. These tests mock httpx so no network/model is needed.
"""
from __future__ import annotations

import asyncio

import httpx

from services import semantic_service as sem


def _run(coro):
    return asyncio.run(coro)


class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)

    def json(self):
        return self._payload


class _FakeClient:
    """Stand-in for httpx.AsyncClient used as an async context manager."""

    def __init__(self, *, result=None, exc=None):
        self._result = result
        self._exc = exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, *a, **k):
        if self._exc is not None:
            raise self._exc
        return self._result


def _patch_client(monkeypatch, **kwargs):
    monkeypatch.setattr(sem.httpx, "AsyncClient", lambda *a, **k: _FakeClient(**kwargs))


def test_empty_lyrics_returns_none_without_calling_service(monkeypatch):
    # Should short-circuit before ever constructing a client.
    def _boom(*a, **k):
        raise AssertionError("should not hit the network for empty input")

    monkeypatch.setattr(sem.httpx, "AsyncClient", _boom)
    assert _run(sem.analyze_semantics("   \n  ")) is None


def test_successful_response_is_passed_through(monkeypatch):
    payload = {
        "coherence_score": 71.5,
        "semantic_surprisal_score": 40.0,
        "lexical_sophistication_score": 55.2,
        "theme_consistency_score": 80.1,
        "metrics": {"line_count": 8.0},
    }
    _patch_client(monkeypatch, result=_FakeResp(payload))
    out = _run(sem.analyze_semantics("some real bars\nmore real bars"))
    assert out == payload


def test_network_error_degrades_to_none(monkeypatch):
    _patch_client(monkeypatch, exc=httpx.ConnectError("refused"))
    assert _run(sem.analyze_semantics("bars\nbars")) is None


def test_timeout_degrades_to_none(monkeypatch):
    _patch_client(monkeypatch, exc=httpx.TimeoutException("slow"))
    assert _run(sem.analyze_semantics("bars\nbars")) is None


def test_http_500_degrades_to_none(monkeypatch):
    _patch_client(monkeypatch, result=_FakeResp({}, status=500))
    assert _run(sem.analyze_semantics("bars\nbars")) is None


def test_non_dict_payload_degrades_to_none(monkeypatch):
    _patch_client(monkeypatch, result=_FakeResp(["not", "a", "dict"]))
    assert _run(sem.analyze_semantics("bars\nbars")) is None
