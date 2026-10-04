"""The front-end HTTP client: one stack (httpx), urllib's old behavior preserved."""

import json

import httpx
import pytest

from ollajev import client


def _resp(status=200, payload=None):
    content = b"" if payload is None else json.dumps(payload).encode()
    return httpx.Response(status, content=content, request=httpx.Request("GET", "http://x/"))


def test_call_returns_parsed_json(monkeypatch):
    monkeypatch.setattr(httpx, "request", lambda *a, **k: _resp(200, {"models": []}))
    assert client.call("GET", "/api/tags") == {"models": []}


def test_call_returns_none_for_an_empty_body(monkeypatch):
    monkeypatch.setattr(httpx, "request", lambda *a, **k: _resp(200))
    assert client.call("GET", "/") is None


def test_call_reports_the_payload_error(monkeypatch):
    monkeypatch.setattr(httpx, "request", lambda *a, **k: _resp(404, {"error": "nope"}))
    with pytest.raises(SystemExit, match=r"^error: nope$"):
        client.call("GET", "/api/show")


def test_server_running_probes_the_health_endpoint(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _resp(200, {"status": "ok"}))
    assert client.server_running() is True

    def refused(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", refused)
    assert client.server_running() is False
