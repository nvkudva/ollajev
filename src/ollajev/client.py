"""Talking to a running Ollajev server over HTTP, for the front ends."""

from __future__ import annotations

import os
from typing import Any

import httpx

from . import config


def server_url() -> str:
    """OLLAJEV_HOST when set, else the address the last `serve` bound, else the default."""
    if os.environ.get("OLLAJEV_HOST"):
        host, port = config.host()
        return f"http://{url_host(host)}:{port}"
    return config.load().get("server_url") or f"http://127.0.0.1:{config.DEFAULT_PORT}"


def auth_headers() -> dict[str, str]:
    headers = {"content-type": "application/json"}
    if key := config.api_key():
        headers["authorization"] = f"Bearer {key}"
    return headers


def call(method: str, path: str, body: dict[str, Any] | None = None, timeout: float = 600) -> Any:
    try:
        resp = httpx.request(
            method, server_url() + path, json=body, headers=auth_headers(), timeout=timeout, follow_redirects=True
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        try:
            payload = exc.response.json()
        except ValueError:
            payload = {}
        raise SystemExit(f"error: {payload.get('error') or payload.get('detail') or exc}") from None
    return resp.json() if resp.content else None


def server_running() -> bool:
    try:
        httpx.get(server_url() + "/", timeout=0.5, follow_redirects=True).raise_for_status()
        return True
    except httpx.HTTPError:
        return False


def need_server() -> None:
    if not server_running():
        raise SystemExit(f"could not connect to Ollajev at {server_url()}; start it with: ollajev serve")


def url_host(host: str) -> str:
    probe = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    return f"[{probe}]" if ":" in probe else probe
