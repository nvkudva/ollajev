"""Ollama-style model management: /api/tags, /api/ps, /api/pull, /api/show, /api/delete, /api/copy, /api/stop."""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from .. import config, store
from ..library import tags
from ..manager import canonical, lookup

__all__ = ["router", "tags"]

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")
_pulling: set[str] = set()
_pulling_guard = threading.Lock()


class ModelRef(BaseModel):
    model: str


class PullRequest(BaseModel):
    model: str
    stream: bool = True


class CopyRequest(BaseModel):
    source: str
    destination: str


@router.get("/tags")
async def api_tags() -> dict[str, Any]:
    """The download scan walks the whole cache, so it runs off the event loop."""
    return {"models": await asyncio.to_thread(tags)}


@router.get("/ps")
def api_ps() -> dict[str, Any]:
    from . import api

    now = time.monotonic()
    models = []
    for slot in api.current_manager().loaded():
        expires = (
            None
            if slot.expires == float("inf")
            else datetime.fromtimestamp(datetime.now().timestamp() + (slot.expires - now), UTC).isoformat()
        )
        models.append(
            {
                "name": slot.name,
                "model": slot.name,
                "device": slot.device,
                "expires_at": expires,
                "details": {"family": slot.resolved.family.name},
            }
        )
    return {"models": models}


def _not_found(exc: LookupError) -> JSONResponse:
    """An unknown model in the Ollama shape ({error}, not the TypeSafe detail list)."""
    return JSONResponse(status_code=404, content={"error": str(exc)})


@router.post("/show")
def api_show(req: ModelRef) -> Any:
    try:
        resolved = store.resolve(lookup(req.model), online=False)
    except LookupError as exc:
        return _not_found(exc)
    return {
        "model": canonical(resolved),
        "repo": resolved.repo_id,
        "revision": resolved.revision,
        "file": resolved.weights,
        "family": resolved.family.name,
        "runs_repo_code": resolved.family.runs_repo_code,
        "trusted": store.is_trusted(resolved),
        "release_date": store.released(resolved.repo_id),
        "limits": resolved.family.limits(resolved),
        "path": store.local_path(resolved),
    }


def report_bytes(resolved: store.Resolved, status: str, events: queue.Queue[dict[str, Any] | None]) -> None:
    """Download `r`, putting a `completed`/`total` byte event on `events` as the cache grows."""
    done = threading.Event()
    try:
        total = store.download_size(resolved)
    except Exception:  # progress is optional; the download itself reports real errors
        total = 0
    start = store.bytes_on_disk(resolved.repo_id)

    def poll() -> None:
        last = -1
        while not done.wait(1.0):
            completed = min(store.bytes_on_disk(resolved.repo_id) - start, total)
            if completed != last:
                last = completed
                events.put({"status": status, "digest": resolved.revision, "total": total, "completed": completed})

    if total:
        threading.Thread(target=poll, daemon=True).start()
    try:
        store.download(resolved)
    finally:
        done.set()


def _untrusted_message(resolved: store.Resolved) -> str:
    name = canonical(resolved)
    return (
        f"{name} runs Python code from its repo; trust is not available over HTTP. "
        f"Review https://huggingface.co/{resolved.repo_id}/tree/{resolved.revision}, then run: "
        f"ollajev pull {name} --trust"
    )


def _claim(repo_id: str) -> bool:
    """Mark `repo_id` as being pulled; False when another pull of it is running."""
    with _pulling_guard:
        if repo_id in _pulling:
            return False
        _pulling.add(repo_id)
        return True


def _release(repo_id: str) -> None:
    with _pulling_guard:
        _pulling.discard(repo_id)


def _download_with_events(resolved: store.Resolved, events: queue.Queue[dict[str, Any] | None]) -> None:
    status = f"downloading {resolved.repo_id}@{resolved.revision[:12]}"
    events.put({"status": status, "digest": resolved.revision})
    report_bytes(resolved, status, events)
    if store.needs_prefetch(resolved):
        events.put({"status": "downloading base model"})
        store.prefetch(resolved)
    events.put({"status": "success", "model": canonical(resolved)})


def _pull(model: str, events: queue.Queue[dict[str, Any] | None]) -> None:
    """Resolve and download `model`, putting status and error events on `events`, then None when done."""
    try:
        events.put({"status": "pulling manifest"})
        resolved = store.resolve(lookup(model))
        if resolved.family.runs_repo_code and not store.is_trusted(resolved):
            events.put({"error": _untrusted_message(resolved)})
            return
        if not _claim(resolved.repo_id):
            events.put({"error": f"{resolved.repo_id} is already being pulled"})
            return
        try:
            _download_with_events(resolved, events)
        finally:
            _release(resolved.repo_id)
    except (LookupError, ValueError) as exc:  # bad name or unknown model: safe to show
        events.put({"error": str(exc)})
    except Exception:
        log.exception("pull of %s failed", model)
        events.put({"error": "pull failed; see the server log"})
    finally:
        events.put(None)


@router.post("/pull")
def api_pull(req: PullRequest) -> Any:
    """Download a model; with stream=true, NDJSON status lines like Ollama's pull."""
    events: queue.Queue[dict[str, Any] | None] = queue.Queue()
    threading.Thread(target=_pull, args=(req.model, events), daemon=True).start()

    def lines():
        while (event := events.get()) is not None:
            yield json.dumps(event) + "\n"

    if req.stream:
        return StreamingResponse(lines(), media_type="application/x-ndjson")
    last = [json.loads(line) for line in lines()][-1]
    return JSONResponse(status_code=400 if "error" in last else 200, content=last)


@router.delete("/delete")
def api_delete(req: ModelRef) -> Any:
    from . import api

    if config.remove_alias(req.model) is not None:
        return {"status": "success"}
    try:
        resolved = store.resolve(lookup(req.model), online=False)
    except LookupError as exc:
        return _not_found(exc)
    api.current_manager().unload(canonical(resolved))
    return {"status": "success", "freed": store.remove(resolved)}


@router.post("/copy")
def api_copy(req: CopyRequest) -> Any:
    config.set_alias(req.destination, lookup(req.source))
    return {"status": "success"}


@router.post("/stop")
def api_stop(req: ModelRef) -> Any:
    from . import api

    try:
        resolved = store.resolve(lookup(req.model), online=False)
    except LookupError as exc:
        return _not_found(exc)
    return {"status": "success" if api.current_manager().unload(canonical(resolved)) else "not loaded"}
