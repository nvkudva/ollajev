"""Jev-compatible (TypeSafe System One) HTTP API, plus the playground page and the Ollama-style admin API."""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import logging
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Body, FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import catalog, config, media, normalize, presets
from ..manager import Manager, NotDownloaded, NotEnoughMemory, NotTrusted, default_model
from . import admin

JSONContent = str | dict[str, Any] | list[Any]


class NoulCriteria(BaseModel):
    true: JSONContent | None = None
    false: JSONContent | None = None


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: JSONContent | None = None
    criteria: NoulCriteria | None = None


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: JSONContent | None = None
    criteria: dict[str, JSONContent | None] = Field(min_length=2)


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: JSONContent | None = None
    criteria: list[JSONContent] = Field(min_length=2)


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class SystemOneRequest(BaseModel):
    state: JSONContent
    model: str = "jev-latest"  # manager.DEFAULT_ALIASES: means the default model
    questions: dict[str, Question] = Field(min_length=1)
    # Base64 data URLs, read by models whose limits list the matching input (image, audio, video).
    images: list[str] | None = None
    audio: list[str] | None = None
    videos: list[str] | None = None


manager: Manager | None = None

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global manager
    manager = Manager()
    if preload:
        # Load before uvicorn accepts connections, so a reply from any route means ready to decide.
        manager.get(preload, keep_alive=-1 if pin_preload else None)
    yield
    manager.unload_all()


def current_manager() -> Manager:
    if manager is None:
        raise RuntimeError("the server has not started")
    return manager


preload: str | None = None  # set by `serve` before startup
pin_preload = False
# Host header names `serve` accepts when bound to a loopback address; blocks DNS rebinding.
# None means no check, as for a non-loopback bind (which requires an API key instead).
allowed_hosts: frozenset[str] | None = None


def configure(
    *,
    preload_model: str | None = None,
    pin_preload_model: bool = False,
    allowed: frozenset[str] | None = None,
) -> None:
    """Startup wiring for `serve`, in one call instead of poking module globals: what to load before
    accepting connections, whether it stays loaded, and which Host headers a loopback bind accepts."""
    global preload, pin_preload, allowed_hosts
    preload = preload_model
    pin_preload = pin_preload_model
    allowed_hosts = allowed


PLAYGROUND_CSP = (
    "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
    "object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
)
OPEN_PATHS = ("/", "/playground", "/demo", "/static/")  # the health probe and the playground page need no key

log = logging.getLogger(__name__)

try:
    _version = version("ollajev")
except PackageNotFoundError:  # running from a source tree that was never installed
    _version = "0+unknown"

app = FastAPI(title="Ollajev", version=_version, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _hostname(header: str) -> str:
    """`localhost:8000` -> `localhost`, `[::1]:8000` -> `[::1]`."""
    header = header.lower()
    if header.startswith("["):
        return header[: header.find("]") + 1]
    return header.partition(":")[0]


@app.middleware("http")
async def guard(request: Request, call_next: Any) -> Any:
    if allowed_hosts is not None:
        host = _hostname(request.headers.get("host", ""))
        if host not in allowed_hosts:
            return _invalid([], "host not allowed", "forbidden_host", 403)
    length = request.headers.get("content-length", "")
    if length.isdigit() and int(length) > config.max_body_bytes():
        return _invalid([], "request body too large", "payload_too_large", 413)
    key = config.api_key()
    path = request.url.path
    if key and path != "/" and not path.startswith(OPEN_PATHS[1:]):
        sent = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        if not secrets.compare_digest(sent.encode(), key.encode()):
            return _invalid([], "missing or wrong API key", "unauthorized", 401)
    return await call_next(request)


def _invalid(loc: list[str | int], msg: str, kind: str = "value_error", status: int = 422) -> JSONResponse:
    return JSONResponse(status_code=status, content={"detail": [{"loc": loc, "msg": msg, "type": kind}]})


def _wire(question: Question) -> dict[str, Any]:
    out = question.model_dump(exclude_none=True)
    if isinstance(question, ChoiceQuestion):
        out["criteria"] = question.criteria  # keep None descriptions; exclude_none would drop those options
    return out


def _model_error(exc: Exception) -> JSONResponse:
    if isinstance(exc, NotDownloaded):
        return _invalid(["body", "model"], str(exc), "model_not_found", 404)
    if isinstance(exc, NotTrusted):
        return _invalid(["body", "model"], str(exc), "model_not_trusted", 403)
    if isinstance(exc, NotEnoughMemory):
        return _invalid(["body", "model"], str(exc), "model_out_of_memory", 503)
    return _invalid(["body", "model"], str(exc))


@app.get("/")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "default_model": default_model(),
        "loaded": [s.name for s in current_manager().loaded()],
        "ui": "/playground",
    }


@app.get("/playground")
async def playground() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "playground.html", media_type="text/html", headers={"content-security-policy": PLAYGROUND_CSP}
    )


@app.get("/demo")
async def demo() -> RedirectResponse:
    """The playground's old address, kept so earlier links and bookmarks still open it."""
    return RedirectResponse("/playground", status_code=308)


@app.get("/ui/presets")
async def ui_presets() -> dict[str, Any]:
    return presets.examples()


@app.get("/ui/catalog")
async def ui_catalog() -> dict[str, Any]:
    """The curated models, downloaded or not, for the playground's model picker."""
    return {"models": [dataclasses.asdict(e) for e in catalog.CATALOG]}


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    """Every downloaded model, default first. TypeSafe clients read name, description and release_date."""
    default = default_model()
    out = []
    for m in await asyncio.to_thread(admin.tags):
        out.append(
            {
                "name": m["name"],
                "description": m["description"],
                "release_date": m["release_date"],
                "default": m["name"] == default,
                "limits": m["limits"],
            }
        )
    out.sort(key=lambda m: not m["default"])
    return {"models": out}


@app.post("/v1/systemone")
async def system_one(req: Annotated[SystemOneRequest, Body()], response: Response) -> Any:
    questions = {name: _wire(q) for name, q in req.questions.items()}
    inputs: dict[str, list[bytes]] = {}
    for field in media.KINDS:
        try:  # up to OLLAJEV_MAX_BODY_BYTES of base64: off the event loop
            inputs[field] = await asyncio.to_thread(media.decode, field, getattr(req, field) or [])
        except media.Invalid as exc:
            return _invalid(["body", field], str(exc))
    manager = current_manager()
    requested_at = time.time()
    try:
        started = time.monotonic()
        # One resolve: run() loads when needed, so no separate get() (which re-resolved offline) first.
        # Inference blocks on the per-model slot lock, so it runs off the event loop and health/tags
        # stay responsive while a question answers.
        if any(inputs.values()):
            run = functools.partial(manager.run, req.model, req.state, questions, inputs=inputs)
        else:
            run = functools.partial(manager.run, req.model, req.state, questions)
        slot, result = await asyncio.to_thread(run)
        finished = time.monotonic()
        answers = normalize.answers(questions, result["answers"])
    except (NotDownloaded, NotTrusted, NotEnoughMemory) as exc:
        return _model_error(exc)
    except media.Invalid as exc:
        return _invalid(["body", exc.field], await asyncio.to_thread(_with_readers, exc))
    except ValueError as exc:
        return _invalid(["body", "questions"], str(exc))
    if slot.loaded_at >= requested_at:
        # This request loaded the model: loaded_at (wall clock) minus the request start is the load.
        load_seconds = max(0.0, slot.loaded_at - requested_at)
        run_seconds = max(0.0, (finished - started) - load_seconds)
    else:
        load_seconds, run_seconds = None, finished - started
    response.headers["server-timing"] = _server_timing(
        load_seconds=load_seconds,
        run_seconds=run_seconds,
    )
    usage = result.get("usage") or {}
    return {
        "model": slot.name,
        "answers": answers,
        "usage": {
            "input_tokens": int(usage.get("input_tokens", 0)),
            "output_tokens": int(usage.get("output_tokens", 0)),
        },
    }


def _with_readers(exc: media.Invalid) -> str:
    """A media error, plus the downloaded models that read that kind of media when the chosen one reads none."""
    if exc.kind is None:
        return str(exc)
    readers = [m["name"] for m in admin.tags() if exc.kind in (m["limits"] or {}).get("inputs", ["text"])]
    if readers:
        return f"{exc} Downloaded models that read {media.PLURALS[exc.kind]}: {', '.join(readers)}."
    return (
        f"{exc} None of your downloaded models reads {media.PLURALS[exc.kind]}; "
        "`ollajev show <model>` lists what a model reads."
    )


def _server_timing(load_seconds: float | None, run_seconds: float) -> str:
    """A Server-Timing header: `load` when this request loaded the model into memory, `run` for the answer itself.
    A header, so the response body stays exactly the Jev wire format."""
    parts = []
    if load_seconds is not None:
        parts.append(f"load;dur={load_seconds * 1000:.0f}")
    parts.append(f"run;dur={run_seconds * 1000:.0f}")
    return ", ".join(parts)


app.include_router(admin.router)


@app.exception_handler(StarletteHTTPException)
def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Starlette answers 404/405 with a bare string; give them the same shape as every other error."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": [{"loc": [], "msg": exc.detail, "type": "http_error"}]},
        headers=exc.headers,  # 405 carries Allow
    )


@app.exception_handler(Exception)
def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    log.error("unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
    return JSONResponse(
        status_code=500, content={"detail": [{"loc": ["body"], "msg": "internal error", "type": "internal_error"}]}
    )
