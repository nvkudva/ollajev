"""Loaded models, Ollama style: load on first request, unload after keep_alive, at most N at once."""

from __future__ import annotations

import contextlib
import gc
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import psutil

from . import config, media, names, registry, store
from .adapters import fast_cpu_convs, pick_device
from .adapters.base import Adapter

log = logging.getLogger(__name__)

# typesafe-sdk's DEFAULT_MODEL and Laya's stock name. Clients that never pass `model=` send one of
# these, so they mean "the default model".
DEFAULT_ALIASES = frozenset({"jev-latest", "laya"})


class NotDownloaded(LookupError):
    pass


class NotTrusted(PermissionError):
    pass


class NotEnoughMemory(MemoryError):
    pass


WEIGHT_SUFFIXES = (".safetensors", ".gguf", ".onnx", ".onnx_data", ".data", ".bin", ".pt")
MEMORY_HEADROOM = 1.1  # activations and runtime buffers on top of the weights


_weights_memo: dict[str, tuple[float, int]] = {}  # snapshot dir -> (dir mtime, weights bytes)


def weights_size(path: str) -> int:
    """Bytes of weight files under `path`. A load stats a multi-GB snapshot, so a result is reused
    while the snapshot dir's mtime is unchanged (downloads write new files, which bump it)."""
    try:
        stamp = os.path.getmtime(path)
    except OSError:
        return 0
    if (memo := _weights_memo.get(path)) is not None and memo[0] == stamp:
        return memo[1]
    size = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(path)
        for f in files
        if f.endswith(WEIGHT_SUFFIXES)
    )
    _weights_memo[path] = (stamp, size)
    return size


def free_memory(device: str) -> int:
    if device.startswith("cuda"):
        import torch

        return torch.cuda.mem_get_info()[0]
    return psutil.virtual_memory().available


def canonical(resolved: store.Resolved) -> str:
    """One name per weight file: `repo`, `repo:QUANT` or `repo:fp16`, whatever spelling the request used."""
    if resolved.weights is None:
        return resolved.repo_id
    return f"{resolved.repo_id}:{names.label_of(resolved.weights)}"


def default_model() -> str:
    return config.load().get("default_model") or config.DEFAULT_MODEL


def lookup(name: str | None) -> str:
    """Request model name -> the name to resolve, following `cp` aliases and the default aliases, ignoring case."""
    if not name or name.lower() in DEFAULT_ALIASES:
        name = default_model()
    return config.alias(name)


@dataclass
class Slot:
    name: str
    adapter: Adapter
    resolved: store.Resolved
    device: str
    lock: threading.Lock = field(default_factory=threading.Lock)
    expires: float = 0.0  # monotonic deadline; inf = never
    pinned: bool = False  # the model `serve` preloaded stays until the server stops
    loaded_at: float = field(default_factory=time.time)
    closed: bool = False  # set under `lock` by unload; a request that sees it asks the manager again


class Manager:
    def __init__(self) -> None:
        self._slots: dict[str, Slot] = {}
        self._guard = threading.RLock()
        self._load_lock = threading.Lock()  # one load at a time: loads are memory spikes
        self._stop = threading.Event()
        self._reaper = threading.Thread(target=self._reap, name="ollajev-reaper", daemon=True)
        self._reaper.start()

    def stop(self) -> None:
        """Wake the reaper so it exits; the thread is a daemon, so this is only for tests."""
        self._stop.set()

    def resolve(self, name: str | None) -> store.Resolved:
        try:
            return store.resolve(lookup(name), online=False)
        except LookupError as exc:
            raise NotDownloaded(str(exc)) from None

    def get(self, name: str | None, keep_alive: float | None = None) -> Slot:
        resolved = self.resolve(name)
        key = canonical(resolved)
        with self._guard:
            slot = self._slots.get(key)
        if slot is None:
            with self._load_lock:
                with self._guard:
                    slot = self._slots.get(key)
                if slot is None:
                    slot = self._load(key, resolved)
        self._touch(slot, keep_alive)
        return slot

    def _load(self, key: str, resolved: store.Resolved) -> Slot:
        if not registry.is_trusted(resolved):
            raise NotTrusted(
                f"{key} runs Python code from its repo at {resolved.revision[:12]}; review it, then run: "
                f"ollajev pull {key} --trust"
            )
        path = store.local_path(resolved)
        if path is None:
            raise NotDownloaded(f"{key} is not downloaded; run: ollajev pull {key}")
        while True:
            with self._guard:
                if len(self._slots) < max(1, config.max_loaded_models()):
                    break
                victim = min(self._slots.values(), key=lambda s: s.expires).name
            self.unload(victim)
        device = pick_device(config.device())
        need, free = int(weights_size(path) * MEMORY_HEADROOM), free_memory(device)
        while need > free and (idle := [s for s in self.loaded() if not s.pinned and not s.lock.locked()]):
            self.unload(min(idle, key=lambda s: s.expires).name)  # like the count limit: least recently used first
            free = free_memory(device)
        if need > free:
            busy = ", ".join(s.name for s in self.loaded())
            raise NotEnoughMemory(
                f"{key} needs about {need / 2**30:.1f} GiB but only {free / 2**30:.1f} GiB is free on {device}"
                + (f" with {busy} still loaded" if busy else "")
                + ". Close other apps, or pick a smaller model or quant."
            )
        log.info("Loading %s on %s …", key, device)
        started = time.monotonic()
        adapter = resolved.family.load(path, resolved, device)
        fast_cpu_convs()
        adapter.name = key
        log.info("Loaded %s in %.1f s", key, time.monotonic() - started)
        slot = Slot(key, adapter, resolved, getattr(adapter, "device", None) or device)
        with self._guard:
            self._slots[key] = slot
        return slot

    def _touch(self, slot: Slot, keep_alive: float | None) -> None:
        if keep_alive is not None and keep_alive < 0:
            slot.pinned = True
        if slot.pinned:
            slot.expires = float("inf")
            return
        seconds = config.keep_alive() if keep_alive is None else keep_alive
        slot.expires = float("inf") if seconds < 0 else time.monotonic() + seconds

    def unload(self, name: str) -> bool:
        with self._guard:
            slot = self._slots.pop(name, None)
        if slot is None:
            return False
        with slot.lock:  # let a running request finish
            slot.closed = True
            close = getattr(slot.adapter, "close", None)
            if close:
                close()
        del slot
        gc.collect()
        _empty_device_cache()
        log.info("Unloaded %s", name)
        return True

    def unload_all(self) -> None:
        for name in list(self._slots):
            self.unload(name)

    def loaded(self) -> list[Slot]:
        with self._guard:
            return list(self._slots.values())

    def _reap(self) -> None:
        while not self._stop.wait(1 if self._slots else 5):
            now = time.monotonic()
            for slot in self.loaded():
                if slot.expires <= now and not slot.lock.locked():
                    self.unload(slot.name)

    def run(
        self,
        name: str | None,
        state: Any,
        questions: dict[str, dict[str, Any]],
        keep_alive: float | None = None,
        inputs: dict[str, list[bytes]] | None = None,
    ) -> tuple[Slot, dict[str, Any]]:
        if any((inputs or {}).values()):  # before a load: media a model cannot read must not load it
            resolved = self.resolve(name)
            media.check(resolved.family.limits(resolved), inputs or {})
        while True:
            slot = self.get(name, keep_alive)
            check_limits(slot.adapter.limits, questions)
            media.check(slot.adapter.limits, inputs or {})  # the loaded model's own limits are the final word
            # One forward pass per model at a time: on MPS concurrent forwards abort the process with a
            # Metal command-buffer assertion, and several adapters keep per-call state.
            with slot.lock:
                if slot.closed:  # unloaded between get and here: load it again
                    continue
                if any((inputs or {}).values()):
                    result = slot.adapter.system_one(state, questions, inputs)
                else:
                    result = slot.adapter.system_one(state, questions)
            break
        self._touch(slot, keep_alive)
        return slot, result


def check_limits(limits: dict[str, Any], questions: dict[str, dict[str, Any]]) -> None:
    """Reject what the loaded model cannot take before it runs. Token limits stay with the model."""
    if (n := limits.get("max_questions")) and len(questions) > n:
        raise ValueError(
            f"this model takes at most {n} questions per request, got {len(questions)}. "
            "Split them across requests, or pick a model that takes more."
        )
    for qid, q in questions.items():
        count = len(q.get("criteria") or ())
        if q["type"] == "choice" and (n := limits.get("max_options")) and count > n:
            raise ValueError(
                f"question {qid!r}: this model takes at most {n} choice options, got {count}. "
                "Merge or drop options, or pick a model that takes more."
            )
        if q["type"] == "score" and (n := limits.get("max_levels")) and count > n:
            raise ValueError(
                f"question {qid!r}: this model takes at most {n} score levels, got {count}. "
                "Use fewer levels, or pick a model that takes more."
            )


def _empty_device_cache() -> None:
    if "mlx.core" in sys.modules:  # MLX keeps freed Metal buffers in its own cache until told to drop them
        with contextlib.suppress(Exception):
            sys.modules["mlx.core"].clear_cache()
    if "torch" not in sys.modules:  # a GGUF or ONNX model never imported it; importing it now costs seconds
        return
    with contextlib.suppress(Exception):  # cache release is best effort
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()


def canonical_or(name: str) -> str:
    try:
        return canonical(store.resolve(name, online=False))
    except (LookupError, ValueError):
        return name
