"""Settings from OLLAJEV_* environment variables, plus the saved config file."""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import os
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import platformdirs

log = logging.getLogger(__name__)
_lock = threading.RLock()

DEFAULT_MODEL = "Mapika/decider-4b-GGUF:Q4_K_M"
DEFAULT_PORT = 8000


APP = "ollajev"


def config_dir() -> Path:
    """OLLAJEV_HOME, else ~/.ollajev."""
    home = os.environ.get("OLLAJEV_HOME")
    return Path(home) if home else Path.home() / ".ollajev"


def log_dir() -> Path:
    return config_dir() / "logs"


def config_path() -> Path:
    return config_dir() / "config.json"


def host() -> tuple[str, int]:
    """OLLAJEV_HOST as host or host:port, like OLLAMA_HOST. Default 127.0.0.1:8000."""
    value = os.environ.get("OLLAJEV_HOST", "")
    value = value.removeprefix("http://").removeprefix("https://").rstrip("/")
    if not value:
        return "127.0.0.1", DEFAULT_PORT
    if value.startswith("["):  # [::1]:8000
        addr, _, port = value[1:].partition("]")
        return addr, _port(port.lstrip(":") or str(DEFAULT_PORT))
    if value.count(":") == 1:
        addr, port = value.split(":")
        return addr or "127.0.0.1", _port(port)
    return value, DEFAULT_PORT


def _port(text: str) -> int:
    if not text.isdigit() or not 0 < int(text) < 65536:
        raise ValueError(f"OLLAJEV_HOST has an invalid port {text!r}; use host:port, e.g. 127.0.0.1:8000")
    return int(text)


def api_key() -> str | None:
    """OLLAJEV_API_KEY: when set, the server requires `Authorization: Bearer <key>` on every API route."""
    return os.environ.get("OLLAJEV_API_KEY") or None


def is_loopback(addr: str) -> bool:
    return addr in ("localhost", "::1") or addr.startswith("127.")


def models_dir() -> str | None:
    """OLLAJEV_MODELS overrides where weights are stored; None means the shared Hugging Face cache."""
    return os.environ.get("OLLAJEV_MODELS") or None


def _setting(env: str, key: str, default: Any) -> Any:
    """An OLLAJEV_* environment variable, else the saved setting, else the default."""
    return os.environ.get(env) or load().get(key) or default


def keep_alive() -> float:
    """Seconds an idle model stays loaded: OLLAJEV_KEEP_ALIVE, else the saved setting, else 5m. Accepts 300, 5m,
    1h, or -1 for forever."""
    value = _setting("OLLAJEV_KEEP_ALIVE", "keep_alive", "5m")
    try:
        return parse_duration(value)
    except ValueError:
        raise ValueError("keep alive (OLLAJEV_KEEP_ALIVE) must be seconds or a duration like 5m, 1h, -1") from None


def max_loaded_models() -> int:
    """How many models stay in memory at once: OLLAJEV_MAX_LOADED_MODELS, else the saved setting, else 1."""
    value = _setting("OLLAJEV_MAX_LOADED_MODELS", "max_loaded_models", 1)
    try:
        return int(value)
    except ValueError:
        raise ValueError("models in memory (OLLAJEV_MAX_LOADED_MODELS) must be a whole number") from None


def max_body_bytes() -> int:
    """Largest request body the API accepts. OLLAJEV_MAX_BODY_BYTES overrides the 8 MiB default."""
    try:
        return int(os.environ.get("OLLAJEV_MAX_BODY_BYTES", 8 * 1024 * 1024))
    except ValueError:
        raise ValueError("OLLAJEV_MAX_BODY_BYTES must be a whole number") from None


def device() -> str | None:
    """OLLAJEV_DEVICE, else the device saved by setup, forces cpu, mps or cuda; None picks the best one."""
    value = _setting("OLLAJEV_DEVICE", "device", None)
    return None if value in (None, "", "auto") else value


def parse_duration(text: str | float | int) -> float:
    if isinstance(text, (int, float)):
        return float(text)
    text = text.strip()
    units = {"s": 1, "m": 60, "h": 3600}
    if text and text[-1] in units:
        return float(text[:-1]) * units[text[-1]]
    return float(text)


def _legacy_path() -> Path:
    """Where releases before 0.2 kept config.json: the OS config folder."""
    return Path(platformdirs.user_config_dir(APP, appauthor=False)) / "config.json"


_load_memo: tuple[str, int, int, dict[str, Any]] | None = None  # path, mtime_ns, size, data


def load() -> dict[str, Any]:
    """The saved settings. The file is re-read only when its size or mtime changed, so the several
    reads per request (pins, aliases, keep-alive) do one parse; the result is a copy, mutate it via edit()."""
    global _load_memo
    path = config_path()
    if not path.exists() and "OLLAJEV_HOME" not in os.environ and _legacy_path().exists():
        path = _legacy_path()  # first run after the move to ~/.ollajev; the next save writes the new file
    try:
        stat = path.stat()
    except OSError:
        return {}
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    with _lock:
        if _load_memo is not None and _load_memo[:3] == key:
            return copy.deepcopy(_load_memo[3])
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        backup = path.with_suffix(".json.bad")
        path.replace(backup)
        log.warning("%s is not valid JSON; moved it to %s and started with empty settings", path, backup)
        with _lock:
            _load_memo = None
        return {}
    with _lock:
        _load_memo = (*key, copy.deepcopy(data))
    return data


def save(data: dict[str, Any]) -> None:
    global _load_memo
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix="config.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps(data, indent=2, sort_keys=True) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        try:
            stat = path.stat()
        except OSError:
            _load_memo = None
        else:
            _load_memo = (str(path), stat.st_mtime_ns, stat.st_size, copy.deepcopy(data))


@contextlib.contextmanager
def edit() -> Iterator[dict[str, Any]]:
    """Read, change and save the config as one step, so threads cannot overwrite each other's changes."""
    with _lock:
        data = load()
        yield data
        save(data)


def aliases() -> dict[str, str]:
    """Saved short names -> model names. Keys are lower case, as `set_alias` stores them."""
    return {short.lower(): target for short, target in load().get("aliases", {}).items()}


def alias(name: str) -> str:
    """`name` as its alias points, or `name` itself. Aliases are matched ignoring case."""
    return aliases().get(name.lower(), name)


def set_alias(short: str, target: str) -> str:
    """Save `short` as a name for `target` and return the stored (lower case) short name."""
    short = short.strip().lower()
    with edit() as data:
        saved = data.setdefault("aliases", {})
        for existing in [key for key in saved if key != short and key.lower() == short]:
            del saved[existing]  # an older save may have used other capitals
        saved[short] = target
    return short


def remove_alias(name: str) -> str | None:
    """Forget the alias `name` when it has one and return the alias that was removed, else None."""
    with edit() as data:
        stored = {short.lower(): short for short in data.get("aliases", {})}
        key = stored.get(name.lower())
        if key is not None:
            data["aliases"].pop(key)
    return key


def update(**fields: Any) -> dict[str, Any]:
    with edit() as data:
        data.update(fields)
    return data
