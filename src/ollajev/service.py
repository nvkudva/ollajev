"""Run the server in the background at login: launchd on macOS, a systemd user unit on Linux."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from . import config

LABEL = "com.ollajev.server"
UNIT = "ollajev.service"


def executable() -> str:
    """The installed `ollajev` command, so the service survives the shell that installed it."""
    found = shutil.which("ollajev")
    if found:
        return str(Path(found).resolve())
    return str(Path(sys.argv[0]).resolve())


def plist(exe: str, log_dir: Path) -> str:
    args = "".join(f"\n    <string>{escape(a)}</string>" for a in (exe, "serve", "--no-browser"))
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>{LABEL}</string>
  <key>ProgramArguments</key>
  <array>{args}
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>{escape(str(log_dir / "service.out.log"))}</string>
  <key>StandardErrorPath</key>
  <string>{escape(str(log_dir / "service.err.log"))}</string>
</dict>
</plist>
"""


def unit(exe: str) -> str:
    return f"""[Unit]
Description=Ollajev System One decision model server
After=network-online.target

[Service]
ExecStart="{exe}" serve --no-browser
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
"""


def _plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def _unit_path() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "systemd" / "user" / UNIT


def _run(*cmd: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a launchctl or systemctl command. With `check`, a failure ends with its own message rather than a
    traceback: e.g. `systemctl --user` in a container or SSH session with no user bus."""
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or f"exit code {result.returncode}"
        raise SystemExit(f"{' '.join(cmd)} failed: {detail}")
    return result


def _unsupported() -> SystemExit:
    return SystemExit(
        "background service is supported on macOS (launchd) and Linux (systemd); on Windows run: ollajev serve"
    )


def install() -> str:
    exe = executable()
    if sys.platform == "darwin":
        log_dir = config.log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        path = _plist_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        domain = f"gui/{os.getuid()}"
        _run("launchctl", "bootout", f"{domain}/{LABEL}", check=False)  # replace an older install
        path.write_text(plist(exe, log_dir))
        _run("launchctl", "bootstrap", domain, str(path))
        return f"installed {path}; the server starts now and at every login"
    if sys.platform.startswith("linux"):
        if not shutil.which("systemctl"):
            raise SystemExit("systemctl not found; run `ollajev serve` under your own supervisor instead")
        path = _unit_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(unit(exe))
        _run("systemctl", "--user", "daemon-reload")
        _run("systemctl", "--user", "enable", "--now", UNIT)
        return f"installed {path}; the server starts now and at every login (`loginctl enable-linger` keeps it after logout)"
    raise _unsupported()


def uninstall() -> str:
    if sys.platform == "darwin":
        path = _plist_path()
        _run("launchctl", "bootout", f"gui/{os.getuid()}/{LABEL}", check=False)
        path.unlink(missing_ok=True)
        return "service removed"
    if sys.platform.startswith("linux"):
        _run("systemctl", "--user", "disable", "--now", UNIT, check=False)
        _unit_path().unlink(missing_ok=True)
        _run("systemctl", "--user", "daemon-reload", check=False)
        return "service removed"
    raise _unsupported()


def status() -> tuple[bool, str]:
    """(running, description)."""
    if sys.platform == "darwin":
        if not _plist_path().exists():
            return False, "not installed"
        out = _run("launchctl", "print", f"gui/{os.getuid()}/{LABEL}", check=False)
        if out.returncode != 0:
            return False, "installed, not loaded"
        state = next(
            (line.split("=", 1)[1].strip() for line in out.stdout.splitlines() if "state =" in line), "unknown"
        )
        return state == "running", f"installed, {state}"
    if sys.platform.startswith("linux"):
        if not _unit_path().exists():
            return False, "not installed"
        out = _run("systemctl", "--user", "is-active", UNIT, check=False)
        state = out.stdout.strip() or "unknown"
        return state == "active", f"installed, {state}"
    return False, "not supported on this platform"


def log_command() -> list[str]:
    if sys.platform.startswith("linux"):
        return ["journalctl", "--user", "-u", UNIT, "-n", "100", "-f"]
    return ["tail", "-n", "100", "-f", str(config.log_dir() / "server.log")]
