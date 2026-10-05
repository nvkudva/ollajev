"""CLI parsing, service file generation and config locations. No server, no models."""

from __future__ import annotations

import plistlib
import sys
from pathlib import Path

import pytest

from ollajev import config, service
from ollajev.ui import cli


@pytest.mark.parametrize(
    "argv,func",
    [
        ([], "cmd_serve"),
        (["serve", "user/repo"], "cmd_serve"),
        (["pull", "a/b", "c/d:Q8_0", "--trust"], "cmd_pull"),
        (["ls"], "cmd_list"),
        (["service", "status"], "cmd_service"),
    ],
)
def test_commands_route_to_their_handlers(argv, func):
    args = cli.build_parser().parse_args(argv)
    assert args.func.__name__ == func


def test_pull_takes_several_models():
    args = cli.build_parser().parse_args(["pull", "a/b", "c/d:Q8_0", "--trust"])
    assert args.model == ["a/b", "c/d:Q8_0"] and args.trust


def test_unknown_service_action_is_a_usage_error():
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(["service", "restart"])
    assert exc.value.code == 2


def test_version_flag(capsys):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--version"])
    assert capsys.readouterr().out.startswith("ollajev ")


def test_launchd_plist_is_valid_and_runs_serve(tmp_path):
    data = plistlib.loads(service.plist("/opt/bin/ollajev", tmp_path).encode())
    assert data["Label"] == service.LABEL
    assert data["ProgramArguments"] == ["/opt/bin/ollajev", "serve", "--no-browser"]
    assert data["RunAtLoad"] and data["KeepAlive"]
    assert data["StandardErrorPath"] == str(tmp_path / "service.err.log")


def test_plist_escapes_paths(tmp_path):
    data = plistlib.loads(service.plist("/Users/a&b/ollajev", tmp_path).encode())
    assert data["ProgramArguments"][0] == "/Users/a&b/ollajev"


def test_systemd_unit_quotes_the_executable():
    text = service.unit("/home/me/my tools/ollajev")
    assert 'ExecStart="/home/me/my tools/ollajev" serve --no-browser' in text
    assert "WantedBy=default.target" in text


def test_ollajev_home_overrides_config_and_logs(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAJEV_HOME", str(tmp_path))
    assert config.config_path() == tmp_path / "config.json"
    assert config.log_dir() == tmp_path / "logs"


def test_default_config_dir_is_dot_ollajev(monkeypatch):
    monkeypatch.delenv("OLLAJEV_HOME", raising=False)
    assert config.config_dir() == Path.home() / ".ollajev"
    assert config.log_dir() == Path.home() / ".ollajev" / "logs"


@pytest.mark.parametrize(
    "value,expected",
    [
        ("", ("127.0.0.1", 8000)),
        ("0.0.0.0", ("0.0.0.0", 8000)),
        ("127.0.0.1:9000", ("127.0.0.1", 9000)),
        ("http://localhost:9000/", ("localhost", 9000)),
        ("[::1]:9000", ("::1", 9000)),
    ],
)
def test_ollajev_host(monkeypatch, value, expected):
    monkeypatch.setenv("OLLAJEV_HOST", value)
    assert config.host() == expected


@pytest.mark.parametrize("text,seconds", [("300", 300), ("5m", 300), ("1h", 3600), ("-1", -1), (30, 30)])
def test_keep_alive_durations(text, seconds):
    assert config.parse_duration(text) == seconds


def test_a_failing_service_command_ends_with_its_message(monkeypatch):
    import subprocess

    def run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, "", "Failed to connect to bus: No medium found\n")

    monkeypatch.setattr(service.subprocess, "run", run)
    with pytest.raises(SystemExit, match="systemctl --user daemon-reload failed: Failed to connect to bus"):
        service._run("systemctl", "--user", "daemon-reload")
    assert service._run("systemctl", "--user", "is-active", "x", check=False).returncode == 1


@pytest.mark.skipif(sys.platform == "win32", reason="install.sh is for macOS and Linux")
def test_install_sh_on_linux_without_a_compiler_says_what_to_install(tmp_path):
    import shutil
    import subprocess

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "uname").write_text("#!/bin/sh\necho Linux\n")
    (bin_dir / "uname").chmod(0o755)
    for tool in ("dirname", "grep", "sed", "cat"):
        (bin_dir / tool).symlink_to(shutil.which(tool))
    root = Path(__file__).resolve().parent.parent
    result = subprocess.run(  # noqa: S603 our own install script, fixed arguments
        ["/bin/sh", str(root / "install.sh")],
        env={"PATH": str(bin_dir), "HOME": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1
    assert "needs a C and C++ compiler" in result.stderr and "sudo apt install build-essential" in result.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="install.sh is for macOS and Linux")
@pytest.mark.parametrize(
    ("tools_installed", "expected"),
    [(False, "needs the Xcode Command Line Tools"), (True, "cannot compile C++ (headers missing)")],
)
def test_install_sh_on_macos_without_working_command_line_tools_says_what_to_do(tmp_path, tools_installed, expected):
    import shutil
    import subprocess

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stubs = {"uname": "echo Darwin", "c++": "exit 1", "xcode-select": "exit 0" if tools_installed else "exit 2"}
    for name, body in stubs.items():
        (bin_dir / name).write_text(f"#!/bin/sh\n{body}\n")
        (bin_dir / name).chmod(0o755)
    for tool in ("dirname", "grep", "sed", "cat"):
        (bin_dir / tool).symlink_to(shutil.which(tool))
    root = Path(__file__).resolve().parent.parent
    result = subprocess.run(  # noqa: S603 our own install script, fixed arguments
        ["/bin/sh", str(root / "install.sh")],
        env={"PATH": str(bin_dir), "HOME": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1
    assert expected in result.stderr and "xcode-select --install" in result.stderr
