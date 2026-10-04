"""Run System One decision models from Hugging Face behind the Jev API.

Commands mirror Ollama's. With no command it serves, and the first run opens a
setup screen to pick a model.
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import socket
import sys
import threading
import time
import webbrowser
from importlib.metadata import version

import httpx

from .. import client, config, store
from ..manager import canonical, canonical_or, default_model, lookup

# ---- model management -----------------------------------------------------------------------------


def confirm_trust(resolved: store.Resolved, assume_yes: bool) -> None:
    """Families that import Python from the model repo run it with your privileges. Ask once per commit."""
    if store.is_trusted(resolved):
        return
    code = sorted(f for f in resolved.files if f.endswith(".py"))
    print(f"\n{canonical(resolved)} runs Python code from its Hugging Face repo, with your user's privileges.")
    print(f"  commit  {resolved.revision}")
    print(f"  review  https://huggingface.co/{resolved.repo_id}/tree/{resolved.revision}")
    if code:
        print(f"  files   {', '.join(code[:8])}{' …' if len(code) > 8 else ''}")
    if not assume_yes:
        if not sys.stdin.isatty():
            raise SystemExit("not trusted; review the code, then pull again with --trust")
        if input("\nTrust this exact commit? Type 'yes': ").strip() != "yes":
            raise SystemExit("not trusted; nothing downloaded")
    store.trust(resolved)


def pull(name: str, trust: bool = False) -> store.Resolved:
    resolved = store.resolve(lookup(name))
    confirm_trust(resolved, trust)
    print(f"==> pulling {canonical(resolved)} ({resolved.family.name}) at {resolved.revision[:12]}", flush=True)
    store.download(resolved)
    if store.needs_prefetch(resolved):
        print("==> pulling base model", flush=True)
        store.prefetch(resolved)
    print(f"==> success: {canonical(resolved)}", flush=True)
    return resolved


def cmd_pull(args: argparse.Namespace) -> None:
    for name in args.model:
        pull(name, args.trust)


def cmd_list(args: argparse.Namespace) -> None:
    from ..library import tags

    rows = tags()
    if not rows:
        print("no models downloaded; try: ollajev pull " + config.DEFAULT_MODEL)
        return
    default = canonical_or(default_model())
    width = max(len(m["name"]) for m in rows)
    print(f"{'NAME':<{width}}  {'FAMILY':<16} {'SIZE':>8}  MODIFIED")
    for m in rows:
        mark = " *" if m["name"] == default else ""
        print(
            f"{m['name']:<{width}}  {m['details']['family']:<16} {m['size'] / 1e9:>6.2f} GB  {m['modified_at'][:10]}{mark}"
        )


def cmd_show(args: argparse.Namespace) -> None:
    try:
        resolved = store.resolve(lookup(args.model), online=False)
    except LookupError as exc:
        raise SystemExit(str(exc)) from None
    print(f"  model        {canonical(resolved)}")
    print(f"  family       {resolved.family.name}")
    print(f"  revision     {resolved.revision}")
    if resolved.weights:
        print(f"  file         {resolved.weights}")
    print(f"  released     {store.released(resolved.repo_id) or '-'}")
    print(f"  repo code    {store.trust_label(resolved)}")
    for key, value in resolved.family.limits(resolved).items():
        print(f"  {key:<12} {value}")
    print(f"  path         {store.local_path(resolved)}")


def cmd_rm(args: argparse.Namespace) -> None:
    for name in args.model:
        if client.server_running():
            client.call("DELETE", "/api/delete", {"model": name})
        else:
            removed = config.remove_alias(name)
            if removed is None:
                store.remove(store.resolve(lookup(name), online=False))
        print(f"deleted '{name}'")


def cmd_cp(args: argparse.Namespace) -> None:
    short = config.set_alias(args.destination, lookup(args.source))
    print(f"copied '{args.source}' to '{short}'")


def cmd_ps(args: argparse.Namespace) -> None:
    client.need_server()
    rows = client.call("GET", "/api/ps")["models"]
    if not rows:
        print("no models loaded")
        return
    width = max(len(m["name"]) for m in rows)
    print(f"{'NAME':<{width}}  {'FAMILY':<16} {'PROCESSOR':<10} UNTIL")
    for m in rows:
        until = m["expires_at"][11:19] if m["expires_at"] else "forever"
        print(f"{m['name']:<{width}}  {m['details']['family']:<16} {m['device']:<10} {until}")


def cmd_stop(args: argparse.Namespace) -> None:
    client.need_server()
    print(client.call("POST", "/api/stop", {"model": args.model})["status"])


# ---- serve ------------------------------------------------------------------------------------------

LISTEN_BACKLOG = 128
ANNOUNCE_TIMEOUT = 900  # seconds to wait for the server to answer before giving up the banner
ANNOUNCE_POLL = 0.5  # seconds between readiness probes while the server starts


def bind(host: str, port: int, scan: bool, tries: int = 50) -> tuple[socket.socket, int]:
    """Claim the port before the model loads, so a busy port fails in the first second."""
    candidates = range(port, port + tries) if scan else [port]
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    for candidate in candidates:
        sock = socket.socket(family)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, candidate))
        except OSError:
            sock.close()
            continue
        sock.listen(LISTEN_BACKLOG)
        sock.set_inheritable(True)
        return sock, candidate
    if not scan:
        raise SystemExit(f"port {port} is already in use on {host}; pick another with --port")
    raise SystemExit(f"no free port in {port}..{port + tries - 1}")


def configure_logging(path: str) -> None:
    """Everything at INFO to the file, warnings and worse to the console."""
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    file = logging.handlers.RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    file.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s  %(message)s"))
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.handlers = [file, console]
    # Model loads and unloads are what a person watching the server wants to see, e.g. after switching
    # models in the demo. Only INFO here: warnings already reach the console through the root handler.
    events = logging.StreamHandler()
    events.setFormatter(logging.Formatter("==> %(message)s"))
    events.addFilter(lambda record: record.levelno == logging.INFO)
    manager_log = logging.getLogger("ollajev.manager")
    manager_log.handlers = [events]


def listen_address(args: argparse.Namespace) -> tuple[str, int, bool]:
    """Host, port, and whether the port was asked for explicitly (an explicit port is not scanned)."""
    data = config.load()
    env_host, env_port = config.host()
    host_from_env = bool(os.environ.get("OLLAJEV_HOST"))
    host = args.host or data.get("host") or env_host
    explicit = args.port is not None or host_from_env
    if args.port:
        port = args.port
    elif host_from_env:
        port = env_port
    else:
        port = data.get("port") or env_port
    return host, port, explicit


def preload_name(model: str) -> str | None:
    """The canonical name to load at startup, or None (with a message) when it cannot load yet."""
    try:
        resolved = store.resolve(lookup(model), online=False)
    except LookupError:
        print(f"==> {model} is not downloaded; serving without a model. Pull one with: ollajev pull {model}")
        return None
    name = canonical(resolved)
    if not store.is_trusted(resolved):
        print(f"==> {name} runs repo code and is not trusted yet; run: ollajev pull {name}")
        return None
    return name


def exit_now() -> None:
    """End the process without waiting for the model manager's background threads. Python would otherwise join
    each pending Hugging Face request or download first, which is what made quitting slow. Settings are already
    saved, and a cut-off download resumes on the next pull."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def cmd_serve(args: argparse.Namespace) -> None:
    first_run = "default_model" not in config.load() and sys.stdin.isatty() and not args.model
    if getattr(args, "setup", False) or first_run:
        from .tui import manage

        if not manage():
            exit_now()
    # Importing the server stack takes a few seconds; say something before the terminal looks stuck.
    print("==> Starting the server …", flush=True)
    import uvicorn

    from ..server import api

    for check in (config.keep_alive, config.max_loaded_models, config.max_body_bytes):
        check()  # fail on a bad value now, not on a request
    host, port, explicit = listen_address(args)
    model = args.model or default_model()

    if not config.is_loopback(host) and not config.api_key():
        raise SystemExit(
            f"refusing to listen on {host}: the API can download, delete and load models, so it needs a key. "
            "Set OLLAJEV_API_KEY, or bind to 127.0.0.1."
        )
    args.log_file = args.log_file or str(config.log_dir() / "server.log")
    configure_logging(args.log_file)
    sock, port = bind(host, port, scan=not explicit)
    base = f"http://{client.url_host(host)}:{port}"
    config.update(server_url=base)

    name = preload_name(model)
    api.configure(
        preload_model=name,
        pin_preload_model=bool(name),  # the manager prints "Loading …" itself
        allowed=frozenset({"localhost", "127.0.0.1", "[::1]", host}) if config.is_loopback(host) else None,
    )
    threading.Thread(
        target=_announce_when_ready, args=(base, name, not args.no_browser, args.log_file), daemon=True
    ).start()
    uvicorn.Server(uvicorn.Config(api.app, log_config=None, log_level="info")).run(sockets=[sock])


def banner(base: str, model: str | None, log_file: str) -> str:
    key = "$OLLAJEV_API_KEY" if config.api_key() else "local"
    serving = model or f"none yet; pull one with: ollajev pull {config.DEFAULT_MODEL}"
    rows = [
        ("Serving", serving),
        ("Demo", f"{base}/demo"),
        ("API", "POST /v1/systemone   answer questions about a state"),
        ("", "GET  /v1/models      downloaded models"),
        ("Manage", "GET /api/tags · /api/ps · /api/show · POST /api/pull · DELETE /api/delete"),
        ("SDK", f"export TYPESAFE_BASE_URL={base}"),
        ("", f"export TYPESAFE_API_KEY={key}"),
        ("Logs", log_file),
    ]
    lines = ["", f"==> Ready on {base}", ""]
    for label, value in rows:
        lines.append(f"    {label:<9}{value}")
    lines += ["", "    A request can name any downloaded model; loads and unloads show below. Ctrl-C to stop.", ""]
    return "\n".join(lines)


def _announce_when_ready(base: str, model: str | None, open_browser: bool, log_file: str) -> None:
    deadline = time.monotonic() + ANNOUNCE_TIMEOUT
    while time.monotonic() < deadline:
        try:
            httpx.get(f"{base}/", timeout=1).raise_for_status()
            break
        except httpx.HTTPError:
            time.sleep(ANNOUNCE_POLL)
    else:
        return
    print(banner(base, model, log_file), flush=True)
    if open_browser:
        webbrowser.open(f"{base}/demo")


def cmd_setup(args: argparse.Namespace) -> None:
    args.setup = True
    cmd_serve(args)


def cmd_run(args: argparse.Namespace) -> None:
    from .repl import run

    run(args.model)


# ---- parser -----------------------------------------------------------------------------------------


EXAMPLES = """\
examples:
  ollajev                                  start the server (first run opens setup)
  ollajev pull SupersonicLabs/Julia-1      download a model
  ollajev pull Mapika/decider-2b-GGUF:Q8_0 download one quantized file
  ollajev run                              ask the default model questions
  ollajev service install                  run the server in the background at login

environment:
  OLLAJEV_HOST, OLLAJEV_KEEP_ALIVE, OLLAJEV_MAX_LOADED_MODELS, OLLAJEV_MODELS,
  OLLAJEV_DEVICE, OLLAJEV_HOME   (see README)
"""


def cmd_service(args: argparse.Namespace) -> None:
    from .. import service

    if args.action == "install":
        print(service.install())
    elif args.action == "uninstall":
        print(service.uninstall())
    elif args.action == "status":
        running, text = service.status()
        print(text)
        if not running:
            raise SystemExit(1)
    else:
        os.execvp(service.log_command()[0], service.log_command())


def build_parser() -> argparse.ArgumentParser:
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(prog="ollajev", description=__doc__, epilog=EXAMPLES, formatter_class=formatter)
    parser.add_argument("-V", "--version", action="version", version=f"ollajev {version('ollajev')}")
    parser.set_defaults(func=cmd_serve, model=None)

    def serve_options(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument("--host", help="bind address (default: 127.0.0.1, or OLLAJEV_HOST)")
        command_parser.add_argument("--port", type=int, help="port (default: the first free one from 8000)")
        command_parser.add_argument("--no-browser", action="store_true", help="do not open the demo page")
        command_parser.add_argument(
            "--log-file", help="request and error log (default: server.log in the OS log folder)"
        )

    serve_options(parser)
    sub = parser.add_subparsers(dest="command", metavar="<command>", title="commands")

    def command(name: str, help_text: str, example: str, func, aliases: tuple[str, ...] = ()):
        command_parser = sub.add_parser(
            name,
            help=help_text,
            description=help_text,
            aliases=list(aliases),
            epilog=f"example:\n  {example}",
            formatter_class=formatter,
        )
        command_parser.set_defaults(func=func)
        return command_parser

    command_parser = command(
        "serve", "start the server", "ollajev serve Mapika/decider-4b-GGUF:Q4_K_M --port 8000", cmd_serve
    )
    command_parser.add_argument("model", nargs="?", help="model to load at start (default: the saved default)")
    serve_options(command_parser)

    serve_options(
        command(
            "setup",
            "manage models (download, switch, ask, delete) and start the server",
            "ollajev setup",
            cmd_setup,
            aliases=("tui",),
        )
    )

    command_parser = command(
        "run", "ask a model questions from the terminal", "ollajev run SupersonicLabs/Julia-1", cmd_run
    )
    command_parser.add_argument("model", nargs="?", help="model (default: the saved default)")

    command_parser = command("pull", "download models", "ollajev pull Mapika/decider-2b-GGUF:Q8_0 --trust", cmd_pull)
    command_parser.add_argument("model", nargs="+", help="<user>/<repo>[:<quant>|:<file.gguf>]")
    command_parser.add_argument("--trust", action="store_true", help="trust the repo's Python code without asking")

    command("list", "list downloaded models", "ollajev list", cmd_list, aliases=("ls",))
    command("ps", "list loaded models", "ollajev ps", cmd_ps)

    command_parser = command(
        "show", "show a model's family, pinned commit and limits", "ollajev show SupersonicLabs/Julia-1", cmd_show
    )
    command_parser.add_argument("model")

    command_parser = command("rm", "delete downloaded models", "ollajev rm jaredpalmer/kev-0.6b", cmd_rm)
    command_parser.add_argument("model", nargs="+")

    command_parser = command("stop", "unload a running model", "ollajev stop SupersonicLabs/Julia-1", cmd_stop)
    command_parser.add_argument("model")

    command_parser = command("cp", "give a model another name", "ollajev cp SupersonicLabs/Julia-1 julia", cmd_cp)
    command_parser.add_argument("source")
    command_parser.add_argument("destination")

    command_parser = command(
        "service", "run the server in the background at login (macOS, Linux)", "ollajev service install", cmd_service
    )
    command_parser.add_argument("action", choices=["install", "uninstall", "status", "logs"])
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (KeyboardInterrupt, EOFError):
        print()
    except (LookupError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from None
