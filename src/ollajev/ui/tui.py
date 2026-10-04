"""Model manager: download, switch, ask, unload and delete models, and start the server, on one screen."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar, NamedTuple

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, ProgressBar, Static

from .. import client, config, names, registry, service, store
from ..catalog import CATALOG
from ..library import tags
from ..manager import canonical, canonical_or, default_model, lookup
from . import dialogs, repl

try:
    import termios
except ImportError:  # Windows: no termios, and its console needs none of this
    termios = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

CSS = """
Screen { background: $background; }
#brand { height: 1; margin: 0 1; padding: 0 1; background: $panel; }
#brand-name { width: auto; text-style: bold; color: $accent; margin-right: 2; }
#brand-gap { width: 1fr; }
#brand Button.menu { width: 14; min-width: 0; padding: 0; content-align: left middle; text-align: left; background: transparent; border: none; }
#brand Button.menu:focus { background: transparent; text-style: none; }
#brand Button.menu:hover { background: $primary 40%; }
#models-panel {
    height: 1fr; margin: 0 1; background: $surface; border: round $primary 60%;
    border-title-color: $text; border-title-style: bold; border-subtitle-color: $text-muted;
}
DataTable { height: 1fr; background: $surface; }
DataTable > .datatable--header { background: $surface; color: $text-muted; text-style: bold; }
DataTable > .datatable--cursor { background: $primary 30%; text-style: bold; }
DataTable > .datatable--hover { background: $boost; }
#empty { height: 1fr; content-align: center middle; color: $text-muted; background: $surface; display: none; }
#status-row { height: 1; margin: 0 1; padding: 0 1; background: $panel; }
#status { width: 1fr; height: 1; color: $text-muted; }
#progress { width: 24; height: 1; margin-left: 2; display: none; }
#progress.show { display: block; }
#progress Bar { width: 1fr; }
#status.error { color: $error; }
#status.busy { color: $warning; }
#status.ok { color: $success; }
#selection-panel {
    height: auto; margin: 0 1; padding: 0 1; background: $surface; border: round $primary 60%;
    border-title-color: $text; border-title-style: bold;
}
#selection-panel.loaded { border: round $success 60%; }
#selection-row { height: auto; }
#selection-info { width: 1fr; height: 2; text-wrap: nowrap; text-overflow: ellipsis; }
#selection-panel .buttons { width: auto; margin-top: 0; }
#server-panel {
    height: auto; margin: 0 1; padding: 0 1; background: $surface; border: round $success 60%;
    border-title-color: $text; border-title-style: bold; display: none;
}
#server-panel.stopped { border: round $error 60%; }
#server-row { height: auto; }
#server-info { width: 1fr; }
#server-panel .buttons { width: auto; margin-top: 0; }

ModalScreen { align: center middle; background: $background 60%; }
.dialog {
    width: 68; max-width: 96%; height: auto; max-height: 90%; padding: 0 2 1 2; background: $panel;
    border: round $primary; border-title-align: left; border-title-style: bold; border-title-color: $text;
}
.dialog.danger { border: round $error; }
.hint { color: $text-muted; margin-top: 1; }
.dialog Input, .dialog Select { margin-bottom: 1; }
.field { height: auto; margin-bottom: 1; }
.field Static { width: 26; color: $text-muted; }
.field Input, .field Select { width: 1fr; margin-bottom: 0; }
.dialog TextArea { height: 6; margin-bottom: 1; }
.buttons { height: auto; margin-top: 1; align-horizontal: right; }
.buttons Button { width: 14; min-width: 0; padding: 0; content-align: left middle; text-align: left; margin-left: 1; background: transparent; border: none; }
.buttons Button:focus { text-style: bold; background: transparent; }
.buttons Button:hover { background: $primary 40%; }
.buttons Button.-primary { color: $accent; text-style: bold; }
.buttons Button.-error { color: $error; text-style: bold; }
.buttons Button.-success { color: $success; text-style: bold; }
.buttons .hint { width: 1fr; margin-top: 0; }
.buttons .gap { width: 1fr; }
#answers-box { height: auto; max-height: 14; border: round $primary 40%; padding: 0 1; }
#answers { color: $text; }
#ask-hint { color: $text-muted; }
#note { color: $text-muted; }
.wide { width: 112; }
#results { height: 20; }
"""

LANGUAGE_SHORT = {"English": "en", "Multilingual": "multi", "100+ languages": "100+"}


def terminal_background() -> tuple[float, float, float] | None:
    """The terminal's background colour as red, green, blue in 0..1, by asking it (OSC 11), or None when it does
    not answer within 0.3 s. Only before the app starts: after that Textual reads the terminal's replies."""
    if termios is None or not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None
    import select
    import tty

    stdin = sys.stdin.fileno()
    saved = termios.tcgetattr(stdin)
    reply = b""
    try:
        tty.setcbreak(stdin)
        sys.stdout.write("\x1b]11;?\x1b\\")
        sys.stdout.flush()
        deadline = time.monotonic() + 0.3
        while not (reply.endswith(b"\x07") or reply.endswith(b"\x1b\\")):
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([stdin], [], [], remaining)[0]:
                break
            reply += os.read(stdin, 64)
    finally:
        termios.tcsetattr(stdin, termios.TCSADRAIN, saved)
    # e.g. ESC ] 11 ; rgb:ffff/ffff/ffff BEL, with 1 to 4 hex digits a channel
    match = re.search(rb"rgb:([0-9a-fA-F]+)/([0-9a-fA-F]+)/([0-9a-fA-F]+)", reply)
    if not match:
        return None
    return tuple(int(channel, 16) / (16 ** len(channel) - 1) for channel in match.groups())  # type: ignore[return-value]


@functools.cache
def system_theme() -> str:
    """ansi-light or ansi-dark: Textual's themes that draw with the terminal's own background and palette, so the
    app follows the terminal's colour theme. Light or dark is the terminal's own background colour when it answers,
    else what it says in COLORFGBG, else the OS appearance. Asked once per process, before the app starts."""
    background_colour = terminal_background()
    if background_colour is not None:
        red, green, blue = background_colour
        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        return "ansi-light" if luminance > 0.5 else "ansi-dark"
    colours = os.environ.get("COLORFGBG", "")  # "15;0": foreground 15 on background 0
    background = colours.rpartition(";")[2]
    if background.isdigit():
        return "ansi-light" if int(background) in (7, 15) else "ansi-dark"
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["/usr/bin/defaults", "read", "-g", "AppleInterfaceStyle"], capture_output=True, text=True, timeout=1
            )
        except (OSError, subprocess.SubprocessError):
            return "ansi-dark"
        return "ansi-dark" if result.stdout.strip() == "Dark" else "ansi-light"
    return "ansi-dark"


def duration(seconds: float) -> str:
    """A time left, coarse: 45s, 12m, 1h 05m."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m"


def open_terminal(command: list[str]) -> bool:
    """Run `command` in a new tab of the terminal this app runs in: a tmux window, a WezTerm or iTerm2 tab, else
    a Terminal window on macOS. False when there is no known way, so the caller can say what to run instead."""
    line = shlex.join(command)
    program = os.environ.get("TERM_PROGRAM", "")
    if os.environ.get("TMUX"):
        launch = ["tmux", "new-window", line]
    elif program == "WezTerm":
        launch = ["wezterm", "cli", "spawn", "--", *command]
    elif sys.platform == "darwin":
        quoted = line.replace("\\", "\\\\").replace('"', '\\"')
        if program == "iTerm.app":
            script = f'tell application "iTerm2" to tell current window to create tab with default profile command "{quoted}"'
        else:
            script = f'tell application "Terminal"\nactivate\ndo script "{quoted}"\nend tell'
        launch = ["osascript", "-e", script]
    else:
        return False
    try:
        subprocess.Popen(  # noqa: S603 a fixed launcher with our own command
            launch,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        return False
    return True


def hugging_face_page(repo_id: str) -> str:
    return f"https://huggingface.co/{repo_id}"


class NoWaitExecutor(ThreadPoolExecutor):
    """The event loop's default executor. On exit asyncio waits for every thread in it to finish, and a Hugging Face
    lookup still running would keep the terminal blank after `s` or `q`. This one lets them finish on their own."""

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        super().shutdown(wait=False, cancel_futures=True)


class Row(NamedTuple):
    name: str  # what pull, rm and the config call it
    label: Text  # what the Model column shows
    size: str
    estimated: bool  # the size is the catalog's or Hugging Face's figure, not the files on disk
    downloads: str  # Hugging Face download count, empty until it is read or when it cannot be
    runtime: str  # what runs the weights: llama.cpp, ONNX or PyTorch
    languages: str  # the catalog's languages, empty for a model outside the catalog


# App actions on the top bar as (key, label, action): these on the left, MENU_END on the right.
# Actions on one model live in the Selected panel, on the server in the Server panel.
MENU = [
    ("n", "Add", "add"),
    ("/", "Filter", "filter"),
]
MENU_END = [
    ("o", "Settings", "options"),
    ("q", "Quit", "quit_app"),
]


# The Selected panel's buttons: (label, action, variant). Each shows only when it applies to the cursor row.
SELECTION_BUTTONS = [
    ("Download", "pull", "primary", "p"),
    ("Serve", "serve_model", "success", "s"),
    ("Default", "set_default", "default", "d"),
    ("Unload", "unload", "default", "u"),
    ("Delete", "remove", "error", "x"),
    ("Info", "info", "default", "i"),
]
SERVER_BUTTONS = [
    ("Demo", "open_demo", "primary", "w"),
    ("Logs", "logs", "default", "l"),
    ("Restart", "restart_server", "default", "R"),
    ("Stop", "stop_server", "error", "S"),
]
DESCRIPTIONS = {entry.name.partition(":")[0]: entry.description for entry in reversed(CATALOG)}


class Models(App[bool]):
    TITLE = "ollajev"
    CSS = CSS
    ENABLE_COMMAND_PALETTE = False  # its main use is picking a theme; the app follows the terminal's instead
    # The status row shows the keys for the cursor row; ? lists them all.
    BINDINGS: ClassVar = [
        Binding("d", "set_default", "Default"),
        Binding("i", "info", "Info"),
        Binding("n", "add", "Add"),
        Binding("slash", "filter", "Filter"),
        Binding("question_mark", "help", "Help"),
        Binding("q", "quit_app", "Quit"),
        Binding("p", "pull", "Pull", show=False),
        Binding("x", "remove", "Remove", show=False),
        Binding("s", "serve_model", "Serve", show=False),
        Binding("R", "restart_server", "Restart", show=False),
        Binding("S", "stop_server", "Stop", show=False),
        Binding("u", "unload", "Unload", show=False),
        Binding("a", "alias", "Alias", show=False),
        Binding("o", "options", "Options", show=False),
        Binding("b", "service", "Service", show=False),
        Binding("w", "open_demo", "Demo", show=False),
        Binding("l", "logs", "Logs", show=False),
        Binding("ctrl+r", "reload_list", "Refresh", show=False),
        Binding("e", "last_error", "Error", show=False),
        Binding("escape", "cancel_job", "Cancel", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.names: list[str] = []
        self.rows: dict[str, Row] = {}  # model name -> its row as listed
        self.summary = ""  # the idle status line: disk use and filter
        self.filter_text = ""  # `/` shows only the models whose name contains it
        self.fetching_quants = False
        self.busy = False
        self.cancel = threading.Event()  # set by Esc; a download in progress stops at its next update
        self.downloading = False  # only downloads can be cancelled; loads and asks run to the end
        self.local: tuple[str, Any, Any] | None = None  # model, ask, release: loaded in this process
        self.last_error: str | None = None  # kept on the status line until the next job succeeds
        self.loading: str | None = None  # the model being loaded into this process, if any
        # One load at a time: two Ask dialogs opened in a row must not hold two models in memory.
        self.load_lock = threading.Lock()
        self.quants: dict[str, list[store.Variant]] = {}  # catalog GGUF repo -> all its quants on Hugging Face
        self.totals: dict[str, int] = {}  # repo -> its download count on Hugging Face

        self.server: subprocess.Popen[bytes] | None = None  # a server this window started, if any
        self.server_model: str | None = None  # what that server was started with
        self.snapshot_cache: tuple[dict[str, Any], str, bool, set[str]] | None = None  # to redraw on a theme change

    # ---- layout and data --------------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Horizontal(id="brand"):
            yield Static("🦒 ollajev", id="brand-name")
            for key, text, action in MENU:
                yield self.menu_item(key, text, action)
            yield Static(id="brand-gap")
            for key, text, action in MENU_END:
                yield self.menu_item(key, text, action)
        with Vertical(id="models-panel") as panel:
            panel.border_title = "Models"
            yield DataTable(cursor_type="row")
            yield Static("", id="empty")
        with Vertical(id="selection-panel") as selection_panel:
            selection_panel.border_title = "Selected"
            with Horizontal(id="selection-row"):
                yield Static("", id="selection-info")
                yield dialogs.buttons(*SELECTION_BUTTONS)
        with Vertical(id="server-panel") as server_panel:
            server_panel.border_title = "Server"
            with Horizontal(id="server-row"):
                yield Static("", id="server-info")
                yield dialogs.buttons(*SERVER_BUTTONS)
        with Horizontal(id="status-row"):
            yield Static("", id="status")
            yield ProgressBar(id="progress", show_eta=False, show_percentage=False)

    @staticmethod
    def menu_item(key: str, text: str, action: str) -> Button:
        item = Button(dialogs.label(text, key), id=f"do-{action}", compact=True, classes="menu")
        item.can_focus = False  # the list keeps the keyboard; the menu is for the mouse and its keys
        return item

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """A key button runs its action like the key; then the list takes the keys again."""
        await dialogs.Clickable.on_button_pressed(self, event)  # type: ignore[arg-type]
        self.query_one(DataTable).focus()

    def on_mount(self) -> None:
        asyncio.get_running_loop().set_default_executor(NoWaitExecutor())
        self.theme = system_theme()
        self.reload()
        self.say_idle()
        self.load_quants()
        self.load_downloads()
        self.set_interval(3, self.auto_refresh)

    def colour(self, role: str) -> str:
        """A colour of the current theme (success, warning, error, accent…) as a Rich colour: one of the terminal's
        own palette colours, as the theme is an ANSI one."""
        value = self.get_css_variables().get(role, "")
        if value.startswith("ansi_"):
            return value.removeprefix("ansi_")
        return value if value.startswith("#") else "default"

    @work(group="quants")
    async def load_quants(self) -> None:
        """List every quant of the catalog's GGUF repos, not only the curated ones. Offline, the list stays as is."""
        self.fetching_quants = True
        self.say_idle()
        repos = list(dict.fromkeys(e.name.partition(":")[0] for e in CATALOG if ":" in e.name))
        found = await asyncio.gather(*(asyncio.to_thread(dialogs.variants, repo) for repo in repos))
        self.quants = {repo: vs for repo, vs in zip(repos, found, strict=True) if vs}
        self.fetching_quants = False
        self.reload()

    @work(group="downloads")
    async def load_downloads(self) -> None:
        """Fill the Downloads column for each listed repo. Offline, it stays empty."""
        repos = list(dict.fromkeys(row.partition(":")[0] for row in self.names or [e.name for e in CATALOG]))
        found = await asyncio.gather(*(asyncio.to_thread(store.listing, repo) for repo in repos))
        self.totals = {repo: total for repo, (total, _) in zip(repos, found, strict=True) if total is not None}
        self.reload()

    def say(self, text: str | Text, kind: str = "busy") -> None:
        """Put `text` on the status row: busy in the warning colour, error in red, ok in green, idle muted."""
        icon = {"busy": "◌ ", "error": "✕ ", "ok": "✓ "}.get(kind, "")
        status = self.query_one("#status", Static)
        status.set_classes(kind)
        status.update(Text.assemble((icon, "bold"), text))

    def say_idle(self) -> None:
        """The status row between jobs: the last error until a job succeeds; else what is going on (a filter,
        quants being fetched), then the keys for what the cursor row can do."""
        if self.last_error:
            first_line = self.last_error.splitlines()[0]
            self.say(Text.assemble(first_line, ("  ·  ", "dim"), self.key_hints([("e", "for details")])), "error")
            return
        line = Text()
        if self.fetching_quants:
            line.append("fetching quants …  ", style=self.colour("warning"))
        if self.filter_text:
            line.append(f"▽ '{self.filter_text}'  ", style=f"bold {self.colour('accent')}")
        line.append_text(self.key_hints(self.selection_keys()))
        self.say(line, "idle")

    def selection_keys(self) -> list[tuple[str, str]]:
        """(key, what it does) for the cursor row, in the order a user tends to need them."""
        keys = [("↑↓", "pick")]
        name = self.selected()
        if name and self.snapshot_cache:
            have, default, _, loaded = self.snapshot_cache
            if name not in have:
                keys += [("enter", "download & use"), ("p", "download only")]
            else:
                if name != default:
                    keys += [("enter", "make default")]
                keys += [("s", "serve")]
                if name in loaded:
                    keys += [("u", "unload")]
                keys += [("a", "short name"), ("x", "delete")]
            keys += [("i", "info")]
        if self.filter_text:
            keys += [("esc", "clear filter")]
        return [*keys, ("?", "all keys")]

    def key_hints(self, keys: list[tuple[str, str]]) -> Text:
        """Keys in bold accent, what they do muted, spaced apart."""
        accent = self.colour("accent")
        return Text("   ").join(Text.assemble((key, f"bold {accent}"), " ", (what, "dim")) for key, what in keys)

    def action_help(self) -> None:
        self.push_screen(dialogs.Info("Keys", keys_help(self.colour("accent")), about=True))

    def action_last_error(self) -> None:
        if self.last_error:
            self.push_screen(dialogs.Info("Last error", self.last_error, danger=True))
        else:
            self.notify("No errors so far")

    def loaded(self, server_up: bool) -> set[str]:
        if server_up:
            try:
                return {m["name"] for m in client.call("GET", "/api/ps")["models"]}
            except SystemExit:
                return set()
        return {self.local[0]} if self.local else set()

    def snapshot(self) -> tuple[dict[str, Any], str, bool, set[str]]:
        """What the list shows: downloads, default, server up, loaded models. Slow: disk scan and server probes."""
        have = {m["name"]: m for m in tags()}
        default = canonical_or(default_model())
        server_up = client.server_running()  # each probe can wait 0.5 s, so probe once per reload
        return have, default, server_up, self.loaded(server_up)

    def reload(self) -> None:
        self.render_list(*self.snapshot())

    @work(thread=True, exclusive=True, group="auto-refresh")
    def auto_refresh(self) -> None:
        """Every 3 s: pick up a server, download or default changed elsewhere, off the UI thread."""
        if self.busy or self.loading or len(self.screen_stack) > 1:
            return
        data = self.snapshot()
        self.call_from_thread(self.render_list, *data)

    def visible_columns(self) -> list[str]:
        """Column keys shown at the current width. Narrow terminals keep Status, Model and Size;
        Downloads/Runtime/Lang return as the window widens (progressive disclosure)."""
        try:
            width = self.size.width or 160
        except Exception:
            width = 160
        if width < 90:
            return ["state", "model", "size"]
        if width < 110:
            return ["state", "model", "size", "downloads"]
        return ["state", "model", "size", "downloads", "runtime", "lang"]

    def render_list(self, have: dict[str, Any], default: str, server_up: bool, loaded: set[str]) -> None:
        self.snapshot_cache = (have, default, server_up, loaded)
        rows = self.list_rows(have)
        if self.filter_text:
            rows = [row for row in rows if self.filter_text.lower() in row.name.lower()]
        table = self.query_one(DataTable)
        keep = self.selected()
        table.clear(columns=True)
        visible = self.visible_columns()
        model_width = min(50, max([len(row.label) for row in rows] + [12]))
        if "state" in visible:
            table.add_column("Status", width=13, key="state")
        if "model" in visible:
            table.add_column("Model", width=model_width, key="model")
        if "size" in visible:
            table.add_column(Text("Size", justify="right"), width=9, key="size")
        if "downloads" in visible:
            table.add_column(Text("Downloads", justify="right"), width=10, key="downloads")
        if "runtime" in visible:
            table.add_column("Runtime", width=10, key="runtime")
        if "lang" in visible:
            table.add_column("Lang", width=6, key="lang")
        self.names, self.rows = [], {}
        for row in rows:
            on_disk, is_loaded = row.name in have, row.name in loaded
            self.rows[row.name] = row
            cells: dict[str, Any] = {
                "state": self.state_pills(row.name == default, is_loaded, on_disk),
                "model": row.label,
                "size": Text(row.size, justify="right", style="dim italic" if row.estimated else ""),
                "downloads": Text(row.downloads, justify="right", style="dim"),
                "runtime": Text(row.runtime, style="dim"),
                "lang": Text(LANGUAGE_SHORT.get(row.languages, row.languages), style="dim"),
            }
            table.add_row(*(cells[key] for key in visible), key=row.name)
            self.names.append(row.name)
        table.display = bool(rows)
        empty = self.query_one("#empty", Static)
        empty.display = not rows
        if self.filter_text:
            empty.update(f"No models match '{self.filter_text}'  ·  / to change it, esc to clear it")
        else:
            empty.update("No models yet  ·  n adds one from Hugging Face")
        if keep in self.names:
            table.move_cursor(row=self.names.index(keep))
        self.show_server_panel(server_up, loaded)
        self.show_selection(default, loaded, have)
        self.summary = self.describe(have)
        self.query_one("#models-panel").border_subtitle = self.summary
        if not self.busy:
            self.say_idle()

    def show_selection(self, default: str, loaded: set[str], have: dict[str, Any]) -> None:
        """The Selected panel: the cursor row's model, its state and facts, and buttons for what applies to it."""
        name = self.selected()
        row = self.rows.get(name) if name else None
        on_disk, is_loaded, is_default = name in have, name in loaded, name == default
        applies = {
            "pull": not on_disk,
            "serve_model": on_disk,
            "set_default": on_disk and not is_default,
            "unload": is_loaded,
            "remove": on_disk,
            "info": True,
        }
        for _, action, _, _ in SELECTION_BUTTONS:
            self.query_one(f"#do-{action}").display = row is not None and applies[action]
        self.query_one("#selection-panel").set_class(is_loaded, "loaded")
        info = Text()
        if row is None:
            info.append("No model selected", style="dim")
        else:
            repo, _, quant = row.name.partition(":")
            info.append(repo, style="bold")
            if quant:
                info.append(f":{quant}", style="dim")
            info.append("   ")
            info.append_text(self.state_pills(is_default, is_loaded, on_disk))
            size = f"{row.size} on disk" if on_disk else f"~{row.size} to download"
            downloads = f"{row.downloads} downloads" if row.downloads else ""
            facts = [size, row.runtime, row.languages, downloads, DESCRIPTIONS.get(repo, "")]
            info.append("\n" + "  ·  ".join(fact for fact in facts if fact), style="dim")
        self.query_one("#selection-info", Static).update(info)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Cursor moved: show the new row in the Selected panel and its keys on the status row."""
        if self.snapshot_cache:
            have, default, _, loaded = self.snapshot_cache
            self.show_selection(default, loaded, have)
            if not self.busy:
                self.say_idle()

    def state_pills(self, is_default: bool, is_loaded: bool, on_disk: bool) -> Text:
        """★ default and ● loaded in colour; ✓ downloaded or ○ available (not downloaded) dim. A loaded default
        shows as ★● loaded, so the column stays narrow."""
        labels = Text()
        if is_default and is_loaded:
            labels.append("★", style=f"bold {self.colour('warning')}")
            labels.append("● loaded", style=f"bold {self.colour('success')}")
        elif is_default:
            labels.append("★ default", style=f"bold {self.colour('warning')}")
        elif is_loaded:
            labels.append("● loaded", style=f"bold {self.colour('success')}")
        elif on_disk:
            labels.append("✓ downloaded", style="dim")
        else:
            labels.append("○ available", style="dim")
        return labels

    def action_serve_model(self) -> None:
        """Serve the selected model: it becomes the default and the server starts (or restarts) with it."""
        name = self.selected()
        if not name or self.refuse_while_busy():
            return
        if not self.downloaded(name):
            self.notify("Download it first (p or Enter)", severity="warning")
            return
        config.update(default_model=canonical_or(name))
        self.action_serve()

    def list_rows(self, have: dict[str, Any]) -> list[Row]:
        """The catalog, each GGUF repo's other quants under its last curated entry, then any other download."""
        curated = {entry.name for entry in CATALOG}
        rows = []
        for index, entry in enumerate(CATALOG):
            repo, _, quant = entry.name.partition(":")
            label = Text(repo)
            if quant:
                label.append(f":{quant}", style="dim")
            size, estimated = self.size_of(entry.name, have, entry.size_gb * 1e9)
            rows.append(
                Row(
                    entry.name,
                    label,
                    size,
                    estimated,
                    self.count_of(entry.name),
                    self.runtime_of(entry.name, have),
                    entry.languages,
                )
            )
            later_entries = CATALOG[index + 1 :]
            last_of_repo = all(other.name.partition(":")[0] != repo for other in later_entries)
            if not last_of_repo:
                continue
            others = [variant for variant in self.quants.get(repo, []) if variant.name not in curated]
            for position, variant in enumerate(others):
                branch = "└" if position == len(others) - 1 else "├"
                label = Text(f"  {branch} {variant.name.partition(':')[2] or variant.name}", style="dim")
                size, estimated = self.size_of(variant.name, have, variant.size)
                rows.append(
                    Row(
                        variant.name,
                        label,
                        size,
                        estimated,
                        self.count_of(variant.name),
                        self.runtime_of(variant.name, have),
                        entry.languages,
                    )
                )
        listed = {row.name for row in rows}
        for name, model in have.items():
            if name not in listed:
                rows.append(
                    Row(
                        name,
                        Text(name),
                        dialogs.human(model["size"]),
                        False,
                        self.count_of(name),
                        self.runtime_of(name, have),
                        "",
                    )
                )
        return rows

    def runtime_of(self, name: str, have: dict[str, Any]) -> str:
        """The Runtime cell: what runs the weights. A downloaded one has the format of the file its family loads
        recorded, which is authoritative; a listed one is read from its tag."""
        if name in have:
            return names.RUNTIMES.get(have[name]["details"]["format"], names.RUNTIMES["safetensors"])
        return names.runtime_of(name.partition(":")[2] or None)

    def count_of(self, name: str) -> str:
        """The Downloads cell for a model name: empty until read, or when Hugging Face has no count for it."""
        total = self.totals.get(name.partition(":")[0])
        return dialogs.count(total) if total is not None else ""

    @staticmethod
    def size_of(name: str, have: dict[str, Any], estimate: float) -> tuple[str, bool]:
        """The size on disk when downloaded, else the estimate, and whether it is one."""
        if name in have:
            return dialogs.human(have[name]["size"]), False
        return dialogs.human(estimate), True

    def describe(self, have: dict[str, Any]) -> str:
        on_disk = sum(model["size"] for model in have.values())
        return f"{len(have)} downloaded · {dialogs.human(on_disk)}"

    @work
    async def action_filter(self) -> None:
        """`/`: show only matching models; an empty filter shows them all again."""
        text = await self.push_screen_wait(dialogs.Prompt("Filter models", "part of a name; empty shows all"))
        self.filter_text = text or ""
        self.reload()

    def action_reload_list(self) -> None:
        self.reload()

    def selected(self) -> str | None:
        table = self.query_one(DataTable)
        if not self.names:
            return None
        row = min(max(table.cursor_row, 0), len(self.names) - 1)
        return self.names[row]

    def downloaded(self, name: str) -> bool:
        try:
            store.resolve(lookup(name), online=False)
        except (LookupError, ValueError):
            return False
        return True

    def connection(self, model: str) -> Any:
        """The ask function for `model`, loading it in this process unless a server is running."""
        with self.load_lock:
            if self.local and self.local[0] != model:
                self.release()
            if self.local is None:
                self.loading = model
                try:
                    ask, release = repl.connect(model, lambda text: None)
                finally:
                    self.loading = None
                self.local = (model, ask, release)
            return self.local[1]

    def refuse_while_busy(self) -> bool:
        """True, with a message, when a job or a model load is running and the action must wait."""
        if self.busy:
            self.notify("Wait for the current job to finish (esc cancels a download)", severity="warning")
            return True
        if self.loading:
            self.notify(f"Wait for {self.loading} to finish loading", severity="warning")
            return True
        return False

    def release(self) -> None:
        if self.local:
            self.local[2]()
            self.local = None

    # ---- jobs: one at a time, off the UI thread ---------------------------------------------------

    async def job(self, text: str, work_fn: Any) -> Any:
        if self.busy:
            self.notify("Wait for the current job to finish", severity="warning")
            return None
        self.busy = True
        self.cancel.clear()
        self.say(text)
        try:
            result = await work_fn()
            self.last_error = None
            return result
        except store.Cancelled:
            self.notify("Download cancelled; the next pull resumes it", severity="warning")
            return None
        except Exception as exc:
            log.exception("%s failed", text)
            self.last_error = f"{text.rstrip(' …')} failed:\n{exc or type(exc).__name__}"
            return None  # say_idle puts it on the status row until the next job succeeds
        finally:
            self.busy = False
            self.reload()
            self.say_idle()

    def action_cancel_job(self) -> None:
        if self.downloading:
            self.cancel.set()
            self.say("Cancelling download …")
        elif self.busy:
            self.notify("Only downloads can be cancelled", severity="warning")
        elif self.filter_text:
            self.filter_text = ""
            self.reload()

    async def fetch(self, name: str) -> store.Resolved | None:
        resolved = await asyncio.to_thread(lambda: store.resolve(lookup(name)))
        if resolved.family.runs_repo_code and not registry.is_trusted(resolved):
            code = sorted(f for f in resolved.files if f.endswith(".py"))
            body = (
                f"{canonical(resolved)} runs Python code from its Hugging Face repo, with your user's privileges.\n\n"
                f"commit  {resolved.revision}\nreview  https://huggingface.co/{resolved.repo_id}/tree/{resolved.revision}\n"
                f"files   {', '.join(code[:6])}{' …' if len(code) > 6 else ''}\n\nTrust this exact commit?"
            )
            if not await self.push_screen_wait(dialogs.Confirm("This model runs repo code", body)):
                self.notify("Not trusted; nothing downloaded", severity="warning")
                return None
            registry.trust(resolved)
        self.downloading = True
        progress = await self.show_progress(resolved)
        try:
            await asyncio.to_thread(store.download, resolved, self.cancel)
            if store.needs_prefetch(resolved):
                self.say("Downloading the base model …")
                await asyncio.to_thread(store.prefetch, resolved, self.cancel)
        finally:
            progress.stop()
            self.downloading = False
        return resolved

    async def show_progress(self, resolved: store.Resolved) -> Any:
        """Put the download's bytes, percent, speed and time left on the status row, with a bar at its right end,
        every half second; returns the timer (stopping it also hides the bar). An unknown size pulses the bar."""
        name = canonical(resolved)
        try:
            total = await asyncio.to_thread(store.download_size, resolved)
        except Exception:  # progress is optional; the download itself reports real errors
            total = 0
        start = store.bytes_on_disk(resolved.repo_id)
        started = time.monotonic()
        bar = self.query_one("#progress", ProgressBar)
        bar.update(total=total or None, progress=0)
        bar.add_class("show")

        def update() -> None:
            if self.cancel.is_set():  # the download stops at its next check; say so until it does
                self.say("Cancelling download …")
                return
            done = store.bytes_on_disk(resolved.repo_id) - start
            speed = done / max(time.monotonic() - started, 0.001)
            if total:
                percent = min(100, done * 100 // total)
                amount = f"{dialogs.human(done)} / {dialogs.human(total)} · {percent}%"
                left = f" · {duration((total - done) / speed)} left" if speed and done < total else ""
            else:
                amount, left = dialogs.human(done), ""
            self.say(f"Downloading {name}: {amount} · {dialogs.human(speed)}/s{left} · esc cancels")
            with contextlib.suppress(Exception):  # the app may be closing
                bar.update(progress=min(done, total) if total else done)

        def hide() -> None:
            with contextlib.suppress(Exception):
                bar.remove_class("show")

        update()
        timer = self.set_interval(0.5, update)
        stop = timer.stop

        def stop_and_hide() -> None:
            stop()
            hide()

        timer.stop = stop_and_hide  # type: ignore[method-assign]
        return timer

    # ---- actions ----------------------------------------------------------------------------------

    @work
    async def on_data_table_row_selected(self) -> None:
        """Enter: download if needed, then make it the default."""
        name = self.selected()
        if not name:
            return
        on_disk = self.downloaded(name)
        if not on_disk:
            size = self.rows[name].size if name in self.rows else "an unknown size"
            question = dialogs.Confirm(
                f"Download {name}?", f"It is {size}. It becomes the default model.", default=True
            )
            if not await self.push_screen_wait(question):
                return

        async def use() -> None:
            if resolved := await self.fetch(name):
                config.update(default_model=canonical(resolved))
                if not on_disk:  # a finished download is news; the new default shows in the list and panel
                    self.notify(f"Downloaded {canonical(resolved)}; it is now the default")

        await self.job(f"Preparing {name} …", use)

    def action_set_default(self) -> None:
        name = self.selected()
        if not name:
            return
        if not self.downloaded(name):
            self.notify("Download it first (p or Enter)", severity="warning")
            return
        config.update(default_model=canonical_or(name))
        self.reload()

    @work
    async def action_pull(self) -> None:
        if name := self.selected():

            async def pull() -> None:
                if resolved := await self.fetch(name):
                    self.notify(f"Downloaded {canonical(resolved)}")

            await self.job(f"Preparing {name} …", pull)

    @work
    async def action_add(self) -> None:
        name = await self.push_screen_wait(dialogs.AddModel())
        if name:

            async def pull() -> None:
                if resolved := await self.fetch(name):
                    self.notify(f"Downloaded {canonical(resolved)}")

            await self.job(f"Preparing {name} …", pull)

    @work
    async def action_ask(self) -> None:
        name = self.selected()
        if not name:
            return
        if not self.downloaded(name):
            self.notify("Download the model first (Enter)", severity="warning")
            return
        await self.push_screen_wait(dialogs.Ask(name))
        self.reload()

    @work
    async def action_unload(self) -> None:
        name = self.selected()
        if not name or self.refuse_while_busy():
            return
        if client.server_running():
            await asyncio.to_thread(client.call, "POST", "/api/stop", {"model": name})
        elif self.local and self.local[0] == name:
            await asyncio.to_thread(self.release)
        else:
            self.notify(f"{name} is not loaded")
        self.reload()

    @work
    async def action_remove(self) -> None:
        name = self.selected()
        if not name or self.refuse_while_busy():
            return
        if not self.downloaded(name):
            self.notify(f"{name} is not downloaded; nothing to delete")
            return
        if not await self.push_screen_wait(
            dialogs.Confirm(f"Delete {name}?", "The downloaded weights are removed from disk.")
        ):
            return

        async def remove() -> None:
            if self.local and self.local[0] == name:
                await asyncio.to_thread(self.release)
            if client.server_running():
                await asyncio.to_thread(client.call, "DELETE", "/api/delete", {"model": name})
            else:
                resolved = await asyncio.to_thread(store.resolve, lookup(name), online=False)
                await asyncio.to_thread(store.remove, resolved)

        await self.job(f"Deleting {name} …", remove)

    @work
    async def action_alias(self) -> None:
        name = self.selected()
        if not name:
            return
        short = await self.push_screen_wait(dialogs.Prompt(f"Short name for {name}", "julia"))
        if not short:
            return
        if message := self.short_name_error(short):
            self.notify(message, severity="error")
            return
        current = config.aliases().get(short.lower())
        if current and current != name:
            replace = dialogs.Confirm(f"Replace '{short.lower()}'?", f"It means {current} now.")
            if not await self.push_screen_wait(replace):
                return
        config.set_alias(short, name)
        self.notify(f"'{short.lower()}' now means {name}")

    def short_name_error(self, short: str) -> str | None:
        """Why `short` cannot be a short name, or None when it can. Aliases match ignoring case, so a name that
        differs from a model's own name only by case is refused too."""
        if any(char.isspace() for char in short):
            return f"'{short}' cannot be a short name: no spaces in it"
        if short.lower() in {name.lower() for name in self.names}:
            return f"'{short}' is already a model's own name"
        return None

    @work
    async def action_info(self) -> None:
        name = self.selected()
        if not name:
            return
        try:
            resolved = store.resolve(lookup(name), online=False)
        except (LookupError, ValueError):
            entry = next((e for e in CATALOG if e.name == name), None)
            await self.push_screen_wait(
                dialogs.Info(
                    name,
                    f"{entry.description if entry else 'unknown model'}\n\nNot downloaded. Press Enter or Download to get it.",
                    link=hugging_face_page(name.partition(":")[0]),
                )
            )
            return
        trusted = registry.trust_label(resolved)
        limits = ", ".join(f"{k} {v}" for k, v in resolved.family.limits(resolved).items()) or "none recorded"
        body = (
            f"family   {resolved.family.name}\ncommit   {resolved.revision}\nfile     {resolved.weights or 'safetensors'}\n"
            f"code     {trusted}\nlimits   {limits}\npath     {store.local_path(resolved)}"
        )
        await self.push_screen_wait(dialogs.Info(canonical(resolved), body, link=hugging_face_page(resolved.repo_id)))

    @work
    async def action_options(self) -> None:
        if values := await self.push_screen_wait(dialogs.Settings()):
            config.update(**values)
            if self.server_alive():
                self.notify("Restart the server (R) to use the new settings")

    @work
    async def action_service(self) -> None:
        if self.refuse_while_busy():
            return
        try:
            running, _ = await asyncio.to_thread(service.status)
            if running:
                if await self.push_screen_wait(dialogs.Confirm("Background service", "Stop and remove it?")):
                    self.notify(await asyncio.to_thread(service.uninstall))
            elif await self.push_screen_wait(
                dialogs.Confirm("Background service", "Run ollajev in the background at login?", default=True)
            ):
                self.notify(await asyncio.to_thread(service.install))
        except SystemExit as exc:
            self.notify(str(exc), severity="error")

    def action_serve(self) -> None:
        """Start the server for the default model in the background; the manager stays open."""
        if self.refuse_while_busy():
            return
        if self.server_alive():  # e.g. Serve on another row: restart with the new default
            self.stop_server(then_start=canonical_or(default_model()))
            return
        if client.server_running():
            self.notify(f"A server is already running at {client.server_url()}", severity="warning")
            return
        self.start_server(canonical_or(default_model()))

    def server_alive(self) -> bool:
        return self.server is not None and self.server.poll() is None

    def start_server(self, model: str) -> None:
        """Run `ollajev serve` as a child process. Its log goes where `ollajev serve` always writes it; what it
        prints goes to server-console.log next to it."""
        self.release()  # the server loads its own copy; this window's copy would only hold memory
        log_dir = config.log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        console = (log_dir / "server-console.log").open("wb")
        command = [sys.executable, "-m", "ollajev", "serve", model, "--no-browser"]
        self.server = subprocess.Popen(  # noqa: S603 our own CLI, with a model name the user picked
            command, stdin=subprocess.DEVNULL, stdout=console, stderr=subprocess.STDOUT, start_new_session=True
        )
        console.close()  # the child holds its own handle
        self.server_model = model
        self.show_server_panel(False, set())

    @work(thread=True, exclusive=True, group="server")
    def stop_server(self, then_start: str | None = None) -> None:
        """Stop the server this window started, off the UI thread; restart it with `then_start` if given."""
        server = self.server
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        self.server = None
        if then_start:
            self.call_from_thread(self.start_server, then_start)
        else:
            self.call_from_thread(self.notify, "Server stopped")
            self.call_from_thread(self.reload)

    def action_stop_server(self) -> None:
        if self.server_alive():
            self.stop_server()
        elif client.server_running():
            self.notify("This server was started outside this window; stop it there (or: ollajev service)")
        else:
            self.notify("No server is running")

    def action_restart_server(self) -> None:
        if self.server_alive() or self.server is not None:
            self.stop_server(then_start=self.server_model or canonical_or(default_model()))
        else:
            self.notify("No server from this window to restart", severity="warning")

    def action_logs(self) -> None:
        """Follow the server's log and console output in a new terminal tab, or say what to run when it cannot."""
        log_dir = config.log_dir()
        command = ["tail", "-F", str(log_dir / "server.log"), str(log_dir / "server-console.log")]
        if not open_terminal(command):
            self.notify(f"Run in another terminal: {shlex.join(command)}", timeout=15)

    def action_open_demo(self) -> None:
        """Open the demo page of the running server in the browser."""
        if not client.server_running():
            self.notify("Start the server first (s, or Serve on a model)", severity="warning")
            return
        webbrowser.open(f"{client.server_url()}/demo")

    def show_server_panel(self, server_up: bool, loaded: set[str]) -> None:
        """The Server panel: shown while a server runs or this window's server starts or has stopped."""
        panel = self.query_one("#server-panel")
        ours = self.server is not None
        starting = self.server_alive() and not server_up
        crashed = ours and not self.server_alive()
        panel.display = server_up or ours
        panel.set_class(crashed, "stopped")
        for button_id in ("#do-stop_server", "#do-restart_server"):
            self.query_one(button_id).display = ours
        self.query_one("#do-open_demo").display = server_up
        url = client.server_url()
        log_file = str(config.log_dir() / "server.log").replace(str(Path.home()), "~", 1)
        info = Text()
        if crashed:
            info.append("● stopped", style=f"bold {self.colour('error')}")
            info.append(f"   the server exited (code {self.server.returncode}); see {log_file}")  # type: ignore[union-attr]
        elif starting:
            info.append("◌ starting …", style=f"bold {self.colour('warning')}")
            info.append(f"   loading {self.server_model}")
        else:
            info.append("● running", style=f"bold {self.colour('success')}")
            info.append(f"   {url}", style="bold")
            info.append("" if ours else "   started outside this window", style="dim")
            serving = ", ".join(sorted(loaded))
            info.append("\nmodel ", style="dim")
            if serving:
                info.append(serving, style=f"bold {self.colour('success')}")
            else:
                info.append("no model loaded yet", style="dim")
            info.append(f"  ·  log {log_file}", style="dim")
        self.query_one("#server-info", Static).update(info)

    @work
    async def action_quit_app(self) -> None:
        if self.downloading:
            question = dialogs.Confirm("A download is running", "Quit anyway? The next pull resumes it.")
            if not await self.push_screen_wait(question):
                return
        if self.server_alive():
            question = dialogs.Confirm("The server is running", "Quit and stop the server?")
            if not await self.push_screen_wait(question):
                return
            self.server.terminate()  # type: ignore[union-attr]
        # Stop a download at its next update. A loaded model is not unloaded: the process is about to end,
        # which frees it at once, while unloading it here would freeze the screen first.
        self.cancel.set()
        self.exit(False)

    async def action_quit(self) -> None:
        """ctrl+q: the same as q."""
        self.action_quit_app()


# (key, what it does, the CLI command it matches), in three groups.
KEYS: list[tuple[str, list[tuple[str, str, str]]]] = [
    (
        "Models",
        [
            ("enter", "download if needed, then make it the default", "pull"),
            ("p", "download only, keep the current default", "pull"),
            ("d", "make a downloaded model the default", ""),
            ("i", "family, commit, limits, path", "show"),
            ("u", "unload it from memory", "stop"),
            ("x", "delete the download from disk", "rm"),
            ("a", "give it a short name", "cp"),
            ("n", "add any Hugging Face repo by name", "pull"),
        ],
    ),
    (
        "Server",
        [
            ("s", "serve the selected model: it becomes the default, the server starts or restarts", "serve"),
            ("R", "restart the server", ""),
            ("S", "stop the server", ""),
            ("w", "open the demo page in the browser", ""),
            ("l", "follow the server logs in a new terminal tab", ""),
            ("o", "settings: device, address, port, memory", ""),
            ("b", "install or remove the background service", "service"),
        ],
    ),
    (
        "App",
        [
            ("/", "filter the list by name; esc clears the filter", ""),
            ("ctrl+r", "refresh the list", ""),
            ("e", "the last error in full", ""),
            ("esc", "cancel a running download first, else clear the filter", ""),
            ("q", "quit (asks first when busy)", ""),
        ],
    ),
]


def keys_help(accent: str) -> Text:
    """The ? screen: keys in the accent colour, the matching CLI command dim. A dim size is an estimate."""
    help_text = Text()
    for group, keys in KEYS:
        help_text.append(f"{group}\n", style="bold")
        for key, description, command in keys:
            help_text.append(f"  {key:<8}", style=f"bold {accent}")
            help_text.append(description)
            if command:
                help_text.append(f"  ({command})", style="dim")
            help_text.append("\n")
        help_text.append("\n")
    help_text.append("★ default  ● loaded  ✓ downloaded  ○ available  ·  a dim size is an estimate", style="dim")
    return help_text


# Plain keys (kitty keyboard flags 0), every mouse report mode off, bracketed paste off, cursor shown.
TERMINAL_RESET = "\x1b[=0;1u\x1b[?1000l\x1b[?1002l\x1b[?1003l\x1b[?1006l\x1b[?1015l\x1b[?2004l\x1b[?25h"


def restore_terminal() -> None:
    """Make sure the shell gets a plain terminal back. Textual resets it on exit, but some terminals keep the
    kitty keyboard mode per screen, so Enter then arrives as ESC[13u and prints "u"; and key releases or query
    replies that arrive after the app stopped reading would land in the shell as stray text."""
    if sys.stdout.isatty():
        sys.stdout.write(TERMINAL_RESET)
        sys.stdout.flush()
    if termios is not None and sys.stdin.isatty():
        with contextlib.suppress(termios.error):
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)


def manage() -> bool:
    """Open the model manager. True when the user chose to serve."""
    system_theme()  # ask the terminal for its colours now, while nothing else is reading its replies
    try:
        return bool(Models().run())
    finally:
        restore_terminal()
