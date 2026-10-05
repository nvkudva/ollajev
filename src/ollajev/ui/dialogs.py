"""The model manager's dialogs: prompts, model search, confirmations, info, options and the ask screen."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import time
import webbrowser
from typing import Any, ClassVar

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Input, Select, Static, TextArea

from .. import client, config, names, store
from . import repl

log = logging.getLogger(__name__)

GITHUB = "https://github.com/nvkudva/ollajev"
GITHUB_BUTTON = ("GitHub", "github", "default", "g")


def label(text: str, key: str) -> str:
    """A button's label as [<key>] label, the key dimmed: every button in the app shows its key the same way."""
    return f"[dim]\\[{key}][/] {text}"


def buttons(
    *specs: tuple[str, str, str, str], start: tuple[str, str, str, str] | None = None, row_id: str | None = None
) -> Horizontal:
    """A row of clickable buttons, each (label, action, variant, key), on the right; `start`, if given, on the left.
    A click runs the action its key would."""

    def button(text: str, action: str, variant: str, key: str) -> Button:
        return Button(label(text, key), id=f"do-{action}", variant=variant, compact=True)  # type: ignore[arg-type]

    row = [button(*spec) for spec in specs]
    if start:
        row = [button(*start), Static(classes="gap"), *row]
    return Horizontal(*row, classes="buttons", id=row_id)


class Clickable:
    """Runs the action named by a `buttons` button, so the mouse does what the keys do."""

    def action_github(self) -> None:
        webbrowser.open(GITHUB)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id.startswith("do-"):
            event.stop()
            await self.run_action(button_id.removeprefix("do-"))  # type: ignore[attr-defined]


class Prompt(Clickable, ModalScreen[str | None]):
    BINDINGS: ClassVar = [("escape", "cancel", "Cancel")]

    def __init__(self, title: str, placeholder: str = "") -> None:
        super().__init__()
        self.heading, self.placeholder = title, placeholder

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog") as box:
            box.border_title = self.heading
            yield Input(placeholder=self.placeholder)
            yield buttons(("OK", "submit", "primary", "enter"), ("Cancel", "cancel", "default", "esc"))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.action_submit()

    def action_submit(self) -> None:
        self.dismiss(self.query_one(Input).value.strip() or None)

    def action_cancel(self) -> None:
        self.dismiss(None)


def human(size: float, units: tuple[str, ...] = ("B", "KB", "MB", "GB", "TB")) -> str:
    for unit in units[:-1]:
        if size < 1000:
            return f"{size:.0f} {unit}" if unit == units[0] else f"{size:.1f} {unit}"
        size /= 1000
    return f"{size:.1f} {units[-1]}"


def count(value: int) -> str:
    """A download count as Hugging Face writes it: 85k, 2.6M, 308M."""
    return human(value, ("", "k", "M", "B")).replace(" ", "")


def runtime(variant: str) -> str:
    """What runs a listed variant: its tag names the weight file, which names the runtime."""
    return names.runtime_of(variant.partition(":")[2] or None)


class AddModel(Clickable, ModalScreen[str | None]):
    """Search Hugging Face and list every quant of every matching repo in one table. Returns the name `pull`
    takes."""

    BINDINGS: ClassVar = [("escape", "cancel", "Cancel")]

    def __init__(self) -> None:
        super().__init__()
        self.supported: set[str] = set()

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog wide") as box:
            box.border_title = "Add a model from Hugging Face"
            yield Input(placeholder="search words, user/repo or a huggingface.co link", id="query")
            yield DataTable(id="results", cursor_type="row")
            yield Static("", id="note")
            with Horizontal(classes="buttons"):
                yield Static("type to search · ↑↓ moves · enter picks · esc closes", classes="hint")
                yield Button(label("Cancel", "esc"), id="do-cancel", compact=True)

    def on_mount(self) -> None:
        table = self.query_one("#results", DataTable)
        table.add_column("Model", width=64)
        table.add_column("Size", width=8)
        table.add_column("Downloads", width=6)
        table.add_column("Runtime", width=9)
        self.query_one("#query", Input).focus()

    def note(self, text: str) -> None:
        self.query_one("#note", Static).update(text)

    def on_input_changed(self, event: Input.Changed) -> None:
        self.search(event.value.strip(), delay=0.4)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.search(event.value.strip(), delay=0)
        self.query_one("#results", DataTable).focus()

    @work(exclusive=True, group="search")
    async def search(self, query: str, delay: float) -> None:
        await asyncio.sleep(delay)
        if not query:
            self.note("Type words, user/repo, or paste a huggingface.co link.")
            return
        self.note(f"Searching for '{query}' …")
        try:
            hits = await asyncio.to_thread(store.search, query, 25)
        except Exception as exc:
            log.exception("search failed")
            self.note(f"error: {exc}")
            return
        if not hits:
            self.show_results(hits, {}, set())
            self.note(f"No models found for '{query}' — try fewer words or user/repo.")
            return
        # The search already names every quant; only their download sizes need a call per repo, so the rows show
        # at once and each repo's sizes fill in as they arrive.
        quants = {hit.repo_id: store.listed_variants(hit) for hit in hits}
        sized: set[str] = set()
        self.show_results(hits, quants, sized)

        async def read_sizes(hit: store.Hit) -> None:
            found = await asyncio.to_thread(variants, hit.repo_id)
            if found:
                quants[hit.repo_id] = found
            sized.add(hit.repo_id)
            self.show_results(hits, quants, sized)
            self.note(f"Reading sizes: {len(sized)}/{len(hits)} models")

        self.note(f"Reading sizes: 0/{len(hits)} models")
        await asyncio.gather(*(read_sizes(hit) for hit in hits))
        files = sum(len(v) for v in quants.values())
        self.note(
            f"{len(hits)} models · {files} files — ↑↓ moves, enter picks, esc closes."
            " Rows marked ✗ unsupported cannot run here."
        )

    def show_results(self, hits: list[store.Hit], quants: dict[str, list[store.Variant]], sized: set[str]) -> None:
        table = self.query_one("#results", DataTable)
        cursor = table.cursor_row  # rows keep arriving while the user moves through them
        table.clear()
        self.supported = set()
        for hit in hits:
            # A repo no family runs has no runtime; its rows are marked and unpickable.
            downloads = count(hit.downloads)
            for variant in quants.get(hit.repo_id, []):
                name = f"✗ {variant.name}" if not hit.family else variant.name
                runs = runtime(variant.name) if hit.family else ""
                size = human(variant.size) if hit.repo_id in sized else "…"
                table.add_row(name, size, downloads, runs, key=variant.name)
                if hit.family:
                    self.supported.add(variant.name)
        if table.row_count:
            table.move_cursor(row=min(cursor, table.row_count - 1))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()  # the manager behind this dialog downloads on its own row selection
        if event.row_key.value not in self.supported:
            self.notify("Ollajev has no adapter for this model", severity="warning")
            return
        self.dismiss(event.row_key.value)

    def action_cancel(self) -> None:
        self.dismiss(None)


def variants(repo_id: str) -> list[store.Variant]:
    """A repo's quants, or none when Hugging Face cannot list them; one failing repo must not fail a search."""
    try:
        return store.variants(repo_id)
    except Exception:
        log.exception("could not read %s", repo_id)
        return []


class Confirm(Clickable, ModalScreen[bool]):
    """Yes or no. Enter picks `default`: yes for harmless steps, no for anything that deletes or discards."""

    BINDINGS: ClassVar = [("y", "yes", "Yes"), ("n,escape", "no", "No"), ("enter", "default", "Default")]

    def __init__(self, title: str, body: str, default: bool = False) -> None:
        super().__init__()
        self.heading, self.body, self.default = title, body, default

    def compose(self) -> ComposeResult:
        # Anything whose safe answer is no (delete, trust code, quit mid-download) gets a red frame.
        with Vertical(classes="dialog" if self.default else "dialog danger") as box:
            box.border_title = self.heading
            yield Static(self.body)
            yes_variant = "primary" if self.default else "error"
            yield buttons(("Yes", "yes", yes_variant, "y"), ("No", "no", "default", "n"))

    def on_mount(self) -> None:
        # The safe answer has the focus, so Enter (or a stray click on nothing) picks it.
        self.query_one("#do-yes" if self.default else "#do-no", Button).focus()

    def action_default(self) -> None:
        self.dismiss(self.default)

    def action_yes(self) -> None:
        self.dismiss(True)

    def action_no(self) -> None:
        self.dismiss(False)


class Info(Clickable, ModalScreen[None]):
    BINDINGS: ClassVar = [("escape,enter,q", "close", "Close"), ("o", "open_link", "Open"), ("g", "github", "GitHub")]

    def __init__(
        self, title: str, body: str | Text, danger: bool = False, link: str | None = None, about: bool = False
    ) -> None:
        """`about`: a wide dialog about the app itself, with the GitHub button at its bottom left."""
        super().__init__()
        self.heading, self.body, self.danger, self.link, self.about = title, body, danger, link, about

    def compose(self) -> ComposeResult:
        classes = "dialog" + (" danger" if self.danger else "") + (" wide" if self.about else "")
        with Vertical(classes=classes) as box:
            box.border_title = self.heading
            yield Static(self.body)
            start = GITHUB_BUTTON if self.about else None
            if self.link:
                yield buttons(
                    ("HF page", "open_link", "primary", "o"), ("Close", "close", "default", "esc"), start=start
                )
            else:
                yield buttons(("Close", "close", "primary", "esc"), start=start)

    def action_open_link(self) -> None:
        if self.link:
            webbrowser.open(self.link)

    def action_close(self) -> None:
        self.dismiss(None)


def valid_host(host: str) -> bool:
    """A host name or an IP address, with no port: a colon is only allowed inside an IPv6 address."""
    if not host or any(char.isspace() for char in host):
        return False
    if ":" not in host:
        return True
    try:
        ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return False
    return True


class Settings(Clickable, ModalScreen[dict[str, Any] | None]):
    """Server settings saved in the config file. An OLLAJEV_* environment variable still wins over a saved value."""

    BINDINGS: ClassVar = [("escape", "cancel", "Cancel"), ("g", "github", "GitHub")]

    def compose(self) -> ComposeResult:
        saved = config.load()
        with Vertical(classes="dialog") as box:
            box.border_title = "Settings"
            with Horizontal(classes="field"):
                yield Static("Device")
                yield Select(
                    [(device, device) for device in ("auto", "mps", "cuda", "cpu")],
                    value=saved.get("device", "auto"),
                    allow_blank=False,
                    id="device",
                    compact=True,
                )
            with Horizontal(classes="field"):
                yield Static("Address")
                yield Input(saved.get("host", "127.0.0.1"), id="host", compact=True)
            with Horizontal(classes="field"):
                yield Static("Port")
                yield Input(str(saved.get("port", config.DEFAULT_PORT)), id="port", type="integer", compact=True)
            with Horizontal(classes="field"):
                yield Static("Keep idle model loaded")
                yield Input(str(saved.get("keep_alive", "5m")), id="keep_alive", placeholder="5m, 1h, -1", compact=True)
            with Horizontal(classes="field"):
                yield Static("Models in memory")
                yield Input(
                    str(saved.get("max_loaded_models", 1)), id="max_loaded_models", type="integer", compact=True
                )
            yield Static(f"Saved in {config.config_path()}; an OLLAJEV_* variable overrides it.", classes="hint")
            yield buttons(
                ("Save", "save", "primary", "enter"), ("Cancel", "cancel", "default", "esc"), start=GITHUB_BUTTON
            )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.action_save()

    def value(self, field: str) -> str:
        return self.query_one(f"#{field}", Input).value.strip()

    def action_save(self) -> None:
        port = self.value("port")
        if not port.isdigit() or not 0 < int(port) < 65536:
            self.notify("Port must be 1-65535", severity="error")
            return
        host = self.value("host") or "127.0.0.1"
        if not valid_host(host):
            self.notify("Address must be a host name or IP address, without a port", severity="error")
            return
        if not config.is_loopback(host) and not config.api_key():
            self.notify(f"Serving on {host} needs OLLAJEV_API_KEY set first", severity="error")
            return
        keep_alive = self.value("keep_alive") or "5m"
        try:
            config.parse_duration(keep_alive)
        except ValueError:
            self.notify("Keep loaded must be seconds or a duration like 5m, 1h, or -1", severity="error")
            return
        models = self.value("max_loaded_models")
        if not models.isdigit() or int(models) < 1:
            self.notify("Models in memory must be 1 or more", severity="error")
            return
        self.dismiss(
            {
                "device": self.query_one("#device", Select).value,
                "host": host,
                "port": int(port),
                "keep_alive": keep_alive,
                "max_loaded_models": int(models),
            }
        )

    def action_cancel(self) -> None:
        self.dismiss(None)


class Ask(Clickable, ModalScreen[None]):
    """Ask a model questions. One question per line, in the same form `ollajev run` takes."""

    BINDINGS: ClassVar = [("escape", "close", "Close"), Binding("ctrl+s,ctrl+r", "send", "Ask", priority=True)]

    def __init__(self, model: str) -> None:
        super().__init__()
        self.model = model
        self.ask: Any = None
        self.history: list[str] = []  # earlier answers, newest first, kept while the dialog is open

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog") as box:
            box.border_title = f"Ask {self.model}"
            yield Static("State — what the model should decide about")
            yield TextArea(id="state")
            yield Static("Questions, one per line — noul:, choice: … | a, b, or score: … | low, high")
            yield TextArea(
                id="questions",
                placeholder="noul: The customer asks for a refund.\nchoice: Which team? | billing, support, sales",
            )
            yield Static("ctrl+s asks · esc closes · answers newest-first below", id="ask-hint")
            with VerticalScroll(id="answers-box"):
                yield Static("", id="answers")
            yield buttons(("Ask", "send", "primary", "^s"), ("Close", "close", "default", "esc"))

    def on_mount(self) -> None:
        self.show(f"Loading {self.model} …")
        self.connect()
        self.query_one("#state", TextArea).focus()

    def show(self, text: str) -> None:
        # A load or answer can finish after Esc closed this dialog; there is nothing left to update then.
        if self.is_attached:
            self.query_one("#answers", Static).update(text)

    @work(thread=True)
    def connect(self) -> None:
        where = "on the server" if client.server_running() else "into this process (no server running)"
        self.app.call_from_thread(self.show, f"Loading {self.model} {where} …")
        try:
            self.ask = self.app.connection(self.model)  # type: ignore[attr-defined]
            self.app.call_from_thread(self.show, "Ready. Press ctrl+s to ask.")
        except Exception as exc:
            log.exception("could not load %s", self.model)
            self.app.call_from_thread(self.show, f"error: {exc}")

    def action_send(self) -> None:
        if self.ask is None:
            self.show("Still loading the model…")
            return
        state = self.query_one("#state", TextArea).text.strip()
        questions: dict[str, Any] = {}
        for lineno, line in enumerate(self.query_one("#questions", TextArea).text.splitlines(), start=1):
            if line.strip():
                parsed = repl.parse_question(line.strip())
                if parsed is None:
                    self.query_one("#answers", Static).update(
                        f"Line {lineno} is not a question: {line.strip()}\n"
                        "Use noul: …, choice: … | a, b, or score: … | low, high."
                    )
                    return
                questions[f"q{len(questions) + 1}"] = parsed[1]
        if not state or not questions:
            missing = "a state" if not state else "at least one question"
            self.query_one("#answers", Static).update(
                f"Enter {missing} first — e.g. a state plus 'choice: Which team? | billing, support'."
            )
            return
        self.query_one("#answers", Static).update("Thinking… (esc closes, the answer lands below)")
        self.send(state, questions)

    @work(thread=True, exclusive=True)
    def send(self, state: str, questions: dict[str, Any]) -> None:
        started = time.monotonic()
        try:
            text = "\n".join(repl.format_answers(self.ask(state, questions)["answers"]))
        except Exception as exc:
            log.exception("ask failed")
            text = f"error: {exc}"
        seconds = time.monotonic() - started
        self.history.insert(0, f"── {self.model} · {seconds:.1f} s\n{text}")
        self.app.call_from_thread(self.show, "\n\n".join(self.history))

    def action_close(self) -> None:
        self.dismiss(None)
