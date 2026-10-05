"""The model manager screen: it mounts, lists the catalog, opens its dialogs and exits with the right answer."""

from __future__ import annotations

import asyncio

import pytest
from textual.widgets import DataTable

from ollajev import client
from ollajev.catalog import CATALOG
from ollajev.ui import tui


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAJEV_HOME", str(tmp_path))
    monkeypatch.setenv("OLLAJEV_MODELS", str(tmp_path / "models"))
    monkeypatch.setattr(client, "server_running", lambda: False)
    monkeypatch.setattr(tui.dialogs, "variants", lambda repo: [])  # offline: the catalog's quants are not listed
    # Offline: download counts never reach Hugging Face, so slow lookups cannot starve the shared
    # thread pool that the dialogs under test also use.
    monkeypatch.setattr(tui.store, "listing", lambda repo: (None, None))
    return tui.Models()


def drive(app, keys):
    async def go():
        async with app.run_test(size=(120, 36)) as pilot:
            table = app.query_one(DataTable)
            assert table.row_count == len(CATALOG)
            for key in keys:
                await pilot.press(key)
                await pilot.pause()
            return [type(screen).__name__ for screen in app.screen_stack][1:]

    return asyncio.run(go())


class FakeServer:
    """Stands in for the `ollajev serve` child process."""

    started: list = []

    def __init__(self, command, **kwargs):
        FakeServer.started.append(command)
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = -9


def test_serve_starts_the_server_inside_the_manager(app, monkeypatch):
    monkeypatch.setattr(tui, "system_theme", lambda: "ansi-dark")  # it runs a subprocess on macOS
    monkeypatch.setattr(tui.subprocess, "Popen", FakeServer)
    monkeypatch.setattr(tui.Models, "downloaded", lambda self, name: True)  # s serves the selected model
    FakeServer.started = []

    async def go():
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.press("s")
            await pilot.pause()
            panel = app.query_one("#server-panel")
            info = str(app.query_one("#server-info").render())
            running_after_s = (app.is_running, panel.display, "starting" in info)
            await pilot.click("#do-stop_server")
            await app.workers.wait_for_complete()
            await pilot.pause()
            return running_after_s, app.server

    (still_open, panel_shown, starting), server = asyncio.run(go())
    assert still_open and panel_shown and starting
    assert FakeServer.started and FakeServer.started[0][-3:-1] == ["serve", tui.canonical_or(tui.default_model())]
    assert server is None


def test_quit_asks_before_stopping_a_running_server(app, monkeypatch):
    monkeypatch.setattr(tui, "system_theme", lambda: "ansi-dark")  # it runs a subprocess on macOS
    monkeypatch.setattr(tui.subprocess, "Popen", FakeServer)
    monkeypatch.setattr(tui.Models, "downloaded", lambda self, name: True)

    async def go():
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.press("s")
            await pilot.pause()
            await pilot.press("q")
            await pilot.pause()
            return type(app.screen).__name__

    assert asyncio.run(go()) == "Confirm"


@pytest.mark.parametrize("key", ["q", "ctrl+q"])
def test_quit_keys_return_false(app, key):
    drive(app, [key])
    assert app.return_value is False


@pytest.mark.parametrize(
    ("key", "screen"), [("o", "Settings"), ("a", "AddModel"), ("c", "Prompt"), ("i", "Info"), ("question_mark", "Info")]
)
def test_keys_open_their_dialog_and_escape_closes_it(app, key, screen):
    assert drive(app, [key]) == [screen]
    assert drive(tui.Models(), [key, "escape"]) == []


def test_ask_is_hidden_for_now(app):
    assert drive(app, ["r"]) == []
    assert not any(action == "ask" for _, action, _, _ in tui.SELECTION_BUTTONS)


def quants(repo):
    return [tui.store.Variant(f"{repo}:{q}", q, n) for q, n in (("Q4_K_M", 2_700_000_000), ("Q8_0", 4_500_000_000))]


def test_add_model_lists_every_quant_and_returns_the_picked_one(app, monkeypatch):
    hits = [tui.store.Hit("u/ok-GGUF", 1200, "decider"), tui.store.Hit("u/no-GGUF", 5, None)]
    monkeypatch.setattr(tui.store, "search", lambda query, limit: hits)
    monkeypatch.setattr(tui.dialogs, "variants", quants)
    picked = []

    async def go():
        async with app.run_test(size=(120, 36)) as pilot:
            app.push_screen(tui.dialogs.AddModel(), picked.append)
            await pilot.pause()
            await pilot.press(*"ok", "enter")
            results = app.screen.query_one("#results", DataTable)
            for _ in range(100):
                if results.row_count:
                    break
                await pilot.pause(0.1)
            rows = [results.get_row_at(i) for i in range(results.row_count)]
            assert rows[0] == ["u/ok-GGUF:Q4_K_M", "2.7 GB", "1.2k", "llama.cpp"]
            # No family runs u/no-GGUF, so its quant stays marked, unpickable and with no runtime to name.
            assert rows[2] == ["✗ u/no-GGUF:Q4_K_M", "2.7 GB", "5", ""]
            assert [r[0] for r in rows] == [
                "u/ok-GGUF:Q4_K_M",
                "u/ok-GGUF:Q8_0",
                "✗ u/no-GGUF:Q4_K_M",
                "✗ u/no-GGUF:Q8_0",
            ]
            await pilot.press("down", "enter")
            await pilot.pause()

    asyncio.run(go())
    assert picked == ["u/ok-GGUF:Q8_0"]


def test_the_model_list_shows_every_quant_of_a_catalog_gguf_repo(app, monkeypatch):
    monkeypatch.setattr(tui.dialogs, "variants", quants)

    async def go():
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.pause(0.3)
            return list(app.names)

    names = asyncio.run(go())
    repo = next(e.name for e in CATALOG if ":" in e.name).partition(":")[0]
    assert f"{repo}:Q4_K_M" in names and f"{repo}:Q8_0" in names
    assert len(names) == len(set(names))


def test_escape_cancels_a_running_download(app, monkeypatch):
    import threading
    from types import SimpleNamespace

    r = SimpleNamespace(repo_id="u/r", family=SimpleNamespace(runs_repo_code=False), ref=SimpleNamespace(name="u/r"))
    started = threading.Event()

    def resolve(name, online=True):
        if not online:  # the model list asks offline whether each is downloaded
            raise LookupError(name)
        return r

    def download(r, cancel):
        started.set()
        cancel.wait(10)
        raise tui.store.Cancelled

    monkeypatch.setattr(tui.store, "resolve", resolve)
    monkeypatch.setattr(tui.store, "download", download)
    monkeypatch.setattr(tui, "canonical", lambda r: r.repo_id)

    async def go():
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.press("p")
            assert await asyncio.to_thread(started.wait, 5)
            await pilot.press("escape")
            await app.workers.wait_for_complete()
            assert not app.busy

    asyncio.run(go())


@pytest.mark.parametrize("key", ["x", "u", "s", "b"])
def test_actions_wait_while_a_model_loads(app, monkeypatch, key):
    monkeypatch.setattr(tui.store, "remove", lambda resolved: pytest.fail("removed during a load"))
    monkeypatch.setattr(tui.service, "status", lambda: pytest.fail("service touched during a load"))
    app.loading = "some/model"
    assert drive(app, [key]) == []
    assert app.return_value is None


def test_quit_during_a_download_asks_first(app):
    async def go():
        async with app.run_test(size=(120, 36)) as pilot:
            app.downloading = True
            await pilot.press("q")
            await pilot.pause()
            assert type(app.screen).__name__ == "Confirm"
            await pilot.press("n")
            await pilot.pause()
            assert app.return_value is None
            await pilot.press("q")
            await pilot.pause()
            await pilot.press("y")
            await pilot.pause()
        return app.return_value

    assert asyncio.run(go()) is False


def test_a_failed_job_stays_on_the_status_line(app):
    async def boom():
        raise RuntimeError("disk full")

    async def go():
        async with app.run_test(size=(120, 36)) as pilot:
            await app.job("Downloading x …", boom)
            await pilot.pause()
            status = str(app.query_one("#status").render())
            await pilot.press("e")
            await pilot.pause()
            return status, type(app.screen).__name__

    status, screen = asyncio.run(go())
    assert "Downloading x failed" in status and "e for details" in status
    assert screen == "Info"


def test_enter_asks_before_downloading(app, monkeypatch):
    monkeypatch.setattr(tui.store, "download", lambda *a: pytest.fail("downloaded without asking"))
    assert drive(app, ["enter"]) == ["Confirm"]
    assert drive(tui.Models(), ["enter", "n"]) == []


def test_download_progress_shows_on_the_status_line(app, monkeypatch):
    from types import SimpleNamespace

    resolved = SimpleNamespace(repo_id="u/r", family=SimpleNamespace(runs_repo_code=False))
    on_disk = {"bytes": 0}
    monkeypatch.setattr(tui, "canonical", lambda resolved: resolved.repo_id)
    monkeypatch.setattr(tui.store, "download_size", lambda resolved: 2_000_000_000)
    monkeypatch.setattr(tui.store, "bytes_on_disk", lambda repo_id: on_disk["bytes"])

    async def go():
        async with app.run_test(size=(120, 36)) as pilot:
            timer = await app.show_progress(resolved)
            on_disk["bytes"] = 500_000_000
            await pilot.pause(0.7)
            timer.stop()
            return str(app.query_one("#status").render())

    status = asyncio.run(go())
    assert "500.0 MB / 2.0 GB · 25%" in status and "esc cancels" in status


def test_status_line_and_filter(app):
    async def go():
        async with app.run_test(size=(160, 36)) as pilot:
            status = str(app.query_one("#models-panel").border_subtitle)
            await pilot.press("f")
            await pilot.pause()
            for key in "julia":
                await pilot.press(key)
            await pilot.press("enter")
            await pilot.pause()
            return status, list(app.names)

    status, names = asyncio.run(go())
    assert "downloaded" in status
    assert names and all("julia" in name.lower() for name in names)


def test_auto_refresh_picks_up_changes_made_elsewhere(app):
    from ollajev import config

    target = CATALOG[3].name

    async def go():
        async with app.run_test(size=(160, 36)) as pilot:
            config.update(default_model=target)
            app.auto_refresh()
            await app.workers.wait_for_complete()
            await pilot.pause()
            return app.query_one(DataTable).get_cell(target, "state").plain

    assert "default" in asyncio.run(go())


def test_enter_in_a_confirm_takes_its_default():
    async def go(default):
        answers = []

        class Host(tui.App):
            def on_mount(self):
                self.push_screen(tui.dialogs.Confirm("t", "b", default=default), answers.append)

        app = Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
        return answers

    assert asyncio.run(go(True)) == [True]
    assert asyncio.run(go(False)) == [False]


def test_ask_keeps_a_history_of_answers(monkeypatch):
    monkeypatch.setattr(tui.repl, "format_answers", lambda answers: [f"answer {answers['n']}"])
    replies = iter([{"answers": {"n": 1}}, {"answers": {"n": 2}}])

    class Host(tui.App):
        def connection(self, model):
            return lambda state, questions: next(replies)

        def on_mount(self):
            self.push_screen(tui.dialogs.Ask("u/model"))

    async def go():
        app = Host()
        async with app.run_test(size=(120, 40)) as pilot:
            await app.workers.wait_for_complete()
            app.screen.query_one("#state").text = "a state"
            app.screen.query_one("#questions").text = "noul: is it?"
            for _ in range(2):
                await pilot.press("ctrl+s")
                await app.workers.wait_for_complete()
                await pilot.pause()
            return str(app.screen.query_one("#answers").render())

    shown = asyncio.run(go())
    assert shown.index("answer 2") < shown.index("answer 1")
    assert "u/model ·" in shown


@pytest.mark.parametrize(
    ("host", "valid"),
    [("127.0.0.1", True), ("localhost", True), ("::1", True), ("[::1]", True), ("my host", False), ("h:80", False)],
)
def test_options_accepts_only_host_names_and_addresses(host, valid):
    assert tui.dialogs.valid_host(host) is valid


def shown_buttons(app):
    panel = app.query_one("#selection-panel")
    return [str(button.label).split()[1] for button in panel.query("Button") if button.display]


def test_the_selected_panel_has_the_cursor_rows_buttons(app):
    async def go():
        async with app.run_test(size=(160, 36)) as pilot:
            table = app.query_one(DataTable)
            out = []
            for row in range(2):
                table.move_cursor(row=row)
                await pilot.pause()
                out.append((shown_buttons(app), app.query_one("#selection-info").render().plain.split()[0]))
            return out

    first, second = asyncio.run(go())
    assert first == (["Download", "Info"], CATALOG[0].name)
    assert second[0] == ["Download", "Info"]


def test_the_selected_panel_offers_what_a_downloaded_model_can_do(app):
    async def go():
        async with app.run_test(size=(160, 36)) as pilot:
            await pilot.pause()
            name = app.selected()
            app.show_selection("other", set(), {name: {}})
            not_default = shown_buttons(app)
            app.show_selection(name, {name}, {name: {}})
            return not_default, shown_buttons(app), "loaded" in app.query_one("#selection-panel").classes

    not_default, loaded_default, bordered = asyncio.run(go())
    assert not_default == ["Serve", "Default", "Delete", "Info"]
    assert loaded_default == ["Serve", "Unload", "Delete", "Info"] and bordered


def test_confirm_buttons_answer_it():
    async def go(button):
        answers = []

        class Host(tui.App):
            def on_mount(self):
                self.push_screen(tui.dialogs.Confirm("t", "b"), answers.append)

        async with Host().run_test() as pilot:
            await pilot.pause()
            await pilot.click(button)
            await pilot.pause()
        return answers

    assert asyncio.run(go("#do-yes")) == [True]
    assert asyncio.run(go("#do-no")) == [False]


def test_selected_panel_buttons_act_on_the_cursor_row(app):
    async def go():
        async with app.run_test(size=(160, 36)) as pilot:
            app.query_one(DataTable).move_cursor(row=3)
            await pilot.pause()
            await pilot.click("#do-info")
            await pilot.pause()
            return app.selected(), str(app.screen.query_one(".dialog").border_title)

    selected, title = asyncio.run(go())
    assert title == selected


@pytest.mark.parametrize(("colours", "theme"), [("15;0", "ansi-dark"), ("0;15", "ansi-light"), ("0;7", "ansi-light")])
def test_theme_follows_the_terminal_colours(monkeypatch, colours, theme):
    monkeypatch.setenv("COLORFGBG", colours)
    monkeypatch.setattr(tui, "terminal_background", lambda: None)
    tui.system_theme.cache_clear()
    assert tui.system_theme() == theme
    tui.system_theme.cache_clear()


@pytest.mark.parametrize(("background", "theme"), [((1.0, 1.0, 1.0), "ansi-light"), ((0.1, 0.1, 0.12), "ansi-dark")])
def test_theme_follows_the_background_the_terminal_reports(monkeypatch, background, theme):
    monkeypatch.setattr(tui, "terminal_background", lambda: background)
    tui.system_theme.cache_clear()
    assert tui.system_theme() == theme
    tui.system_theme.cache_clear()


def test_info_links_to_the_model_on_hugging_face(app, monkeypatch):
    opened = []
    monkeypatch.setattr(tui.dialogs.webbrowser, "open", opened.append)

    async def go():
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.press("i")
            await pilot.pause()
            await pilot.click("#do-open_link")
            await pilot.pause()

    asyncio.run(go())
    assert opened == [f"https://huggingface.co/{CATALOG[0].name.partition(':')[0]}"]


def test_status_kinds_set_their_style_class(app):
    from textual.widgets import Static

    async def go():
        async with app.run_test(size=(160, 36)) as pilot:
            await pilot.pause()
            status = app.query_one("#status", Static)
            app.say("working", "busy")
            busy = set(status.classes)
            app.say("failed", "error")
            return busy, set(status.classes)

    busy, error = asyncio.run(go())
    assert "busy" in busy and "error" in error


def test_status_row_lists_the_keys_for_the_cursor_row(app):
    async def go():
        async with app.run_test(size=(160, 36)) as pilot:
            await pilot.pause()
            return app.query_one("#status").render().plain

    hint = asyncio.run(go())
    assert "enter download & use" in hint and "? all keys" in hint and "x delete" not in hint


def test_narrow_terminals_keep_only_the_essential_columns(app):
    async def go():
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            table = app.query_one(DataTable)
            return [str(column.label) for column in table.columns.values()]

    labels = asyncio.run(go())
    assert "Status" in labels and "Actions" not in labels
    assert "Adapter" not in labels
    assert "Runtime" not in labels and "Lang" not in labels


def test_download_progress_also_drives_the_bar(app, monkeypatch):
    from types import SimpleNamespace

    from textual.widgets import ProgressBar

    resolved = SimpleNamespace(repo_id="u/r", family=SimpleNamespace(runs_repo_code=False))
    on_disk = {"bytes": 0}
    monkeypatch.setattr(tui, "canonical", lambda resolved: resolved.repo_id)
    monkeypatch.setattr(tui.store, "download_size", lambda resolved: 2_000_000_000)
    monkeypatch.setattr(tui.store, "bytes_on_disk", lambda repo_id: on_disk["bytes"])

    async def go():
        async with app.run_test(size=(160, 36)) as pilot:
            timer = await app.show_progress(resolved)
            assert "show" in app.query_one("#progress").classes
            on_disk["bytes"] = 500_000_000
            await pilot.pause(0.7)
            bar = app.query_one("#progress", ProgressBar)
            progress, status = bar.progress, str(app.query_one("#status").render())
            timer.stop()
            await pilot.pause()
            return progress, status, "show" in app.query_one("#progress").classes

    progress, status, shown_after_stop = asyncio.run(go())
    assert progress == 500_000_000
    assert "500.0 MB / 2.0 GB · 25%" in status
    assert not shown_after_stop


@pytest.mark.parametrize(
    ("env", "platform", "installed", "launcher"),
    [
        ({"TMUX": "1"}, "linux", [], "tmux"),
        ({"TERM_PROGRAM": "WezTerm"}, "linux", [], "wezterm"),
        ({"TERM_PROGRAM": "iTerm.app"}, "darwin", [], "osascript"),
        ({"TERM_PROGRAM": "ghostty"}, "darwin", [], "osascript"),
        ({"GNOME_TERMINAL_SCREEN": "/org/gnome/1"}, "linux", ["gnome-terminal"], "gnome-terminal"),
        ({"KONSOLE_VERSION": "240800"}, "linux", ["konsole"], "konsole"),
        ({}, "linux", ["x-terminal-emulator"], "x-terminal-emulator"),
        ({}, "linux", [], None),
    ],
)
def test_logs_open_in_a_new_terminal_tab_where_it_can(monkeypatch, env, platform, installed, launcher):
    for name in ("TMUX", "TERM_PROGRAM", "GNOME_TERMINAL_SCREEN", "KONSOLE_VERSION"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(tui.sys, "platform", platform)
    monkeypatch.setattr(tui.shutil, "which", lambda name: f"/usr/bin/{name}" if name in installed else None)
    launched = []
    monkeypatch.setattr(tui.subprocess, "Popen", lambda command, **kwargs: launched.append(command))
    opened = tui.open_terminal(["tail", "-F", "/a b/server.log"])
    assert opened is (launcher is not None)
    assert [command[0] for command in launched] == ([launcher] if launcher else [])
    if launched:
        assert "/a b/server.log" in " ".join(launched[0])


def test_logs_key_says_what_to_run_when_no_tab_can_open(app, monkeypatch):
    monkeypatch.setattr(tui, "open_terminal", lambda command: False)
    notes = []
    monkeypatch.setattr(app, "notify", lambda message, **kwargs: notes.append(message))

    async def go():
        async with app.run_test(size=(160, 36)) as pilot:
            await pilot.press("l")
            await pilot.pause()

    asyncio.run(go())
    assert len(notes) == 1 and "tail -F" in notes[0] and "server.log" in notes[0]


@pytest.mark.parametrize("key", ["question_mark", "o"])
def test_keys_and_settings_link_to_github(app, monkeypatch, key):
    opened = []
    monkeypatch.setattr(tui.dialogs.webbrowser, "open", opened.append)

    async def go():
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.press(key)
            await pilot.pause()
            row = app.screen.query_one(".buttons")
            first = row.children[0].id
            await pilot.click("#do-github")
            await pilot.pause()
            return first

    assert asyncio.run(go()) == "do-github"
    assert opened == [tui.dialogs.GITHUB]


@pytest.mark.parametrize(("seconds", "text"), [(45, "45s"), (754, "12m"), (3900, "1h 05m")])
def test_time_left_is_coarse(seconds, text):
    assert tui.duration(seconds) == text


def test_status_row_says_cancelling_until_the_download_stops(app, monkeypatch):
    import threading
    import time
    from types import SimpleNamespace

    r = SimpleNamespace(repo_id="u/r", family=SimpleNamespace(runs_repo_code=False), ref=SimpleNamespace(name="u/r"))
    started = threading.Event()

    def resolve(name, online=True):
        if not online:
            raise LookupError(name)
        return r

    def download(r, cancel):
        started.set()
        cancel.wait(10)
        time.sleep(1.5)  # a slow stop, past a few progress updates
        raise tui.store.Cancelled

    monkeypatch.setattr(tui.store, "resolve", resolve)
    monkeypatch.setattr(tui.store, "download", download)
    monkeypatch.setattr(tui.store, "download_size", lambda r: 0)
    monkeypatch.setattr(tui, "canonical", lambda r: r.repo_id)

    async def go():
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.press("p")
            assert await asyncio.to_thread(started.wait, 5)
            await pilot.press("escape")
            await pilot.pause(1.0)
            during = app.query_one("#status").render().plain
            await app.workers.wait_for_complete()
            return during

    assert "Cancelling download" in asyncio.run(go())


@pytest.mark.parametrize(("width", "narrow"), [(100, True), (160, False)])
def test_cards_stack_their_buttons_under_the_text_when_narrow(app, width, narrow):
    async def go():
        async with app.run_test(size=(width, 36)) as pilot:
            await pilot.pause()
            return ["narrow" in row.classes for row in app.query(".card-row")]

    assert asyncio.run(go()) == [narrow, narrow]


def notes_of(app, monkeypatch):
    """The messages app.notify shows, in order."""
    notes = []
    monkeypatch.setattr(app, "notify", lambda message, **kwargs: notes.append(message))
    return notes


def test_status_row_lists_the_keys_for_a_downloaded_loaded_model(app):
    async def go():
        async with app.run_test(size=(160, 36)) as pilot:
            await pilot.pause()
            name = app.selected()
            app.snapshot_cache = ({name: {}}, "other/model", False, {name})
            app.say_idle()
            return app.query_one("#status").render().plain

    hint = asyncio.run(go())
    for keys in ("enter make default", "s serve", "u unload", "c short name", "x delete", "i info"):
        assert keys in hint
    assert "download" not in hint


def test_the_selected_panel_says_on_disk_or_to_download_and_describes_the_model(app):
    async def go():
        async with app.run_test(size=(160, 36)) as pilot:
            await pilot.pause()
            name = app.selected()
            app.show_selection("other", set(), {})
            remote = app.query_one("#selection-info").render().plain
            app.show_selection("other", set(), {name: {}})
            return remote, app.query_one("#selection-info").render().plain

    remote, local = asyncio.run(go())
    assert "to download" in remote and tui.DESCRIPTIONS[CATALOG[0].name.partition(":")[0]] in remote
    assert "on disk" in local and "to download" not in local


def test_serve_refuses_a_model_that_is_not_downloaded(app, monkeypatch):
    monkeypatch.setattr(tui.subprocess, "Popen", lambda *a, **k: pytest.fail("a server was started"))
    notes = notes_of(app, monkeypatch)
    drive(app, ["s"])
    assert notes == ["Download it first (p or Enter)"]


def test_stop_and_restart_say_when_there_is_no_server(app, monkeypatch):
    notes = notes_of(app, monkeypatch)
    drive(app, ["S", "R"])
    assert notes == ["No server is running", "No server from this window to restart"]


def test_unload_says_when_nothing_is_loaded_and_is_quiet_when_it_unloads(app, monkeypatch):
    notes = notes_of(app, monkeypatch)
    released = []

    async def go():
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.press("u")
            await app.workers.wait_for_complete()
            app.local = (app.selected(), None, lambda: released.append(True))
            await pilot.press("u")
            await app.workers.wait_for_complete()
            return app.local

    assert asyncio.run(go()) is None
    assert released == [True] and len(notes) == 1 and notes[0].endswith("is not loaded")


@pytest.mark.parametrize(("on_disk", "toast"), [(True, False), (False, True)])
def test_enter_makes_the_default_and_toasts_only_a_new_download(app, monkeypatch, on_disk, toast):
    from types import SimpleNamespace

    monkeypatch.setattr(tui.Models, "downloaded", lambda self, name: on_disk)
    monkeypatch.setattr(tui, "canonical", lambda resolved: resolved.repo_id)

    async def fetch(self, name):
        return SimpleNamespace(repo_id=name)

    monkeypatch.setattr(tui.Models, "fetch", fetch)
    notes = notes_of(app, monkeypatch)

    async def go():
        async with app.run_test(size=(120, 36)) as pilot:
            name = app.selected()
            await pilot.press("enter")
            await pilot.pause()
            if not on_disk:
                await pilot.press("enter")  # the Confirm dialog's default: download
            await app.workers.wait_for_complete()
            return name

    name = asyncio.run(go())
    assert tui.config.load()["default_model"] == name
    assert notes == ([f"Downloaded {name}; it is now the default"] if toast else [])


def test_button_labels_put_the_key_in_brackets_before_the_text():
    assert tui.dialogs.label("Add", "n") == "[dim]\\[n][/] Add"
    from textual.content import Content

    assert Content.from_markup(tui.dialogs.label("Close", "esc")).plain == "[esc] Close"
