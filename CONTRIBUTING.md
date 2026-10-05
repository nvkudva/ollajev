# Contributing

## Setup

```sh
git clone https://github.com/nvkudva/ollajev.git
cd ollajev
uv sync                      # Python 3.12 environment with the dev tools
uv run ollajev --help
```

## Checks

Run these before opening a pull request. CI runs the same ones.

```sh
uv run ruff check src tests
uv run ruff format --check src tests
uv run pyright
uv run pytest
```

`uv run pre-commit install` runs ruff on every commit.

The tests load no model weights. To check a change against a real model, run the server and send
it a request:

```sh
OLLAJEV_HOME=/tmp/ollajev-dev uv run ollajev serve SupersonicLabs/Julia-1 --port 8765
```

## Layout

| Path | What it holds |
|---|---|
| `src/ollajev/ui/` | front ends: `cli.py` command line, `repl.py` for `ollajev run`, `tui.py` model manager |
| `src/ollajev/server/` | HTTP routes: `api.py` Jev API, `admin.py` management API, `static/` playground page |
| `src/ollajev/client.py` | talking to a running server, for the front ends |
| `src/ollajev/manager.py` | loading, unloading and running models |
| `src/ollajev/store.py`, `names.py` | model names, pinned downloads, trust |
| `src/ollajev/adapters/` | one module per model family |
| `src/ollajev/normalize.py` | answers in the TypeSafe shape |
| `src/ollajev/_vendor/kev/` | kev's loader, vendored; do not edit except as noted in `VENDORED.md` |

Everything else in `src/ollajev/` is the core. The core never imports `server/` or `ui/`, and `server/`
never imports `ui/`; `tests/test_layout.py` checks this.

## Adding a model family

Add a module under `src/ollajev/adapters/` with a `FAMILY` object that implements the `Family`
protocol in `adapters/__init__.py` (`matches`, `allow_patterns`, `limits`, `load`), and register it
in `families()`. Set `runs_repo_code = True` if it imports Python from the model repo. Run the new
model end to end through `/v1/systemone` before adding it to `catalog.py`.

## Running tests

```sh
uv run pytest
```

## Lint and type check

```sh
uv run ruff check
uv run pyright
```

## Releasing

1. Bump `version` in `pyproject.toml`.
2. Move the `[Unreleased]` entries in `CHANGELOG.md` under the new version.
3. Commit, then tag `vX.Y.Z` and push the tag.

`.github/workflows/release.yml` checks the tag matches `pyproject.toml`, builds, publishes to PyPI
with trusted publishing, and creates the GitHub release.

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:` …) and
add a line to `CHANGELOG.md` under `[Unreleased]` for user-visible changes.

## License

Contributions are made under the Apache-2.0 license, the same as the project. There is no CLA.
Add a `Signed-off-by` line (`git commit -s`) if you like; it is not required.
