# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.3.0] - 2026-10-10

### Added

- `Cloudflare/clef-omni` (Qwen3-Omni 30B-A3B: images, audio, video) and `LiquidAI/d1-omni-600M` (new `d1`
  family: images or a voice clip) in the catalog.
- `/v1/systemone` takes `images`, `audio` and `videos` as base64 data URLs; `limits.inputs` says which a
  model reads. clef and clef-flash now read images and video.
- New `clef-mlx` family: Clef, Clef-Flash and Clef-Omni on MLX for Apple Silicon, in 4 and 8 bits
  (`mlx-community/clef-flash-4bit` 6.2 GB, `clef-4bit` 16.3 GB, `clef-omni-4bit` 19.8 GB, and their `-8bit` copies).
- The playground lists every curated model and downloads one from the page (`GET /ui/catalog`, `/api/pull`).
- The playground attaches images, audio and video for models that read them.

### Fixed

- clef downloads now include the script the family is detected by, so a pulled clef model resolves offline (the
  server failed with "not a supported System One model"). Snapshots pulled before the fix are repaired by
  running `ollajev pull` again.
- Unloading an MLX model releases its Metal memory, so the next model fits.

### Changed

- `OLLAJEV_MAX_BODY_BYTES` defaults to 64 MiB, to fit a video.
- Dependencies: `pillow`, `av` (PyAV) and `torchvision` (the clef processors need it); `mlx-vlm` 0.7.x on
  Apple Silicon Macs only.

## [0.2.0] - 2026-10-05

First release on PyPI: `uv tool install ollajev`.

### Security

- `POST /api/pull` no longer accepts `trust`; trusting a repo's Python code is CLI-only.
- `OLLAJEV_API_KEY` requires a bearer token on every API route. Listening on a non-loopback
  address without it is refused. On loopback, only `localhost`, `127.0.0.1` and `[::1]` are
  accepted as Host (DNS-rebinding guard).
- Model names reject `.`/`..` segments and foreign URL hosts.
- Requests over `OLLAJEV_MAX_BODY_BYTES` (8 MiB) get 413. The demo page sends a Content-Security-Policy.

### Changed

- The demo page is now the playground, at `/playground`; `/demo` redirects there. The model manager's
  button and the server banner say Playground.
- `ollajev setup` (alias `tui`) is now a model manager: download, switch the default, unload, delete,
  alias, inspect and serve from one screen. A Selected card shows the model at the cursor with buttons for
  what applies to it, a Server card shows the running server with Playground, Logs, Restart and Stop, and the
  status row lists the keys. Every button shows its key as `[key] label`. Ask is hidden for now.
- The model manager uses the terminal's own colours and has no theme picker.
- Keys: `a` adds a model, `f` filters, `c` gives a short name, `s` serves the selected model, `R` restarts
  and `S` stops the server, `l` follows the server logs in a new terminal tab.
- The model manager starts faster: listing downloads no longer imports PyTorch.
- The playground opens on the model the server has loaded.

- Config and logs now live in `~/.ollajev` (override with `OLLAJEV_HOME`). A config in the old OS
  folder is read once and moved on the next save.
- Dependencies use compatible ranges in `pyproject.toml`; `uv.lock` keeps exact versions.
- Installers pin to a release tag.

### Fixed

- `install.sh` checks for a C and C++ compiler on Linux, which llama.cpp needs, and says how to install one.
- A failing `systemctl` or `launchctl` call in `ollajev service` ends with its message, not a traceback.
- Config writes are locked and atomic; a corrupt `config.json` is moved to `config.json.bad`.
- A request can no longer run on a model the reaper just unloaded; eviction reads the slot table
  under its lock.
- Two pulls of the same repo cannot run at once; unexpected pull errors are logged, not sent to the client.
- Repo-code imports are serialised.
- Removing one quant no longer deletes a blob another revision still uses.
- A failed download no longer leaves a pin.
- A bad `OLLAJEV_HOST`, `OLLAJEV_KEEP_ALIVE` or `OLLAJEV_MAX_LOADED_MODELS` gives a one-line error naming the variable.

### Added

- Jev / System One API (`GET /v1/models`, `POST /v1/systemone`), drop-in for `typesafe-sdk`.
- Adapters for seven model families: decider (PyTorch and llama.cpp GGUF), laya, Julia,
  open-jev, kev, Intern-Decision and Decision-1.0. Sixteen curated models under 4 GB.
- Ollama-style model names (`repo`, `repo:quant`, `repo:file.gguf`, `hf.co/` prefix), pinned to the
  commit of first download.
- Model manager: load on request, `keep_alive` unload, least-recently-used eviction.
- Trust prompt per repo and commit for families that run code from the model repo.
- Ollama-style management API (`/api/tags`, `ps`, `pull`, `show`, `delete`, `copy`, `stop`) and
  CLI (`serve`, `setup`, `run`, `pull`, `list`, `ps`, `show`, `rm`, `stop`, `cp`, `service`).
- Textual setup screen on first run; interactive `run`.
- Demo page with a model picker and per-model limits.
- `install.sh` / `install.ps1`, and a background service (launchd, systemd user unit).
