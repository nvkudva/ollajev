#!/bin/sh
# Install Ollajev as a command on your PATH, optionally as a background service.
#
#   curl -fsSL https://raw.githubusercontent.com/nvkudva/ollajev/main/install.sh | sh
#   curl -fsSL https://raw.githubusercontent.com/nvkudva/ollajev/main/install.sh | sh -s -- --service
#   ./install.sh                 # from a checkout: installs that checkout
#   ./install.sh --uninstall
#
# Options:
#   --service         also run the server in the background at login (macOS launchd, Linux systemd)
#   --source <spec>   what to install: a path, a git URL, or a PyPI requirement
#   --uninstall       remove the service and the command (your config and models are kept)
set -eu

# Pinned to a release tag, so a piped install never builds an unreviewed branch tip.
REPO_URL="git+https://github.com/nvkudva/ollajev@v0.2.0"
SERVICE=0
UNINSTALL=0
SOURCE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --service) SERVICE=1 ;;
    --uninstall) UNINSTALL=1 ;;
    --source) shift; SOURCE="${1:?--source needs a value}" ;;
    -h|--help) sed -n '2,13p' "$0" 2>/dev/null | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
  shift
done

say() { printf '==> %s\n' "$*"; }

export PATH="$HOME/.local/bin:$PATH"

if [ "$UNINSTALL" = 1 ]; then
  if command -v ollajev >/dev/null 2>&1; then
    ollajev service uninstall >/dev/null 2>&1 || true
  fi
  if command -v uv >/dev/null 2>&1; then
    uv tool uninstall ollajev >/dev/null 2>&1 && say "removed the ollajev command" || say "Ollajev was not installed"
  fi
  say "kept your config and downloaded models; delete them by hand if you want the space back:"
  echo "    config  ~/.ollajev"
  echo "    models  ~/.cache/huggingface/hub (shared with other Hugging Face tools)"
  exit 0
fi

# A checkout installs itself; a piped install takes the GitHub repo.
if [ -z "$SOURCE" ]; then
  here=$(CDPATH= cd -- "$(dirname -- "$0")" 2>/dev/null && pwd || true)
  if [ -n "$here" ] && [ -f "$here/pyproject.toml" ] && grep -q '^name = "ollajev"' "$here/pyproject.toml"; then
    SOURCE="$here"
  else
    SOURCE="$REPO_URL"
  fi
fi

# PyPI ships llama-cpp-python as source only, so llama.cpp is compiled during the install.
if [ "$(uname -s)" = Linux ] && ! { command -v cc >/dev/null 2>&1 && command -v c++ >/dev/null 2>&1; }; then
  echo "Ollajev needs a C and C++ compiler on Linux to build llama.cpp. Install one, then run this again:" >&2
  echo "    Debian, Ubuntu:  sudo apt install build-essential" >&2
  echo "    Fedora:          sudo dnf install gcc gcc-c++" >&2
  echo "    Arch:            sudo pacman -S base-devel" >&2
  exit 1
fi

# Compile a C++ include, not just `xcode-select -p`: an update can leave the tools without their C++ headers.
if [ "$(uname -s)" = Darwin ] && ! printf '#include <mutex>\n' | c++ -x c++ -std=c++17 -fsyntax-only - >/dev/null 2>&1; then
  if xcode-select -p >/dev/null 2>&1; then
    echo "Ollajev builds llama.cpp, but your Xcode Command Line Tools cannot compile C++ (headers missing)." >&2
    echo "Reinstall them, then run this again:" >&2
    echo "    sudo rm -rf /Library/Developer/CommandLineTools" >&2
    echo "    xcode-select --install" >&2
  else
    echo "Ollajev builds llama.cpp, which needs the Xcode Command Line Tools. Install them, then run this again:" >&2
    echo "    xcode-select --install" >&2
  fi
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  say "installing uv (https://docs.astral.sh/uv/)"
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  else
    wget -qO- https://astral.sh/uv/install.sh | sh
  fi
  command -v uv >/dev/null 2>&1 || { echo "uv install failed; see https://docs.astral.sh/uv/" >&2; exit 1; }
fi

say "installing Ollajev from $SOURCE (Python 3.12, its own environment)"
uv tool install --python 3.12 --force "$SOURCE"

case ":$PATH:" in
  *":$(uv tool dir --bin):"*) ;;
  *) uv tool update-shell >/dev/null 2>&1 || true
     say "added $(uv tool dir --bin) to your PATH; open a new terminal to use ollajev" ;;
esac

if [ "$SERVICE" = 1 ]; then
  say "installing the background service"
  "$(uv tool dir --bin)/ollajev" service install
fi

say "done. Next:"
if [ "$SERVICE" = 1 ]; then
  echo "    ollajev pull Mapika/decider-4b-GGUF:Q4_K_M   # download the default model"
  echo "    ollajev run                                  # ask it questions"
else
  echo "    ollajev            # first run: pick a model, then serve"
fi
