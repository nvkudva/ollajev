# Install Ollajev as a command on your PATH (Windows).
#
#   irm https://raw.githubusercontent.com/nvkudva/ollajev/main/install.ps1 | iex
#   .\install.ps1                  # from a checkout: installs that checkout
#   .\install.ps1 -Uninstall
#
# The background service (-Service on macOS/Linux) is not available on Windows; run `ollajev serve`.
param(
  [string]$Source = "",
  [switch]$Uninstall
)
$ErrorActionPreference = "Stop"
# Pinned to a release tag, so a piped install never builds an unreviewed branch tip.
$RepoUrl = "git+https://github.com/nvkudva/ollajev@v0.2.0"

function Say($msg) { Write-Host "==> $msg" }

if ($Uninstall) {
  if (Get-Command uv -ErrorAction SilentlyContinue) { uv tool uninstall ollajev }
  Say "kept your config (~\.ollajev) and models (~\.cache\huggingface\hub)"
  exit 0
}

if (-not $Source) {
  $here = if ($PSScriptRoot) { $PSScriptRoot } else { "" }
  if ($here -and (Test-Path "$here\pyproject.toml") -and (Select-String -Quiet -Pattern '^name = "ollajev"' "$here\pyproject.toml")) {
    $Source = $here
  } else {
    $Source = $RepoUrl
  }
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Say "installing uv (https://docs.astral.sh/uv/)"
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

Say "installing Ollajev from $Source (Python 3.12, its own environment)"
uv tool install --python 3.12 --force $Source
uv tool update-shell | Out-Null

Say "done. Open a new terminal, then run:  ollajev"
