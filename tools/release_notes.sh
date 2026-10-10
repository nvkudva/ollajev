#!/bin/sh
# Print CHANGELOG.md's section for version $1 (no leading v), with each wrapped bullet joined onto one line:
# GitHub release notes render every newline as a line break.
awk -v v="$1" '
  index($0, "## [" v "]") == 1 { f = 1; next }
  /^## \[/ { f = 0 }
  !f { next }
  /^  [^ ]/ && buf != "" { sub(/^ +/, ""); buf = buf " " $0; next }
  { if (started) print buf; buf = $0; started = 1 }
  END { if (started) print buf }
' "${2:-CHANGELOG.md}"
