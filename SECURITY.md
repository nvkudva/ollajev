# Security Policy

## Supported versions

Only the latest 0.x release receives security fixes.

## Reporting a vulnerability

Report privately through a [GitHub security advisory](https://github.com/nvkudva/ollajev/security/advisories/new). Do not open a public issue or pull request. Include the version, how to reproduce, and the impact.

Expect an acknowledgement within 7 days and a status update within 30 days. This is a volunteer project, so these are goals, not guarantees. Fixes are released as a new 0.x version and credited in the advisory unless you prefer otherwise.

## Threat model

- **Local by default.** The server binds `127.0.0.1`. A non-loopback bind requires `OLLAJEV_API_KEY` (a bearer token); without it the server refuses to start.
- **DNS rebinding.** On loopback, requests are checked against a `Host` header allow-list.
- **Trust is never remote.** The HTTP API cannot mark a repo trusted. Trust is set only from the CLI, interactively or with `--trust`, and is keyed to `repo@commit sha`, so a new commit needs new trust.
- **Model code runs unsandboxed.** The families Julia, open-jev, Intern and Decision-1 import Python from the model repo (`trust_remote_code` / `sys.path`) in the server process, with your privileges and no sandbox. Only trust repos and commits you have reviewed.
- **Playground page.** The built-in playground page does not work when an API key is set.
