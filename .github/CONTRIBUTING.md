# Contributing

FSRCNNX-EXT is pre-release software. Keep changes focused and open pull
requests against `develop`; `main` is reserved for stable integration.

## Before opening a pull request

1. Use Node.js 20.11 or newer.
2. Run `npm run check`.
3. For runtime, shader, model, or packaging changes, run
   `npm run package:internal`, then
   `npm run validate:browser -- --extension-root dist/fsrcnnx-ext`.
4. Explain user-visible behavior and validation results in the pull request.

Changes to models, shaders, generated files, or vendored code must record the
exact upstream source, version or commit, SHA-256 digest, license, local
changes, and reproduction procedure. Artifacts with unknown or incompatible
provenance cannot be accepted.

Report security-sensitive findings through the process in
[SECURITY.md](SECURITY.md), not a public issue.
