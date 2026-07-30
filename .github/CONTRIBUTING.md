# Contributing

Flux Fidelity is pre-release software. Keep changes focused and open pull
requests against `develop`; `main` is reserved for stable integration.

## Before opening a pull request

1. Use Node.js 20.11 or newer (`.nvmrc` pins the version CI verifies against).
2. Have **CPython 3.11 or newer on `PATH` as `python`**. `npm run check` invokes
   `python`, not `python3`, so on distributions that ship only `python3` the
   suite fails with an opaque exit code. The CDA feasibility tests additionally
   skip unless `tools/cda-vsr/requirements.txt` is installed; those three skips
   are expected.
3. Run `npm run check`.
4. For runtime, shader, model, or packaging changes, run
   `npm run package:internal`, then
   `npm run validate:browser -- --extension-root dist/flux-fidelity`.
   Browser validation needs Chromium and `xvfb-run` on `PATH`. On Ubuntu 24.04
   and newer it also needs an AppArmor profile permitting Chromium's user
   namespaces; `.github/workflows/ci.yml` shows the exact profile the CI job
   installs. `FSRCNNX_TIMEOUT_SCALE` multiplies the harness wait budgets if your
   machine is slower than the defaults assume.
   Note that `npm run check` compiles no WGSL and touches no GPU, so shader and
   presentation changes are only actually exercised by this step.
5. Explain user-visible behavior and validation results in the pull request.

Changes to models, shaders, generated files, or vendored code must record the
exact upstream source, version or commit, SHA-256 digest, license, local
changes, and reproduction procedure. Artifacts with unknown or incompatible
provenance cannot be accepted.

Unless explicitly stated otherwise, contributions intentionally submitted for
inclusion are licensed under Apache-2.0 without additional terms, as described
in section 5 of [LICENSE](../LICENSE). Do not submit material you do not have
the right to license.

Report security-sensitive findings through the process in
[SECURITY.md](SECURITY.md), not a public issue.
