# FSRCNNX-EXT

FSRCNNX-EXT is a pre-release Chromium extension for local WebGPU video enhancement. It supports FSRCNNX, ArtCNN, and bundled Real-ESRGAN or temporal CDA-VSR ONNX upscaling; optional SSimDownscaler and sharpening; and RIFE or blend frame interpolation.

## Requirements

- A current Chromium-based browser with WebGPU and a compatible GPU.
- A readable, non-DRM BT.709/sRGB SDR video in the top-level page. HDR, wide-gamut, cross-origin, iframe, and page-specific restrictions may leave a video on the browser's native renderer.
- A strict host Content Security Policy can prevent RIFE from starting. Neural runs in an extension-owned frame to avoid that specific restriction.
- Node.js 20.11 or newer and CPython 3.11 or newer for repository checks and
  packaging. The extension itself has no npm or Python dependency.

Processing stays on the device. See [Privacy](PRIVACY.md) for the data and permission boundaries.

## Install and use

1. Open `chrome://extensions`, enable **Developer mode**, and choose **Load unpacked**.
2. Select this repository's root directory.
3. Play a supported video, open the extension popup, choose an engine and its available options, then select **Upscale**.
4. Select **Off** to restore native rendering. Reload the extension after changing the checkout.

Settings are stored per site; local-file pages share one local-file scope. The extension has no configured source-resolution or pixel-area ceiling. Requested inputs, outputs, and model resources must still fit the browser's and GPU adapter's actual limits. Neural offers tiled Real-ESRGAN AnimeVideo XS 2× and temporal CDA-VSR 4×; CDA can present an exact 2× result by downscaling its native 4× output. CDA-VSR targets compressed-video fidelity but is compute- and memory-intensive and may not keep pace with HD playback. Neural pauses frame interpolation while selected.

Performance-driven fallbacks are advanced per-site settings and are off by default. **Automatic quality fallback** lowers FSRCNNX High, ArtCNN, or Neural to standard FSRCNNX after sustained frame drops or GPU backlog; **Automatic blend fallback** replaces RIFE when it cannot maintain a useful frame rate.

## Repository layout

- `src/` — extension entry points, popup, and runtime modules.
- `model/`, `shaders/`, and `vendor/` — runtime assets and pinned third-party sources.
- `docs/compliance/` and `LICENSES/` — release records, rebuilding guidance, and license texts.
- `validate.html` and `validation/` — browser validation entry point and numerical reference fixtures.
- `tests/` — deterministic Node and browser fixtures.
- `tools/` — checks, packaging, reproduction utilities, and transpilers.

## Validate and package

Run the standard offline repository check:

```sh
npm run check
```

This includes dependency-free CDA-VSR contract tests and exact bundled-model
hash checks. [`tools/cda-vsr/`](tools/cda-vsr/) documents conversion,
reproduction, deterministic promotion of the reviewed bundled pair, and an
isolated browser probe for external exports.

Run browser and GPU smoke checks against the checkout in a temporary Chromium profile:

```sh
npm run validate:browser
```

Set `FSRCNNX_BROWSER` if no browser is found automatically. The manually loaded GPU suite is at `chrome-extension://<extension-id>/validate.html`.

`npm run package:internal` creates a deterministic validation archive and staged extension under `dist/`. Validate that exact staged package with:

```sh
npm run validate:browser -- --extension-root dist/fsrcnnx-ext
```

`npm run package` enforces the release-clearance ledger, creates the
deterministic archive, and validates the exact staged CDA-VSR model through
initializer and recurrent real-video runs in a temporary browser profile.

## Licensing

Project-authored source code and documentation are licensed under
[Apache-2.0](LICENSE). This does not relicense third-party or derived
material; see [NOTICE](NOTICE), [Third-party notices](THIRD_PARTY_NOTICES.md),
[Model provenance](docs/compliance/MODEL_PROVENANCE.md), and
[LGPL rebuilding instructions](docs/compliance/LGPL_REBUILDING.md).

Before opening a pull request, read [Contributing](.github/CONTRIBUTING.md). Report vulnerabilities through the [security policy](.github/SECURITY.md).
