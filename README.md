# FSRCNNX-EXT

FSRCNNX-EXT is a pre-release Chromium extension for real-time WebGPU video enhancement. It supports FSRCNNX and ArtCNN upscaling, optional SSimDownscaler and sharpening, RIFE or blend interpolation, and an experimental ONNX super-resolution path.

Public distribution is not cleared because some bundled artifacts still have unresolved provenance or licensing records. Review [Model provenance](docs/compliance/MODEL_PROVENANCE.md) before packaging or publishing.

## Requirements

- A current Chromium-based browser with WebGPU and a compatible GPU.
- A readable, non-DRM BT.709/sRGB SDR video in the top-level page. HDR, wide-gamut, cross-origin, iframe, and page-specific restrictions may leave a video on the browser's native renderer.
- Node.js 20.11 or newer for repository checks and packaging. The extension itself has no npm dependency.

Processing stays on the device. See [Privacy](PRIVACY.md) for the data and permission boundaries.

## Install and use

1. Open `chrome://extensions`, enable **Developer mode**, and choose **Load unpacked**.
2. Select this repository's root directory.
3. Play a supported video, open the extension popup, choose an engine and policy, then select **Upscale**.
4. Select **Off** to restore native rendering. Reload the extension after changing the checkout.

Settings are stored per origin. Available combinations depend on browser support, GPU memory, source resolution, and model cost. The ONNX super-resolution option remains unavailable until a compatible licensed model is added to `model/neural/manifest.json`.

## Repository layout

- `src/` — extension entry points, popup, and runtime modules.
- `model/`, `shaders/`, and `vendor/` — runtime assets and pinned third-party sources.
- `docs/compliance/` and `LICENSES/` — release records, rebuilding guidance, and license texts.
- `validation/` — browser validation page and numerical reference fixtures.
- `tests/` — deterministic Node and browser fixtures.
- `tools/` — checks, packaging, reproduction utilities, and transpilers.

## Validate and package

Run the complete offline check:

```sh
npm run check
```

Run browser and GPU smoke checks against the checkout in a temporary Chromium profile:

```sh
npm run validate:browser
```

Set `FSRCNNX_BROWSER` if no browser is found automatically. The manually loaded GPU suite is at `chrome-extension://<extension-id>/validate.html`.

`npm run package:internal` creates a deterministic validation archive and staged extension under `dist/`. Validate that exact staged package with:

```sh
npm run validate:browser -- --extension-root dist/fsrcnnx-ext
```

`npm run package` also enforces the public-release gate and currently stops on unresolved items in [Model provenance](docs/compliance/MODEL_PROVENANCE.md).

## Licensing

Project-owned source is covered by [LICENSE](LICENSE). Third-party and derived material retains its own terms; see [Third-party notices](THIRD_PARTY_NOTICES.md), [Model provenance](docs/compliance/MODEL_PROVENANCE.md), and [LGPL rebuilding instructions](docs/compliance/LGPL_REBUILDING.md).

Before opening a pull request, read [Contributing](.github/CONTRIBUTING.md). Report vulnerabilities through the [security policy](.github/SECURITY.md).
