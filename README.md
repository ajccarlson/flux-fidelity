# FSRCNNX-EXT

FSRCNNX-EXT is a pre-release Chromium extension for real-time WebGPU video enhancement. It provides FSRCNNX and ArtCNN upscaling, optional SSimDownscaler, sharpening and debanding, RIFE or blend frame interpolation, and an experimental ONNX super-resolution path.

Public distribution is not yet cleared. Several bundled artifacts still have unresolved provenance or licensing records; see [Model provenance](MODEL_PROVENANCE.md) before packaging or publishing the extension.

## Requirements

- A current Chromium-based browser with WebGPU enabled and a compatible GPU driver.
- A readable, non-DRM HTML5 video in the top-level page. Protected video, cross-origin restrictions, iframes, and page-specific rendering can prevent capture.
- Node.js 20.11 or newer only for repository validation and packaging. Runtime use has no npm dependency.

The extension requests access to all sites so it can find and process eligible page videos.

## Install locally

1. Open `chrome://extensions` in Chromium.
2. Enable **Developer mode**.
3. Choose **Load unpacked** and select this directory.
4. Reload the extension from that page after updating the checkout.

## Use

1. Open a page containing a video and start playback.
2. Open the extension popup.
3. Select an upscaling engine and policy, then choose **Upscale**. Optional filters and interpolation can be enabled separately.
4. Choose **Off** to restore normal page rendering.

Settings are stored per origin. GPU memory, source resolution, browser support, and model cost determine which combinations can sustain real-time playback. The bundled `SPAN 2x SMOKE` model has random weights and is only a pipeline test.

## Validate

Run the complete repository check with:

```sh
npm run check
```

Run the production-pipeline and model-inference smoke checks in a temporary local Edge/Chrome/Chromium profile with:

```sh
npm run validate:browser
```

Set `FSRCNNX_BROWSER` to an executable path when no supported browser is found automatically. After loading the extension manually, the GPU validation suite is available at `chrome-extension://<extension-id>/validate.html`; the ID is shown on `chrome://extensions`.

`npm run package` creates a deterministic local archive under `dist/` after running the checks. A successful archive build verifies technical integrity only; `npm run release:check` must also pass before publication and currently reports the unresolved gates in [Model provenance](MODEL_PROVENANCE.md).

## Licensing

The project-owned source is all rights reserved; see [LICENSE](LICENSE). Bundled and derived third-party material remains under its own terms, recorded in [Third-party notices](THIRD_PARTY_NOTICES.md) and [Model provenance](MODEL_PROVENANCE.md).
