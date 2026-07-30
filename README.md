<p align="center">
  <img src="icons/icon-128.png" alt="Flux Fidelity" width="160">
</p>

# Flux Fidelity

Flux Fidelity is a Chromium extension for local WebGPU video enhancement. It
combines real-time upscaling, filtering, and optional frame interpolation while
keeping video processing on the device.

## Core features

- FSRCNNX and ArtCNN WebGPU upscaling
- Bundled Real-ESRGAN and temporal CDA-VSR ONNX upscaling
- RIFE neural or blend frame interpolation
- Optional SSimDownscaler and adaptive sharpening
- Per-site settings with opt-in quality and interpolation fallbacks
- No configured source-resolution or pixel-area ceiling

## Technology

Flux Fidelity uses Chromium Manifest V3, JavaScript, WebGPU, WGSL, ONNX Runtime
Web, Node.js, and Python. The installed extension has no npm or Python runtime
dependency, and all supported processing runs locally.

## Local installation

Requirements:

- A current Chromium-based browser with WebGPU
- A compatible GPU
- A readable, non-DRM BT.709/sRGB SDR video in the top-level page

1. Open `chrome://extensions`.
2. Build the packaged extension: `npm run package:stage`. Loading the repository
   root instead works but also loads the test suite, tooling, and Git metadata.
3. Enable **Developer mode** and select **Load unpacked**.
4. Choose `dist/flux-fidelity`.
5. Play a supported video, open FluFi, choose an engine, and select
   **Upscale**.
6. Select **Off** to restore native rendering. `Alt+Shift+U` toggles
   enhancement without opening the popup, including while fullscreen.

Reload the extension after changing the checkout. HDR, wide-gamut,
cross-origin, iframe, DRM, and page-specific restrictions may leave a video on
the browser's native renderer. A strict host Content Security Policy can
prevent RIFE from starting; neural upscaling runs in an extension-owned frame
to avoid that restriction.

Settings are stored per site, while local-file pages share one local-file
scope. Automatic quality and blend fallbacks are advanced settings and are off
by default.

## Validation

Development requires Node.js 20.11 or newer and CPython 3.11 or newer. Run the
standard offline repository check:

```sh
npm run check
```

Run browser and GPU validation against the checkout:

```sh
npm run validate:browser
```

The complete release gate is:

```sh
npm run release:check
```

This verifies source and generated assets, deterministic model identities,
release records, the packaged extension, and temporal neural inference on real
video. Set `FSRCNNX_BROWSER` when browser auto-detection is insufficient.

## Documentation

- [Privacy](PRIVACY.md)
- [Contributing](.github/CONTRIBUTING.md)
- [Security policy](.github/SECURITY.md)
- [Browser and numerical validation](validation/README.md)
- [Model provenance](docs/compliance/MODEL_PROVENANCE.md)
- [LGPL rebuilding instructions](docs/compliance/LGPL_REBUILDING.md)
- [CDA-VSR conversion and browser probe](tools/cda-vsr/README.md)
- [RIFE FP16 reproduction](tools/model-reproduction/README.md)

## Models and upstream sources

Flux Fidelity includes or derives assets from FSRCNNX, ArtCNN, Real-ESRGAN,
CDA-VSR, RIFE, SSimDownscaler, adaptive-sharpen, and ONNX Runtime Web. Exact
sources, revisions, hashes, transformations, and license evidence are recorded
in [Model provenance](docs/compliance/MODEL_PROVENANCE.md). Required
attributions and license texts are preserved in [NOTICE](NOTICE),
[Third-party notices](THIRD_PARTY_NOTICES.md), and [`LICENSES/`](LICENSES/).

## License

Project-authored source code and documentation are licensed under
[Apache-2.0](LICENSE). This does not relicense third-party or derived material,
which remains subject to its respective licenses and terms.

## Maintainer

[Aaron Carlson](https://github.com/ajccarlson)
