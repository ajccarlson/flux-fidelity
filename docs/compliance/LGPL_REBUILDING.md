# Rebuilding and replacing LGPL-covered portions

This package provides the source and practical replacement information below for its LGPL-3.0-or-later portions. It does not represent a completed legal review or clear the `lgpl-compliance-review` release gate.

## Covered portions and exact sources

- `shaders/upstream/FSRCNNX_x2_16-0-4-1.glsl` and its transformed outputs, `model/FSRCNNX_x2_16-0-4-1.wgsl` and `model/FSRCNNX_x2_16-0-4-1.passes.json`. The source is the FSRCNN-TensorFlow 1.1 release asset at <https://github.com/igv/FSRCNN-TensorFlow/releases/download/1.1/FSRCNNX_x2_16-0-4-1.glsl>, SHA-256 `d5a24a271e5d9a3f7f7a053b150c460a44c25b3cf7f770857d57cc3a2e1c9965`. Release tag commit: `1aa11ab0e1fc12741fdb84cef31da5619a478670`; the asset checksum identifies the later-attached bytes.
- `shaders/upstream/FSRCNNX_x2_56-16-4-1.glsl` and its transformed outputs, `model/FSRCNNX_x2_56-16-4-1.wgsl` and `model/FSRCNNX_x2_56-16-4-1.passes.json`. Archived release-1.1 pages list the High source through 2019-10-21, and the archived parameter bundle plus tag-1.1 generator reproduce the checked-in 362,441-byte LF source exactly. The source SHA-256 is `34cd5d0087ebb6ae5f9bff2578382205457da53baa364d52de8021d6925b7fd6`; the full reproduction record is in [`MODEL_PROVENANCE.md`](MODEL_PROVENANCE.md).
- `shaders/upstream/SSimDownscaler.glsl` and its hand port, `src/core/fsrcnnx-ssimds.js`. The source is gist revision `38992bce7f9ff844f800820df0908692b65bb74a` at <https://gist.githubusercontent.com/igv/36508af3ffc84410fe39761d6969be10/raw/38992bce7f9ff844f800820df0908692b65bb74a/SSimDownscaler.glsl>, SHA-256 `f46f4710a162d17058b9d82ed8610588b0c04d7be07cef6bf2a8c4077828f804`.

The Standard and SSim sources state LGPL-3.0-or-later. The High source has no embedded header; this project classifies the officially released High source and its generated pair as LGPL-3.0-or-later consistently with Standard. That classification and the distribution plan remain subject to qualified review. [`LICENSES/LGPL-3.0.txt`](../../LICENSES/LGPL-3.0.txt) and its GPL-3.0 companion, [`LICENSES/GPL-3.0.txt`](../../LICENSES/GPL-3.0.txt), are included in this package.

## Local modifications

- On 2026-07-21, FSRCNNX-EXT translated the Standard and High FSRCNNX mpv/libplacebo GLSL hook passes into WGSL compute passes plus JSON pass manifests. The model weights and pass order are preserved. `tools/transpile.js` is the complete offline transformation script and records source/license metadata in the outputs.
- In 2026, FSRCNNX-EXT hand-ported SSimDownscaler to WebGPU/WGSL, reorganized the original hook operations into explicit separable passes, and added finite-ratio, work-budget, and zero-weight numerical guards. The port's header and `src/core/fsrcnnx-ssimds.js` identify these changes.

## Offline FSRCNNX regeneration

From the unpacked package directory, install Node.js 20.11 or newer. No npm dependencies or network access are needed. Verify and regenerate the checked-in models with:

```sh
sha256sum shaders/upstream/FSRCNNX_x2_16-0-4-1.glsl
sha256sum shaders/upstream/FSRCNNX_x2_56-16-4-1.glsl
node tools/transpile.js shaders/upstream/FSRCNNX_x2_16-0-4-1.glsl --out model
node tools/transpile.js shaders/upstream/FSRCNNX_x2_56-16-4-1.glsl --out model
```

The hash commands must report the source hashes above. `tools/transpile.js` rejects different bytes under either verified filename. To build a modified version, update the source, its expected hash, and its modification notice in the corresponding verified-source entry inside `tools/transpile.js`, then run the same command. The generated files retain the modified source hash and notice.

## SSimDownscaler modification and substitution

There is no automatic SSimDownscaler transpiler. Use `shaders/upstream/SSimDownscaler.glsl` as the exact upstream basis, make the desired changes, and carry them into `src/core/fsrcnnx-ssimds.js`. A drop-in replacement must continue to export `buildMeanShader`, `buildL2Shader`, `SSIMDS_MR_WGSL`, and `SSIMDS_FINAL_WGSL`, because `src/core/fsrcnnx-ssimds-runtime.js` imports that interface. Record the changes in the replacement's source header.

You may instead replace the generated FSRCNNX `.wgsl` and `.passes.json` files or the SSimDownscaler JavaScript module directly, provided the replacement preserves the filenames and runtime data/interface contracts.

## Loading a replacement in Chromium

1. Extract the extension package to a writable directory containing `manifest.json`.
2. Regenerate or replace the LGPL-covered files as described above.
3. Open `chrome://extensions` (or the Chromium equivalent), enable **Developer mode**, choose **Load unpacked**, and select that directory.
4. Disable any other installed copy of FSRCNNX-EXT, then use **Reload** on the unpacked copy after subsequent changes.

This installs the recipient-modified files directly; no signing, store upload, or project-controlled service is required.
