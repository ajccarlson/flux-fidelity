# Model and generated-asset provenance

Audit date: 2026-07-21.

This inventory is a release gate for bundled model, generated shader, and inference-runtime artifacts. A matching hash proves byte identity, not ownership or permission. `Verified` means the recorded bytes, source, and stated upstream license were checked. `Blocked` means at least one public-distribution surface remains unsafe. A removed artifact remains as a historical tombstone when it is absent from the current tree and package but still exists in private Git history.

The machine-readable gate ledger is [`release-clearance.json`](release-clearance.json). `npm run release:check` must pass before public distribution; it intentionally fails while any recorded gate remains blocked.

## Unresolved public-release gates

- The removed unidentified `model/rife.onnx` still exists in private Git history without an exact authoritative source, model-version record, or license.
- The removed high-quality x2 FSRCNNX source and generated assets still exist in private Git history without an established author, upstream revision, or license.
- The removed x3 and x4 FSRCNNX generated assets still exist in private Git history without retained source shaders or upstream revisions.
- The removed random-weight SPAN smoke model still exists in private Git history without a recorded random seed or exact generation environment.
- The removed deband port still exists in private Git history without an exact source/revision or established license.
- Exact LGPL upstream sources, license texts, source notices, generation tooling, and recipient rebuilding/substitution instructions are bundled, but public distribution of the LGPL-derived files still needs qualified legal review.
- ONNX Runtime's exact upstream MIT license and pinned third-party notice set are bundled, but the notices and obligations applicable to the distributed Web runtime files still need qualified review.

The current extension package still contains material covered by the LGPL and ONNX Runtime review gates. The unidentified RIFE model, random-weight SPAN smoke model, unverified FSRCNNX assets, and unresolved deband port are absent from both the current tree and package, but their private-history tombstones still block repository publication. Therefore, a successful `npm run check` or `npm run package:internal` does **not** make either distribution surface publicly releasable; `npm run package` enforces the blocking public-release check.

## ONNX models

| Artifact | SHA-256 | Evidence | License evidence | Status |
| --- | --- | --- | --- | --- |
| `model/rife_v4.26.onnx` | `af25762dfec02a4bbb949decea63988b01fa56c46c0ff9dc66ac8e2f12cbb661` | Byte-identical to `rife_v2/rife_v4.26.onnx` in [AmusementClub/vs-mlrt `rife_v4.26.7z`](https://github.com/AmusementClub/vs-mlrt/releases/download/external-models/rife_v4.26.7z). Archive SHA-256: `dfdabd84a2a3db773f87604b8cc255e94a6a72f13550d910ccd3b4ee2606cd4f`; member size: `22,748,049` bytes. The release points to Practical-RIFE commit [`de9a989bb7b8a71d94f058297e603633aaa43ad6`](https://github.com/hzwer/Practical-RIFE/tree/de9a989bb7b8a71d94f058297e603633aaa43ad6). The ONNX producer metadata says PyTorch 2.2.0. | The cited Practical-RIFE revision states that linked trained-model content is under the repository's MIT License (Copyright 2021 hzwer). | **Verified**; retain the MIT notice and exact source record. |
| `model/rife_v4.26_fp16.onnx` | `d5672f39b493609220c95c709542d6b99204145a67d9ca496d4500cd8895301f` | Deterministically derived from the preceding FP32 file, SHA-256 `af25762dfec02a4bbb949decea63988b01fa56c46c0ff9dc66ac8e2f12cbb661`, by [`reproduce-rife-fp16.py`](tools/model-reproduction/reproduce-rife-fp16.py). The converter is ONNX Runtime 1.27.0 `OnnxModel.convert_float_to_float16` with symbolic shape inference, FP32 public I/O, and a deterministic topological sort. CPython 3.11.15 and every wheel are hash-locked in [`requirements-rife-fp16.txt`](tools/model-reproduction/requirements-rife-fp16.txt); independent clean reproductions used container `python@sha256:db3ff2e1800a8581e2c48a27c3995339d47bdf046da21c7627accd3d51053a93` and produced identical bytes. The verifier runs the full ONNX checker, requires the exact 640-node inventory and audited cast surgery (six redundant INT64 casts removed and ten boundary casts inserted), checks every shared node's operation and attributes, verifies all 158 initializer conversions and all 22 FLOAT tensor-attribute conversions, and requires the exact 662-entry value-info dtype inventory (353 FLOAT16, 294 INT64, 9 FLOAT, and 6 BOOL). In the pinned environment, a deterministic structured 64×64 input representing a four-pixel horizontal shift at timestep 0.5 measured FP32-versus-FP16 CPU errors of maximum `0.002737591043114662`, mean `0.00009668370330473408`, and p99.9 `0.0016377191059291363`; 100% of samples were within `1/255`. Reproduction enforces maximum ≤ `0.005`, mean ≤ `0.0002`, and p99.9 ≤ `0.003`. Fixed inputs at 64×64, 80×96, and 96×128 are also bit-exact to the superseded historical serialization, SHA-256 `65494496be256b2809b6760f87a941c0d80bbc5cb09de0f3f6795488ae1fcd9d`. | Verified derivative of the preceding Practical-RIFE model; retain the parent's MIT notice and the reproduction evidence. | **Verified**; CI regenerates the model in the pinned environment and requires byte identity. |
| Historical `model/rife.onnx` | `6a31074c0f588648982b5e828aee6c27e005015a712a46ea63da48c65fa9a26b` | Six-channel dynamic ONNX export; producer metadata says PyTorch 2.5.1. It does not match by size or hash any file in the official vs-mlrt `rife_v2_v4.7z` older-model archive, and no authoritative exact-hash match was found. The file, runtime mapping, and user-facing choice have been removed from the current tree and extension package; stored `rife_orig` preferences migrate to verified RIFE 4.26. | Practical-RIFE's code/model license cannot establish that an unidentified binary came from that source. | **Repository publication remains blocked** until the private Git history is purged. |
| Historical `model/neural/span2x_smoke.fp16.onnx` | `ae7642c2b3bdd96e475dfcbf6a9180d8ec41175a1329d9f9b2a402f186c05c25` | Random-weight SPAN fixture associated with a superseded `tools/neural-export/export.py` smoke path; ONNX producer metadata says PyTorch 2.12.1. The current requirements instead pin PyTorch 2.13.0 and no generation seed or environment lock was retained. The file and random-weight export path have been removed, and the bundled neural manifest is intentionally empty. | The exporter depends on Spandrel 0.4.2, tag commit [`724cca389f28c38e1050689d4862a452fd644484`](https://github.com/chaiNNer-org/spandrel/tree/724cca389f28c38e1050689d4862a452fd644484), MIT. The artifact did not embed evidence that this exact dependency built it. | **Repository publication remains blocked** until the private Git history is purged. |

## Generated upscaler shaders

`npm run check:generated` regenerates the standard x2 and three ArtCNN pairs and requires byte-for-byte equality. The standard FSRCNNX manifest and WGSL header retain its source path, SHA-256, LGPL identifier, and modification notice. That proves local reproducibility from the checked-in source bytes; it does not by itself establish ownership or permission. Recipient rebuilding and Chromium substitution instructions are packaged in [`LGPL_REBUILDING.md`](LGPL_REBUILDING.md).

| Generated artifacts | SHA-256 | Source | License and status |
| --- | --- | --- | --- |
| `model/FSRCNNX_x2_16-0-4-1.wgsl` / `.passes.json` | `2b005b9c4e60c59445708b2f503c9afb01fd70ee9efb1615782274e7b7707f26` / `1378fc336deb2588f75ddf8b9ed6ec70109256f2c3fa0477ca742adf830fb3e0` | Packaged [FSRCNN-TensorFlow release 1.1 asset](shaders/upstream/FSRCNNX_x2_16-0-4-1.glsl), source SHA-256 `d5a24a271e5d9a3f7f7a053b150c460a44c25b3cf7f770857d57cc3a2e1c9965`. Authoritative download: <https://github.com/igv/FSRCNN-TensorFlow/releases/download/1.1/FSRCNNX_x2_16-0-4-1.glsl>. Release tag commit: [`1aa11ab0e1fc12741fdb84cef31da5619a478670`](https://github.com/igv/FSRCNN-TensorFlow/tree/1aa11ab0e1fc12741fdb84cef31da5619a478670); the downloadable asset was added later, so its checksum is the immutable identifier. | Source header: Copyright 2017-2021 igv, LGPL-3.0-or-later. The package includes `transpile.js` (SHA-256 `2ad45126cd36d52ce1064e8da1e189e10b5d256d8edc28a9dec3737957f4f631`) and rebuilding/substitution instructions (SHA-256 `b23f3a64a3db81248f1fe99dc1138d4b55aace39b162cfaefaa667836240effe`). **Source and reproduction verified; public-package LGPL compliance remains gated.** |
| Historical `shaders/FSRCNNX_x2_56-16-4-1.glsl`; `model/FSRCNNX_x2_56-16-4-1.wgsl` / `.passes.json` | `34cd5d0087ebb6ae5f9bff2578382205457da53baa364d52de8021d6925b7fd6`; `267ba203867483a467c535fd03c36c62ff9428116111d4d258dc5c295ef8e0d7` / `57395ac668b4cbebea69938a9089c9bea0029ce785f7cc6dad239c4be31d43e7` | The local source reproduced the generated pair, but no authoritative origin was established. All three files have been removed from the current tree and extension package. | **Repository publication remains blocked** because the unlicensed bytes remain in private Git history. Purge that history before making the repository public. |
| Historical `model/FSRCNNX_x3_16-0-4-1.wgsl` / `.passes.json` | `7df8eaae1c6f50bfd2b494d403cce8e1638c7408e0b92362b174cb72fdf816fb` / `b96ed9241b9e5303d17eb05c13c923f2173465046317d44b4d95cf627fecb80b` | No source shader or authoritative upstream revision was retained. Both files have been removed from the current tree and extension package. | **Repository publication remains blocked** until the private Git history is purged. |
| Historical `model/FSRCNNX_x4_16-0-4-1.wgsl` / `.passes.json` | `3be5b66c0d87f222f2b192ceb4c4e66b6b5f6759e00872ab3b70a542c3ee38e3` / `25a5cdacc30aaef84b738bc64981afdaf6a17aabff5e68456a0afa4961f804ea` | No source shader or authoritative upstream revision was retained. Both files have been removed from the current tree and extension package. | **Repository publication remains blocked** until the private Git history is purged. |
| `model/ArtCNN_C4F32.artcnn.wgsl` / `.artcnn.json` | `ab6fe4c88e88eb0cc3b5482e68ca9279c802c0b7844699c40f9f15eb3aac8138` / `4ab29b29a6121e0fa3d3880b890bedabb3ea1f49356ef46704ad1770b143077a` | [`GLSL/ArtCNN_C4F32.glsl`](https://github.com/Artoriuz/ArtCNN/blob/a20445ca420ed9f0c2a807e2d0c186a991115da0/GLSL/ArtCNN_C4F32.glsl) at commit `a20445ca420ed9f0c2a807e2d0c186a991115da0`; source SHA-256 `f773bce6cf5fe7e5e5d599a695edd40df5cd7a20c3d08c4d164d07591d5bead3`. The local compute transformation uses f32 accumulation, rgba16float intermediate storage, and explicit zero padding for out-of-range convolution taps. | MIT, Copyright 2024 Joao Chrisostomo and Kacper Michajłow. **Verified.** |
| `model/ArtCNN_C4F32_DN.artcnn.wgsl` / `.artcnn.json` | `c319ff51ff358558cd4daa1fc897da4bfc0064c175cca3f9fd29052ac29af280` / `b5911c707c83462c79dcf954bcaf422efd2d6b42efd4d08228361ab8ea52fe79` | [`GLSL/ArtCNN_C4F32_DN.glsl`](https://github.com/Artoriuz/ArtCNN/blob/a20445ca420ed9f0c2a807e2d0c186a991115da0/GLSL/ArtCNN_C4F32_DN.glsl) at the same commit; source SHA-256 `6b51a6f7d75826c9492c3f78b5e60acffa24a71928e2d47c4a329423922a143c`. The same local f32, rgba16float, and zero-padding transformation applies. | MIT, same notice. **Verified.** |
| `model/ArtCNN_C4F32_DS.artcnn.wgsl` / `.artcnn.json` | `41a1e37c67bfb76a74ce07b52324d961fb4e9351eee44581fba783f8d69341af` / `f98bbd5e834cbfb2ed66ba07865889f76466279e356bfbd62c33df73e95b30cb` | [`GLSL/ArtCNN_C4F32_DS.glsl`](https://github.com/Artoriuz/ArtCNN/blob/a20445ca420ed9f0c2a807e2d0c186a991115da0/GLSL/ArtCNN_C4F32_DS.glsl) at the same commit; source SHA-256 `a04c9cba6fbb8e6db9239d61848390208aedf8e348ef116e12174c803d22077e`. The same local f32, rgba16float, and zero-padding transformation applies. | MIT, same notice. **Verified.** |

## ONNX Runtime Web

All three files below are byte-identical to their paths in the official [`onnxruntime-web@1.27.0`](https://www.npmjs.com/package/onnxruntime-web/v/1.27.0) npm tarball:

- Tarball: <https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-1.27.0.tgz>
- Tarball SHA-256: `b59c9819434a7519f334f77e8d4bf22b69808d531a57724cabc4bb2c0704c835`
- npm integrity: `sha512-ogDLsqIozHZwifPuN37OproAo0byX6t43/bP8GzeZWBWD6MOGExswFAx3up4NS/vvWBOg2u2PXomDt3rMmdQSg==`
- Source tag commit: [`8f0278c77bf44b0cc83c098c6c722b92a36ac4b5`](https://github.com/microsoft/onnxruntime/tree/8f0278c77bf44b0cc83c098c6c722b92a36ac4b5)
- License: MIT, Copyright Microsoft Corporation

| Artifact | SHA-256 |
| --- | --- |
| `vendor/ort/ort.webgpu.min.mjs` | `46988a5a025f49449850f39f95eb0d21e40e67b3beb13a0b54efd3ab5d83f60e` |
| `vendor/ort/ort-wasm-simd-threaded.asyncify.mjs` | `7236653b8565da4046e459cd0e274123419a1d9f1f8f18fd36c28058346ca655` |
| `vendor/ort/ort-wasm-simd-threaded.asyncify.wasm` | `7e83cd6cee77e478bc96a7e91b198144fb5e4126287daf1f9b54bb195ebcd55a` |

The package also preserves the official MIT license from that source commit as `vendor/ort/LICENSE`, SHA-256 `2f07c72751aed99790b8a4869cf2311df85a860b22ded05fa22803587a48922c`, and `ThirdPartyNotices.txt` byte-for-byte as `vendor/ort/ThirdPartyNotices.txt`, SHA-256 `0e07b95f3a8d6230037707c5c4a2b554d12c4cb67369669ac255635528ffcee2`.

Runtime byte identity and the Microsoft MIT notice are **verified**. The upstream notice set is preserved, but the applicability review remains a public-release gate.

## Hand-ported runtime shaders

These are runtime source modules rather than generated model files, but their derivation claims affect release clearance.

| Local file | SHA-256 | Recorded source | Status |
| --- | --- | --- | --- |
| `fsrcnnx-ssimds.js` | `0f55f8f2b49bea3cb8ee2e4c801a663f21d4dfabb88efaf2de23b709c6ade3c6` | Packaged [`SSimDownscaler.glsl`](shaders/upstream/SSimDownscaler.glsl), SHA-256 `f46f4710a162d17058b9d82ed8610588b0c04d7be07cef6bf2a8c4077828f804`, from [gist revision `38992bce`](https://gist.github.com/igv/36508af3ffc84410fe39761d6969be10/38992bce7f9ff844f800820df0908692b65bb74a), LGPL-3.0-or-later. The local WebGPU port adds finite-ratio validation, a work budget, and zero-weight numerical guards. | Exact source/revision and license header retained with manual substitution instructions; LGPL public-package obligations remain gated. |
| `fsrcnnx-sharpen.js` | `9312f5445791792634679bac74f01d3292e8e776c6fc7e3be348435f2913ef8a` | Packaged [`adaptive-sharpen.glsl`](shaders/upstream/adaptive-sharpen.glsl), gist revision [`572f5909`](https://gist.github.com/igv/8a77e4eb8276753b54bb94c1c50c317e/572f59099cd0e3eb5e321a6da0a3d90a7382e2dc), source SHA-256 `827fb3d662ac9a91b4075e9117fe6e1dbc1c06d85959ba719cdb954dfb7fb8e4`, Copyright 2015-2021 bacondither under its two-clause redistribution notice. The local WebGPU port adds finite strength normalization and flat-field numerical guards. | Exact source/revision and verbatim source notice retained. |
| Historical `fsrcnnx-deband.js` | `56155c7bd5a15b5524ec1b44baeb4b5cb368e57f9adaf5ff8635bd1a2dba3f84` | Comments said it was modeled on mpv's `f_deband` / haasn algorithm, but no exact source file or revision was recorded. The module, runtime path, settings, and user-facing controls have been removed from the current tree and extension package. | **Repository publication remains blocked** until the private Git history is purged. |

## Reproduction checks

```sh
npm run fetch:shader-sources
npm run check:generated
npm run check:rife-reproduction
npm run reference:check
npm run check
npm run release:check
```

The fetch step accepts upstream bytes only when their pinned hashes match. The generated check covers standard x2 and the three ArtCNN variants. The reference check verifies the committed numerical-oracle inventory and hashes; `npm run reference:generate` separately recreates the fixtures with the external mpv/libplacebo/FFmpeg toolchain documented under `validation/`. The RIFE reproduction check uses the hash-locked CPython 3.11.15 environment documented under `tools/model-reproduction/`; CI runs it separately from the JavaScript checks because it installs platform-specific wheels. Removed high x2, x3, and x4 assets are deliberately outside the current generation and package boundaries.
