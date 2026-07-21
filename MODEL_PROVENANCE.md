# Model and generated-asset provenance

Audit date: 2026-07-20.

This inventory is a release gate for bundled model, generated shader, and inference-runtime artifacts. A matching hash proves byte identity, not ownership or permission. `Verified` means the recorded bytes, source, and stated upstream license were checked. `Blocked` means the artifact must be removed from a public package or its missing evidence must be supplied first.

## Unresolved public-release gates

- `model/rife.onnx` has no exact authoritative source or model-version record.
- `model/rife_v4.26_fp16.onnx` has no retained conversion command, converter version, or structural comparison proving its asserted parent.
- The original upstream source and license for `shaders/FSRCNNX_x2_56-16-4-1.glsl` are unknown.
- The x3 and x4 FSRCNNX generated assets have no retained source shader or upstream revision.
- The random-weight SPAN smoke model has no recorded random seed or exact generation environment and is not a production model.
- The exact source/revision and license for the deband port have not been established.
- Public packages containing LGPL-derived files need the complete applicable license text and a reviewed compliance plan; the repository currently records notices and source links but does not carry a standalone LGPL-3.0 text.

The current package builder includes these artifacts. Therefore, a successful `npm run check` or `npm run package` does **not** clear the repository for public distribution.

## ONNX models

| Artifact | SHA-256 | Evidence | License evidence | Status |
| --- | --- | --- | --- | --- |
| `model/rife_v4.26.onnx` | `af25762dfec02a4bbb949decea63988b01fa56c46c0ff9dc66ac8e2f12cbb661` | Byte-identical to `rife_v2/rife_v4.26.onnx` in [AmusementClub/vs-mlrt `rife_v4.26.7z`](https://github.com/AmusementClub/vs-mlrt/releases/download/external-models/rife_v4.26.7z). Archive SHA-256: `dfdabd84a2a3db773f87604b8cc255e94a6a72f13550d910ccd3b4ee2606cd4f`; member size: `22,748,049` bytes. The release points to Practical-RIFE commit [`de9a989bb7b8a71d94f058297e603633aaa43ad6`](https://github.com/hzwer/Practical-RIFE/tree/de9a989bb7b8a71d94f058297e603633aaa43ad6). The ONNX producer metadata says PyTorch 2.2.0. | The cited Practical-RIFE revision states that linked trained-model content is under the repository's MIT License (Copyright 2021 hzwer). | **Verified**; retain the MIT notice and exact source record. |
| `model/rife_v4.26_fp16.onnx` | `65494496be256b2809b6760f87a941c0d80bbc5cb09de0f3f6795488ae1fcd9d` | Repository history describes a local FP16 conversion of the preceding model, but no command, tool version, log, or independently checked parent relationship was retained. | Would inherit the verified parent's terms only after the derivation is proven. | **Blocked**. Recreate from the verified FP32 file with pinned tooling, record the command, and compare model structure and numeric conversion. |
| `model/rife.onnx` | `6a31074c0f588648982b5e828aee6c27e005015a712a46ea63da48c65fa9a26b` | Six-channel dynamic ONNX export; producer metadata says PyTorch 2.5.1. It does not match by size or hash any file in the official vs-mlrt `rife_v2_v4.7z` older-model archive, and no authoritative exact-hash match was found. | Practical-RIFE's code/model license cannot establish that an unidentified binary came from that source. | **Blocked**. Supply the original download/export URL, model version, immutable revision, and applicable license, or remove it. |
| `model/neural/span2x_smoke.fp16.onnx` | `ae7642c2b3bdd96e475dfcbf6a9180d8ec41175a1329d9f9b2a402f186c05c25` | Random-weight SPAN fixture associated with `tools/neural-export/export.py`; ONNX producer metadata says PyTorch 2.12.1. The current requirements instead pin PyTorch 2.13.0 and no generation seed or environment lock was retained. | The exporter currently depends on Spandrel 0.4.2, tag commit [`724cca389f28c38e1050689d4862a452fd644484`](https://github.com/chaiNNer-org/spandrel/tree/724cca389f28c38e1050689d4862a452fd644484), MIT. The artifact does not embed evidence that this exact dependency built it. | **Development-only / blocked for release**. Regenerate deterministically from a documented environment if a smoke fixture must ship. |

## Generated upscaler shaders

`npm run check:generated` regenerates the standard x2, high x2, and three ArtCNN pairs and requires byte-for-byte equality. That proves local reproducibility from the checked-in source bytes; it does not fill an upstream or license gap.

| Generated artifacts | SHA-256 | Source | License and status |
| --- | --- | --- | --- |
| `model/FSRCNNX_x2_16-0-4-1.wgsl` / `.passes.json` | `ff7e001ee85d139eb704be1981cd306e8d515145ca06fc3da26a12f9e78d4755` / `6e58179e009ee8d599019359465566d4525acd65c5669519f0ca63398186df1d` | [FSRCNN-TensorFlow release 1.1 asset](https://github.com/igv/FSRCNN-TensorFlow/releases/download/1.1/FSRCNNX_x2_16-0-4-1.glsl), source SHA-256 `d5a24a271e5d9a3f7f7a053b150c460a44c25b3cf7f770857d57cc3a2e1c9965`. Release tag commit: [`1aa11ab0e1fc12741fdb84cef31da5619a478670`](https://github.com/igv/FSRCNN-TensorFlow/tree/1aa11ab0e1fc12741fdb84cef31da5619a478670); the downloadable asset was added later, so its checksum is the immutable identifier. | Source header: Copyright 2017-2021 igv, LGPL-3.0-or-later. **Source and reproduction verified; public-package LGPL compliance remains gated.** |
| `model/FSRCNNX_x2_56-16-4-1.wgsl` / `.passes.json` | `267ba203867483a467c535fd03c36c62ff9428116111d4d258dc5c295ef8e0d7` / `57395ac668b4cbebea69938a9089c9bea0029ce785f7cc6dad239c4be31d43e7` | Checked-in `shaders/FSRCNNX_x2_56-16-4-1.glsl`, SHA-256 `34cd5d0087ebb6ae5f9bff2578382205457da53baa364d52de8021d6925b7fd6`. No authoritative origin was established. | **Blocked** despite byte-reproducible generation: upstream author, revision, and license are unknown. |
| `model/FSRCNNX_x3_16-0-4-1.wgsl` / `.passes.json` | `7df8eaae1c6f50bfd2b494d403cce8e1638c7408e0b92362b174cb72fdf816fb` / `b96ed9241b9e5303d17eb05c13c923f2173465046317d44b4d95cf627fecb80b` | No source shader or authoritative upstream revision retained. | **Blocked**. |
| `model/FSRCNNX_x4_16-0-4-1.wgsl` / `.passes.json` | `3be5b66c0d87f222f2b192ceb4c4e66b6b5f6759e00872ab3b70a542c3ee38e3` / `25a5cdacc30aaef84b738bc64981afdaf6a17aabff5e68456a0afa4961f804ea` | No source shader or authoritative upstream revision retained. | **Blocked**. |
| `model/ArtCNN_C4F32.artcnn.wgsl` / `.artcnn.json` | `af2b1911fe4ec1f77354b71d5aa6796c93b4d53eb73cd693b59af7e8cfb9d654` / `4ab29b29a6121e0fa3d3880b890bedabb3ea1f49356ef46704ad1770b143077a` | [`GLSL/ArtCNN_C4F32.glsl`](https://github.com/Artoriuz/ArtCNN/blob/a20445ca420ed9f0c2a807e2d0c186a991115da0/GLSL/ArtCNN_C4F32.glsl) at commit `a20445ca420ed9f0c2a807e2d0c186a991115da0`; source SHA-256 `f773bce6cf5fe7e5e5d599a695edd40df5cd7a20c3d08c4d164d07591d5bead3`. | MIT, Copyright 2024 Joao Chrisostomo and Kacper Michajłow. **Verified.** |
| `model/ArtCNN_C4F32_DN.artcnn.wgsl` / `.artcnn.json` | `f204b33d52614e87bc9d8d31ba43822ad3bad1ff75da8d425ca6fbd90a2032a9` / `b5911c707c83462c79dcf954bcaf422efd2d6b42efd4d08228361ab8ea52fe79` | [`GLSL/ArtCNN_C4F32_DN.glsl`](https://github.com/Artoriuz/ArtCNN/blob/a20445ca420ed9f0c2a807e2d0c186a991115da0/GLSL/ArtCNN_C4F32_DN.glsl) at the same commit; source SHA-256 `6b51a6f7d75826c9492c3f78b5e60acffa24a71928e2d47c4a329423922a143c`. | MIT, same notice. **Verified.** |
| `model/ArtCNN_C4F32_DS.artcnn.wgsl` / `.artcnn.json` | `f6de86466a0ae261c178f53d72d2cb79032ade94b8ec452f51fa1315b93be3c5` / `f98bbd5e834cbfb2ed66ba07865889f76466279e356bfbd62c33df73e95b30cb` | [`GLSL/ArtCNN_C4F32_DS.glsl`](https://github.com/Artoriuz/ArtCNN/blob/a20445ca420ed9f0c2a807e2d0c186a991115da0/GLSL/ArtCNN_C4F32_DS.glsl) at the same commit; source SHA-256 `a04c9cba6fbb8e6db9239d61848390208aedf8e348ef116e12174c803d22077e`. | MIT, same notice. **Verified.** |

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

Status: **Verified**. Preserve the MIT notice.

## Hand-ported runtime shaders

These are runtime source modules rather than generated model files, but their derivation claims affect release clearance.

| Local file | SHA-256 | Recorded source | Status |
| --- | --- | --- | --- |
| `fsrcnnx-ssimds.js` | `09a07f30ac718600ea529005d5056d2e54ff6a868b9ddb988683a16057673cbf` | [igv/SSimDownscaler.glsl](https://gist.github.com/igv/36508af3ffc84410fe39761d6969be10/38992bce7f9ff844f800820df0908692b65bb74a), gist revision `38992bce7f9ff844f800820df0908692b65bb74a`, source SHA-256 `ef19b89e82b543a84caa539b91bb8f1a3b853f39ce313abee4d26ee4f1b4cfba`, LGPL-3.0-or-later. The local WebGPU port adds finite ratio validation and zero-weight numerical guards. | Source/revision identified; LGPL public-package obligations remain gated. |
| `fsrcnnx-sharpen.js` | `9b47a6aa2e5cc6294bb7e747d6d54b141aee1cf2a03c95d079e1fa9aeec23f9d` | [igv/adaptive-sharpen.glsl](https://gist.github.com/igv/8a77e4eb8276753b54bb94c1c50c317e/572f59099cd0e3eb5e321a6da0a3d90a7382e2dc), gist revision `572f59099cd0e3eb5e321a6da0a3d90a7382e2dc`, source SHA-256 `0f1d82afa20b1536c45d5b11cddbab1dae5312442e9b6bfb0eb7ae4c43331851`, Copyright 2015-2021 bacondither under its two-clause redistribution notice. The local WebGPU port adds finite strength normalization and flat-field numerical guards. | Source/revision and notice identified. |
| `fsrcnnx-deband.js` | `56155c7bd5a15b5524ec1b44baeb4b5cb368e57f9adaf5ff8635bd1a2dba3f84` | Comments say it was modeled on mpv's `f_deband` / haasn algorithm. The local builder now normalizes non-finite strength inputs, but no exact source file or revision is recorded. | **Blocked** pending a source-and-license record or a documented original/clean-room implementation basis. |

## Reproduction checks

```sh
npm run fetch:shader-sources
npm run check:generated
npm run check
```

The fetch step accepts upstream bytes only when their pinned hashes match. The generated check covers standard x2, high x2, and the three ArtCNN variants; it cannot regenerate x3 or x4.
