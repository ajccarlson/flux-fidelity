# Shader sources

This directory contains source material used to generate runtime JSON/WGSL model assets. Source identity and license are separate from successful generation; unresolved entries below are not cleared for public redistribution.

## Checksum-pinned upstream sources

`npm run fetch:shader-sources` downloads these files into `shaders/upstream/` and rejects any bytes that do not match the pinned SHA-256. `npm run check:generated` transpiles the checked-in sources and compares their output byte-for-byte with `model/`.

| Source | Authoritative location | SHA-256 | License | Generated outputs |
| --- | --- | --- | --- | --- |
| `upstream/FSRCNNX_x2_16-0-4-1.glsl` | [FSRCNN-TensorFlow release 1.1](https://github.com/igv/FSRCNN-TensorFlow/releases/download/1.1/FSRCNNX_x2_16-0-4-1.glsl) | `d5a24a271e5d9a3f7f7a053b150c460a44c25b3cf7f770857d57cc3a2e1c9965` | LGPL-3.0-or-later; Copyright 2017-2021 igv | `model/FSRCNNX_x2_16-0-4-1.{wgsl,passes.json}` |
| `upstream/ArtCNN_C4F32.glsl` | [ArtCNN commit `a20445c`](https://github.com/Artoriuz/ArtCNN/blob/a20445ca420ed9f0c2a807e2d0c186a991115da0/GLSL/ArtCNN_C4F32.glsl) | `f773bce6cf5fe7e5e5d599a695edd40df5cd7a20c3d08c4d164d07591d5bead3` | MIT; Copyright 2024 Joao Chrisostomo and Kacper Michajłow | `model/ArtCNN_C4F32.{artcnn.wgsl,artcnn.json}` |
| `upstream/ArtCNN_C4F32_DN.glsl` | [same commit](https://github.com/Artoriuz/ArtCNN/blob/a20445ca420ed9f0c2a807e2d0c186a991115da0/GLSL/ArtCNN_C4F32_DN.glsl) | `6b51a6f7d75826c9492c3f78b5e60acffa24a71928e2d47c4a329423922a143c` | MIT, same notice | `model/ArtCNN_C4F32_DN.{artcnn.wgsl,artcnn.json}` |
| `upstream/ArtCNN_C4F32_DS.glsl` | [same commit](https://github.com/Artoriuz/ArtCNN/blob/a20445ca420ed9f0c2a807e2d0c186a991115da0/GLSL/ArtCNN_C4F32_DS.glsl) | `a04c9cba6fbb8e6db9239d61848390208aedf8e348ef116e12174c803d22077e` | MIT, same notice | `model/ArtCNN_C4F32_DS.{artcnn.wgsl,artcnn.json}` |

The FSRCNN release tag resolves to commit `1aa11ab0e1fc12741fdb84cef31da5619a478670`, but its shader asset was attached later. The pinned asset checksum, not the tag commit, identifies the exact source bytes.

## Locally reproducible but unverified source

`FSRCNNX_x2_56-16-4-1.glsl` has SHA-256 `34cd5d0087ebb6ae5f9bff2578382205457da53baa364d52de8021d6925b7fd6` and reproduces the checked-in high-quality x2 JSON/WGSL files. Its original upstream URL, author, revision, and license are not established. Reproducibility from local bytes does not resolve that provenance gap.

## Missing source

No source shader or authoritative upstream revision is retained for:

- `model/FSRCNNX_x3_16-0-4-1.wgsl` and `.passes.json`
- `model/FSRCNNX_x4_16-0-4-1.wgsl` and `.passes.json`

Do not publicly redistribute the high x2, x3, or x4 artifacts until their source and license records are completed, or remove them from the public package. See [MODEL_PROVENANCE.md](../MODEL_PROVENANCE.md) for output hashes and all release gates.
