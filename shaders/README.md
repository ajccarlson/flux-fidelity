# Shader sources

This directory contains source material used to generate runtime JSON/WGSL model assets. Source identity and license are separate from successful generation; unresolved entries below are not cleared for public redistribution.

## Checksum-pinned upstream sources

`npm run fetch:shader-sources` downloads these files into `shaders/upstream/` and rejects any bytes that do not match the pinned SHA-256. `npm run check:generated` transpiles the FSRCNNX and ArtCNN sources and compares their output byte-for-byte with `model/`. SSimDownscaler is a hand port; its exact source and substitution procedure are retained in the package instead.

| Source | Authoritative location | SHA-256 | License | Generated outputs |
| --- | --- | --- | --- | --- |
| `upstream/FSRCNNX_x2_16-0-4-1.glsl` | [FSRCNN-TensorFlow release 1.1](https://github.com/igv/FSRCNN-TensorFlow/releases/download/1.1/FSRCNNX_x2_16-0-4-1.glsl) | `d5a24a271e5d9a3f7f7a053b150c460a44c25b3cf7f770857d57cc3a2e1c9965` | LGPL-3.0-or-later; Copyright 2017-2021 igv | `model/FSRCNNX_x2_16-0-4-1.{wgsl,passes.json}` |
| `upstream/SSimDownscaler.glsl` | [gist revision `38992bce`](https://gist.github.com/igv/36508af3ffc84410fe39761d6969be10/38992bce7f9ff844f800820df0908692b65bb74a) | `f46f4710a162d17058b9d82ed8610588b0c04d7be07cef6bf2a8c4077828f804` | LGPL-3.0-or-later | Hand-ported as `fsrcnnx-ssimds.js` |
| `upstream/adaptive-sharpen.glsl` | [gist revision `572f5909`](https://gist.github.com/igv/8a77e4eb8276753b54bb94c1c50c317e/572f59099cd0e3eb5e321a6da0a3d90a7382e2dc) | `827fb3d662ac9a91b4075e9117fe6e1dbc1c06d85959ba719cdb954dfb7fb8e4` | Two-clause redistribution notice; Copyright 2015-2021 bacondither | Hand-ported as `fsrcnnx-sharpen.js` |
| `upstream/ArtCNN_C4F32.glsl` | [ArtCNN commit `a20445c`](https://github.com/Artoriuz/ArtCNN/blob/a20445ca420ed9f0c2a807e2d0c186a991115da0/GLSL/ArtCNN_C4F32.glsl) | `f773bce6cf5fe7e5e5d599a695edd40df5cd7a20c3d08c4d164d07591d5bead3` | MIT; Copyright 2024 Joao Chrisostomo and Kacper Michajłow | `model/ArtCNN_C4F32.{artcnn.wgsl,artcnn.json}` |
| `upstream/ArtCNN_C4F32_DN.glsl` | [same commit](https://github.com/Artoriuz/ArtCNN/blob/a20445ca420ed9f0c2a807e2d0c186a991115da0/GLSL/ArtCNN_C4F32_DN.glsl) | `6b51a6f7d75826c9492c3f78b5e60acffa24a71928e2d47c4a329423922a143c` | MIT, same notice | `model/ArtCNN_C4F32_DN.{artcnn.wgsl,artcnn.json}` |
| `upstream/ArtCNN_C4F32_DS.glsl` | [same commit](https://github.com/Artoriuz/ArtCNN/blob/a20445ca420ed9f0c2a807e2d0c186a991115da0/GLSL/ArtCNN_C4F32_DS.glsl) | `a04c9cba6fbb8e6db9239d61848390208aedf8e348ef116e12174c803d22077e` | MIT, same notice | `model/ArtCNN_C4F32_DS.{artcnn.wgsl,artcnn.json}` |

The FSRCNN release tag resolves to commit `1aa11ab0e1fc12741fdb84cef31da5619a478670`, but its shader asset was attached later. The pinned asset checksum, not the tag commit, identifies the exact source bytes.

## Removed historical assets

The unverified high-quality x2 source and generated pair, and the source-less x3 and x4 generated pairs, have been removed from the current tree and extension package. Their historical hashes are retained as tombstones in [MODEL_PROVENANCE.md](../MODEL_PROVENANCE.md) and [`release-clearance.json`](../release-clearance.json).

Those bytes still exist in private Git history. Do not make that repository history public until it has been purged or authoritative source and license evidence has been established. `npm run check:generated` deliberately covers only the standard x2 and three ArtCNN rows above; `tools/check-models.mjs` separately pins the SSim source and port.
