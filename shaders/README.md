# Shader sources

This directory contains source material used to generate runtime JSON/WGSL model assets. Source identity and license are separate from successful generation; the release ledger remains authoritative for public distribution.

## Checksum-pinned upstream sources

`npm run fetch:shader-sources` downloads the sources that remain directly available into `shaders/upstream/` and rejects bytes that do not match their pinned SHA-256. The retired High source remains checked in with its archived reproduction chain below. `npm run check:generated` transpiles the FSRCNNX and ArtCNN sources and compares their output byte-for-byte with `model/`. SSimDownscaler is a hand port; its exact source and substitution procedure are retained in the package instead.

| Source | Authoritative location | SHA-256 | License | Generated outputs |
| --- | --- | --- | --- | --- |
| `upstream/FSRCNNX_x2_16-0-4-1.glsl` | [FSRCNN-TensorFlow release 1.1](https://github.com/igv/FSRCNN-TensorFlow/releases/download/1.1/FSRCNNX_x2_16-0-4-1.glsl) | `d5a24a271e5d9a3f7f7a053b150c460a44c25b3cf7f770857d57cc3a2e1c9965` | LGPL-3.0-or-later; Copyright 2017-2021 igv | `model/FSRCNNX_x2_16-0-4-1.{wgsl,passes.json}` |
| `upstream/FSRCNNX_x2_56-16-4-1.glsl` | [Archived FSRCNN-TensorFlow release 1.1](https://web.archive.org/web/20191021180715/https://github.com/igv/FSRCNN-TensorFlow/releases) | `34cd5d0087ebb6ae5f9bff2578382205457da53baa364d52de8021d6925b7fd6` | LGPL-3.0-or-later; qualified compliance review pending | `model/FSRCNNX_x2_56-16-4-1.{wgsl,passes.json}` |
| `upstream/SSimDownscaler.glsl` | [gist revision `38992bce`](https://gist.github.com/igv/36508af3ffc84410fe39761d6969be10/38992bce7f9ff844f800820df0908692b65bb74a) | `f46f4710a162d17058b9d82ed8610588b0c04d7be07cef6bf2a8c4077828f804` | LGPL-3.0-or-later | Hand-ported as `src/core/fsrcnnx-ssimds.js` |
| `upstream/adaptive-sharpen.glsl` | [gist revision `572f5909`](https://gist.github.com/igv/8a77e4eb8276753b54bb94c1c50c317e/572f59099cd0e3eb5e321a6da0a3d90a7382e2dc) | `827fb3d662ac9a91b4075e9117fe6e1dbc1c06d85959ba719cdb954dfb7fb8e4` | Two-clause redistribution notice; Copyright 2015-2021 bacondither | Hand-ported as `src/core/fsrcnnx-sharpen.js` |
| `upstream/ArtCNN_C4F32.glsl` | [ArtCNN commit `a20445c`](https://github.com/Artoriuz/ArtCNN/blob/a20445ca420ed9f0c2a807e2d0c186a991115da0/GLSL/ArtCNN_C4F32.glsl) | `f773bce6cf5fe7e5e5d599a695edd40df5cd7a20c3d08c4d164d07591d5bead3` | MIT; Copyright 2024 Joao Chrisostomo and Kacper Michajłow | `model/ArtCNN_C4F32.{artcnn.wgsl,artcnn.json}` |
| `upstream/ArtCNN_C4F32_DN.glsl` | [same commit](https://github.com/Artoriuz/ArtCNN/blob/a20445ca420ed9f0c2a807e2d0c186a991115da0/GLSL/ArtCNN_C4F32_DN.glsl) | `6b51a6f7d75826c9492c3f78b5e60acffa24a71928e2d47c4a329423922a143c` | MIT, same notice | `model/ArtCNN_C4F32_DN.{artcnn.wgsl,artcnn.json}` |
| `upstream/ArtCNN_C4F32_DS.glsl` | [same commit](https://github.com/Artoriuz/ArtCNN/blob/a20445ca420ed9f0c2a807e2d0c186a991115da0/GLSL/ArtCNN_C4F32_DS.glsl) | `a04c9cba6fbb8e6db9239d61848390208aedf8e348ef116e12174c803d22077e` | MIT, same notice | `model/ArtCNN_C4F32_DS.{artcnn.wgsl,artcnn.json}` |

The FSRCNN release tag resolves to commit `1aa11ab0e1fc12741fdb84cef31da5619a478670`, but its shader assets were attached later. The pinned asset checksums, not the tag commit, identify the exact source bytes.

## High x2 source chain

igv's release 1.1 listed `FSRCNNX_x2_56-16-4-1.glsl` as a release asset in archived snapshots from [2018-08-26](https://web.archive.org/web/20180826101521/https://github.com/igv/FSRCNN-TensorFlow/releases) through [2019-10-21](https://web.archive.org/web/20191021180715/https://github.com/igv/FSRCNN-TensorFlow/releases); it was absent by [2020-10-11](https://web.archive.org/web/20201011050558/https://github.com/igv/FSRCNN-TensorFlow/releases/tag/1.1). The archived release notes identify 56-16-4-1 as the image-upscaling model that was too slow for real-time use without MRT support.

The archived `checkpoints_params.7z` (425,502 bytes; SHA-256 `28167f74341256054c790e94c30a10964818f6bdbe7aedb97c6507208123fc10`) contains `params/weights56_16_4_1.txt` (328,634 bytes; SHA-256 `a27f732e1609a0d26e768d63447a42b04acd71918386026e1ca18a937ceea290`). Running tag-1.1 [`gen.py`](https://github.com/igv/FSRCNN-TensorFlow/blob/1aa11ab0e1fc12741fdb84cef31da5619a478670/gen.py) (Git blob `9b1b5cdb6c840c4534d671ccade170e518c6ce4e`; SHA-256 `aa99254fd8001f2d0ac99e93a71f7225d78227e282b727b9c4bf7e5901e601ca`) produces a 364,494-byte CRLF shader with SHA-256 `b507e0ec6c0d9ab22d440736677cd2ccb8a8b5441e190889ca7ec762d53ca063`. Normalizing its 2,053 CRLF line endings to LF yields the checked-in 362,441-byte source exactly, SHA-256 `34cd5d0087ebb6ae5f9bff2578382205457da53baa364d52de8021d6925b7fd6`.

That source deterministically reproduces the High WGSL and pass manifest, SHA-256 `19a5327c8f96b7cb0593512f846f75ef266a3d857a84532c4dc5a374296e3d11` and `4b7512ca17fd9788f4876f2681207fa8fb3b10c46d314ea2b3ce684864fb4d70`. The source has no embedded license header; this project classifies the officially released source and its generated pair as LGPL-3.0-or-later consistently with FSRCNNX Standard. Qualified review of the distribution plan remains required.

## Removed historical assets

The source-less x3 and x4 generated pairs remain removed from the current tree, extension package, and writable branch history. Their hashes are retained as tombstones in [MODEL_PROVENANCE.md](../docs/compliance/MODEL_PROVENANCE.md) and [`release-clearance.json`](../docs/compliance/release-clearance.json). GitHub's retained pull-request refs still expose the pre-rewrite objects, so do not publish the repository until GitHub Support has removed those refs and caches and garbage-collected the objects. `tools/check-models.mjs` separately pins the SSim source and port.
