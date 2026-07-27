# Third-party notices

Flux Fidelity includes or derives from the components below. Exact artifact hashes and unresolved release gates are recorded in [MODEL_PROVENANCE.md](docs/compliance/MODEL_PROVENANCE.md) and the machine-readable [`release-clearance.json`](https://github.com/ajccarlson/fsrcnnx-ext/blob/main/docs/compliance/release-clearance.json). `npm run release:check` remains the authoritative release and distribution gate.

## License not specified by upstream

### CDA-VSR

The bundled `model/neural/cda-vsr-initializer.onnx` and
`model/neural/cda-vsr-recurrent.onnx` are mixed-FP16 conversions of the
CDA-VSR architecture and `best.pth` checkpoint from pinned upstream commit
[`5707d997`](https://github.com/sspBIT/CDA-VSR/tree/5707d997759996f19521c3beaddfb3d1ea965d44).

Upstream does not specify a license for its CDA-specific source or pretrained
checkpoint. The pinned README lists a `LICENSE.txt` that is absent from the
repository, and its license section only acknowledges BasicSR and TMP. The
repository's `NOTICE` is attributed to `xtudbxk`; it does not establish terms for
the CDA additions or checkpoint.

The shipping graphs identify their exact parent exports and describe the
upstream license as `not-specified`. Their hashes and conversion evidence are
recorded in [Model provenance](docs/compliance/MODEL_PROVENANCE.md). Inclusion
is an explicit repository-owner accepted-risk decision and does not claim legal
clearance or grant recipients rights beyond those supplied by the applicable
copyright holders.

## MIT-licensed components

### ONNX Runtime Web 1.27.0

The bundled `ort.webgpu.min.mjs` is byte-identical to the official `onnxruntime-web@1.27.0` npm package. Its single-thread Asyncify module pair was built from the corresponding unmodified ONNX Runtime tag commit `8f0278c77bf44b0cc83c098c6c722b92a36ac4b5`; the generated loader receives one documented hardening edit that removes an unused shared-memory feature probe.

Copyright (c) Microsoft Corporation

Upstream: <https://github.com/microsoft/onnxruntime/tree/8f0278c77bf44b0cc83c098c6c722b92a36ac4b5>

This package preserves the official MIT text from the pinned source commit as `vendor/ort/LICENSE` (SHA-256 `2f07c72751aed99790b8a4869cf2311df85a860b22ded05fa22803587a48922c`). The official npm tarball does not carry a notice file, so the package also preserves `vendor/ort/ThirdPartyNotices.txt` byte-for-byte from that commit (SHA-256 `0e07b95f3a8d6230037707c5c4a2b554d12c4cb67369669ac255635528ffcee2`). The repository owner has deferred external applicability review and accepted the residual risk; the bundled license and notice set remain part of every package.

### ArtCNN

The generated `model/ArtCNN_*` files derive from three GLSL shaders at ArtCNN commit `a20445ca420ed9f0c2a807e2d0c186a991115da0`.

Copyright (c) 2024 João Chrisóstomo<br>
Copyright (c) 2024 Joao Chrisostomo, Kacper Michajłow

Upstream: <https://github.com/Artoriuz/ArtCNN/tree/a20445ca420ed9f0c2a807e2d0c186a991115da0>

### Practical-RIFE 4.26

The verified FP32 `model/rife_v4.26.onnx` is an ONNX export of the Practical-RIFE 4.26 trained model associated with commit `de9a989bb7b8a71d94f058297e603633aaa43ad6`. That revision explicitly states that content linked under “Trained Model” is under the project's MIT License. The bundled `model/rife_v4.26_fp16.onnx` is a deterministically reproduced FP16 derivative of that exact FP32 artifact; its parent and output hashes, pinned conversion environment, structural checks, and numerical validation are recorded in `docs/compliance/MODEL_PROVENANCE.md`.

Copyright (c) 2021 hzwer

Upstream: <https://github.com/hzwer/Practical-RIFE/tree/de9a989bb7b8a71d94f058297e603633aaa43ad6><br>
Verified export archive: <https://github.com/AmusementClub/vs-mlrt/releases/download/external-models/rife_v4.26.7z>

### MIT License text

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## BSD-3-Clause components

### Real-ESRGAN AnimeVideo XS 2x

`model/neural/realesrganv2_animevideo_xsx2.fp16.onnx` is a checked FP16-weight ONNX export of the official `RealESRGANv2-animevideo-xsx2.pth` asset from Real-ESRGAN release v0.2.3.0.

Copyright (c) 2021, Xintao Wang

Upstream release: <https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.3.0><br>
Pinned source: <https://github.com/xinntao/Real-ESRGAN/tree/f07aaffda04c7e69f11e6bfaf8023a6435471459>

The official BSD-3-Clause project publishes the checkpoint without separate asset terms. The package preserves its exact license as [`LICENSES/Real-ESRGAN-BSD-3-Clause.txt`](LICENSES/Real-ESRGAN-BSD-3-Clause.txt). Checkpoint, export, and numerical-validation hashes are recorded in [Model provenance](docs/compliance/MODEL_PROVENANCE.md).

## LGPL-3.0-or-later shader sources

The standard x2 generated FSRCNNX artifacts derive from `FSRCNNX_x2_16-0-4-1.glsl` in FSRCNN-TensorFlow release 1.1.

Copyright (C) 2017-2021 igv

Upstream asset: <https://github.com/igv/FSRCNN-TensorFlow/releases/download/1.1/FSRCNNX_x2_16-0-4-1.glsl>

The High x2 source `FSRCNNX_x2_56-16-4-1.glsl` was also an official release-1.1 asset and is classified LGPL-3.0-or-later consistently with Standard. Its archived release and deterministic reproduction evidence are recorded in [Model provenance](docs/compliance/MODEL_PROVENANCE.md).

The SSimDownscaler WebGPU port derives from the packaged `shaders/upstream/SSimDownscaler.glsl`, gist revision `38992bce7f9ff844f800820df0908692b65bb74a`, SHA-256 `f46f4710a162d17058b9d82ed8610588b0c04d7be07cef6bf2a8c4077828f804`.

Upstream: <https://gist.github.com/igv/36508af3ffc84410fe39761d6969be10/38992bce7f9ff844f800820df0908692b65bb74a>

The Standard and SSim upstream files state that they may be redistributed and/or modified under the GNU Lesser General Public License, version 3 or, at the recipient's option, any later version; they disclaim warranty. High is treated consistently as LGPL-3.0-or-later.

The package includes the exact FSRCNNX and SSim upstream sources, complete official [`LGPL-3.0.txt`](LICENSES/LGPL-3.0.txt) text and [`GPL-3.0.txt`](LICENSES/GPL-3.0.txt) companion, the FSRCNNX transformation script, generated source metadata, and [`LGPL_REBUILDING.md`](docs/compliance/LGPL_REBUILDING.md) with offline rebuilding and Chromium substitution instructions. Sources with embedded notices retain them, and the generated/ported files identify local modifications. The Apache-2.0 license for project-authored material does not restrict modification, reverse engineering, or recombination needed to exercise LGPL rights in these portions.

## Adaptive Sharpen redistribution notice

`src/core/fsrcnnx-sharpen.js` is a WebGPU port of bacondither's adaptive-sharpen shader, version 2021-10-17, from gist revision `572f59099cd0e3eb5e321a6da0a3d90a7382e2dc`. The package retains the exact upstream source as `shaders/upstream/adaptive-sharpen.glsl`, SHA-256 `827fb3d662ac9a91b4075e9117fe6e1dbc1c06d85959ba719cdb954dfb7fb8e4`.

Copyright (c) 2015-2021, bacondither. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer in this position and unchanged.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Upstream: <https://gist.github.com/igv/8a77e4eb8276753b54bb94c1c50c317e/572f59099cd0e3eb5e321a6da0a3d90a7382e2dc>

The source module retains the preceding notice verbatim before its local port description.
