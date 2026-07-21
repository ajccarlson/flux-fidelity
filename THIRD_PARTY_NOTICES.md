# Third-party notices

FSRCNNX-EXT includes or derives from the components below. Exact artifact hashes and unresolved release gates are recorded in [MODEL_PROVENANCE.md](MODEL_PROVENANCE.md) and the machine-readable [`release-clearance.json`](release-clearance.json). `npm run release:check` remains the authoritative pre-publication gate. This file records verified notices; it does not assign a license to artifacts whose origin remains unknown.

## MIT-licensed components

### ONNX Runtime Web 1.27.0

Bundled files under `vendor/ort/` come from the official `onnxruntime-web@1.27.0` npm package and correspond to ONNX Runtime tag commit `8f0278c77bf44b0cc83c098c6c722b92a36ac4b5`.

Copyright (c) Microsoft Corporation

Upstream: <https://github.com/microsoft/onnxruntime/tree/8f0278c77bf44b0cc83c098c6c722b92a36ac4b5>

The official npm tarball does not carry a notice file. This package therefore preserves `vendor/ort/ThirdPartyNotices.txt` byte-for-byte from the pinned source commit (SHA-256 `0e07b95f3a8d6230037707c5c4a2b554d12c4cb67369669ac255635528ffcee2`). Bundling that upstream notice set does not replace the pending review of which notices and obligations apply to the distributed Web runtime files.

### ArtCNN

The generated `model/ArtCNN_*` files derive from three GLSL shaders at ArtCNN commit `a20445ca420ed9f0c2a807e2d0c186a991115da0`.

Copyright (c) 2024 João Chrisóstomo<br>
Copyright (c) 2024 Joao Chrisostomo, Kacper Michajłow

Upstream: <https://github.com/Artoriuz/ArtCNN/tree/a20445ca420ed9f0c2a807e2d0c186a991115da0>

### Practical-RIFE 4.26

The verified FP32 `model/rife_v4.26.onnx` is an ONNX export of the Practical-RIFE 4.26 trained model associated with commit `de9a989bb7b8a71d94f058297e603633aaa43ad6`. That revision explicitly states that content linked under “Trained Model” is under the project's MIT License. The bundled `model/rife_v4.26_fp16.onnx` is a deterministically reproduced FP16 derivative of that exact FP32 artifact; its parent and output hashes, pinned conversion environment, structural checks, and numerical validation are recorded in `MODEL_PROVENANCE.md`.

Copyright (c) 2021 hzwer

Upstream: <https://github.com/hzwer/Practical-RIFE/tree/de9a989bb7b8a71d94f058297e603633aaa43ad6><br>
Verified export archive: <https://github.com/AmusementClub/vs-mlrt/releases/download/external-models/rife_v4.26.7z>

This notice does not cover the unidentified historical `model/rife.onnx`. That file is absent from the current tree and package, but its private-history tombstone remains blocked as described in `MODEL_PROVENANCE.md`.

### Spandrel 0.4.2

The neural export tool is configured to use Spandrel 0.4.2, including its SPAN architecture implementation, at commit `724cca389f28c38e1050689d4862a452fd644484`.

Copyright (c) 2024 The ChaiNNer Organization

Upstream: <https://github.com/chaiNNer-org/spandrel/tree/724cca389f28c38e1050689d4862a452fd644484>

The export tool remains available for locally supplied, appropriately licensed checkpoints. The former random-weight SPAN smoke model was removed because its generation environment was not retained; this attribution does not establish the origin or licensing of those historical bytes.

### MIT License text

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## LGPL-3.0-or-later shader sources

The standard x2 generated FSRCNNX artifacts derive from `FSRCNNX_x2_16-0-4-1.glsl` in FSRCNN-TensorFlow release 1.1.

Copyright (C) 2017-2021 igv

Upstream asset: <https://github.com/igv/FSRCNN-TensorFlow/releases/download/1.1/FSRCNNX_x2_16-0-4-1.glsl>

The SSimDownscaler WebGPU port derives from `SSimDownscaler.glsl`, gist revision `38992bce7f9ff844f800820df0908692b65bb74a`.

Upstream: <https://gist.github.com/igv/36508af3ffc84410fe39761d6969be10/38992bce7f9ff844f800820df0908692b65bb74a>

Both upstream files state that they may be redistributed and/or modified under the GNU Lesser General Public License, version 3 or, at the recipient's option, any later version; they disclaim warranty.

The package includes the complete official [`LGPL-3.0.txt`](LGPL-3.0.txt) text and its [`GPL-3.0.txt`](GPL-3.0.txt) companion. The checked-in LGPL sources retain their upstream license headers, and the SSim module identifies its 2026 WebGPU port modifications. These additions do not clear public distribution: the applicable notice, corresponding-source, relinking/modification, and other obligations still require a completed compliance review.

## Adaptive Sharpen redistribution notice

`fsrcnnx-sharpen.js` is a WebGPU port of bacondither's adaptive-sharpen shader, version 2021-10-17, from gist revision `572f59099cd0e3eb5e321a6da0a3d90a7382e2dc`.

Copyright (c) 2015-2021, bacondither. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer in this position and unchanged.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Upstream: <https://gist.github.com/igv/8a77e4eb8276753b54bb94c1c50c317e/572f59099cd0e3eb5e321a6da0a3d90a7382e2dc>

The source module retains the preceding notice verbatim before its local port description.

## Unresolved material

No license is asserted here for the removed high-quality x2, x3, or x4 FSRCNNX artifacts, unidentified generic RIFE model, or unreproducible SPAN smoke model. They are absent from the current tree and extension package, but their bytes remain in private Git history and block publication of that history until it is purged. The deband port remains a current-tree and package blocker. The LGPL and ONNX Runtime applicability reviews also remain blocked in `release-clearance.json`; merely bundling complete upstream texts and notices does not clear them.
