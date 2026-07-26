# CDA-VSR conversion experiment

This offline toolkit adapts a user-supplied CDA-VSR source file and checkpoint
into two dynamic-spatial ONNX graphs:

- `cda-vsr-initializer.onnx`: RGB frame to 4× output plus two feature states.
- `cda-vsr-recurrent.onnx`: RGB, motion, residual, and prior states to 4× output
  plus updated states.

It does not download, copy, or redistribute upstream source or weights. Review
their provenance and redistribution terms before publishing any derived model.

The verified reference is
[`sspBIT/CDA-VSR@5707d997`](https://github.com/sspBIT/CDA-VSR/tree/5707d997759996f19521c3beaddfb3d1ea965d44):
architecture SHA-256 `0defb80e5fcbaa2abd0eb9cbc4f4f2050a68e94fa6f743aa48a785cc734fd87b`
and checkpoint SHA-256 `afc8745b890289ae421c500279d9ccf2a27c92cf3e71133b20840c7816e86d3e`.
Its README names `LICENSE.txt`, but that file is absent at the pinned revision.

## Convert

First inspect the inputs and copy the two printed hashes:

```sh
python tools/cda-vsr/cda_tool.py inspect \
  --source ../CDA-VSR \
  --checkpoint ../CDA-VSR/pretrained_models/best.pth
```

With CPython 3.11 or newer, create an isolated environment and export:

```sh
python -m venv .venv
.venv/bin/python -m pip install -r tools/cda-vsr/requirements.txt
.venv/bin/python tools/cda-vsr/cda_tool.py export \
  --source ../CDA-VSR \
  --checkpoint ../CDA-VSR/pretrained_models/best.pth \
  --source-sha256 SOURCE_SHA256 \
  --checkpoint-sha256 CHECKPOINT_SHA256 \
  --height 16 --width 16
```

On Windows, use `.venv\Scripts\python.exe`. Dependencies are version-pinned but
their wheels are not hash-locked.

Dynamic height and width are the default. `--height` and `--width` select the
capture and first parity fixture, not a resolution ceiling. Export validates a
second odd, non-square size to catch accidental shape specialization. Use
`--fixed-shape` only for a capture-size feasibility fixture; its receipt marks
it incompatible with a model-catalog entry.

`export` executes the hash-pinned architecture file, loads the checkpoint in
PyTorch's tensor-only mode, lowers the released kernel=1 MMCV operation to
standard ONNX, validates both graphs, runs recurrent CPU parity, and writes the
hashes and results to `cda-vsr-export.json`. No MMCV installation is needed.

The lowering divides the 128 input channels into four deform groups. Each
group is sampled with standard ONNX `GridSample` using MMCV's interleaved
`[y, x]` offset order, multiplied by its modulation mask, concatenated, and
passed through the checkpoint's original 1×1 convolution.

## Recheck

Re-run PyTorch/ONNX sequence parity:

```sh
.venv/bin/python tools/cda-vsr/cda_tool.py parity \
  --source ../CDA-VSR \
  --checkpoint ../CDA-VSR/pretrained_models/best.pth \
  --source-sha256 SOURCE_SHA256 \
  --checkpoint-sha256 CHECKPOINT_SHA256 \
  --height 16 --width 16
```

Verify graph structure, metadata, receipt hashes, and the absence of custom
ONNX domains without loading upstream inputs:

```sh
.venv/bin/python tools/cda-vsr/cda_tool.py verify
```

For a one-time primitive comparison against the original implementation, use
an upstream-compatible CUDA/PyTorch environment with compiled `mmcv-full`:

```sh
python tools/cda-vsr/cda_tool.py dcn-parity --device cuda
```

MMCV is deliberately excluded from the portable requirements because its
binary must match the selected PyTorch and CUDA builds.

## Compare decoded priors

Evaluate the deterministic decoded-frame block matcher against zero priors and,
when available, CDA's processed motion/residual arrays:

```sh
.venv/bin/python tools/cda-vsr/cda_tool.py evaluate \
  --source ../CDA-VSR \
  --checkpoint ../CDA-VSR/pretrained_models/best.pth \
  --source-sha256 SOURCE_SHA256 \
  --checkpoint-sha256 CHECKPOINT_SHA256 \
  --previous previous.png \
  --current current.png \
  --true-motion current-mv.npy \
  --true-residual current-residual.npy \
  --ground-truth current-4x.png \
  --output-dir tmp/cda-vsr/evaluation
```

The offline matcher mirrors the current `decoded-cda-v1` provider defaults
(16-pixel blocks, 8-pixel search radius, and 4-pixel sampling stride), sampled
SAD edge penalty, tie order, float16 history snapshot, and dense residual
sampling. WGSL reduction arithmetic and browser video-to-texture conversion can
still differ at numerical ties. Use `--block-size`, `--search-radius`, and
`--sample-stride` to evaluate non-default provider settings.

True motion and residual must be supplied together. Their exact upstream
residual convention is undocumented, so that comparison requires CDA-VSR's
processed data. With ground truth and true priors, the report also tests the
provisional gate that decoded priors retain at least 60% of the true-prior PSNR
benefit.

## Current boundary

These are FP32 feasibility graphs—not extension assets. They have no artificial
source-resolution ceiling, although device memory, WebGPU limits, and latency
still impose practical bounds. Browser-provider testing at representative
video sizes, safe FP16 conversion, and performance work remain. Temporal
tiling is a possible later memory strategy, not a requirement for variable
source sizes. Regular parity proves that the graphs match the lowered PyTorch
network; `dcn-parity` separately tests the lowering against MMCV.

No checkpoint redistribution license has been established. The report marks
the generated files experimental/local-only, and the tool never adds them to
the extension's shipping model catalog.
