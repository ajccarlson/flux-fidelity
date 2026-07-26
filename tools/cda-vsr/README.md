# CDA-VSR conversion experiment

This offline toolkit adapts a user-supplied CDA-VSR source file and checkpoint
into two fixed-shape ONNX graphs:

- `cda-vsr-initializer.onnx`: RGB frame to 4× output plus two feature states.
- `cda-vsr-recurrent.onnx`: RGB, motion, residual, and prior states to 4× output
  plus updated states.

It does not download, copy, or redistribute upstream source or weights. Review
their provenance and redistribution terms before publishing any derived model.

## Convert

First inspect the inputs and copy the two printed hashes:

```sh
python tools/cda-vsr/cda_tool.py inspect \
  --source ../CDA-VSR \
  --checkpoint ../CDA-VSR/pretrained_models/best.pth
```

With CPython 3.11 or newer, create an isolated environment and export a small
parity fixture:

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

`export` executes the hash-pinned architecture file, loads the checkpoint in
PyTorch's tensor-only mode, replaces only the released kernel=1 MMCV
deformable convolution, checks both ONNX graphs, runs recurrent CPU parity, and
writes `cda-vsr-export.json`, a machine-readable report containing input and
output hashes, fixed tensor shapes, opset, parity tolerances/results, and the
future decoded-prior contract `decoded-cda-v1`. No MMCV installation is needed
for conversion.

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

True motion and residual must be supplied together. Their exact upstream
residual convention is undocumented, so that comparison requires CDA-VSR's
processed data. With ground truth and true priors, the report also evaluates
the provisional gate that decoded priors retain at least 60% of the true-prior
PSNR benefit. This CPU block matcher is a deterministic reference, not the
future WebGPU implementation.

## Current boundary

These are fixed-shape, FP32 feasibility graphs—not extension assets. The tool
does not yet implement temporal tiling, safe FP16 conversion, or browser
runtime inference. Its offline evaluator generates motion/residual proxies, but
does not represent a production estimator. The regular parity check proves the
exported graphs match the lowered PyTorch network; `dcn-parity` separately
tests that lowering against MMCV.

No checkpoint redistribution license has been established. The report marks
the generated files experimental/local-only, and the tool never adds them to
the extension's shipping model catalog.
