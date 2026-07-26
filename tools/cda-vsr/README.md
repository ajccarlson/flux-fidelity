# CDA-VSR conversion experiment

This offline toolkit adapts a user-supplied CDA-VSR source file and checkpoint
into two dynamic-spatial ONNX graphs:

- `cda-vsr-initializer.onnx`: RGB frame to 4× output plus two feature states.
- `cda-vsr-recurrent.onnx`: RGB, motion, residual, and prior states to 4× output
  plus updated states.

It does not download, copy, or redistribute upstream source or weights. Only
run conversion commands on code and weights obtained from a trusted location.
Review their provenance and redistribution terms before publishing any derived
model.

The hash-pinned reference is
[`sspBIT/CDA-VSR@5707d997`](https://github.com/sspBIT/CDA-VSR/tree/5707d997759996f19521c3beaddfb3d1ea965d44):
architecture SHA-256 `0defb80e5fcbaa2abd0eb9cbc4f4f2050a68e94fa6f743aa48a785cc734fd87b`
and checkpoint SHA-256 `afc8745b890289ae421c500279d9ccf2a27c92cf3e71133b20840c7816e86d3e`.
Its README names `LICENSE.txt`, but that file is absent at the pinned revision.

## Convert

First inspect the inputs. This reads and hashes the architecture and checkpoint;
it does not import the architecture or load the checkpoint:

```sh
python tools/cda-vsr/cda_tool.py inspect \
  --source ../CDA-VSR \
  --checkpoint ../CDA-VSR/pretrained_models/best.pth
```

The printed hashes identify the files but do not establish that they are safe,
authentic, or licensed. Conversion commands enforce the canonical hashes above
by default. With CPython 3.11 or newer, create an isolated environment and
export:

```sh
python -m venv .venv
.venv/bin/python -m pip install -r tools/cda-vsr/requirements.txt
.venv/bin/python tools/cda-vsr/cda_tool.py export \
  --source ../CDA-VSR \
  --checkpoint ../CDA-VSR/pretrained_models/best.pth \
  --height 16 --width 16
```

On Windows, use `.venv\Scripts\python.exe`. Dependencies are version-pinned but
their wheels are not hash-locked.

Dynamic height and width are the default. `--height` and `--width` select the
capture and first parity fixture, not a resolution ceiling. Export validates a
second odd, non-square size to catch accidental shape specialization. Use
`--fixed-shape` only for a capture-size feasibility fixture; its receipt marks
it incompatible with a model-catalog entry.

`export` imports and executes the selected architecture as Python code without
a sandbox, loads the checkpoint in PyTorch's tensor-only mode, lowers the
released kernel=1 MMCV operation to standard ONNX, validates both graphs, runs
recurrent CPU parity, and writes the input, tool, graph, and parity identities
to `cda-vsr-export.json`. No MMCV installation is needed.

To work with another source/checkpoint pair that you have independently
reviewed, add `--source-sha256 <exact-hash>`,
`--checkpoint-sha256 <exact-hash>`, and `--allow-unpinned-inputs`. That flag
acknowledges departure from the canonical reference; it does not make the inputs
trustworthy.

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
  --height 16 --width 16
```

Verify local graph bytes and structure against the receipt, the exact runtime
ABI and metadata, parity-receipt self-consistency, current toolkit identity,
and the absence of custom ONNX domains without loading upstream inputs:

```sh
.venv/bin/python tools/cda-vsr/cda_tool.py verify
```

Receipts are deliberately tied to the exact toolkit files that produced them.
Verification checks local self-consistency; it does not authenticate the
original inputs or establish provenance, safety, or licensing.

## Probe the graphs in Chromium

After `verify` succeeds, run the external graphs through the real extension
integration without adding them to the repository:

```sh
npm run probe:cda-browser -- --onnx-dir tmp/cda-vsr/onnx
```

Set `FSRCNNX_BROWSER` when browser auto-detection is insufficient. The probe
rechecks the dynamic, local-only export receipt and exact graph hashes, builds
the normal package into an operating-system temporary directory, appends a
local CDA-VSR manifest entry, selects that exact model for real-video
validation, requires successful initializer and recurrent executions, and
removes the temporary package afterward. It never changes the shipping neural
manifest or copies the graphs into the repository.

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

These are FP32 feasibility graphs—not extension assets. Dynamic shapes remove
exact-size specialization, not physical WebGPU limits. At a 1280×720 source,
each 64-channel FP32 state is 225 MiB and the 4× RGB output is 168.75 MiB, so
individual tensors exceed the common 128 MiB storage-binding limit. A validated
tiling/state strategy, FP16 conversion, or model redesign—and representative
browser-provider tests—is therefore required before catalog inclusion. This is
a hardware/runtime blocker, not an artificial source-resolution policy.

Regular parity checks sampled numerical agreement between the graphs and the
lowered PyTorch network; `dcn-parity` separately samples the lowering against
MMCV.

`npm run check:cda-feasibility` runs the dependency-free contract tests and is
part of the normal repository check. The optional ML environment and actual
source/checkpoint conversion remain explicit local validation steps.

No license for the architecture source, nor checkpoint license or redistribution
clearance, has been established. The receipt marks generated files
experimental/local-only, and the tool never adds them to the extension's
shipping model catalog.
