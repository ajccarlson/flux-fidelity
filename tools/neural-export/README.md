# Neural model export

`export.py` converts a Spandrel-supported RGB super-resolution checkpoint to
the extension ABI: one dynamic NCHW float32 input and output, with float16
weights by default. It checks the graph, runs a CPU comparison against the
source model, lowers unsupported PReLU nodes to audited ORT WebGPU operators,
updates `model/neural/manifest.json`, and prints the output hash.

Use an isolated environment:

```sh
python -m venv .venv
.venv/bin/python -m pip install -r tools/neural-export/requirements.txt
```

Then export with the expected source identity pinned:

```sh
.venv/bin/python tools/neural-export/export.py \
  --pth checkpoint.pth \
  --checkpoint-sha256 SHA256 \
  --expected-scale 2 \
  --key model-id \
  --label "Model label" \
  --source-url https://example.com/checkpoint.pth
```

On Windows, use `.venv\Scripts\python.exe`. Verify the checkpoint's provenance
and redistribution license before placing the generated model in a release.

## Bundled Real-ESRGAN model

The checked export used CPython 3.12.10 on Windows x64 and the exact versions
in `requirements.txt`. Those versions are pinned, but their wheels are not
hash-locked.

```sh
python tools/neural-export/export.py \
  --pth RealESRGANv2-animevideo-xsx2.pth \
  --checkpoint-sha256 27985aa2198711ecd72f9bb274ec7b164e018fc9ce2933daaa7c7ab36a2bd3fe \
  --expected-scale 2 \
  --expected-arch "RealESRGAN Compact" \
  --source-url https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.3.0/RealESRGANv2-animevideo-xsx2.pth \
  --key realesrganv2_animevideo_xsx2 \
  --label "Real-ESRGAN AnimeVideo XS 2x"
```

Two clean runs produced SHA-256
`f674a410b528aec55bb9f9f594cb1aaea580237adb29abd9dc32296d34b690a0`.
