# Neural model export

This tool converts a Spandrel-supported PyTorch checkpoint into a dynamically sized ONNX model for FSRCNNX-EXT. Exported models use float32 input/output tensors; weights may be converted to float16.

Create an isolated Python environment and install the pinned dependencies:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r tools/neural-export/requirements.txt
```

Export a checkpoint:

```sh
python tools/neural-export/export.py --pth checkpoint.pth --key model-id --label "Model label"
```

Output is written to `model/neural/`, and its entry is added to `model/neural/manifest.json`. Check the checkpoint's provenance and redistribution license before adding the generated ONNX file to a release boundary. No neural model is bundled by default.
