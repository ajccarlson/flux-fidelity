# FSRCNNX-EXT

FSRCNNX-EXT is a Chromium extension for real-time video enhancement with WebGPU. It combines FSRCNNX and ArtCNN upscaling, optional SSimDownscaler, debanding and sharpening filters, GPU-based RIFE frame interpolation, and experimental ONNX super-resolution support.

## Status

This project is pre-release and under active reconstruction. It currently targets Chromium browsers with WebGPU and non-DRM video sources that the page permits the extension to read. The bundled neural super-resolution model is a random-weight smoke-test model and is not intended for normal viewing.

## Project structure

- `fsrcnnx-main.js` coordinates video discovery, rendering, settings, and feature modules.
- `fsrcnnx-runtime.js` and `fsrcnnx-artcnn-runtime.js` execute generated WGSL model passes.
- `fsrcnnx-interpolate.js`, `fsrcnnx-rife.js`, and `fsrcnnx-rife-gpu.js` implement frame interpolation.
- `fsrcnnx-neural.js` runs full-RGB ONNX super-resolution models through ONNX Runtime Web.
- `model/` contains generated shader manifests, WGSL programs, and runtime models.
- `transpile.js`, `transpile-artcnn.js`, and `tools/neural-export/` generate runtime assets from source models.

## Development workflow

Create feature branches from `develop`, merge completed features into `develop`, and promote tested releases from `develop` to `main`.

```text
feature/* -> develop -> main
```

Do not commit credentials, internal planning documents, raw training checkpoints, or local validation output.
