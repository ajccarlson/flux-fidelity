# Third-party notices

FSRCNNX-EXT includes or derives from third-party software and model artifacts. This file records licensing information; it does not replace the upstream license texts.

## ONNX Runtime Web 1.27.0

The files under `vendor/ort/` are distributed by Microsoft as part of ONNX Runtime and are licensed under the MIT License. The bundled JavaScript header identifies version 1.27.0.

Upstream: <https://github.com/microsoft/onnxruntime>

## ArtCNN

The ArtCNN shader weights represented by the generated files under `model/ArtCNN_*` originate from Artoriuz/ArtCNN, which is licensed under the MIT License.

Upstream: <https://github.com/Artoriuz/ArtCNN>

## RIFE

RIFE is developed by hzwer and contributors. The upstream implementation is licensed under the MIT License. The exact lineage of each bundled ONNX export is recorded separately in `MODEL_PROVENANCE.md`; an upstream code license does not automatically establish redistribution rights for an independently obtained model file.

Upstream: <https://github.com/hzwer/ECCV2022-RIFE>

## FSRCNNX and shader ports

Generated FSRCNNX weights and the SSimDownscaler, adaptive-sharpen, and deband ports are derived from community mpv/libplacebo shader implementations. Their precise source revisions and redistribution terms must be recorded in `MODEL_PROVENANCE.md` before a public binary release.
