import { GENERATED_MODEL_ASSET_PATHS } from "../src/core/fsrcnnx-model-catalog.js";

export const EXPECTED_PACKAGE_FILE_COUNT = 91;

// These files let recipients inspect, rebuild, and substitute the LGPL-covered
// shader portions and retain exact licenses for bundled third-party components.
// Keep the inventory separate so package tests can enforce it explicitly.
export const REQUIRED_COMPLIANCE_FILES = Object.freeze([
  "LICENSES/GPL-3.0.txt",
  "LICENSES/LGPL-3.0.txt",
  "LICENSES/Real-ESRGAN-BSD-3-Clause.txt",
  "docs/compliance/LGPL_REBUILDING.md",
  "shaders/upstream/FSRCNNX_x2_16-0-4-1.glsl",
  "shaders/upstream/FSRCNNX_x2_56-16-4-1.glsl",
  "shaders/upstream/SSimDownscaler.glsl",
  "shaders/upstream/adaptive-sharpen.glsl",
  "tools/package.json",
  "tools/transpile.js",
  "vendor/ort/LICENSE",
]);

// These assets are selected indirectly from runtime model catalogs, so static
// import/getURL scanning cannot prove their exact package membership. Keep this
// list independent of PACKAGE_FILES so an accidental allowlist omission fails
// validation instead of degrading only the affected model at runtime.
export const REQUIRED_RUNTIME_MODEL_FILES = Object.freeze([
  ...GENERATED_MODEL_ASSET_PATHS,
  "model/neural/cda-vsr-initializer.onnx",
  "model/neural/cda-vsr-recurrent.onnx",
  "model/neural/realesrganv2_animevideo_xsx2.fp16.onnx",
  "model/rife_v4.26.onnx",
  "model/rife_v4.26_fp16.onnx",
]);

// This list is the package boundary. Additions and removals must be deliberate:
// package creation and package reference validation both consume this exact set.
export const PACKAGE_FILES = Object.freeze([
  "LICENSE",
  "LICENSES/GPL-3.0.txt",
  "LICENSES/LGPL-3.0.txt",
  "LICENSES/Real-ESRGAN-BSD-3-Clause.txt",
  "NOTICE",
  "PRIVACY.md",
  "THIRD_PARTY_NOTICES.md",
  "docs/compliance/LGPL_REBUILDING.md",
  "docs/compliance/MODEL_PROVENANCE.md",
  "icons/icon-128.png",
  "icons/icon-16.png",
  "icons/icon-32.png",
  "icons/icon-48.png",
  "icons/icon-off-128.png",
  "icons/icon-off-16.png",
  "icons/icon-off-32.png",
  "icons/icon-off-48.png",
  "manifest.json",
  "model/ArtCNN_C4F32.artcnn.json",
  "model/ArtCNN_C4F32.artcnn.wgsl",
  "model/ArtCNN_C4F32_DN.artcnn.json",
  "model/ArtCNN_C4F32_DN.artcnn.wgsl",
  "model/ArtCNN_C4F32_DS.artcnn.json",
  "model/ArtCNN_C4F32_DS.artcnn.wgsl",
  "model/FSRCNNX_x2_16-0-4-1.passes.json",
  "model/FSRCNNX_x2_16-0-4-1.wgsl",
  "model/FSRCNNX_x2_56-16-4-1.passes.json",
  "model/FSRCNNX_x2_56-16-4-1.wgsl",
  "model/neural/cda-vsr-initializer.onnx",
  "model/neural/cda-vsr-recurrent.onnx",
  "model/neural/manifest.json",
  "model/neural/realesrganv2_animevideo_xsx2.fp16.onnx",
  "model/rife_v4.26.onnx",
  "model/rife_v4.26_fp16.onnx",
  "popup.html",
  "shaders/upstream/FSRCNNX_x2_16-0-4-1.glsl",
  "shaders/upstream/FSRCNNX_x2_56-16-4-1.glsl",
  "shaders/upstream/SSimDownscaler.glsl",
  "shaders/upstream/adaptive-sharpen.glsl",
  "src/background.js",
  "src/content.js",
  "src/core/fsrcnnx-artcnn-runtime.js",
  "src/core/fsrcnnx-cda-priors.js",
  "src/core/fsrcnnx-color-support.js",
  "src/core/fsrcnnx-color.js",
  "src/core/fsrcnnx-frame-signature.js",
  "src/core/fsrcnnx-gpu-timing.js",
  "src/core/fsrcnnx-grab.js",
  "src/core/fsrcnnx-images.js",
  "src/core/fsrcnnx-interpolate.js",
  "src/core/fsrcnnx-main.js",
  "src/core/fsrcnnx-model-bundle.js",
  "src/core/fsrcnnx-model-catalog.js",
  "src/core/fsrcnnx-neural-frame-bridge.js",
  "src/core/fsrcnnx-neural-temporal-atlas.js",
  "src/core/fsrcnnx-neural-temporal-tiling.js",
  "src/core/fsrcnnx-neural.js",
  "src/core/fsrcnnx-performance.js",
  "src/core/fsrcnnx-rife-gpu.js",
  "src/core/fsrcnnx-rife.js",
  "src/core/fsrcnnx-runtime.js",
  "src/core/fsrcnnx-setting-contract.js",
  "src/core/fsrcnnx-settings-store.js",
  "src/core/fsrcnnx-sharpen.js",
  "src/core/fsrcnnx-ssimds-runtime.js",
  "src/core/fsrcnnx-ssimds.js",
  "src/core/fsrcnnx-video-controller.js",
  "src/frame/neural-frame-runtime.js",
  "src/frame/neural-frame.html",
  "src/popup.js",
  "tools/package.json",
  "tools/transpile.js",
  "validate.html",
  "validation/README.md",
  "validation/fsrcnnx-reference-validation.js",
  "validation/fsrcnnx-validation.js",
  "validation/reference-fixtures.js",
  "validation/reference-fixtures.json",
  "validation/references/ArtCNN_C4F32.rgb16le",
  "validation/references/ArtCNN_C4F32_DN.rgb16le",
  "validation/references/ArtCNN_C4F32_DS.rgb16le",
  "validation/references/FSRCNNX_x2_16-0-4-1.rgb16le",
  "validation/references/FSRCNNX_x2_56-16-4-1.rgb16le",
  "validation/references/sharpen.rgb16le",
  "validation/references/ssimds.rgb16le",
  "validation/validate.js",
  "vendor/ort/LICENSE",
  "vendor/ort/ThirdPartyNotices.txt",
  "vendor/ort/ort-wasm-simd.asyncify.mjs",
  "vendor/ort/ort-wasm-simd.asyncify.wasm",
  "vendor/ort/ort.webgpu.min.mjs",
]);

function assertPackageBoundary() {
  if (PACKAGE_FILES.length !== EXPECTED_PACKAGE_FILE_COUNT) {
    throw new Error(`Package allowlist must contain exactly ${EXPECTED_PACKAGE_FILE_COUNT} entries`);
  }
  if (new Set(PACKAGE_FILES).size !== PACKAGE_FILES.length) {
    throw new Error("Package allowlist contains duplicate entries");
  }
  if (PACKAGE_FILES.some((file) =>
    !file || file.startsWith("/") || file.includes("\\") || file.split("/").includes("..")
  )) {
    throw new Error("Package allowlist entries must be normalized relative paths");
  }
  const sorted = [...PACKAGE_FILES].sort();
  if (PACKAGE_FILES.some((file, index) => file !== sorted[index])) {
    throw new Error("Package allowlist must remain sorted");
  }
  const sortedRuntimeModels = [...REQUIRED_RUNTIME_MODEL_FILES].sort();
  if (new Set(REQUIRED_RUNTIME_MODEL_FILES).size !== REQUIRED_RUNTIME_MODEL_FILES.length ||
      REQUIRED_RUNTIME_MODEL_FILES.some((file, index) => file !== sortedRuntimeModels[index])) {
    throw new Error("Required runtime model files must be unique and sorted");
  }
  for (const file of REQUIRED_RUNTIME_MODEL_FILES) {
    if (!PACKAGE_FILES.includes(file)) {
      throw new Error(`Package allowlist is missing required runtime model asset ${file}`);
    }
  }
  const sortedComplianceFiles = [...REQUIRED_COMPLIANCE_FILES].sort();
  if (new Set(REQUIRED_COMPLIANCE_FILES).size !== REQUIRED_COMPLIANCE_FILES.length ||
      REQUIRED_COMPLIANCE_FILES.some((file, index) => file !== sortedComplianceFiles[index])) {
    throw new Error("Required compliance files must be unique and sorted");
  }
  for (const file of REQUIRED_COMPLIANCE_FILES) {
    if (!PACKAGE_FILES.includes(file)) {
      throw new Error(`Package allowlist is missing required compliance material ${file}`);
    }
  }
}

assertPackageBoundary();
