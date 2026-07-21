import { GENERATED_MODEL_ASSET_PATHS } from "../fsrcnnx-model-catalog.js";

export const EXPECTED_PACKAGE_FILE_COUNT = 59;

// These files let recipients inspect, rebuild, and substitute the LGPL-covered
// shader portions and retain the exact license for the vendored ONNX Runtime.
// Keep the inventory separate so package tests can enforce it explicitly.
export const REQUIRED_COMPLIANCE_FILES = Object.freeze([
  "LGPL_REBUILDING.md",
  "shaders/upstream/FSRCNNX_x2_16-0-4-1.glsl",
  "shaders/upstream/SSimDownscaler.glsl",
  "transpile.js",
  "vendor/ort/LICENSE",
]);

// These assets are selected indirectly from runtime model catalogs, so static
// import/getURL scanning cannot prove their exact package membership. Keep this
// list independent of PACKAGE_FILES so an accidental allowlist omission fails
// validation instead of degrading only the affected model at runtime.
export const REQUIRED_RUNTIME_MODEL_FILES = Object.freeze([
  ...GENERATED_MODEL_ASSET_PATHS,
  "model/rife_v4.26.onnx",
  "model/rife_v4.26_fp16.onnx",
]);

// This list is the package boundary. Additions and removals must be deliberate:
// package creation and package reference validation both consume this exact set.
export const PACKAGE_FILES = Object.freeze([
  "GPL-3.0.txt",
  "LGPL-3.0.txt",
  "LGPL_REBUILDING.md",
  "LICENSE",
  "MODEL_PROVENANCE.md",
  "THIRD_PARTY_NOTICES.md",
  "background.js",
  "content.js",
  "fsrcnnx-artcnn-runtime.js",
  "fsrcnnx-color.js",
  "fsrcnnx-grab.js",
  "fsrcnnx-images.js",
  "fsrcnnx-interpolate.js",
  "fsrcnnx-main.js",
  "fsrcnnx-model-bundle.js",
  "fsrcnnx-model-catalog.js",
  "fsrcnnx-neural.js",
  "fsrcnnx-rife-gpu.js",
  "fsrcnnx-rife.js",
  "fsrcnnx-runtime.js",
  "fsrcnnx-settings-store.js",
  "fsrcnnx-sharpen.js",
  "fsrcnnx-ssimds-runtime.js",
  "fsrcnnx-ssimds.js",
  "fsrcnnx-validation.js",
  "fsrcnnx-video-controller.js",
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
  "model/neural/manifest.json",
  "model/rife_v4.26.onnx",
  "model/rife_v4.26_fp16.onnx",
  "popup.html",
  "popup.js",
  "release-clearance.json",
  "shaders/upstream/FSRCNNX_x2_16-0-4-1.glsl",
  "shaders/upstream/SSimDownscaler.glsl",
  "transpile.js",
  "validate.html",
  "validate.js",
  "vendor/ort/LICENSE",
  "vendor/ort/ThirdPartyNotices.txt",
  "vendor/ort/ort-wasm-simd-threaded.asyncify.mjs",
  "vendor/ort/ort-wasm-simd-threaded.asyncify.wasm",
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
